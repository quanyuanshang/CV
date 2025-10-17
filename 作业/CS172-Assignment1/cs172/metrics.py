import torch
from torchmetrics import Metric

# Total Accuracy for identify one image 
class ImageAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.total += target.size(0)
        preds = preds.reshape(-1, 5, 10).argmax(axis=-1)
        target = target.reshape(-1, 5, 10).argmax(axis=-1)
        cmp = (preds.type(target.dtype) == target).sum(axis=-1)
        cmp = cmp == torch.ones_like(cmp, device=cmp.device) * 5
        self.correct += int(cmp.type(target.dtype).sum())

    def compute(self):
        return self.correct / self.total
    
# Digit accuracy in all images
class DigitAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        self.total += target.size(0) * 5
        preds = preds.reshape(-1, 10).argmax(axis=-1)
        target = target.reshape(-1, 10).argmax(axis=-1)
        cmp = preds.type(target.dtype) == target
        self.correct += int(cmp.type(target.dtype).sum())

    def compute(self):
        return self.correct / self.total