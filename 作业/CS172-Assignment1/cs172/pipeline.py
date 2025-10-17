import torch
from tqdm.notebook import tqdm

from torchmetrics import MetricCollection
from cs172.metrics import ImageAccuracy, DigitAccuracy

def train(model, device, dataloader, lr = 1e-3, weight_decay = 0.05, num_epoch = 10):
    metric_collection = MetricCollection({
        "image_accuracy": ImageAccuracy(),
        "digit_accuracy": DigitAccuracy()
    }).to(device)
    

    model.to(device)
    model.train()

    # ================== TO DO START ====================
    # define the optimizer and loss_func
    # Adam is a recommended optimizer, you can try different learning rate and weight_decay
    # You can use cross entropy as a loss function
    # ===================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.CrossEntropyLoss()
    # =================== TO DO END =====================
    
    # If you implement the previous code correctly, 10 epoch should be enough
    for epoch in range(num_epoch):
        print(f"Starting epoch {epoch}")
        sum_loss = 0
        
        
        # metric_collection.reset()  # 重要：每个epoch重置指标
        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)
            # ================== TO DO START ====================
            # Get the prediction through the model and call the optimizer
            # ===================================================
            pred = model(img) # passforward
            B = pred.shape[0]
            pred_digits = pred.view(B, 5, 10)          # [B,5,10]
            pred_for_loss = pred_digits.reshape(-1, 10)  # [B*5, 10]
            label_digits = label.argmax(dim=2)         # [B,5]
            label_for_loss = label_digits.reshape(-1)  # [B*5]
            loss= loss_func(pred_for_loss, label_for_loss)
            


            optimizer.zero_grad()  # 清除梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数
            
            # =================== TO DO END =====================
            sum_loss += loss.item()
            metric_collection.update(pred, label)
        print(f"loss for epoch {epoch}:", sum_loss/len(dataloader))
        for key, value in metric_collection.compute().items():
            print(f"{key} for epoch {epoch}:", value.item())



def test(model, device, dataloader):
    metric_collection = MetricCollection({
        "image_accuracy": ImageAccuracy(),
        "digit_accuracy": DigitAccuracy()
    }).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img, label = img.to(device), label.to(device)
            pred = model(img)
            metric_collection.update(pred, label)
    return metric_collection.compute()
