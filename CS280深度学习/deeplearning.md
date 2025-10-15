# Artificial neuron

神经元：

![image-20250220161321917](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20250220161321917.png)

输入特征点乘权重，再放入非线性系统计算，最后得到输出。

- 如何找到感知机最好的权重w参数：

预测：符号函数：sign函数，w^Tx正的为1，负的为-1

![image-20250220161715354](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20250220161715354.png)

这里的wT相当于plane方向向量，点到直线的距离公式

**感知机Perceptron 算法：**

1. w:初始化为0

2. 预测是正是负

3. 预测错了：

   -  原本应该是正的，预测错成负的+=x
   - 负的-=x

   如何理解：根据点击公式展开里面的cos

4. 

为什么最终一定都会分类对:

两个claim：先假设存在最优解w*

1. ![image-20250220111001344](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20250220111001344.png)

   

   从而能推出w会越来越趋近于w*

单层神经网络

每个神经元都有一个权重限量，最终形成一个输出向量。

1. independent feature

2. competition![image-20250220111251288](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20250220111251288.png)

3. 

   softmax：![image-20250220111304916](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20250220111304916.png)

   每个神经元的权重分布

损失函数：

![image-20250220112207930](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20250220112207930.png)

![image-20250220112428820](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20250220112428820.png)



L1正则化：权重松散，模型capacity大，要解决就要减少模型的特征，那么就是把一些权重设置为0.（underfit）

L2正则化：模型对噪声很敏感。使任何一个特征的权重变小，那么就不会有任何一个模型的噪声造成很大的影响。（overfit）

w也能可视化成图像，如果w与x的图像更相似，那么就能划入这一类。