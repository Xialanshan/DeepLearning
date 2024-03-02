#### 项目简介
1. 项目名称: Style Transfer 图像风格迁移任务
2. 参考论文: A Neural Algorithm of Artistic Style（CVPR 2015）
3. 项目概要: 本项目依据上述论文，旨在探索深度学习在图像处理领域的应用，重点在于用卷积神经网络 (CNN) 实现图像风格迁移任务

#### 具体任务
 1. 准备图片数据集：包括模型训练的特定图片和拓展功能使用的微型图片数据集（Image_mini、Image)
 2. 模型选择和实现: 原文使用的模型基于VGG19模型，本项目在复现的基础上，尝试替换预训练模型，最终选定InceptionV3模型，并加入了总变差损失来更新优化目标 (Model)
 3. 开发拓展功能: 本项目尝试了基于权重的多图片风格迁移和随机风格迁移功能 (Extend)
   

#### 备注
1. 模型调整选择过程在layers文件夹中
2. texture_transfer.py文件实现了原文中提到的纹理迁移，用来进行效果对比
3. 此文件即为简略，只提供对项目的基本介绍

#### 效果

##### 新模型效果
![image](https://github.com/Xialanshan/DeepLearning/assets/110965468/d7785fbc-702b-44d9-b04a-7cd34a6ae8bd)

（左图为内容图片，右图为风格图片）

![image](https://github.com/Xialanshan/DeepLearning/assets/110965468/eeac993e-72af-4fed-a7ab-482b5fce75df)

（图中红框标记处效果对比较为明显）

##### 拓展功能一效果

![image](https://github.com/Xialanshan/DeepLearning/assets/110965468/c8d1d9c0-2964-486f-8a40-227474ad394a)

![image](https://github.com/Xialanshan/DeepLearning/assets/110965468/07863f8d-a1db-4168-be08-94f569c787af)


##### 拓展功能二效果

![image](https://github.com/Xialanshan/DeepLearning/assets/110965468/2ad712e7-fc03-47c5-b6cc-cdaaae3c674b)



