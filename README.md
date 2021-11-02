### 使用pytorch训练测试自己的数据，并将训练好的分类器封装成类以供调用。本项目选择的训练模型是官方提供的resnet50，任务为对汽车类型分类（SUV和轿车）。可以通过这套代码训练任何自己需要的数据。

本项目基于以下版本可以通过测试：  
pytorch == 1.0.0  
opencv == 3.4.0

### 准备数据
准备好自己的数据集，按照以下目录结构进行保存。  
|--data  
|　|--train  
|　|　|--class0  
|　|　|　|--img0.jpg  
|　|　|　|--img1.jpg  
|　|　|　|--...  
|　|　|--class1  
|　|　|　|--img0.jpg  
|　|　|　|--img1.jpg  
|　|　|　|--...  
|　|--test  
本项目所用的汽车数据集已经上传到了百度网盘，如有需要，可以自行下载并解压到data目录下。该数据集包含了SUV和轿车两类车型，训练数据各有1400张，测试数据各有30张。  
百度网盘  
链接:https://pan.baidu.com/s/1PvDT_XesSQqE6RgQE3Z2Wg  密码:a8te  


### 训练数据
1.使用GenList.py脚本生成训练数据的list，可以通过设置rate的值来分配训验证集占总的数据集的比例。本项目设置的rate=0.1，即2800张图像中选取了280张作为验证集。  
2.使用train.py训练数据，最终得到的模型文件保存在model目录下。我使用resnet50进行训练，读者可以自行更换训练网络。由于我的训练数据比较少，而分类的图片类型比较复杂，最终的验证集精度为88.9%。

### 使用模型
为了方便使用，我将调用模型的代码封装成了一个类以供调用，用时只需要进行初始化和调用两个步骤，具体代码可参考prediction.py文件，使用该类进行预测时，输入为opencv读取的numpy格式的图像，输出是分类结果。通过运行prediction脚本，我们可以看到，使用训练的模型进行预测正确率可达到85%以上。我提供了训练好的模型供大家测试使用，模型下载地址  
百度网盘  
链接:https://pan.baidu.com/s/1sPC2LAJE2cyjj9youyPhyw  密码:tqrz

### fine-tunning
可以通过设置'pretrained'和'pretrained_model'两个选项来加载预训练模型，本项目提供了官方提供的resnet50预训练模型。  
百度网盘  
链接:https://pan.baidu.com/s/1-MyBSPE5r3QyG7lBGaPX-Q  密码:jq73  
使用预训练模型进行fine-tuning后，训练速度明显提高，验证集精度从原来的88.9%提升到了99.6%，效果提升明显。读者可以下载微调后的训练模型与之前的训练模型进行比较。  
百度网盘  
链接:https://pan.baidu.com/s/14mrtEh33iMoXt0VgyqD15w  密码:avgy

### CenterLoss
增加CenterLoss函数以实现自定义损失函数功能，读者可以用train_centerloss.py脚本进行训练，经测试，训练精度不如softmax。  
CenterLoss参考代码  
[MNIST_center_loss_pytorch](https://github.com/jxgu1016/MNIST_center_loss_pytorch)

### TODO
~~1.增加加载预训练模型以实现fine-tune功能。~~  
~~2.增加centerloss函数以实现自定义损失函数功能。~~  
3.增加SGD梯度下降训练方法。
