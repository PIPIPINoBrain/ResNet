### 使用pytorch训练测试自己的数据，并将训练好的分类器封装成类以供调用。本项目选择的训练模型是官方提供的resnet50，任务为对箭头和轮毂以及锈斑进行分类。可以通过这套代码训练任何自己需要的数据。

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

### little trick
已有数据的可以不用看这个文件
数据截取增广，对图像进行点标注，旋转截取一定范围数据，本项目是要分辨箭头，采用这种方案，
arrow_get.py 包含两种方式，一种先旋转后截取，一种先截取再旋转，
数据使用需要更改下路径，具体使用看程序中的注释，给出了项目中图像以及json标注文件的示例

### 项目使用
将数据按照分类以及准备数据的要求存储，
对create_label_files.py文件的get_dile_list分类进行更改并运行；
对train.py文件里的num_class进行修改并运行
predict.py文件也需要进行修改，可根据自己要求进行改写

### 权重
链接：https://pan.baidu.com/s/1rMMW0hTzgvK6s-O6xUz60w 
提取码：mab6
