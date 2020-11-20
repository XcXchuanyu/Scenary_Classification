## Scenary Calssification
This task is the final work of DSA5203 in NUS, we need to classify 15 category pics,
and in each category there are 100 pictures.

We use keras in Tensorflow2.0 to build our model.

### How to run it? 

train command in CMD:

```shell script
python .\scene_recog_cnn.py --phase="train" --train_data_dir=".\data\train\" --test_data_dir='.\data\test\' --model_dir='.\'
```

train command in CMD:

```shell script
python .\scene_recog_cnn.py --phase="train" --train_data_dir=".\data\train\" --test_data_dir='.\data\test\' --model_dir='.\'
```



<div align=center>![image](./readme_file/image4.png)</div>

<div align=center>
<img src="./readme_file/image4.png" width="100" height="100" />
</div>

### Our procedure
1. Dataset Clean

The existence of dirty data often affects the classification accuracy of the model, so the first thing we do is to clean out the dirty data in the dataset provided. In fact, in our data set, many scenes have high similarity, so it is difficult to distinguish them by simple CNN, and it is even difficult for human eyes to distinguish the corresponding pictures of these scenes. For example, the image_0271.jpg in OpenCountry can also be classified as forest.

![image](./readme_file/image1.jpg)

2. Data augmentation



BrightnessEnhancement	1.5 times

ContrastEnhancement	1.5 times

RandomRotation	Randomly in [-90, +90, 180]

Flip	Filp image in left and right

3. Model design

![image](./readme_file/image2.png)

In addition, optimizer "Adam" and Loss Function "categorical_crossentropy" is recognized choice of multiple classification problems.
Besides we use dropout function to prohibit overfitting problem.


4.	Result Analysis

We randomly choose 15% of our picture from dataset in each category to seperated as train_dataset and validation_dataset.

The final accuracy of our model can be seen as below. Here we randomly choose 15% of image as our validation data. With the procession of training, we obtained an accuracy of 93% on training set and 60% on the validation set. There are two reasons for the low accuracy of the prediction of validation set. The main reason is because the number of images of the training set is not enough, which leads to overfitting. Another reason is because of the similarity of different categories，such as forest and OpenCountry, bedroom and living room.

![image](./readme_file/image3.png)

