Sneaker-Classifier
====

A model that can be used to classify the brand of sneakers

Hi! This is Xuetao Liu and this project is about how I do the following
  * Collect the images of different brands of sneakers
  * How to import them into AutoML Vision
  * After finishing training the model, we export them as
    * CoreML model
    * TensorFlow Lite model
    
## Collecting the dataset


As we all know that the quality of our dataset determines the quality of our model, so it is indeed the crucial part in conducting our project. We decide to use Google Images to collect our dataset, with the help of useful tool [`gi2ds`](https://github.com/toffebjorkskog/ml-tools/blob/master/gi2ds.md#bookmarklet) created by **Christoffer Björkskog**, which can let us easily collect the url link of images shown in the Google Images and place them into our txt file. Below is the video demo about how I use this tool to collect the images.[![Figure1](https://i.ibb.co/VJc7BTt/2020-01-3117-19-41.png)](https://youtu.be/DMCR8EVWK7A) 

### Download the images from the txt file

With the help of [`Fast.ai`](https://course.fast.ai/start_colab.html) library, we can easily download the image using the function `download_images` to download the images from our txt file automatically. Here is the notebook I create the actually shows how to implement the relavant procedures. [https://colab.research.google.com/drive/15TOopbc8evJvt4sJVXzpPtfYyUGSHXLi](https://colab.research.google.com/drive/15TOopbc8evJvt4sJVXzpPtfYyUGSHXLi).

To doing so, we first need to import the google drive and fast.ai library

```
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

import os
os.chdir("/content/gdrive/My Drive/Colab Notebooks/FinalProject/Data/images/")
!ls
```
```
from google.colab import auth
auth.authenticate_user()

project_id = 'sneaker-project-264720'
!gcloud config set project {project_id}
!gsutil ls
```
```
from fastai.vision import *
import matplotlib.pyplot as plt
import numpy as np
```
and set up the base directory
```
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'Colab Notebooks/FinalProject/Data/'
path = Path(base_dir + 'images') # The Path function create a folder named "images"
```

After creating the folders, we place our txt files into the the corresponding folder. Then we come to download the images into our folders **Remember to use the same category name as your txt file, otherwise it would gets wrong.**
```
classes = ['424_dipped', 'Adidas_yeezy_350', 'Adidas_yeezy_500', 'Adidas_yeezy_700', 
           'Adidas_nmd', 'Adidas_ozweego', 'Adidas_raf_simons', 'Adidas_stan_smith',
           'Adidas_superstar', 'Adidas_ultraboost', 'Adidas_zx', 'airjordan_1',
           'airjordan_2', 'airjordan_3', 'airjordan_4', 'airjordan_5', 'airjordan_6',
           'airjordan_7', 'airjordan_8', 'airjordan_9', 'airjordan_10', 'airjordan_11',
           'airjordan_12', 'airjordan_13', 'airjordan_14', 'airjordan_15', 'airjordan_16',
           'airjordan_17', 'airjordan_18', 'airjordan_19', 'airjordan_20', 'airjordan_21',
           'airjordan_22', 'airjordan_23', 'airjordan_24', 'airjordan_25', 'airjordan_26',
           'airjordan_27', 'airjordan_28', 'airjordan_29', 'airjordan_30', 'airjordan_31',
           'airjordan_32', 'airjordan_33', 'airjordan_34', 'Balenciaga_speed_trainer', 
           'Balenciaga_track', 'Balenciaga_triple_s', 'Bape_star', 'converse', 
           'Dior_sneaker_high', 'Dior_sneaker_low', 'Dr_martens_1460', 'Dr_martens_1461', 
           'Dr_martens_chelsea', 'Gucci_embroidery_sneaker', 'Hoka_tor_ultra_high', 
           'Hoka_tor_ultra_low', 'Li_ning_essence', 'Li_ning_furious_ride',
           'Louis_vuitton_archlight_trainers', 'Louis_vuitton_virgil_abloh_sneakers',
           'Mcqueen_oversized_sneaker', 'New_balance_574', 'New_balance_990', 'New_balance_993',
           'New_balance_997', 'New_balance_1600', 'Nike_air_footscape', 'Nike_airforce_1', 
           'Nike_airmax_0', 'Nike_airmax_1', 'Nike_airmax_90', 'Nike_airmax_95', 
           'Nike_airmax_96', 'Nike_airmax_97', 'Nike_airmax_98', 'Nike_airmax_99', 
           'Nike_airmax_720', 'Nike_dunksb_high', 'Nike_dunksb_low', 'Nike_epic_react',
           'Nike_foamposite', 'Nike_free', 'Nike_joyride', 'Nike_react_element',
           'Nike_vapormax', 'Puma_suede', 'Reebok_pump_fury', 'Reebok_supreme', 
           'Vans_old_skool_hi', 'Vans_old_skool_low', 'Visvim_fbt_runner', 'Visvim_virgil_boot']
```
We use the loop to download the images. Noticed that there maybe occur some error when downloading the images, so if you do not want to download it again, you can delete the labels that has already downloaded and rerun the loop. **The error may occurs more than once**, so if something wrong, do not worry and rerun the loop.
```
for class_name in classes:
  folder = class_name
  file = class_name + '.txt' #That's the reason why we have to use the same name
  dest = path/folder
  dest.mkdir(parents=True, exist_ok=True)
  print(class_name + '.txt')
  download_images(path/file, dest, max_pics=700)  #The function provided by fast.ai can help us download the images from txt file
```

### Clean the invalid images

After download all the images to our google drive, at least there would some images that cannot be opened or anything else, so we need to wipe them out. Here is the link of tutorial. https://zhuanlan.zhihu.com/p/56567350

```
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)
```

### Create the csv file that GCS AutoML Vision needs

* By documentation of AutoML Vision, we have two ways to import the iamges, since our dataset is too large so we use GCS storage to import them.
* And it needs us to give the csv file that contains the directory of each images in the GCS storage.
* This code is given by the AutoML tutorial. https://gist.github.com/yufengg/984ed8c02d95ce7e95e1c39da906ee04

We first import the libraies
```
import os
import pandas as pd
```
and check some of the images
```
filenames = [os.listdir(f) for f in classes]
[print(f[1]) for f in filenames]
[len(f) for f in filenames]
```
Zip up the files
```
files_dict = dict(zip(classes, filenames))
```
Set up the GCS storage base directory. **Noticed that when you transfrom the folder from drive to storage, the folder name would be the same where contains your images.**
```
base_gcs_path = 'gs://sneaker_us_central1/images/'
```
Create the csv file by looping them.
```
data_array = []

for (dict_key, files_list) in files_dict.items():
    for filename in files_list:
        if '.jpg' not in filename: 
            continue # don't include non-photos

        label = dict_key
        
        data_array.append((base_gcs_path + dict_key + '/' + filename , label))
```
We can see some of the results
```
data_array
```
We then use Pandas library to **create the csv file**
```
dataframe = pd.DataFrame(data_array)
```
Finally, we transform the data frame into csv file
```
dataframe.to_csv('Final.csv', index=False, header=False)  #Remember to let the parameter index and header false
```
Then transfer it from drive to GCS storage
```
bucket_name = 'sneaker_us_central1'

!gsutil -m cp -r /content/gdrive/My\ Drive/Colab\ Notebooks/FinalProject/Data/images/Final.csv gs://{bucket_name}/
```

### Transfer the images from Google drive to GCS storage

* After we finish the previous steps, we come to the last step of preparing the dataset, transfer them to GCS storage directly.
* This method is given by https://medium.com/@philipplies/transferring-data-from-google-drive-to-google-cloud-storage-using-google-colab-96e088a8c041.
```
bucket_name = 'sneaker_us_central1'

!gsutil -m cp -r /content/gdrive/My\ Drive/Colab\ Notebooks/FinalProject/Data/images/ gs://{bucket_name}/
```

### Frequently errors

* Remember to place the txt file into the correct folder.
* Check the directory of GCS storage and drive carefully, it cannot work if something wrong.
* If the csv file cannot work, carefully check the content again, and compared it with the directory in your storage.

---

## Deploy the model to ios devices

Assume that we all can successfully import the images to GCS storage and have already trained the model, we now give two ways of deploy the model to our iPhone, but one of the ways have some flaws.
  1. CoreML provided by Apple. Here is the example of using coreML https://developer.apple.com/documentation/vision/classifying_images_with_vision_and_core_ml. One drawback of it is that it cannot display the labels correctly, and I have not find the solution yet.
  2. TensorFlow Lite provided by Google. Here is the documentation and examples of how to using it and substitute your customized model in it. https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios

The exact steps are shown in the notebooks `SneakerDetector.ipynb`.

Here is the video about how my model works on my ios devices.
  
[![Delpoy the model on my ios devices](https://i.ibb.co/WWftRSv/2020-01-3116-28-49.png)](https://youtu.be/DMCR8EVWK7A)
  
---

## Strengths and Shortages
After that, I also discuss about the strengths and shortage about my project

 * Strengths
   * Collect a relatively large dataset (nearly 25,000 and nearly 100 categories) and delete all the invalid images by hand
   * The accuracy is relatively high and can classify most of the sneakers correctly
 * Shortages
   * The dataset is not large enough that we can achieve very high accuracy. For example, there are some cases that our classifier cannot detect the sneakers correctly due to the perspectives of the shoes.
   * We know that there are more than one colour for different categories of shoes, but we just classfy them into one category instead of different categories.
   * Some of the categories do not has as much as imgaes as other categories, which may caused the lack of accuracy.


## Conclusion

At the very first beginning of the lecture, I was astonished by the powerful of some vision edge devices such as Amazon DeepLens and Raspberry Pi that enable us to deploy our model in it instead of deploying them online, which is very convenient. What is more, I actually love shoes so I initiated the idea that making a sneaker classifier which can show the brand and price of the shoes. I know it is a challenging task and my buddies support me a lot when I am struggling to make this classifier. My passion for shoes also plays a key role in conducting this project and give me the initial inspirations.

As we all know that collecting data is the most important procedure during all of our project and the quality of your dataset determines the quality of your model. You would not be able have an excellent model if your dataset is poor. We know that there usually would have many colors of the same shoes, but the market price of different color also varies a lot. Considering the time is not enough for me to collect such detailed dataset so I made a tradeoff that just collect the images for different shoes and classify the different colors as the same brand. When I am searching the relative information online, I found a very useful tool called ‘gi2ds’which enables us to collect the url link of different images from Google Images. Therefore, I use this tool on Google Images and spend nearly four hours in collecting my datasets, which includes approximately 100 categories. After that, I have to guarantee the quality and purity of my dataset, so I spend another two hours in removing the redundant and incorrect images. In conclusion I have collected roughly 25,000 images and divided them into about 100 categories.

As the lecture progressing, we were taught and introduced many online machine learning platforms such as GCS Vision, AWS Machine Learning, Uber Ludwig and so on. What touched me the most is the GCS Vision, a very powerful online machine learning platform that can help us conduct the tasks of training our model and we do not write any code. So of course, I use it as the tool to create my model. It is worth mentioning that GCS Vision provides many approaches to export the model like CoreML or TensorFlow Lite, both of them were easy to deploy but I finally choose TensorFlow due to some technical problems. One of the most important advantages of GCS Vision is that it can help us train our model without writing any codes so that we can focus on where and how to collect our dataset, which is also the ultimate target and the future of AutoML..

Generally speaking, there does not exist a so called “perfect”model but only the difference between accuracy, speed or sizes of the model, which means that our goal is to find the “best”balances between the three. In our model, our dataset contains about 25,000 images and include nearly 100 categories, a relatively large dataset, so it would spend a great amount of time in training the model. Therefore, I do not choose the “higher accuracy” when training the model but the “best tradeoff”one, which significantly reduce the time for training the model and the accuracy has not drop too much as well. Another tradeoff I made is about the categories, it is very time-consuming for us to divide the categories by colors and the images that correspond to each category would also poor, so I only classify them by the style and brands. Fairly speaking, the accuracy and speed of our model is quite acceptable and even excellent.

