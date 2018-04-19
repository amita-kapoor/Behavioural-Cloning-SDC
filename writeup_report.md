# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py contains the script to create and train the model, it also contains the utilities needed for preprocessing and Data Generation and Augmentation.
* video.mp4 a video showing the car running in autonomous mode on track 1.
* drive.py is unchanges, exactly as provided by the Udacity.
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

As the base model I used Nvidia model described in the paper [End-to-End Deep Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf). 

The figure below is the model architecture that I used
![Model summary](model_summary.png)

The model consists of 5 convolutional layers, the first three have kernel size of ```5 x 5``` ensuring a wider look into the input image, and 24, 36, and 48 filters respectively. The last two convolutional layers have the kernel size ```3 x 3``` and 64 filters each.  All the convolutional layers use **Elu** activation function to ensure non-linearity. The output of the last convolutional layer is Flattened. Next we add fully connected layes to perform the task of regression based on the features extracted by convolutional layers.
There are four fully connected layers with neurons 100, 50 10 and 1 respectively.

I also hard wired both Cropping (model.py line ) and Normalization (model.py line) in my model so that the process can make use of the GPU speed. 

The complete model is defined in the model.py file in lines 104-131. 


#### 2. Attempts to reduce overfitting in the model

Overfitting is a common problem, to ensure that overfitting does not happen, from the starting I used following methods:

* Between the convolutional layers and Dense layers I added a Dropout layer, according to the paper [Dropout:  A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf?utm_content=buffer79b43&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer). the dropout layer helps in overcoming overfitting. During training some of the connections are off, thus ensuring remaining weights learn the feature represenataion.

* The dataset was divided into training and validation set. As the training went both training and validation loss were considered, and the model with best performance, that is least validation loss was saved.

* I also used Data Augmentation on the training dataset  (function ```data_generator``` defined in model.py lines 58-93), this helps by making the system robust. According to this paper [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf) augmenation improves the performance by taking care of insufficient information, also since random distortions are introduced in the training dataset this ensures that model does not overfit. While according to the paper neural augmentation gives best result for simplicity I used traditional augmentation methods and performed random horizontal flip, rotation, horizontal and vertical shift and change in brightness (the random distortions were added using function ```distort_random``` defined in model.py lines 29-54). 

After the network was trained the generated model was used to test on track 1, below is the you tube link to that video. As you can see, the car succesfully completed the lap, but there are many places where it is almost on the boundary and might leave the track, had I been sitting in the car would have been terrified :smiley:.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=wdtUOWjxPIQ
" target="_blank"><img src="http://img.youtube.com/vi/wdtUOWjxPIQ/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, along with it the left and right camera positions with high correction in the steering angle to train the model.

In my opinion sufficient amount of this sort of data should be sufficient for end to end training.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Initially I experimented with a very simple MLP model and used only center camera image, it was utter failure the car just kept moving in circles. Then as I added the CNN layers the car started to stay more on road, but was still not able to handle curves. With the introduction of **Nvidia Dave model** the performance improved further, but now car was failing at sharp turns.
This forced me to look at my training data. From the plot of training and validation loss, I knew the problem is not of overfitting, but instead the model has not learned to take sharp turns, which were at times necessary.

![Loss](Figure_1.png)


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model summary](model_summary.png)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first had two laps on track-one recorded using center lane driving. Here is an example image of center lane driving:

![Center](center_track_1.jpg)

I repetaed the center lane driving on 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.