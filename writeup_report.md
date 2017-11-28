# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./images/image1.jpg
[image2]: ./images/image2.jpg
[image3]: ./images/image3.jpg
[image4]: ./images/image4.jpg
[networkVisualization]: ./images/networkVisualization.png

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I did the following:
1. crop image
2. use keras lambda to normalize
3. use two 5x5 filter for convolution with relu for activation and depth of 6 and 16, respectively
4. use 2x2 max pool
5. use two densely-connected neural network with sigmoid for activation and outputs of 100 and 200, respectively

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers. I added partial laps around the areas that confused the model.

I used .2 (or 20%) test/validation split.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data
I used 2 full laps (all around the track) for initial training. Once I determine where the model fails to stay on track, I ran more laps around the places where it fails to stay on track. I adjust the steering angle so that it steers harder to left or right. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find something that will work well with images with poor lighting.

I used LeNet architecture because it did a good job detecting features even with images that are not well lit.

Since the test and validation sets fluctuate, I save the model at every epoch. I try out every model and save the one that performed the best. The model that is chosen is determined by how well the vehicle recovers when it moves towards the edge and the number of times it drives off the track.

I improve the driving behavior by running a few laps around the areas where the car fails to stay on track. I retrain using the saved model with all the previous laps in addition to the new lap.

I reduce overfitting by adding laps if the vehicle goes off track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a chart of the architecture I used

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   						| 
| Crop Image     	    | outputs 320x80x3 RGB image 					|
| Normalize             | divide all values by 255 and subtract 0.5 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 316x76x6 	|
| RELU					| Zero out negative values						|
| Max pooling	      	| 2x2 stride, valid padding, outputs 158x38x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 154x34x16 	|
| RELU					| Zero out negative values						|
| Max pooling	      	| 2x2 stride, valid padding, outputs 77x17x16	|
| Flatten               | Flattens a 77x17x16 array to 20944 elements	|
| Fully connected		| 20944x200 weights, 0 bias 					|
| Sigmoid				| Self-explanatory								|
| Fully connected		| 200x100 weights, 0 bias						|
| Sigmoid				| Self-explanatory								|
| Fully connected		| 100x1 weights, 0 bias								| 

Here's a visualization of the neural network
![alt text][networkVisualization]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle on the left and right side of the road. In the code, I adjust the steering 0.5 and -0.5 which steers the car to the right and left respectively.

Here are some of the images the car on the left and right side of the track

![alt text][image2]
![alt text][image3]
![alt text][image4]

In summary, I do the following to collect data:
1. run 2 full laps around the track
2. train with 5 epochs. (Note: most of the time additional epoch does not improve train/validation numbers)
3. choose the model that performs the best
4. see where the car falls of track
5. depending on the turn, I run laps on the left/right side of the track
6. repeat steps 2 thru 5. Train until it stays on track. (Note: Be sure to retrain with all previous data)