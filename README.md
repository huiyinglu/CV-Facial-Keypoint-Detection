# CV-Facial-Keypoint-Detection
This project will be all about defining and training a convolutional neural network to perform facial keypoint detection, and using computer vision techniques to transform images of faces. 

Facial keypoints (also called facial landmarks) are the small magenta dots shown on each of the faces in our data image. In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face. These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, and so on. 

This facial keypoints dataset we are using consists of 5770 color images. All of these images are separated into either a training or a test set of data: 3462 of these images are training images, we may use to create a model to predict keypoints and 2308 are test images, which will be used to test the accuracy of our model.

The major steps implemented in this project include:
1) Read in training/testing images in batches and transform the images (normalize, rescale, randomcrop, covert to Pytorch tensor)
2) Feed the images to our model (implemented in Pytorch using CNN) and train the model.
3) Predict the facial keypoint on testing images.
4) Select an image during run time, perform facial detection (use OpenCV's pre-trained Haar Cascade classifiers) and run keypoint detection.

The performance of our model is pretty good: the keypoints detected by our model on testing images and on run time selected image are closely match the keypoints on the faces of the images.

