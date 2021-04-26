# CNN_ImageClassifier
Medium Depth CNN Image Classifier on Fashion-MNIST dataset 

Download the Fashion Mnist data from http://fashion-mnist.s3-website.eu-central-1.amazonaws.com

Put your train/test gz files under data/fashion/

To train/test for fixed number of 50 epochs. 

Primary motive of this simple CNN text classifier is to introduce ML engineers/students to a CNN image classifie in Pytorch from first principles and hence a lot of high level methods for SGD and cross-validation are not used. 

We perform classifical SGD by taking the gradients and updating the weights and run the traininng for fixed number of epochs (40), followed by testing on indepednent official test-set of Fashion-Mnist. 

Accuracy: 87.87%

 

