# Fine-Tuning a Pre-Trained Network

In tried the first option:Write a Python function to be used at the end of training that generates HTML output showing
each test image and its classification scores.

I used PyTorch to build a convolutional neural network. I then trained the CNN on the CIFAR-10 data set to be able to classify images from the CIFAR-10 testing set into the ten categories present in the data set

The classes are: 'plane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck' (see in my code: classes)

The images are 3x32x32, i.e., 3 channels (red, green, blue) each of size 32x32 pixels.


Here are the steps taken with reference to the code for each specific task:

1. Using torchvision library, I loaded and normalized the CIFAR10 training and test datasets (see in the code: transform, trainset, trainloader, testset, testloader)

More details on step 1:

In a two-step process, we prepare the data for use with the CNN. First step is to convert Python Image Library format to PyTorch tensors. Second step is to normalize the data by specifying a  mean and standard deviation for each of the three channels. This will convert the data from [0,1] to [-1,1]
Normalization of data should help speed up conversion and reduce the chance of vanishing gradients with certain activation functions.


We feed the inputs to the network and optimize the weights:


2. Our defined Convolutional Neural Network (see in the code: class BasicBlock(nn.Module)) was trained on the training dataset (see the code: for epoch in range(NUM_EPOCH):) given the defined loss function and optimizer (see the code: criterion, optimizer). 


3. I then saved my trained model (see the code: torch.save)

4. Finally, I tested the network on the test data. I wrote a Python function to generate HTML output showing each test image and its classification scores. To see the classification scores, I used nn.functional.softmax(outputs)








