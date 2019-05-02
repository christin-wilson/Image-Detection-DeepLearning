# Image-Detection-DeepLearning
A Deep Learning Model to Detect Images with Cyberbully Actions

Network Structure:
To classify bullying and non-bullying in images we used a convolutional neural network and followed the VGG16 architecture. The model used a 3 x 3 filter through the network to increase its depth and has 13 convolutional layers and 5 pooling layers. The number of activation maps starts at 64 and increased by a factor of 2 after each max-pooling layer until it gets to 512.

Process:
We downloaded non-cyberbullying images to increase the image class to 10. The addition of the non-bully images decreased the accuracy but not by a large margin. Training was carried out using mini-batch gradient descent with momentum and learning rate set to 0.001 initially. we started with an epoch of 2 to get a sense of what the loss if and gradually increased the epoch to 7, to and 74. While training with epoch set to 74 we also set weight decay to 0.0005 and we started with a learning rate of 0.01 and decreased it by a factor of 10 every 7 epoch.
We applied horizontal flips and normalization and subtracted mean from the images to increase the dataset size.

Code Usages:
In terms of code usage, we rewrote the vgg16 code and borrowed some ideas from the following sources:
https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet /main.py
https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

• To run the code,
1. Add the "best_model.pt" file to the weights folder.
  https://drive.google.com/open?id=1tHsYCQfQbzadhBOQRgNztK4NiyybR-Tp 
2. Add the "wt.weights" file from the same drive link to the darknet folder.  
3. Input the command: "python main.py -p 1 -i my_image.jpg"

Reference Papers:
We followed the following paper: very deep convolutional networks for large-scale image recognition
