from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import argparse
from experiments import secnet
from experiments import test
from experiments import validate
from experiments import train_model
from experiments import save_checkpoint
from load_data import device

parser = argparse.ArgumentParser(description='cyberbully image prediction')
parser.add_argument('-p', '--pretrained', help='use pre-trained model')
parser.add_argument('-i', '--image', help="path to image")


best_prediction = 0

'''- TO USE FROM COMMAND LINE ENTER: python train.py -p 1 -i applauding_005.jpg
   - If you enter only python train.py the else statement will execute and training from scratch will start.
   TODO: FIX THE ISSUE BELOW IF YOU CAN!!!
   - The else statement keeps printing things in load_data.py 4 times because batch_size is 4 when train_model function is called. Why is it being printed my guess is "I have some issue in the way I am importing it".
'''

if __name__ == '__main__':
    args = parser.parse_args()
#    print(vars(args))
    args = vars(args) #convert arguments to a dict
    if args['pretrained']:
        # process image
        img = Image.open(args["image"])
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                        ])
        img = transform(img)
        img = img.unsqueeze(0)
        img = Variable(img)

        net = secnet(pretrained=True)
        net = net.to(device)

        prediction = test(net, img)
        print("{}".format(prediction))
    else:
        #load model that is not pre-trained
        net = secnet(pretrained=False)
        net = net.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        lr_reduction_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        train_model(net, criterion, optimizer, lr_reduction_scheduler, epoch_size=2)
        current_prediction = validate(net, criterion, dataloaders)

        is_best = current_prediction > best_prediction
        best_prediction = max(current_prediction, best_prediction)
        save_checkpoint(net, is_best)
        print("All done here!")
