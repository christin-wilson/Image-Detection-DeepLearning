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
from skimage import io#, transform
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

'''- TO USE FROM COMMAND LINE ENTER: python main.py -p 1 -i applauding_005.jpg'''

if __name__ == '__main__':
    args = parser.parse_args()
    #print(vars(args))
    args = vars(args) #convert arguments to a dict
    if args['pretrained']:
        # process image
        img = io.imread(args['image'])
        img = img.transpose((2, 0, 1))  #convert from HxWxC to CxHxW

        img = transforms.ToPILImage()(img)  #convert to a PIL image
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                        ])
        img = transform(img)
        img = img.unsqueeze(0)  #since its a single image let batch size be 1 [b,c,H,W]
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
