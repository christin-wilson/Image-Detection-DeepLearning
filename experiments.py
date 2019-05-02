import shutil
import numpy as np
from load_data import *
from model import SEC_NET
import subprocess
import os

def secnet(pretrained=False, **kwargs):
    '''
    Args: pretrained - boolean. if True return our pretrained model, else return our architecture and initialize weights
    '''
    if pretrained:
        kwargs['weights'] = False
    model = SEC_NET(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load("weights/best_model.pt", map_location=device))
    return model

def test(net, img):
    outputs = net(img)
    _, model_prediction = torch.max(outputs.data, 1)
    prediction = class_names[torch.max(model_prediction).item()]
    print(prediction)
    if prediction == 'not-bully':
        return prediction
    else:
        result = obj_detection(net, img)
        return result

def obj_detection(net, img):
    #result = subprocess.Popen(["./darknet", "detect", "cfg/yolov3.cfg", "yolov3.weights", "data/dog.jpg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #output, errors = result.communicate()
    result = os.system("./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg")
    return result

def validate(net, criterion, dataloaders):

    net.eval()
    validation_loss = 0.0
    correct_predictions = 0

    for i, data in enumerate(dataloaders['val']):
        test_x, test_y = data
        test_x = test_x.to(device)
        test_y = test_y.to(device)

        outputs = net(test_x)
        _, model_prediction = torch.max(outputs.data, 1)
        loss = criterion(outputs, test_y)

        #statistics
        validation_loss = loss.item() * test_x.size(0)
        correct_predictions += torch.sum(model_prediction == test_y)#.item()

        print("{} Loss: {:.4f}".format('val', validation_loss))
        print()
        print(test_y)
        print(model_prediction)
        for j in range(len(test_x)):
            print('correct label {} which is {} is predicted as {} {}'.format(test_y[j], class_names[test_y[j]], class_names[model_prediction[j]], model_prediction[j]))
        print("Model Prediction: {}".format(class_names[torch.max(model_prediction)]))

        print()
        if i % 50 == 49:
            print("Accuracy over {} testset is: {:2.2%}".format(i+1, correct_predictions/dataset_sizes['val']))
        print()


    print(correct_predictions)
    print(dataset_sizes['val'])

    validation_acc = float(correct_predictions) / dataset_sizes['val']
    print("Accuracy of the network on {} testset is {:2.2%}".format(dataset_sizes['val'], validation_acc))
    return validation_acc

def train_model(net, criterion, optimizer, scheduler, epoch_size=2):
    print("Training Model")
    for epoch in range(epoch_size):  # loop over the dataset multiple times
        #lr_reduction_scheduler.step()
        #training_correct = 0
        training_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            # get the inputs
            train_x, train_y = data
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(train_x)
            loss = criterion(output, train_y)
            loss.backward()
            optimizer.step()

            # print statistics
            training_loss += loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, training_loss / 50))
                training_loss = 0.0

    print('Finished Training')

def save_checkpoint(net, is_best):
    if is_best:
        print("Saving model...")
        torch.save(net.state_dict(), '/content/gdrive/My Drive/Colab Notebooks/best_model.pt')
        print("Model saved")
