
def train(net, learning_rate=0.001, momentum=0.9, weight_decay=0.0005, epoch_size=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    lr_reduction_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(epoch_size):  # loop over the dataset multiple times
      print('Epoch {}/{}'.format(epoch, epoch_size - 1))
      print('-' * 10)

      for phase in ['train', 'val']:
          if phase == 'train':
              lr_reduction_scheduler.step()
              net.train()
          else:
              net.eval()

          running_loss = 0.0
          running_corrects = 0

          for inputs, labels in dataloaders[phase]:
              # get the inputs
              inputs = inputs.to(device)
              labels = labels.to(device)

              # zero the parameter gradients
              optimizer.zero_grad()

              with torch.set_grad_enabled(phase == 'train'):
                  # forward
                  output = net(inputs)
                  _, model_prediction = torch.max(outputs.data, 1)
                  loss = criterion(output, train_y)

                  # backward + optimize
                  if phase == 'train':
                      loss.backward()
                      optimizer.step()

              # print statistics
              print(inputs.size(0))
              running_loss += loss.item() * inputs.size(0)
              running_corrects += torch.sum(model_prediction == labels.data).item()

          epoch_loss = running_loss / dataset_sizes[phase]
          epoch_acc = running_corrects.double() / dataset_sizes[phase]
          print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

          # deep copy the model
          if phase == 'val' and epoch_acc > best_acc:
              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(net.state_dict())
          print()
    print('Best val Acc: {:4f}'.format(best_acc))
    #load best model weights
    net.load_state_dict(best_model_wts)
    return net

print(len(class_names))

model_conv = torchvision.models.vgg16_bn(pretrained=True)
print(model_conv.classifier[6].out_features)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_features = model_conv.classifier[6].in_features
print(num_features)

features = list(model_conv.classifier.children())[:-1] # Remove last layer
print(features)
features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
print(features)
model_conv.classifier = nn.Sequential(*features) # Replace the model classifier

model_conv = model_conv.to(device)
#print(model_conv)


# Parameters of newly constructed modules have requires_grad=True by default

model_conv = train(model_conv)
