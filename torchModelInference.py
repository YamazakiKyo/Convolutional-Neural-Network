import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import time
import matplotlib.pyplot as plt
import numpy as np

" http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py "
" http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html "

def imshow(img):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * np.transpose(img.numpy(), (1, 2, 0)) + mean # unnormalize
    plt.imshow(img, interpolation='nearest')
    plt.show()

def CIFAR10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader

def ImageNet():
    transform = transforms.Compose(
        [   transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )
    trainset = datasets.ImageFolder(root='./ImageNet/hymenoptera_data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    testset = datasets.ImageFolder(root='./ImageNet/hymenoptera_data/val', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    return trainloader, testloader

# trainloader, testloader = CIFAR10()
trainloader, testloader = ImageNet()
# images, labels = iter(trainloader).next()

resnet18 = models.resnet152(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False
alexnet = models.alexnet(pretrained=True)
for param in alexnet.parameters():
    param.requires_grad = False
vgg19 = models.vgg19(pretrained=True)
for param in vgg19.parameters():
    param.requires_grad = False
densenet = models.densenet161(pretrained=True)
for param in densenet.parameters():
    param.requires_grad = False
inception = models.inception_v3(pretrained=True)
for param in inception.parameters():
    param.requires_grad = False

benchmark_models = [alexnet, resnet18, vgg19, densenet]

start_time = time.time()
for model in benchmark_models:
    model.eval
    correct = 0
    total = 0
    counter = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            counter += 1
            if counter % 30 == 0:
                print(counter)
                print('Inference Time: %d' %(time.time() - start_time))
                print('Accuracy: %d' % (100 * correct / total))
                break
