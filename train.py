import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import numpy as np


np.random.seed(37)
torch.manual_seed(37)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pretrained=True
num_classes = 10

def load_dataset(root_path='./data'):
    transform = transforms.Compose(
                            [transforms.ToTensor()])

    batch_size = 256

    trainset = torchvision.datasets.CIFAR10(root=root_path, train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root_path, train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return train_loader, test_loader

def train(dataloader, model, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        optimizer.step()
        model.train()

        running_loss = 0.0
        running_corrects = 0

        n = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            n += len(labels)

        epoch_loss = running_loss / float(n)
        epoch_acc = running_corrects.double() / float(n)

        print(f'epoch {epoch}/{num_epochs} : {epoch_loss:.5f}, {epoch_acc:.5f}')

def sum_t(tensor):
    return tensor.float().sum().item()

def evaluate(dataloader, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct, total = 0.0, 0.0

    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size
        predicted = outputs[:, :10].max(1)[1]
        total += batch_size
        correct_mask = (predicted == targets)
        correct += sum_t(correct_mask)

    print("Accuracy: %.2f%%" % (100. * correct / total))

if __name__ == '__main__':
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_loader, test_loader = load_dataset()

    train(train_loader, model, criterion, optimizer, num_epochs=10)
    evaluate(test_loader, model)

    CKPT_PATH = './ckpt/resnet18.pth'
    torch.save(model.state_dict(), CKPT_PATH)

    args = torch.randn(1, 3, 32, 32, device=device)
    ONNX_PATH = './ckpt/resnet18.onnx'
    torch.onnx.export(model, args, ONNX_PATH, verbose=False)    