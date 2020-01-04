import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from input import getargs
from dataloader import get_train_test, plot_loss


# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 70
num_classes = 5
learning_rate = 0.001
upper_k = 10
down_k = 10
ratio = 0.62
avg_lost_list = []
liner_number = 5120


# prepare train and test data
args, commands = getargs()
if commands[0] in ['-h', '--help']:
    raise Exception("input params error\n")

train_loader, test_loader = get_train_test(upper_k, down_k, args.input_path, args.target, args.type, ratio)


# Convolutional neural network (two convolutional layers)
# 2K*128
# padding=(f-1)/2 same
# output_size = (input_size-kener_size + 2*padding)/stride + 1
# padding can use tuple, means add in different direction
class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            # torch.nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            # torch.nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     #torch.nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=1))
        #
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     # torch.nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=1))

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     # torch.nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     # torch.nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=1))

        self.fc1 = nn.Linear(liner_number, num_classes)
        #self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.layer6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        #out = self.fc2(out)
        return out


class MLP(nn.Module):
    def __init__(self, num_classes=5):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            torch.nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            torch.nn.Dropout(0.25)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, num_classes),
            torch.nn.Dropout(0.25)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
# model.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    avg_lost_list.append(loss.item())

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Test the train data
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the train images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), '/data2/users/zengys/data/train_mode_params/cnn_model.ckpt')
plot_loss(num_epochs, list(range(num_epochs)), avg_lost_list)
