'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

# The embedding architecture returning the 
# output of the penultimate layer
class LeNetEmbed(nn.Module):
    def __init__(self,embedding_dim=84):
        super(LeNetEmbed, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, embedding_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
    
class LeNet(nn.Module):
    def __init__(self,embedding_dim, classifier, activation =F.relu):
        super(LeNet, self).__init__()
        self.embed = LeNetEmbed(embedding_dim=embedding_dim)
        self.activation = activation
        self.classifier = classifier

    def forward(self, x):
        out = self.embed(x)
        out = self.activation(out)
        out = self.classifier(out)
        return out
