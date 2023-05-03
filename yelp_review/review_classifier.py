import torch.nn as nn 
import torch.nn.functional as F

class ReviewClassifier(nn.Module):
    """ a simple perceptron-based classifier """
    
    def __init__(self, num_features):
        """
        Args:
        num_features (int): the size of the input feature vector
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=250)
        self.fc2 = nn.Linear(in_features=250, out_features=125)
        self.fc3 = nn.Linear(in_features=125, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, apply_sigmoid=False):
        """The forward pass of the classifier
        Args:
        x_in (torch.Tensor): an input data tensor
        x_in.shape should be (batch, num_features)
        apply_sigmoid (bool): a flag for the sigmoid activation
        should be false if used with the cross-entropy losses
        Returns:
        the resulting tensor. tensor.shape should be (batch,).
        """
        fc1 = self.relu(self.fc1(x))
        fc2 = self.relu(self.fc2(fc1))
        fc3 = self.fc3(fc2).squeeze()
        if apply_sigmoid: return F.sigmoid(fc3)
        return fc3