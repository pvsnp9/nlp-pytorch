import torch 
import torch.nn as nn
import torch.nn.functional as F

class CBOWClassifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, padding_idx=0):
        """
        Args:
        vocabulary_size (int): number of vocabulary items, controls the
        number of embeddings and prediction vector size
        embedding_size (int): size of the embeddings
        padding_idx (int): default 0; Embedding will not use this index
        """
        super(CBOWClassifier, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.fc1 = nn.Linear(in_features=embedding_size, out_features=vocabulary_size)
        
    def forward(self, x, apply_softmax=False):
        """The forward pass of the classifier
        Args:
        x_in (torch.Tensor): an input data tensor
        x_in.shape should be (batch, input_dim)
        apply_softmax (bool): a flag for the softmax activation
        should be false if used with the cross-entropy losses
        Returns:
        the resulting tensor. tensor.shape should be (batch, output_dim).
        """
        
        x_embedded_sum = F.dropout(self.embedding(x).sum(dim=1), 0.2)
        fc1 = self.fc1(x_embedded_sum)
        
        if apply_softmax: return F.softmax(fc1, dim=1)
        return fc1
    
    
    
    
"""
It invovles three essential step:
    First, indices representing words are used with an Embedding layer to create vectors for each word in the context.
    Second, the vectors are combined in some way to capture the overall context. In this example, they are summed, but other options include taking the max, 
            average, or using a Multilayer Perceptron on top.
    Third, the context vector is used with a Linear layer to compute a prediction vector, which is a probability distribution over the entire vocabulary.
    
The largest value in the prediction vector indicates the likely prediction for the target word, which is the center word missing from the context.
"""