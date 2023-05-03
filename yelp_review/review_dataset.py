from torch.utils.data import Dataset
import pandas as pd
from review_vectorizer import ReviewVectorizer

class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        Args:
        review_df (pandas.DataFrame): the dataset
        vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """
        self.review_df = review_df
        self._vectorizer = vectorizer
        
        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)
        
        self.val_df = self.review_df[self.review_df.split=='val']
        self.validation_size = len(self.val_df)
        
        self.test_df = self.review_df[self.review_df.split=='test']
        self.test_size = len(self.test_df)
        
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)
                            }
    
        self.set_split('train')
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """Load dataset and make a new vectorizer from scratch
        Args:
        review_csv (str): location of the dataset
        Returns:
        an instance of ReviewDataset
        """
        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))
    

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split='train'):
        """ selects the splits in the dataset using a column in the dataframe
        Args:
        split (str): one of "train", "val", or "test"
        """
        self._target_splt = split
        self._target_df, self._target_size = self._lookup_dict[split]
        
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
            Args:
            index (int): the index to the data point
            Returns:
            a dict of the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        
        review_vector = self._vectorizer.vectorize(row.review)
        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)
        
        return {
            'x_data': review_vector,
            'y_target': rating_index
        }
        
    def get_num_bathces(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
            Args:
            batch_size (int)
            Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
    
    
# if __name__ =="__main__":
#     dataset = ReviewDataset.load_dataset_and_make_vectorizer("/home/whiskey/Documents/DevBase/Learnings/NLP/Learning form Books/NLP with Pytorch/Data/reviews_with_splits_lite.csv")
#     vectorizer = dataset.get_vectorizer()