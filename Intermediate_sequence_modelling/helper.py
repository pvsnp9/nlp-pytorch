import torch
import numpy as np
import os
import re
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Helper(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def column_gather(y_out, x_lengths):
        '''Get a specific vector from each batch datapoint in `y_out`.

        More precisely, iterate over batch row indices, get the vector that's at
        the position indicated by the corresponding value in `x_lengths` at the row
        index.

        Args:
            y_out (torch.FloatTensor, torch.cuda.FloatTensor)
                shape: (batch, sequence, feature)
            x_lengths (torch.LongTensor, torch.cuda.LongTensor)
                shape: (batch,)

        Returns:
            y_out (torch.FloatTensor, torch.cuda.FloatTensor)
                shape: (batch, feature)
        '''
        x_lengths = x_lengths.long().detach().cpu().numpy() - 1

        out = []
        for batch_index, column_index in enumerate(x_lengths):
            out.append(y_out[batch_index, column_index])

        return torch.stack(out)
    
    
    def generate_batches(dataset, batch_size, shuffle=True, drop_last=True,  device='cuda:0'): 
        """
        A generator function which wraps the PyTorch DataLoader. It will 
        ensure each tensor is on the write device location.
        """
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict
    
    @staticmethod
    def make_train_state(args):
        return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file
        }
    @staticmethod
    def update_train_state(args, model, train_state):
        """Handle the training state updates.

        Components:
        - Early Stopping: Prevent overfitting.
        - Model Checkpoint: Model is saved if the model is better

        :param args: main arguments
        :param model: model to train
        :param train_state: a dictionary representing the training state values
        :returns:
            a new train_state
        """

        # Save one model at least
        if train_state['epoch_index'] == 0:
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['stop_early'] = False

        # Save model if performance improved
        elif train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= train_state['early_stopping_best_val']:
                # Update step
                train_state['early_stopping_step'] += 1
            # Loss decreased
            else:
                # Save the best model
                if loss_t < train_state['early_stopping_best_val']:
                    torch.save(model.state_dict(), train_state['model_filename'])

                # Reset early stopping step
                train_state['early_stopping_step'] = 0

            # Stop early ?
            train_state['stop_early'] = \
                train_state['early_stopping_step'] >= args.early_stopping_criteria

        return train_state
    
    
    @staticmethod
    def load_glove_from_file(glove_filepath):
        """
        Load the GloVe embeddings 
        
        Args:
            glove_filepath (str): path to the glove embeddings file 
        Returns:
            word_to_index (dict), embeddings (numpy.ndarary)
        """

        word_to_index = {}
        embeddings = []
        with open(glove_filepath, "r") as fp:
            for index, line in enumerate(fp):
                line = line.split(" ") # each line: word num1 num2 ...
                word_to_index[line[0]] = index # word = line[0] 
                embedding_i = np.array([float(val) for val in line[1:]])
                embeddings.append(embedding_i)
        return word_to_index, np.stack(embeddings)

    @staticmethod
    def make_embedding_matrix(glove_filepath, words):
        """
        Create embedding matrix for a specific set of words.
        
        Args:
            glove_filepath (str): file path to the glove embeddigns
            words (list): list of words in the dataset
        """
        word_to_idx, glove_embeddings = Helper.load_glove_from_file(glove_filepath)
        embedding_size = glove_embeddings.shape[1]
        
        final_embeddings = np.zeros((len(words), embedding_size))

        for i, word in enumerate(words):
            if word in word_to_idx:
                final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
            else:
                embedding_i = torch.ones(1, embedding_size)
                torch.nn.init.xavier_uniform_(embedding_i)
                final_embeddings[i, :] = embedding_i

        return final_embeddings
    
    @staticmethod
    def normalize_sizes(y_pred, y_true):
        """Normalize tensor sizes
        
        Args:
            y_pred (torch.Tensor): the output of the model
                If a 3-dimensional tensor, reshapes to a matrix
            y_true (torch.Tensor): the target predictions
                If a matrix, reshapes to be a vector
        """
        if len(y_pred.size()) == 3:
            y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
        if len(y_true.size()) == 2:
            y_true = y_true.contiguous().view(-1)
        return y_pred, y_true

    
    @staticmethod
    def compute_accuracy(y_pred, y_true, mask_index):
        y_pred, y_true = Helper.normalize_sizes(y_pred, y_true)

        _, y_pred_indices = y_pred.max(dim=1)
        
        correct_indices = torch.eq(y_pred_indices, y_true).float()
        valid_indices = torch.ne(y_true, mask_index).float()
        
        n_correct = (correct_indices * valid_indices).sum().item()
        n_valid = valid_indices.sum().item()

        return n_correct / n_valid * 100

    @staticmethod 
    def sequence_loss(y_pred, y_true, mask_index):
        y_pred, y_true = Helper.normalize_sizes(y_pred, y_true)
        return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)
    
    @staticmethod
    def handle_dirs(dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            
    @staticmethod
    def set_seed_everywhere(seed, cuda):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def preprocess_text(text):
        text = ' '.join(word.lower() for word in text.split(" "))
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
        return text
    
    @staticmethod
    def pretty_print(results):
        """
        Pretty print embedding results.
        """
        for item in results:
            print ("...[%.2f] - %s"%(item[1], item[0]))
    
    @staticmethod
    def get_closest(target_word, word_to_idx, embeddings, n=5):
        """
        Get the n closest
        words to your word.
        """

        # Calculate distances to all other words
        
        word_embedding = embeddings[word_to_idx[target_word.lower()]]
        distances = []
        for word, index in word_to_idx.items():
            if word == "<MASK>" or word == target_word:
                continue
            distances.append((word, torch.dist(word_embedding, embeddings[index])))
        
        results = sorted(distances, key=lambda x: x[1])[1:n+2]
        return results
    
    @staticmethod
    def sample_from_model(model, vectorizer, num_samples=1, sample_size=20, 
                      temperature=1.0):
        """Sample a sequence of indices from the model
        
        Args:
            model (SurnameGenerationModel): the trained model
            vectorizer (SurnameVectorizer): the corresponding vectorizer
            num_samples (int): the number of samples
            sample_size (int): the max length of the samples
            temperature (float): accentuates or flattens 
                the distribution. 
                0.0 < temperature < 1.0 will make it peakier. 
                temperature > 1.0 will make it more uniform
        Returns:
            indices (torch.Tensor): the matrix of indices; 
            shape = (num_samples, sample_size)
        """
        begin_seq_index = [vectorizer.char_vocab.begin_seq_index 
                        for _ in range(num_samples)]
        begin_seq_index = torch.tensor(begin_seq_index, 
                                    dtype=torch.int64).unsqueeze(dim=1)
        indices = [begin_seq_index]
        h_t = None
        
        for time_step in range(sample_size):
            x_t = indices[time_step]
            x_emb_t = model.char_emb(x_t)
            rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
            prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
            probability_vector = F.softmax(prediction_vector / temperature, dim=1)
            indices.append(torch.multinomial(probability_vector, num_samples=1))
        indices = torch.stack(indices).squeeze().permute(1, 0)
        return indices

    @staticmethod
    def decode_samples(sampled_indices, vectorizer):
        """Transform indices into the string form of a surname
        
        Args:
            sampled_indices (torch.Tensor): the inidces from `sample_from_model`
            vectorizer (SurnameVectorizer): the corresponding vectorizer
        """
        decoded_surnames = []
        vocab = vectorizer.char_vocab
        
        for sample_index in range(sampled_indices.shape[0]):
            surname = ""
            for time_step in range(sampled_indices.shape[1]):
                sample_item = sampled_indices[sample_index, time_step].item()
                if sample_item == vocab.begin_seq_index:
                    continue
                elif sample_item == vocab.end_seq_index:
                    break
                else:
                    surname += vocab.lookup_index(sample_item)
            decoded_surnames.append(surname)
        return decoded_surnames
    
    
    #========================== code only for conditioned (hidden state) =======================
    def conditioned_sample_from_model(model, vectorizer, nationalities, sample_size=20, temperature=1.0):
        """Sample a sequence of indices from the model
        
        Args:
            model (SurnameGenerationModel): the trained model
            vectorizer (SurnameVectorizer): the corresponding vectorizer
            nationalities (list): a list of integers representing nationalities
            sample_size (int): the max length of the samples
            temperature (float): accentuates or flattens 
                the distribution. 
                0.0 < temperature < 1.0 will make it peakier. 
                temperature > 1.0 will make it more uniform
        Returns:
            indices (torch.Tensor): the matrix of indices; 
            shape = (num_samples, sample_size)
        """
        num_samples = len(nationalities)
        begin_seq_index = [vectorizer.char_vocab.begin_seq_index 
                        for _ in range(num_samples)]
        begin_seq_index = torch.tensor(begin_seq_index, 
                                    dtype=torch.int64).unsqueeze(dim=1)
        indices = [begin_seq_index]
        nationality_indices = torch.tensor(nationalities, dtype=torch.int64).unsqueeze(dim=0)
        h_t = model.nation_emb(nationality_indices)
        
        for time_step in range(sample_size):
            x_t = indices[time_step]
            x_emb_t = model.char_emb(x_t)
            rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
            prediction_vector = model.fc(rnn_out_t.squeeze(dim=1))
            probability_vector = F.softmax(prediction_vector / temperature, dim=1)
            indices.append(torch.multinomial(probability_vector, num_samples=1))
        indices = torch.stack(indices).squeeze().permute(1, 0)
        return indices

    def conditioned_decode_samples(sampled_indices, vectorizer):
        """Transform indices into the string form of a surname
        
        Args:
            sampled_indices (torch.Tensor): the inidces from `sample_from_model`
            vectorizer (SurnameVectorizer): the corresponding vectorizer
        """
        decoded_surnames = []
        vocab = vectorizer.char_vocab
        
        for sample_index in range(sampled_indices.shape[0]):
            surname = ""
            for time_step in range(sampled_indices.shape[1]):
                sample_item = sampled_indices[sample_index, time_step].item()
                if sample_item == vocab.begin_seq_index:
                    continue
                elif sample_item == vocab.end_seq_index:
                    break
                else:
                    surname += vocab.lookup_index(sample_item)
            decoded_surnames.append(surname)
        return decoded_surnames