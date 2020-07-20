"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first                   #
        ########################################################################
        index_iterator = iter(np.random.permutation(len(self.dataset)))
        batch=[]
        new_batch={}
        
        l=0
        Total = len(self.dataset)
        spanned = len(self.dataset)%self.batch_size
        x=[]
        
        
        for index in index_iterator:
            batch.append(self.dataset[index])
            l=l+1
            if len(batch) == self.batch_size:
                for sample in batch:
                    for key,value in sample.items():
                        if key not in new_batch:
                            new_batch[key]=[]
                        new_batch[key].append(value)
                    
                yield new_batch
                batch = []
                new_batch = {}
            if l>=(Total - spanned +1 ):
                x.append(index)
        
        batch = []
        new_batch = {}
    
        
        
        if self.drop_last == False:
            index_iterator_2 = iter(np.random.permutation(x))
            
            for index in index_iterator_2:
                batch.append(self.dataset[index])
                
            for sample in batch:
                    for key,value in sample.items():
                        if key not in new_batch:
                            new_batch[key]=[]
                        new_batch[key].append(value)
            
            yield new_batch
                
        
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset  #
        ########################################################################
        length = int((len(self.dataset))/ (self.batch_size))
        if self.drop_last == False:
            length = length + 1
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
