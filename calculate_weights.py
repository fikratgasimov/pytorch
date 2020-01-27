import os
from tqdm import tqdm
import numpy as np
from mypath import Path

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader) 
    print('Calculating classes weights')
    # sample already initialized in difference dataset, this is the same or ?
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        # num_classes already defined in dataset
        mask = (y >= 0) & (y < num_classes)
        # modify datatype
        labels = y[mask].astype(np.uint8)
        # count the number of occurences of each value in array of non negative ints
        count_l = np.bincount(labels, minlength=num_classes)
        # increment
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    # introduce class_weights of array
    class_weights = []
    # frequency initialized and loop through z:
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        # accordingly append class_weights into class_weights
        class_weights.append(class_weight)
        # perform class_weights as an array
    ret = np.array(class_weights)
    # eventually, determine path of class_weights_path:join dataset and dataset + '_classes_weights.npy'
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    # Save outcomes :
    np.save(classes_weights_path, ret)

    return ret
