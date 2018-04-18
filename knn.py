# Compute k nearest neighbors for COCO images based on feature maps
# Requires pre-computed feature maps (e.g., data/coco/extract/.../trainset.hdf5)

import argparse
import h5py
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Ex: 'data/coco/extract/arch,fbresnet152_size,448'
parser.add_argument('base_dir', type=str, help='path to dir containing extracted hdf5 features')
parser.add_argument('--hdf5_file', type=str, default='trainset.hdf5')

parser.add_argument('--save_dir', type=str, default='data/coco/extract')
parser.add_argument('--save_file', type=str, default='knn_results_trainset.npy')

parser.add_argument('-k', '--k', default=25, type=int)
parser.add_argument('-b', '--batch_size', default=10, type=int)

def main():
    args = parser.parse_args()
    
    load_path = os.path.join(args.base_dir, args.hdf5_file)
    assert(os.path.exists(load_path))
    
    assert(os.path.isdir(args.save_dir))
    save_path = os.path.join(args.save_dir, args.save_file)
    if os.path.exists(save_path):
        print("Warning: {} already exists and will be overwritten.".format(save_path))
    print("Saving results to {}".format(save_path))
    
    f = h5py.File(load_path, 'r')
    dataset = f.get('noatt')
    features = np.array(dataset)
    print("Loaded features as array of size {}".format(features.shape))

    print("Fitting KNN model for k={}".format(args.k))
    nbrs = NearestNeighbors(n_neighbors=args.k)
    nbrs.fit(features)

    print("Starting KNN computations...")
    indices_list = []
    distances_list = []
    for i in tqdm(range(0, features.shape[0], args.batch_size)):
        if i + args.batch_size < features.shape[0]:
            distances, indices = nbrs.kneighbors(features[i:i+args.batch_size, :])
        else:
            distances, indices = nbrs.kneighbors(features[i:, :])
        indices_list.append(indices)
        distances_list.append(distances)

    all_indices = np.concatenate(indices_list)
    all_distances = np.concatenate(distances_list)
    
    np.save(save_path, {"indices": all_indices, "distances": all_distances})

if __name__ == '__main__':
    main()
    