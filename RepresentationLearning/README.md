# Metric Learning vs Classification

My goal is to see the benefit of metric learning for retrieval tasks as opposed to a simple classification task. I will train a simpleCNN model with different metric losses to see how it affects the clustering of the data. 

I chose CIFAR10 dataset for carrying out this experiment and visualize the embeddings using UMAP. This experiment was motivated by reading the wonderful survey -- https://hav4ik.github.io/articles/deep-metric-learning-survey

## Experiments:
- [Classification](./train_classification.ipynb)
- [Metric Learning with Contrastive Loss](./train_contrastive.ipynb)
- [Metric Learning with Triplet Loss](./train_triplet.ipynb)
<!-- - [Metric Learning with Triplet Loss and Hard Negative Mining](./train_triplet_hard_neg_mining.ipynb) -->
<!-- - [Metric Learning with Triplet Loss, Center Loss and Hard Negative Mining](./train_triplet_hard_neg_mining_and_center_loss.ipynb) -->
<!-- - [Metric Learning with Quadruplet Loss](./train_quadruplet.ipynb) -->