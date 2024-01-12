# Siamese-Face-Detection-CNN

An implementation of a convolutional neural network (CNN) for one-shot facial recognition using a siamese network approach, based on the [work](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) of Koch et al.
The authors' paper explores a method for learning siamese neural networks to rank similarity between input images. In this work, I recreate the siamese network architecture and train it to discriminate between pairs of faces from a [dedicated dataset](https://talhassner.github.io/home/projects/lfwa/index.html) with a predefined train/test split. 

I experimented with a wide range of hyperparameters, including those proposed in the original paper as well as my own modifications. The experiments resulted in an accuracy score of 0.706 on the test set, using a configuration that included similar
convolutional filters and kernels as in the paper, but with differences in batch size, learning rate, and weight initialization.

## Dataset
The dataset used in this assignment consists of a folder containing images of faces of people,
along with two text files that specify which image pairs should be used for comparison in the
train and test sets. The images have an original size of 250 x 250 pixels. The dataset comprises a
total of 2200 pairs of images in the training set and 1000 pairs in the test set, with the class labels
being binary (1 for same person, 0 for not the same person), and the classes are evenly
distributed in the dataset.
<p>
To handle the dataset, I implemented a customized function <code>load_data</code> that reads the
text files and prepares train and test NumPy arrays of the images. Within this function, I resized
the images to 105x105 pixels to match the network architecture presented in the paper.
Additionally, I scaled the pixel values of the images to fall within the range of 0 to 1.
Furthermore, I split the training data into 80% for training and 20% for validation to evaluate
the performance of the model during training.
</p>
Randomly sampled pairs of images are presented in the figure below, arranged vertically, and labeled with their corresponding class values:

<img width="700" alt="image" src="https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/ba1762fd-117f-47ed-8cd6-1078d384f315">
