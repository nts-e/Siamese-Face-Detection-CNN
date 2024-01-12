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


## Network Architecture
The siamese network was recreated based on the architecture presented in the paper. The twin branch of the network consists of four 2D convolution layers,
each followed by max pooling (except for the last layer), with a rectified linear unit (ReLU) activation, followed by a dense fully connected layer. Each image of the pairs is propagated
through the network, and the resulting feature vectors are then passed through a Lambda layer that performs an absolute subtraction of the feature vectors. Subsequently, a single neuron with a
sigmoid activation function is used for making the prediction.

The implementation of the network architecture can be found in the "Model" section of the Jupyter notebook. The main functions used in the implementation are as follows:
<p>
a. <code>conv_layer</code>: This function adds a set of Conv2D, BatchNormalization, Activation, and MaxPooling2D layers.<br>
b. <code>shared_network</code>: This function creates a twin branch composed of four convolution layers and a single fully connected layer with 4096 neurons.<br>
c. <code>create_model</code>: This function creates an instance of the shared network, and then feeds it with two images (face 1 and face 2) for comparison. It evaluates the distance between the feature vectors
using a Lambda layer as described above and outputs a sigmoid prediction.<br>
</p>


## Experiments
Six sets of hyperparameters have been defined, some of which were suggested in the paper, while others
are new based on the specific task of face image recognition. The task here is different from the one in the paper, which focused on simpler images. The
hyperparameters I experimented with included batch sizes (8, 16, 32, 64), learning rates
(1e-3, 1e-4, 5*1e-5, 1e-5), dropout rates for the fully connected layer (0, 0.2, 0.3), different
convolution layer weights and biases initializations (including the ones from the paper with
weights of 0 mean and standard deviation of 0.01, bias of 0.5 mean and standard deviation of
0.01, as well as alternative ones using the default glorot_uniform weights and zero bias), and
different dense layer weights and biases initializations (including the ones from the paper with
weights of 0 mean and standard deviation of 0.2, bias of 0.5 mean and standard deviation of 0.01,
as well as alternative ones using the default glorot_uniform weights and zero bias). I also
experimented with two convolution configurations of kernels, filters, and max pools, with one
being the architecture from the paper and the other being a new suggestion of a more standard
architecture.

In total, 768 different hyperparameter sets have been tested. I used the Adam optimizer,
binary_crossentropy loss, and measured accuracy as the evaluation metric. I also employed
two callbacks during training: EarlyStopping, which stops training when validation loss has
stopped improving with a patience of 10 epochs and min_delta of 0, and ReduceLROnPlateau,
which reduces the learning rate by 0.1 when validation loss has not improved for at least 5
epochs.

Notebook ran on Google Colab, with a Premium GPU and High RAM. The total training
time (768 fit operations) was 5.8 hours, with an average of 27.3 seconds per run.

Based on the validation accuracy, I tested the three best-performing configurations,
which had a validation accuracy higher than 0.752. Then, the classifier was retrained using these
three configurations on the entire training set (train + validation) until it reached the epoch that
had the best performance during the initial validation.

The top 25 best-performing configurations (out of the 768 configurations) are described in the
table below, with the first three rows being the ones used for the final testing.

<img width="1000" alt="image" src="https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/6fcabb9a-eb98-4df9-b75d-a6e00608ea54">

Among the 3 best configurations, configuration 60 had a final test accuracy of
0.706. It is implemented with a batch size of 8, an initial learning rate of 10e-4 and no drop outs.
In addition, both its weight and biases have been initialized with the default keras initializations
and not as recommended in the paper. Best epoch was the 9th. The plots below show the
accuracy and the training and validation metrics during training, and the final test values.

<img width="500" alt="image" src="https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/8ad6f8af-58b5-4142-b910-f9d27c33c353">

## Best Model Analysis

The confusion matrix results indicate that the best model achieved a relatively high true negative
and true positive count, but also had a notable false positive and false negative count. The
analysis of the false positives and false negatives revealed interesting insights, such as the
importance of nose shape and other factors in determining the correct classification of image
pairs.

![image](https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/4693d9b7-2c60-4a0a-880e-c57975030c56)

The figure below presents the test set prediction probabilities histogram, where the blue color
represents the negative class cases and the orange color represents the positive class cases. Gray
areas indicate overlapping regions. Interestingly, both classes exhibited a relatively high severe
error rate, with the positive class having many cases between 0-0.1 and the negative class having
many cases between 0.9-1.0. This indicates that the classifier is quite deterministic, and
adjusting the prediction threshold may not be effective in improving its performance.

![image](https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/6689d04a-6c98-4af1-b1c8-f1eec94ced36)

To gain further insights into the successes and failures of the classifier, I printed six pairs of
images for four different cases.
First, I examined the true negatives with high certainty, where the classifier accurately
determined that the persons in the images were not the same, as the faces appeared significantly
different from each other.

![image](https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/a48c642f-4196-481a-a8ed-c5e32800f937)

Next, I looked at the true positives with high certainty, where the classifier correctly identified
that the persons in the images were the same, as the faces appeared very similar to each other.
Notably, we can observe an image pair featuring Pierce Brosnan with a woman, although a different
woman in each image, which added an interesting aspect to the results.

![image](https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/f9310831-b431-49d2-b023-a29674d36dc9)

Examination of the false negatives with the highest certainty shows the biggest mistakes
in not identifying that the persons were the same in both images. I found that differences such
as aging, sunglasses, and different hairstyles were evident in these cases, although image pairs
270 and 553 were particularly surprising as they appeared quite similar to the human eye. I
speculate that this discrepancy might be related to lighting conditions or camera angles.

![image](https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/d2496ad1-2cec-4d63-8059-25e166c2cdd4)

Finally, I examined the false positives with the highest certainty, which were the biggest
mistakes in not identifying that the persons were not the same in both images. Notice that
the open mouth of the persons in image pair 152 might have confused the algorithm. Other
examples were also surprising, with the nose shape appearing to be a crucial factor in some
cases. For instance, in image pair 545, the nose shape of the two persons was similar, which
could have led to the misclassification.

![image](https://github.com/nts-e/Siamese-Face-Detection-CNN/assets/107881111/1323aab9-c07d-45e3-923f-8d76c363df6c)







