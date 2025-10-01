# DL-Final-project---CNN-and-CIFAR-10-Classification
This is my submission for the final project of Introduction to Deep Learning. I use several CNN architectures to attempt to classify CIFAR-10 images.

In this notebook we attempt to classify photos from the widely used [CIFAR10](https://keras.io/api/datasets/cifar10/) dataset. The dataset contains 60,000 images (50,000 training set and 10,000 test set) in 10 categories. The categories are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck and the training and test image sets are divided equally between the categories (5,000 training, 1,000 test each). The images in the dataset are 32x32 pixel RGB images.

With such small images and high intra-class variability, accurate classification is a challenging task. To accomplish this. We train a convolutional neural network (CNN) to attempt to classify the images into the 10 categories. Convolutional Neural Networks (CNNs) are particularly well-suited to image classification tasks because they automatically learn hierarchical features such as edges, textures, and object parts. Our goal is to evaluate the performance of a custom CNN architecture and analyze its strengths and weaknesses.

Data source: https://keras.io/api/datasets/cifar10/

# Data Inspection

The class names and their corresponding indices were taken directly from the documentation. While the documentation also specifies the dimensions and counts of images, we need to check to be certain. An inspection of the shape of the data confirms that we have 50,000 training and 10,000 test images of 32 by 32 pixels and 3 chanels for RGB.

# Exploratory Data Analysis

We also check the class counts to ensure the classes are well balanced. As the bar charts confirm, both the trainign and test sets are perfectly balanced across classes, so there is no need for any class balancing techniques.

<img width="989" height="590" alt="92b53759-5bb1-45d8-adce-c66ca9684c9e" src="https://github.com/user-attachments/assets/f0f0585e-5947-4e06-84ac-33b6c4c2353d" />

<img width="989" height="590" alt="e22a70cc-b059-47ca-9f5c-87a05f5d9fdc" src="https://github.com/user-attachments/assets/ed23744a-a481-4d86-87a0-a33c309dddfa" />

We visualize random examples of each of the classes as a sanity check. As one would expect, the 1,024 pixel images are not very clear, but for most images they are clear enough for a human to categorize them. There are some noteable exceptions however, where minus the class label, we would have difficulty saying which class the image belongs to with any certainty.

<img width="950" height="410" alt="1f6e3fb1-106d-4534-b1ed-eb234d8bb9b1" src="https://github.com/user-attachments/assets/424ae07e-3b91-47e1-8281-e6dfae2c999a" />

Finally, we examine the color channels of the images by class to see if there are any obvious trends. The blue channel intensity tends to be higher in airplane and ship images, which is likely explained by more frequent sky and water backgrounds respectively. Airplane images in general tend to be more intense across all channels. A tenuous explanation for this could be that airplane photos tend to be taken towards the sky allowing more sunlight into the camera, but this is merely our amateur theory. Apart from these, there don't seem to be any obvious trends in class versus color meaning our model may have to rely mainly on shapes and physical patterns in the image rather than color.

<img width="1489" height="490" alt="b30d2d9c-2c00-4db8-bdc9-fc44c33d99ee" src="https://github.com/user-attachments/assets/efff2da6-30f6-4dd8-b1d6-65f4d0d54bf4" />

# Models

Our main strategy was to start with a simple "baseline" CNN to give an idea of what a very simple model could do. From there we would add (or later subtract) layers to find a model that could provide over 0.8 accuracy.

We chose a CNN architecture for this project because it is specifically designed to handle image data efficiently and accurately. CNNs are excellent at detecting patterns in images and that is precisely what this task requires. Importantly CNNs are good at recognizing these patterns even if they appear in different parts of images or are flipped or rotated by some degree. As this is an introduction to deep learning class, the CNNs we will use will be relatively simple, but among the architectures we studied, CNNs are unrivaled at computer vision tasks.

To begin we prepared the data by splitting the training data into a training (0.8) and validation (0.2) set during training. The trained model would then be used on the test data from the original dataset. In order to avoid overfitting we introduced some data augmentation techniques. We introduced random flips and across the y-axis and random translations. This improved model performance. Other data augmentation techniques such as random rotations and random were tried, but these seemed to significantly harm both the model's convergence and its generalizability. Our explanation for this is the size of the dataset images. With such small images, these distortions likely introduce a more significant change to the image than a similarly scaled distortion on a large image. In other words, the model already doesn't have much to work with, taking anything away makes the job of classification significantly harder.

## Simple model (baseline)

We started with a simple CNN model based on the basic model demonstrated in the class lectures. It consists of 3 convolutional layers joined by 2D max pooling layers. The convolutional layers are our feature detectors and the max pooling to keep only the strongest detected features. This is follwed by a 2 layer fully connected network with an output vector of length 10. The 10 variables in the output vector represent the model's estimation of the the probabilty that the image contains that class.

<img width="724" height="398" alt="image" src="https://github.com/user-attachments/assets/23433283-b03f-433c-8be1-a40e497a3b7b" />

### Results

As expected the model performed much better than random chance, but still left a lot to be desired. For such a lightweight, shallow network it achieved a respectable 0.76 accuracy in classifying the test set. As we can see in the confusion matrix below, the model did struggle slightly in distinguishing cats and dogs. In fact, the model had a little more difficulty in identifying cats compared to any other class, as demonstrated in the bar chart of misclassifications below. With this model as a starting point, we attempted to expand the model's layers to achieved a better predictive model, our personal goal being above 0.8 accuracy.

<img width="981" height="449" alt="3861211b-3e49-497a-9e12-42855728e91c" src="https://github.com/user-attachments/assets/44bee9de-003c-45f4-bda4-23e97f9ffb90" />

<img width="523" height="374" alt="image" src="https://github.com/user-attachments/assets/e649ba3f-4a5e-4ce7-af5b-0171ba5d8ba2" />

<img width="584" height="477" alt="f7c90dcd-36b7-4609-a06b-784d511319a8" src="https://github.com/user-attachments/assets/76846784-5ef9-4984-996e-d886ffa509a5" />

<img width="990" height="590" alt="aa6fb52c-fb36-4dd4-ab42-1725b971c509" src="https://github.com/user-attachments/assets/8110105e-a776-421a-8cfe-b924831de1bc" />

## Improved model

With our baseline results set, we attempted to develop a model with significantly better performance. The first step was to extend each convolutional block by adding a second convolutional layer and introduce batch normalization after each one. As stated above the convolutional layers learn the features and the batch normalization was included to stabilize the training. The presence of a second convolutional layer in each block should allow the model to learn more abstract and complex features which should allow it to better recognize each of the 10 classes. We also included dropout layers at the end of each block in order to prevent overfitting.

<img width="735" height="1513" alt="image" src="https://github.com/user-attachments/assets/5308fe45-40ff-406f-90fa-95beab6c08c7" />

### Results

As expected the model performed much better than random chance, but still left a lot to be desired. For such a lightweight, shallow network it achieved a respectable 0.76 accuracy in classifying the test set. As we can see in the confusion matrix below, the model did struggle slightly in distinguishing cats and dogs. In fact, the model had a little more difficulty in identifying cats compared to any other class, as demonstrated in the bar chart of misclassifications below. With this model as a starting point, we attempted to expand the model's layers to achieved a better predictive model, our personal goal being above 0.8 accuracy.

<img width="981" height="449" alt="3861211b-3e49-497a-9e12-42855728e91c" src="https://github.com/user-attachments/assets/bc5fca53-f6aa-4961-8dca-690846a0e6b3" />

<img width="509" height="370" alt="image" src="https://github.com/user-attachments/assets/56427ce1-3fd5-4d50-8955-a54401c39967" />

<img width="584" height="477" alt="f7c90dcd-36b7-4609-a06b-784d511319a8" src="https://github.com/user-attachments/assets/a713f383-1eae-40b3-aab7-409c96f33e75" />

<img width="990" height="590" alt="aa6fb52c-fb36-4dd4-ab42-1725b971c509" src="https://github.com/user-attachments/assets/b8ac0e38-6a1e-4cef-8991-9c782065a213" />

## Resnet-Style model

While researching for this project, the authors came across an article titled "[A Practical Comparison Between CNN and ResNet Architectures: A Focus on Attention Mechanisms
](https://medium.com/@leonardofonseca.r/a-practical-comparison-between-cnn-and-resnet-architectures-a-focus-on-attention-mechanisms-cee7ec8eca55)" that compared a standard CNN to a ResNet like architecture for classifying cat and dog images. This inspired us to adapt their approach for our own dataset. The key architectural difference between our earlier CNN and this ResNet implementation lies in the introduction of residual blocks with skip connections. In a conventional CNN, each convolutional block processes its input and passes the transformed output directly to the next layer. In contrast, in a ResNet block the output of the convolutional path is added to the original input before applying the activation function. This “shortcut” connection allows the network to preserve information from earlier layers, reducing the risk of losing important features and mitigate the vanishing gradient problem.

Each residual block ressembles our convolutional blocks from the previous model with two convolutional layers, each followed by batch normalization and dropout. The key difference is the skip path which is included after the second convolution-batch normalization pair. The dense layer mirrors our CNN exactly.

<img width="502" height="1380" alt="image" src="https://github.com/user-attachments/assets/330c9869-a037-4c0d-b3f5-13a4be627a5a" />

### Results

<img width="981" height="449" alt="063a5e80-faeb-4e1a-844e-a517220e5108" src="https://github.com/user-attachments/assets/ef6dfb70-0d96-45f0-b52c-fe995faffa82" />

<img width="509" height="374" alt="image" src="https://github.com/user-attachments/assets/43fa8f25-f540-4b88-89f6-81cc13cfae1e" />

<img width="584" height="477" alt="ae6d0997-7dd3-4cae-bd7a-71a775093427" src="https://github.com/user-attachments/assets/10f7cc52-4ac3-4d8a-b68b-78b819195482" />

<img width="990" height="590" alt="7f9d8ed1-862a-49d6-a8df-265d6953aa39" src="https://github.com/user-attachments/assets/52811f75-d11d-45a7-a349-354f08988e79" />

# Results and Discussion

Our results show that a relatively simple CNN can achieve strong performance on CIFAR-10. This highlights the effectiveness of convolutional layers in extracting visual features, even from small, low-resolution images where those features are less distinct compared to larger datasets.

We also experimented with residual connections inspired by the ResNet architecture. In our case, these connections did not provide clear performance benefits, which contrasts with findings from other studies, including the article on which our ResNet-inspired model was based. However, our architecture was relatively lightweight; it is possible that larger, deeper, or more state-of-the-art networks would benefit more from residual connections. Importantly, the addition of residual connections did not harm performance in any significant way.

A key limitation observed was the model’s difficulty with fine-grained distinctions between visually similar classes. For example, cats and dogs share many features and often appear against similar backgrounds, leading to higher misclassification rates between these classes. Similarly, cars and trucks were sometimes confused, as they share structural similarities and background contexts. Notably, cats were misclassified more often than any other class.

To better understand these misclassifications, we examined Grad-CAM visualizations of several randomly selected cat images. Grad-CAM overlays a heatmap onto the original image, highlighting the regions that most strongly influenced the model’s decision. Warmer colors (red, orange, yellow) indicate areas of high influence, while cooler colors (blue, purple) indicate low influence. In many of the cat images, the heatmaps suggested that the model was paying more attention to the background than to the animal itself. This suggests that the CNN may be relying too heavily on contextual cues rather than focusing on the defining features of the object.

Overall, these findings suggest that while our CNN is effective at learning general visual patterns in CIFAR-10, it still struggles with subtle, fine-grained distinctions and occasionally relies on background context rather than the object itself. This highlights both the strengths and limitations of our approach, setting the stage for potential improvements through more advanced architectures, more robust data augmentation, or transfer learning.

<img width="359" height="1965" alt="b81b69bb-bbb4-4211-846b-57dd8771584e" src="https://github.com/user-attachments/assets/a2b2de37-d191-4e0e-9262-4d4e8750f5c9" />

# Conclusions

This project demonstrated that CNNs can achieve strong performance on the CIFAR-10 dataset, even with relatively simple architectures. While residual connections did not significantly improve results in our lightweight model, they remain a promising direction for deeper networks.

The main challenges were in fine-grained class distinctions (e.g., cats vs. dogs) and the model’s tendency to rely on background features, as revealed by Grad-CAM visualizations. These findings suggest that while the approach is effective overall, there is room for improvement.

Future work should explore more robust data augmentation, further hyperparameter tuning, and transfer learning with deeper architectures to further improve accuracy and robustness.

# References

CIFAR-10 Data source: https://keras.io/api/datasets/cifar10/

[A Practical Comparison Between CNN and ResNet Architectures: A Focus on Attention Mechanisms
](https://medium.com/@leonardofonseca.r/a-practical-comparison-between-cnn-and-resnet-architectures-a-focus-on-attention-mechanisms-cee7ec8eca55)
