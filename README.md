Siddarth Rudraraju
sr88018@usc.edu

In this project, I developed a binary classification model using computer vision techniques to differentiate between images of muffins and chihuahuas. Inspired by a popular meme that humorously highlights the challenge of distinguishing these visually similar items, this project delves into the potential of computer vision in identifying differences within similar-looking classes.

Data: 
The dataset has two classes: 'muffin' and 'chihuahua'. For standardization, images were normalized to a uniform size, with pixel values appropriately scaled. To mitigate overfitting, I explored several data augmentation techniques, including random horizontal flipping, color jittering, rotations, affine transformations, and Gaussian blurs. However, upon observing a decrease in model accuracy with complex augmentations, I only did random horizontal flipping, which yielded the highest accuracy.

Model: 

I used the ResNet18 model due to its ability to efficiently learn through its deep architecture and residual blocks, addressing the vanishing gradient problem. This capability was crucial for learning the nuanced differences between muffins and chihuahuas. This depth is beneficial for our task because it allows the network to learn both the low-level features (such as edges and textures) and high-level features (like the overall shape of a muffin or a chihuahua) that are crucial for accurate classification. The final layer of ResNet18 was modified to suit our binary classification task. 

The training involved experimenting with epoch counts (10, 15, 20, 25) and learning rates (0.01, 0.1, 0.001). Doing an 80:20 split, I used validation data at each epoch to see how well the model was being trained to minimise to risk of overfitting. I finally decided that 20 epochs with a learning rate of 0.001 using the Adam optimizer was effective. Thus, I used these parameters and used the entire training data to train the model. Finally, I used the model for the test data. 

Model Evaluation/Results:

Accuracy was chosen as the evaluation metric due to the balanced nature of the dataset. I also used the cross-entropy loss function as it can measure the confidence of the predictions. With the given model parameters and features, the model finally achieved a 97.21% accuracy along with 0.0812 loss when used on the test data. When training, epoch 18 generated the best result at 94.13% accuracy and 0.1563 loss. I will save the model at the epoch with the highest accuracy which is the 18th epoch in this case. 



Discussion:
 
The model's high accuracy demonstrates its potential utility in practical applications. However, the project's limitations include potential biases in the dataset and the model's reliance on visual cues. Future work could explore more extensive datasets that cover more classes and also contain different types of data where context can be important such as videos. Although this project doesn’t necessarily solve social problems, it is a great educational example due to its interesting context which can potentially make more people interested in AI. This project also shows the potential that computer vision has and how it can be applied to more critical tasks where objects should be differentiated such as in medical imaging or in disaster detection. 

Continuation of this project would involve detailed hyperparameter tuning, incorporating a broader range of data augmentation techniques, and evaluating the model against a more diverse set of images. I hope to continue exploring how computer vision can be used in more human-centered or critical projects to solve social problems. My next steps would be to investigate more pre-trained models like YOLO, investigate this model with more classes, and try different data augmentation techniques. I tried using this model for ASL classification but due to the large amount of data the project has, it was hard for me to find the effective parameters due to the model training time length. I hope to continue seeing the model’s implications on these types of projects in the future. 

