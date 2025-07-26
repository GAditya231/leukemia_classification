# leukemia_classification
Leukemia Classification using ResNet50
Project Overview
This project develops an image classification model to detect Acute Lymphoblastic Leukemia (ALL) from microscopic blood smear images. It utilizes transfer learning with the ResNet50 convolutional neural network to classify images as either ALL (leukemia) or HEM (healthy).

Key Features
Automated Image Preprocessing: Includes steps like CLAHE for contrast enhancement, Otsu's thresholding for segmentation, smart cropping to focus on cells, and resizing.

Transfer Learning: Leverages a pre-trained ResNet50 model (on ImageNet) as a powerful feature extractor.

Deep Learning Model: A Keras sequential model built on ResNet50 with custom classification layers and dropout for regularization.

Optimized Training: Uses Adam optimizer, binary_crossentropy loss, Early Stopping, and Model Checkpointing.

Performance Evaluation: Provides test accuracy, classification report, and confusion matrix.

Training Visualization: Plots model accuracy and loss history.

Dataset
The model is trained on the C-NMC Leukemia dataset, containing ALL and HEM blood smear images.

Dataset Source: Kaggle - Leukemia Classification (C-NMC)

Results
The model effectively classifies leukemia images, achieving a test accuracy of approximately 99.91%.

Test Loss: 0.0051
Test Accuracy: 0.9991

Classification Report:

              precision    recall  f1-score   support

           0       1.00      0.99      1.00       678
           1       1.00      1.00      1.00      1455

    accuracy                           1.00      2133
   
Confusion Matrix:

True Negatives (Correctly predicted HEM): 672

False Positives (Predicted ALL, but was HEM): 6

False Negatives (Predicted HEM, but was ALL): 0

True Positives (Correctly predicted ALL): 1455


![Model confusion_matrix Plot](confusion_matrix_heatmap.plot.png)

 ![Model Accuracy Plot](model_accuracy_plot.png)
 
 ![Model Loss Plot](model_loss_plot.png)

Future Work
Explore advanced data augmentation techniques.

Experiment with other CNN architectures.

Implement cross-validation for more robust evaluation.

Contact
Name: GADITY231

Email: galinkiaditya231@gmail.com
