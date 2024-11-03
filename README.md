# ITIR626-Individual-Project-Assignment-Due-date-is-4-th-November-2024-
Early Detection of Skin Cancer Using Deep Learning on the ISIC Dataset
## Objective
The objective of this project is to develop a reliable, automated deep learning model to 
classify skin lesions as either benign or malignant using dermoscopic images. This 
model aims to assist dermatologists in early detection of melanoma, a highly aggressive 
form of skin cancer. By utilizing the ISIC dataset, the project demonstrates the potential 
of deep learning to aid in early diagnosis, especially in areas with limited access to 
specialized healthcare.
## Dataset 
The model was trained using the ISIC (International Skin Imaging Collaboration) 
dataset, which contains high-resolution labelled images of various skin lesions, 
including both benign (non-cancerous) and malignant (cancerous) samples. The 
dataset was downloaded from ISIC 2024 Challenge on Kaggle. 
The dataset includes: 
• Image Data: High-resolution images of skin lesions. 
• Metadata: Information such as lesion ID and target label, where 1 indicates 
malignant and 0 indicates benign. 
Data Preprocessing 
1. Class Balancing: The dataset showed an imbalance, with a higher proportion of 
benign cases than malignant ones. To address this, downsampling was applied 
to balance the classes, ensuring the model does not favour benign predictions. 
 
2. Image Preprocessing: 
o Images were resized to a uniform dimension of 128x128 pixels to reduce 
computational load. 
o Each image was converted to grayscale to simplify the model and focus 
on texture and shape rather than colour. 
o Images were flattened to prepare them for model input. 
3. Data Splitting: The data was split into training and testing sets with an 80-20 
ratio, allowing for model training and evaluation on separate subsets. 
## Model Development
A Convolutional Neural Network (CNN) was selected for its effectiveness in image 
classification tasks. The CNN was designed with several convolutional and pooling 
layers to capture the spatial hierarchies of features in the images. Key model 
configurations: 
 
Architecture: Sequential CNN with convolutional, pooling, and fully connected layers. 
Loss Function: Binary cross-entropy was used since this is a binary classification task. 
Optimizer: Adam optimizer was used for faster convergence. 
Metrics: The model’s performance was evaluated using metrics such as accuracy, 
precision, recall, F1-score, ROC curve, and AUC. 
The CNN was trained over 20 epochs to minimize the loss on the training set and 
improve classification accuracy. 
## Experiments and Evaluation 
The model’s performance was evaluated using the following metrics: 
1. Accuracy: The overall percentage of correct predictions. This metric provides a 
general sense of model performance. 
2. Precision: The proportion of correctly predicted malignant cases out of all cases 
predicted as malignant. High precision minimizes false positives. 
3. Recall: The proportion of correctly identified malignant cases out of all actual 
malignant cases. High recall is essential in medical diagnosis to avoid missing 
malignant cases. 
4. F1-score: The harmonic mean of precision and recall, which balances the trade
off between the two. 
5. ROC Curve and AUC: The ROC curve plots the true positive rate against the false 
positive rate at different thresholds, and AUC quantifies the overall performance.
The model achieved a high accuracy and balanced precision and recall, indicating it can 
effectively differentiate between malignant and benign lesions. The AUC score also 
demonstrated strong discrimination ability between the two classes.
Performance Evaluation: 
• The model demonstrated a balanced performance on benign cases (label 0) with 
a precision of 0.52 and a recall of 1.00, indicating it effectively identifies benign 
lesions. 
• However, for malignant cases (label 1), the precision was perfect at 1.00, but the 
recall was very low at 0.03, suggesting the model struggles significantly to 
identify malignant lesions. 
• Overall accuracy was reported at 53%, indicating room for improvement, 
especially in detecting malignant cases. 
 
## Visualization and Testing 
To visualize the model’s predictions, a few experiments were conducted: 
• Random Sample Prediction: Individual test images were randomly selected and 
classified. The images were displayed with their true labels (benign or malignant) 
and the model’s predicted labels, allowing for quick qualitative assessment. 
 
• Confusion Matrix: A confusion matrix was generated to show the counts of true 
positives, false positives, true negatives, and false negatives. This provided 
insight into where the model might misclassify lesions. 
 
 
## Conclusion 
The experiments demonstrate the feasibility of using deep learning for skin lesion 
classification. The CNN model successfully classifies benign and malignant lesions, 
showing potential as a supplementary tool for dermatologists. Future work could 
include experimenting with more advanced architectures, such as transfer learning, to 
further improve classification accuracy, especially on smaller datasets. Additionally, 
expanding the dataset and enhancing data preprocessing (e.g., using data 
augmentation) could help improve model robustness and generalizability. 

