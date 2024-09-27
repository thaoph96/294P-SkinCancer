# DETECTING SKIN CANCER USING LESSION IMAGE
Jasmine Pham
### Executive Summary
The primary objective is to develop a binary classification AI algorithm capable of accurately detecting skin tumors using diagnostically labeled images, patient demographics, and lesion feature data. The images resemble close-up smartphone photos commonly used in telehealth applications. Skin cancer can be deadly if not detected early, but many populations lack access to specialized dermatologic care. Studying this dataset can act as the pre-diagnostic which help identify individuals who need to see the clinic for further diagnostic and receive treatment early
### Introduction
Skin cancer is one of the most common cancers globally, with early detection being crucial for successful treatment. However, access to specialized dermatological care can be limited, especially in remote or underserved areas. In such settings, telehealth has emerged as a valuable tool, allowing patients to submit photos of skin lesions for remote evaluation. The quality of these images, often captured by smartphones, is usually adequate for preliminary assessments but poses challenges for accurate diagnosis.
To address this gap, we propose developing image-based algorithms designed to identify histologically confirmed skin cancer cases using single-lesion crops from 3D total body photos (TBP). These photos, which closely resemble the smartphone images submitted for telehealth purposes, offer a promising avenue for enhancing early detection and triage in settings lacking specialized care.
By focusing on binary classification algorithms, our goal is to create a tool that can reliably distinguish between malignant and benign lesions, even in environments with limited resources. Such an algorithm would be instrumental in prioritizing patients who require urgent care, thereby improving outcomes through earlier intervention.
### Methodology
1.	__Data Description:__
•	The dataset used for this study is sourced from a Kaggle competition focused on Skin Cancer Detection using 3D Total Body Photography (3D-TBP) technology. This dataset contains a large collection of images, including single-lesion crops that simulate close-up smartphone photos.
•	Train-image (SLICE-3D): Contains 401,059 JPEG images of skin lesion crops extracted from 3D TBP used for training
•	Train-metadata.csv: Includes 401,059 metadata entries including labels (target), age, sex, general anatomical site, patient identifiers, clinical size, and other relevant fields obtained from the TBP Lesion Visualizer. These attributes are additional information about the patients and the lesions that may be useful for training.
•	Test-image: Data stored in a single HDF5 file with isic_id as the key, with 3 test examples used for inference validation. An additional hidden test set of approximately 500,000 images is used for evaluation when submitting on Kaggle.
•	Test-metadata: Metadata for the test subset consisting of 3 entries of information of images in test-image.
2.	__Model Description:__
    - __Data Preprocessing__: Data preprocessing and augmentation are crucial for improving model performance. 
o	Image Preprocessing: We utilized the Albumentations library to apply a series of transformations, including resizing, normalization, and various augmentations such as CLAHE, flipping, affine transformations, rotation, sharpening, and brightness/contrast adjustments. This preprocessing ensures that the model can generalize well across diverse image variations.
o	Metadata preprocessing: Drop unnecessary data features. Missing values in age_approx, sex, and anatom_site_general are imputed with the median or 'Unknown' respectively, to ensure completeness. New features are engineered, such as sex_anatom_site, combining gender and anatomical site information, and tbp_lv_mean and tbp_lv_std to summarize TBP columns if present. The age_group feature categorizes age into bins, and the size_to_area_ratio feature quantifies the relationship between clinical size and TBP area if relevant columns exist. Finally, categorical variables are converted into dummy variables to facilitate machine learning modeling.
•	CNN (Resnet 18): To enhance the detection of skin cancer, we implemented a Convolutional Neural Network (CNN) using PyTorch. This model leverages a pre-trained ResNet-18 architecture, tailored for binary classification to differentiate between cancerous and non-cancerous skin lesions. The model's effectiveness is evaluated through both training and validation phases, with accuracy metrics visualized to assess performance.
o	Training and Evaluation: The model is trained using a standard training loop with PyTorch, incorporating both training and validation phases. During training, we track and plot the training and validation accuracy to monitor the model’s performance over epochs.
o	Training Results: After training the model for the specified number of epochs, the following results were obtained:
o	Training Accuracy: 67.61%
o	Validation Accuracy: 69.43%
o	These results reflect the model’s performance in correctly classifying skin lesions into cancerous or non-cancerous categories.
o	Visualization: To provide further insights into the model's performance, accuracy metrics were plotted over the training epochs. The visualization below shows the progression of both training and validation accuracy:
![Training and Validation Accuracy](../294P-SkinCancer/image.png)
This plot demonstrates how the model's accuracy improved over time, with the validation accuracy being slightly higher than the training accuracy. This suggests that the model generalizes well to the validation set and can identify skin cancer cases with a reasonable degree of accuracy.

•	Balanced Random Forest Classifier (BRF) Results: A Balanced Random Forest Classifier (BRF) was applied to the dataset for comparison. This model is well-suited for handling imbalanced datasets and was evaluated using accuracy and AUC-ROC metrics. The Classifier was configured with a random state of 42 to ensure reproducibility. The model was trained using the metadata dataset.
o	Results
	AUC-ROC Score: 0.9922
	Accuracy Score: 0.8677
The AUC-ROC score of 0.9922 indicates exceptional performance in distinguishing between cancerous and non-cancerous cases, with the model being able to correctly rank the probability of cancerous cases nearly perfectly. The accuracy score of 86.77% further highlights the model’s strong performance in correctly classifying cases.

•	Logistic Regression Model Results
In addition to the Convolutional Neural Network (CNN) and Balanced Random Forest (BRF) models, a Logistic Regression model was applied to the preprocessed metadata dataset. This model was evaluated based on accuracy and AUC-ROC metrics to assess its performance in skin cancer detection.
o	Results
	Training Accuracy: 99.90%
	Test Accuracy: 99.91%
	Test AUC-ROC: 0.8799
These results demonstrate the Logistic Regression model's exceptional performance in predicting skin cancer cases based on metadata. The extremely high accuracy values indicate that the model performs exceedingly well on both training and test datasets. The AUC-ROC score of 0.8799 highlights the model’s strong capability in distinguishing between cancerous and non-cancerous cases.

Claims and Evidence
Claim 1: Machine learning model on metadata provide a high accuracy tool to predict skin cancer 
Both models (Balanced random forest and Logistic Regression) are giving high accuracy score on predicting the chances of having cancer in patients. The BRF classifier achieves an AUC-ROC score of 0.9922 and an accuracy score of 86.77%. While the Logistic regression have AUC-ROC score of 0.8799 and an accuracy score of 99.91%.
Claim 2: Further Tuning Required for CNN Models
The current Convolutional Neural Network (CNN) model requires further tuning or alternative approaches to improve its performance in skin cancer detection. The CNN model, when applied to the image dataset, achieved a training accuracy of 67.61% and a validation accuracy of 69.31%. These results indicate that the model’s performance is below the desired threshold for reliable skin cancer detection. The relatively modest accuracy scores suggest that the CNN model may not yet be fully optimized for the specific characteristics of the skin cancer image data. This performance could be due to various factors such as model architecture, hyperparameters, or the quality of the data. Therefore, additional model tuning, including adjustments to hyperparameters, changes to the network architecture, or enhanced data augmentation techniques, may be necessary to improve performance.
Claim 3: High-Quality Image Preprocessing Improves Model Performance
Claim: Effective image preprocessing techniques significantly enhance the performance of CNN models for skin cancer detection.
•	Preprocessing Techniques: The CNN model utilized a series of preprocessing steps, including CLAHE (Contrast Limited Adaptive Histogram Equalization), resizing, normalization, and various augmentations such as rotation, flipping, and brightness/contrast adjustments.
•	Impact on Performance: Proper preprocessing improves the quality of input images, making the features more discernible and aiding the model in learning better representations. The inclusion of these techniques in the preprocessing pipeline likely contributed to the CNN's ability to extract relevant features from the images.
Image preprocessing is crucial in preparing data for CNN models. By enhancing image quality and augmenting data, preprocessing techniques help the model generalize better and improve its performance. Effective preprocessing can mitigate issues like image noise and variability, leading to better accuracy and robustness in detecting skin cancer.
4. Addressing Previous Action Items
In the previous presentation, there was a need for more comprehensive evidence regarding the analysis conducted. Since then, I have successfully completed the following actions:
1.	Data Exploration: I have conducted an exploratory analysis of the data, examining its characteristics and underlying patterns.
2.	Logistic Regression Model: I have trained and evaluated a Logistic Regression model on the metadata, achieving significant results that demonstrate its effectiveness in skin cancer prediction.
3.	Convolutional Neural Network: I was still in the process of training the Convolutional Neural Network (CNN). 
These steps address the previous gaps and offer a comprehensive view of the analysis and model performance.
Conclusion
This report has explored various machine learning models for detecting skin cancer, focusing on metadata-driven and image-based approaches. Our findings demonstrate that:
1.	Metadata-Driven Models: The Balanced Random Forest (BRF) classifier and Logistic Regression model showcase exceptional performance in predicting skin cancer based on patient metadata. The BRF classifier achieved an outstanding AUC-ROC score of 0.9922 and an accuracy of 86.77%, while the Logistic Regression model delivered near-perfect accuracies of 99.90% (training) and 99.91% (test), along with an AUC-ROC of 0.8799. These results highlight the efficacy of metadata-driven models in classifying skin cancer, particularly when dealing with imbalanced datasets.
2.	Image-Based Models: The Convolutional Neural Network (CNN) applied to image data demonstrated lower performance, with a training accuracy of 67.61% and a validation accuracy of 69.31%. This suggests that the current CNN model requires further tuning or alternative approaches to enhance its performance. Effective image preprocessing has been shown to improve CNN performance, underscoring the importance of high-quality data preparation.
