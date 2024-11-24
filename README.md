# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO.: 212223240070</H3>
<H3>EX. NO.7</H3>
<H3>DATE: 07.10.24</H3>

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load Data**  
   Import and prepare the dataset to initiate the analysis workflow.

2. **Explore Data**  
   Examine the data to understand key patterns, distributions, and feature relationships.

3. **Select Features**  
   Choose the most impactful features to improve model accuracy and reduce complexity.

4. **Split Data**  
   Partition the dataset into training and testing sets for validation purposes.

5. **Scale Features**  
   Normalize feature values to maintain consistent scales, ensuring stability during training.

6. **Train Model with Hyperparameter Tuning**  
   Fit the model to the training data while adjusting hyperparameters to enhance performance.

7. **Evaluate Model**  
   Assess the model’s accuracy and effectiveness on the testing set using performance metrics.

## Program:
```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items_binary.csv"
data = pd.read_csv(url)

# Select features and target variable
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'

X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Not Suitable', 'Suitable'], yticklabels=['Not Suitable', 'Suitable'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/9c6c88a9-5aac-458c-a1da-39e44f32caad)
![image](https://github.com/user-attachments/assets/128a45ba-3fff-4bd1-ac08-d84c45bd4060)


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
