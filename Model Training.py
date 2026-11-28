#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install scikit-learn joblib')


# In[8]:


import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading master data...")
df = pd.read_csv('master_custom_data.csv')

# Create groups so augmented versions stay with their original image
df['Group_ID'] = df['Image_Path'].str.replace('_aug', '')
X = df.drop(columns=['Label', 'Image_Path', 'Group_ID']).values
y = df['Label'].values
groups = df['Group_ID'].values

# Split 80% Train, 20% Test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training on {len(X_train)} images...")
print(f"Testing on {len(X_test)} unseen images...")

# Train SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_model.fit(X_train, y_train)

# Test
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=====================================")
print(f"TRUE CUSTOM ACCURACY: {accuracy * 100:.2f}%")
print("=====================================")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Save
joblib.dump(svm_model, 'master_svm_model.pkl')
print("Model saved as 'master_svm_model.pkl'")


# In[1]:


import pandas as pd
df = pd.read_csv('master_custom_data.csv')
df['Group_ID'] = df['Image_Path'].str.replace(r'_aug\d*', '', regex=True)
print(df.groupby('Label')['Group_ID'].nunique())


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
print("Loading data...")
df = pd.read_csv('custom_alphabet_data.csv')
X = df.drop(columns=['Label']).values
y = df['Label'].values

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} samples...")
print(f"Testing on {len(X_test)} samples...")

# Train SVM with RBF Kernel (Best for landmarks)
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Test accuracy
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=====================================")
print(f"OFFLINE TEST ACCURACY: {accuracy * 100:.2f}%")
print("=====================================")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(svm_model, 'svm_alphabet_model.pkl')
print("Model saved as 'svm_alphabet_model.pkl'")


# In[5]:


import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the data
print("Loading data...")
df = pd.read_csv('custom_alphabet_data.csv')

# We need a Group ID to keep original and augmented images together.
# We will create a fake group ID based on the row index (every 9 rows belong to one original image)
# Since we created 1 original + 8 augmented = 9 rows per image.
df['Group_ID'] = df.index // 9 

X = df.drop(columns=['Label', 'Group_ID']).values
y = df['Label'].values
groups = df['Group_ID'].values

# 2. GroupShuffleSplit (Prevents leakage)
print("Splitting data with GroupShuffleSplit...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Training on {len(X_train)} samples...")
print(f"Testing on {len(X_test)} samples...")

# 3. Train the SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 4. Test
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=====================================")
print(f"TRUE OFFLINE ACCURACY (No Leakage): {accuracy * 100:.2f}%")
print("=====================================")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 5. Save
joblib.dump(svm_model, 'svm_alphabet_model.pkl')
print("Model saved as 'svm_alphabet_model.pkl'")


# In[ ]:




