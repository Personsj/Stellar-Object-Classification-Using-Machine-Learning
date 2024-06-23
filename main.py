import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# Load data
data = pd.read_csv('star_classification.csv')


#if any value is not defined, it will replace with 0
data.fillna(0, inplace=True)


# Select features and target
# target: "class"
# features: in this case everything besides "class" and "spec_obj_ID" (Can be changed to experiment with accuracy)
feature_names = data.drop(columns=["class", "spec_obj_ID"], axis=1).columns
features = data[feature_names]

labels, unique = pd.factorize(data['class'])


# You can add these labels back to the DataFrame if needed
data['NumericLabels'] = labels


# 'unique' contains the mapping of categories to numbers
print("Numeric Labels:", labels)
print("The input features are: {} \nThere are {} input features in total.".format(feature_names, len(feature_names)))
print("The output feature is \'class\'. \nUnique Categories within \'class\' column: {}, and number of unique categories is {}".format(unique, len(unique)))


galaxies = data[(data['class'] == unique[0])]
qsos = data[(data['class'] == unique[1])]
stars = data[(data['class'] == unique[2])]


print("The number of datapoints classified as galaxy, quasar, or star are {}, {}, and {} respectively.".format(len(galaxies), len(qsos), len(stars)))


# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)


# # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# # Create and train the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)


# Get feature importances
importances = clf.feature_importances_


# Sort the features by their importance and select the top 10
indices = np.argsort(importances)[::-1]
top_10_feature_indices = indices[:10]
top_2_feature_indices = indices[:2]
top_10_feature_names = feature_names[top_10_feature_indices]
top_2_feature_names = feature_names[top_2_feature_indices]


# Create a new training dataset using only the top 10 features
X_train_top10 = X_train[:, top_10_feature_indices]




# Create and train a new decision tree classifier on the reduced dataset
#clf_top10 = DecisionTreeClassifier(random_state=42)
clf_top10 = RandomForestClassifier(n_estimators=100)
clf_top10.fit(X_train_top10, y_train)




# Evaluate the classifier with full features
y_pred_full = clf.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred_full)
print(f"Accuracy with full features: {accuracy_full * 100:.2f}%")




# Prepare the test set for top 10 features
# Make sure to use the same top 10 features indices as used for training
X_test_top10 = X_test[:, top_10_feature_indices]
print('Shape of test_top10 (the test set for the top 10 features) is',X_test_top10.shape)




# Evaluate the classifier with top 10 features
y_pred_top10 = clf_top10.predict(X_test_top10)
accuracy_top10 = accuracy_score(y_test, y_pred_top10)
print(f"Accuracy with top 10 features: {accuracy_top10 * 100:.2f}%")


# Sort and display features by their importance
sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
    print(f'{feature}: {importance}')








# Separate the feature names and their importances for plotting
sorted_feature_names, sorted_importances = zip(*sorted_features)
plt.figure(figsize=(12, 8))  # Increase figure size for better spacing
plt.bar(sorted_feature_names[:10], sorted_importances[:10])
plt.title('Top 10 Feature Importances')
plt.ylabel('Importance')
plt.xlabel('Features')




# Rotate feature names and optionally adjust font size
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and adjust font size
plt.tight_layout()  # Adjust layout for a better fit
plt.show()




# Assuming 'X' is your feature set, 'y' is your class labels
# and 'top_2_feature_indices' is a list or array containing the indices of the top two features
# For example, top_2_feature_indices = [index_of_first_top_feature, index_of_second_top_feature]




# Create a DataFrame for easy plotting
plot_data = pd.DataFrame({
    'Top Feature 1': X_train[:, top_2_feature_indices[0]],
    'Top Feature 2': X_train[:, top_2_feature_indices[1]],
    'Class': y_train
})




# Create the scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=plot_data, x='Top Feature 1', y='Top Feature 2', hue='Class', palette='Set1')




# Adding titles and labels
plt.title('Visualization of Top Two Features with Class Labels')
plt.xlabel('{}'.format(top_2_feature_names[0]))
plt.ylabel('{}'.format(top_2_feature_names[1]))
plt.legend(title='Class', loc='upper left')




plt.show()


# Assuming 'data' is your DataFrame and 'labels' is your target column
one_hot_encoder = OneHotEncoder(sparse_output=False)
encoded_labels = one_hot_encoder.fit_transform(data[['class']])
y = encoded_labels  # Use the encoded labels here
print('labels before {} and after {}'.format(data['class'][31000], encoded_labels[31000]))
#sys.exit(0)




# Split data into training and testing sets
X_train, X_test, y_train, y_test, = train_test_split(features, y, test_size=0.2, random_state=42)




# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)




input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
print('input dimensions: {}, Out dim: {}'.format(input_dim, output_dim))




# Create TensorDataset and DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)




test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32)




import torch.nn as nn
import torch.nn.functional as F




class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        #adjust number of layers as necessary
        self.fc1 = nn.Linear(input_dim, 100)  # 4 input features
        self.fc11 = nn.Linear(100, 100)  
        self.fc12 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, output_dim)  # Binary classification


    def forward(self, x):
        #adjust based on number of layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x

model = NeuralNet()


import torch.optim as optim


# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




# Training loop
losses = []
num_epochs = 100

for epoch in range(num_epochs):  # number of epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)  # Remove extra dimension from outputs
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    #print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Now plot the training loss
plt.figure(figsize=(10, 6))
plt.title("Training Loss Over Epochs")
plt.plot(range(num_epochs), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Switch to evaluation mode
model.eval()

with torch.no_grad():
    predictions = model(X_test).squeeze()  # Get predictions
    predictions = predictions.round()  # Convert to binary
    accuracy = (predictions == y_test).float().mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")

