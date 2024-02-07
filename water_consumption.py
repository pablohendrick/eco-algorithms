import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the data using pandas
data = pd.read_csv('water_consumption.csv')

print(data.head())

print(data.describe())

# Clustering the data into Residential, Industrial, and Agricultural categories
X_cluster = data[['location', 'consumption']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(X_cluster)

# Visualizing the clustered data
plt.scatter(data['location'], data['consumption'], c=data['cluster'], cmap='viridis')
plt.title('Clustered Data')
plt.xlabel('Location')
plt.ylabel('Consumption')
plt.show()

# Visualizing the count of data points in each cluster
data['cluster'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Count of Data Points in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Displaying a histogram of water consumption by category
data.hist(column='consumption', by='usage_type', bins=30, figsize=(10, 6))
plt.suptitle('Water Consumption Histogram by Category')
plt.show()

# Splitting the dataset into training, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Converting the datasets to PyTorch tensors
def dataframe_to_tensor(df):
    df['usage_type'] = df['usage_type'].astype('category').cat.codes
    inputs = torch.tensor(df[['location', 'consumption']].values, dtype=torch.float32)
    labels = torch.tensor(df['usage_type'].values, dtype=torch.long)
    return inputs, labels

train_inputs, train_labels = dataframe_to_tensor(train_data)
val_inputs, val_labels = dataframe_to_tensor(val_data)
test_inputs, test_labels = dataframe_to_tensor(test_data)

# Model neurons
class WaterConsumptionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(WaterConsumptionModel, self).__init__()
        self.fc = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        self.output = nn.Linear(8, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return self.output(x)

# Instantiating the model, loss function, and optimizer
input_size = 2
output_size = 3
model = WaterConsumptionModel(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model on the validation set
model.eval()
with torch.no_grad():
    val_outputs = model(val_inputs)
    val_loss = criterion(val_outputs, val_labels)
    _, val_predictions = torch.max(val_outputs, 1)
    accuracy = (val_predictions == val_labels).float().mean()
    print(f'Validation Loss: {val_loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Evaluating the model on the test set
test_outputs = model(test_inputs)
test_loss = criterion(test_outputs, test_labels)
_, test_predictions = torch.max(test_outputs, 1)
test_accuracy = (test_predictions == test_labels).float().mean()
print(f'Test Loss: {test_loss.item():.4f}, Accuracy: {test_accuracy.item():.4f}')

# Comparative graph of water volume spent by each category
model.eval()
with torch.no_grad():
    all_outputs = model(torch.tensor(data[['location', 'consumption']].values, dtype=torch.float32))
    _, all_predictions = torch.max(all_outputs, 1)
    data['predicted_usage_type'] = all_predictions.numpy()

plt.figure(figsize=(12, 6))
for usage_type in data['usage_type'].unique():
    subset = data[data['usage_type'] == usage_type]
    plt.scatter(subset['location'], subset['consumption'], label=f'Category {usage_type}', alpha=0.5)

plt.scatter(data['location'], data['predicted_usage_type'], label='Prediction', marker='x', color='black')
plt.xlabel('Location')
plt.ylabel('Water Consumption')
plt.title('Comparison of Water Volume Spent by Category')
plt.legend()
plt.show()
