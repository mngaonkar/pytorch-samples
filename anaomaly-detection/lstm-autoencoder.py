import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Step 1: Generate or Load Sample Time Series Data
# For demonstration, we'll create synthetic normal and anomalous data
def generate_data(num_samples=1000, seq_length=10):
    # Normal sine wave data
    time = np.linspace(0, 10 * np.pi, num_samples)
    normal_data = np.sin(time) + np.random.normal(0, 0.1, num_samples)
    
    # Introduce anomalies in test data
    anomalous_data = normal_data.copy()
    anomalous_data[200:210] += 5  # Spike anomaly
    
    # Reshape into sequences
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    train_sequences = create_sequences(normal_data[:800], seq_length)
    test_sequences = create_sequences(anomalous_data[:300], seq_length)
    
    # Normalize data
    scaler = MinMaxScaler()
    train_sequences = scaler.fit_transform(train_sequences.reshape(-1, 1)).reshape(train_sequences.shape)
    test_sequences = scaler.transform(test_sequences.reshape(-1, 1)).reshape(test_sequences.shape)
    
    return train_sequences, test_sequences, scaler

seq_length = 10
train_data, test_data, scaler = generate_data(seq_length=seq_length)

# Convert to PyTorch tensors
train_tensor = torch.FloatTensor(train_data).unsqueeze(2)  # Add feature dimension
test_tensor = torch.FloatTensor(test_data).unsqueeze(2)

# DataLoader
train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=32, shuffle=True)

# Step 2: Define LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        # Encoder
        _, (hidden, _) = self.encoder(x)
        # Repeat hidden state for decoder input
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        # Decoder
        output, _ = self.decoder(decoder_input)
        return output

# Initialize model, loss, optimizer
model = LSTMAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3: Train the Model
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, _ in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

# Step 4: Anomaly Detection
model.eval()
with torch.no_grad():
    # Reconstruction on train data to find threshold
    train_recon = model(train_tensor)
    train_errors = torch.mean((train_recon - train_tensor)**2, dim=[1,2]).numpy()
    threshold = np.mean(train_errors) + 3 * np.std(train_errors)
    
    # Reconstruction on test data
    test_recon = model(test_tensor)
    test_errors = torch.mean((test_recon - test_tensor)**2, dim=[1,2]).numpy()

# Detect anomalies
anomalies = test_errors > threshold
print(f'Number of anomalies detected: {np.sum(anomalies)}')
print(f'Anomaly indices: {np.where(anomalies)}')

# Optional: Plot results (requires matplotlib)
import matplotlib.pyplot as plt

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# First subplot: Reconstruction Error
ax1.plot(test_errors, label='Reconstruction Error')
ax1.axhline(threshold, color='r', linestyle='--', label='Threshold')
ax1.set_title('Anomaly Detection with LSTM Autoencoder')
ax1.set_ylabel('Reconstruction Error')
ax1.set_xlabel('Sample Index')
ax1.legend()

# Second subplot: Train and Test Data
ax2.plot(train_data[:300].flatten(), label='Train Data', alpha=0.7)
ax2.plot(test_data.flatten(), label='Test Data', alpha=0.7)
ax2.set_title('Original Time Series Data')
ax2.set_ylabel('Value')
ax2.set_xlabel('Sample Index')
ax2.legend()

plt.tight_layout()
plt.show()