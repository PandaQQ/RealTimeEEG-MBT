import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

FREQUENCY = 125
PCA_COMPONENTS = 20

# Define your model architecture (same as used during training)
class my_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(24, 50, (PCA_COMPONENTS, 1))
        self.bn1 = nn.BatchNorm2d(50, affine=False)
        self.bn2 = nn.BatchNorm2d(10, affine=False)
        self.bn3 = nn.BatchNorm2d(1, affine=False)
        self.pool1 = nn.MaxPool2d((1, 20))
        self.conv2 = nn.Conv2d(10, 10, (1, 10))
        self.conv3 = nn.Conv2d(20, 1, (1, 1))
        self.pool2 = nn.MaxPool2d((1, 20))
        self.pool3 = nn.MaxPool2d((1, 1))
        self.fc1 = nn.Linear(50 * (FREQUENCY // 20), 32)
        self.fc2 = nn.Linear(32, 2)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class EEGCNNPredictor:
    def __init__(self, model_path, device=None):
        """
        Initializes the EEG CNN Predictor.

        Parameters:
        - model_path (str): Path to the saved PyTorch model (.pth file).
        - device (torch.device, optional): Device to run the model on. Defaults to CPU or GPU if available.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize and load the model
        self.model = my_cnn()
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

        # Define class labels
        self.class_labels = {0: 'Resting', 1: 'Active'}

    def preprocess(self, power_data, n_channels=10):
        """
        Preprocesses the CWT power data for the CNN.

        Parameters:
        - power_data (numpy.ndarray): CWT power data with shape (1, n_channels, n_frequencies, n_times).
        - n_channels (int): Number of channels to select for the CNN input.

        Returns:
        - torch.Tensor: Preprocessed tensor ready for CNN input.
        """
        power_tensor = torch.tensor(power_data, dtype=torch.float32)

        # Select channels
        actual_channels = power_tensor.shape[1]
        if actual_channels >= n_channels:
            selected_channels = list(range(n_channels))
        else:
            # Duplicate channels to reach n_channels
            repeats = n_channels // actual_channels + 1
            selected_channels = (list(range(actual_channels)) * repeats)[:n_channels]

        power_selected = power_tensor[:, selected_channels, :, :]
        power_selected = power_selected.to(self.device)
        return power_selected

    def predict(self, power_data):
        """
        Makes a prediction on the input power data.

        Parameters:
        - power_data (numpy.ndarray): CWT power data with shape (n_channels, n_frequencies, n_times).

        Returns:
        - dict: Dictionary containing predicted class and probabilities.
        """
        import torch
        import torch.nn.functional as F

        # Ensure the input has the correct dimensions for the model
        if power_data.ndim == 3:
            # Add batch dimension
            power_data = power_data[np.newaxis, ...]

        # Convert to PyTorch tensor and ensure it's on the same device as the model
        input_tensor = torch.tensor(power_data, dtype=torch.float32, device=self.device)

        # Perform prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            probabilities_percentage = probabilities * 100

            # Extract prediction results
            pred_class_idx = predicted_classes[0].item()
            pred_class_label = self.class_labels.get(pred_class_idx, f"Unknown({pred_class_idx})")
            resting_prob = probabilities_percentage[0, 0].item()
            active_prob = probabilities_percentage[0, 1].item()

        # Return results
        result = {
            'predicted_class_index': pred_class_idx,
            'predicted_class_label': pred_class_label,
            'probabilities': {
                'Resting': resting_prob,
                'Active': active_prob
            }
        }
        return result
