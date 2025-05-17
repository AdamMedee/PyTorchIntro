# Import the libraries we need
import torch                          # Core PyTorch library
import torch.nn as nn                # For building neural networks
import torch.nn.functional as F      # For functions like ReLU
import torch.optim as optim          # For the optimizer (SGD)
from torchvision import datasets, transforms  # For loading image data and transforming it
from PIL import Image                # For loading images from your computer


##################################################################################


# Define how to convert and prepare the images
transform = transforms.Compose([
    transforms.ToTensor(),            # Turn image into a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize image values to be between -1 and 1
])

# Load the training and testing digit data (MNIST)
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split the data into batches (small groups of 64 images at a time)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


##################################################################################


# Build the neural network
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # First layer: input size (28x28 pixels) to 16 neurons
        # Input size, number of neurons
        self.fc1 = nn.Linear(28 * 28, 16)
        # Second layer: 16 neurons to 16 neurons
        self.fc2 = nn.Linear(16, 16)
        # Output layer: 16 neurons to 10 outputs (for digits 0–9)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        # Flatten the image into a single row of numbers
        x = x.view(-1, 28 * 28)
        # Pass through first layer and apply ReLU (activation function)
        x = F.relu(self.fc1(x))
        # Pass through second layer and apply ReLU again
        x = F.relu(self.fc2(x))
        # Final layer gives 10 outputs (one for each digit)
        x = self.fc3(x)
        return x

# Create the model
model = DigitRecognizer()

# Define the loss function (to measure how wrong the model is)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (how the model learns)
optimizer = optim.SGD(model.parameters(), lr=0.01)


##################################################################################


# Train the model over the dataset
for test in range(10):  # Go through the training data 10 times
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()              # Clear old gradients
        outputs = model(images)            # Make a prediction
        loss = criterion(outputs, labels)  # Calculate how wrong it was
        loss.backward()                    # Backpropagation: find how to improve
        optimizer.step()                   # Update weights
        total_loss += loss.item()          # Keep track of total loss
    print(f"Test {test+1}, Loss: {total_loss:.4f}")


##################################################################################


# Test how well the model does on new images it hasn't seen
correct = 0
total = 0
with torch.no_grad():  # We’re not training here, just testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Pick the most likely digit
        total += labels.size(0)                    # Count all the images
        correct += (predicted == labels).sum().item()  # Count correct guesses

print(f"Test Accuracy: {100 * correct / total:.2f}%")


##################################################################################


# Try predicting your own handwritten digit images
for i in range(10):
    image_path = "digit_%d.jpg" % i  # Use file names like digit_0.png, digit_1.png, ...
    img = Image.open(image_path).convert("L")  # Open the image in grayscale ("L")

    # Same transformation as training data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),         # Resize image to 28x28
        transforms.ToTensor(),               # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize to match training data
    ])

    img_tensor = transform(img)       # Apply transformation
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension: [1, 1, 28, 28]

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Don’t track gradients while predicting
        output = model(img_tensor)     # Predict the digit
        _, predicted = torch.max(output.data, 1)
        print(f"Predicted Digit: {predicted.item()}")
