import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Load the saved model
model = torch.load("model.pt")
model = Net()
model.load_state_dict(model_dict)


# Set the model to evaluation mode
model.eval()

# Load the test data
test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Get the first 10 test examples
test_data = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)
test_iter = iter(test_data)
images, labels = test_iter.next()

# Run the examples through the model
with torch.no_grad():
    outputs = model(images)

# Print the predictions and correct labels for each example
for i in range(10):
    print("Example", i+1)
    print("Predicted:", ", ".join([f"{out:.2f}" for out in outputs[i]]))
    print("Max output index:", torch.argmax(outputs[i]).item())
    print("Correct label:", labels[i].item())

# Plot the first 9 digits with predictions
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    if i <= 9:
        plt.imshow(images[i-1].squeeze(), cmap="gray")
        pred = torch.argmax(outputs[i-1]).item()
        plt.title(f"Prediction: {pred}")
plt.show()

