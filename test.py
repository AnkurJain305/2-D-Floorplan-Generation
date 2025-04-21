import torch

# Load the model checkpoint
checkpoint = torch.load(r"D:\floor plan\GATNET-model\Checkpoints\best_model.pth", map_location=torch.device("cpu"))

# Print metrics
print("Epoch:", checkpoint.get("epoch"))
print("Final Training Loss:", checkpoint.get("train_loss"))
print("Final Validation Loss:", checkpoint.get("val_loss"))

# Optional: plot training/validation loss
import matplotlib.pyplot as plt
train_losses = checkpoint.get("train_losses", [])
val_losses = checkpoint.get("val_losses", [])

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
