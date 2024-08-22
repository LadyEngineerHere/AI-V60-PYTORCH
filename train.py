import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset, collate_fn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 1e-3
batch_size = 4
num_epochs = 10

# Load Data
annotation_file = "/Users/amandanassar/Desktop/V60 NOK AI/annotations.json"
img_dirs = [
    "/Users/amandanassar/Desktop/V60 NOK AI/Data images/camerafault",
    "/Users/amandanassar/Desktop/V60 NOK AI/Data images/external",
    "/Users/amandanassar/Desktop/V60 NOK AI/Data images/realNOK"
]

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = CustomDataset(json_file=annotation_file, img_dirs=img_dirs, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

# Modify the model's head for our specific number of classes
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {losses.item():.4f}')

# Accuracy Check
def check_accuracy(loader, model):
    model.eval()
    num_images = len(loader.dataset)
    print(f"Number of images: {num_images}")
    
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            # For now, just print the output sizes
            for output in outputs:
                print(f"Detected {len(output['labels'])} objects.")

# After training, check accuracy
print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
