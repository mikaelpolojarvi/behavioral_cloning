from model import CNN
from data_loader import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    EPOCHS = 10
    DATASETS = 15 # Number of data sets in the TRAINING_DATA_PATH -directory. Excepts that data is saved in a separate np files for each run named "run_x.np", where x is the count of the run.
    TRAINING_IMG_PATH = ""
    TRAINING_DATA_PATH = ""
    SAVE_MODEL_PATH = ""
    MODEL_NAME = ""

    num_actions = 4
    num_channels = 3

    totalActions = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0
    }
    outActions = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0
    }

    transform = transforms.Compose([
        # resize
        transforms.Resize((320, 160)),

        # converting to tensor
        transforms.ToTensor(),
    ])

    # Initialize the CNN
    model = CNN((320, 160), num_channels, num_actions)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    # Arrays for graphs
    statsArrLoss = []
    statsArrAccuracy = []
    epochsArr = []
    for i in range(EPOCHS):
        epochsArr.append(i + 1)

    for epoch in range(EPOCHS):

        loader = DataLoader(TRAINING_DATA_PATH, DATASETS)
        model.train()

        stats = {
            "epoch": epoch + 1,
            "loss": 0,
            "accuracy": 0
        }

        correct = 0
        total = 0
        loss = 0

        print("Training epoch number", epoch + 1, "...")
        for batch, data in enumerate(loader):

            # Get image as x and label as y
            image = Image.fromarray(data["image"])
            image_rgb = image.convert("RGB")
            x = transform(image_rgb).to(device)
            action = np.array([int(data["action"])])
            y = torch.Tensor(action).to(device).long()

            optimizer.zero_grad()

            # Model output from image x
            out = model(x)

            # Loss calculated from comparing the result to label y
            loss = F.cross_entropy(out, y)

            if out.argmax().item() == y.item():
                correct += 1

            totalActions[str(y.item())] += 1
            outActions[str(out.argmax().item())] += 1

            # Backpropagate loss
            loss.backward()
            optimizer.step()
            total = batch

        # Statistics for the epoch trained
        stats["loss"] = loss
        statsArrLoss.append(loss.item())
        stats["accuracy"] = correct / total
        statsArrAccuracy.append(correct / total)
        print(stats)
        print(totalActions)
        print(outActions)

    # Save the model to given path
    torch.save(model, SAVE_MODEL_PATH + MODEL_NAME)

    # Print statistic and save images of the model's loss and accuracy to the given path
    print(statsArrAccuracy)
    print(statsArrLoss)

    plt.plot(epochsArr, statsArrAccuracy, 'g')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(TRAINING_IMG_PATH + MODEL_NAME + '_accuracy.png')
    plt.close()

    plt.plot(epochsArr, statsArrLoss, 'g')
    plt.title('Training  loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-entropy loss')
    plt.savefig(TRAINING_IMG_PATH + MODEL_NAME + '_loss.png')
