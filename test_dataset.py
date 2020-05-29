import matplotlib.pyplot as plt

from dataset import CNRParkDataset

dataset = CNRParkDataset("./PATCHES", "./LABELS")


for i in range(len(dataset)):
    sample = dataset[i]
    images = sample["images"]
    labels = sample["labels"]

    sample_size = len(images)
    print(f"{i}: {sample_size} images")
    grid_length = 8

    for j, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, grid_length + 1, j + 1)
        plt.tight_layout()
        plt.title({label})
        plt.axis('off')
        plt.imshow(image)
        if j == grid_length:
            break
    plt.savefig(f"dataset_test{i}.pdf")
    input()

    if i == 3:
        break
