"""
Let's create a minimal example of customizing a Hugging Face transformer layer architecture
for the MNIST dataset. We will create a custom transformer-based model
for classifying handwritten digits from the MNIST dataset.
"""

# pip install torch transformers datasets

##

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader

import numpy as np
from torchvision import transforms


class CustomConfig(PretrainedConfig):
    model_type = "custom_model"

    def __init__(self, hidden_size=256, num_hidden_layers=4, num_attention_heads=8, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_classes = num_classes

##

class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Linear(28 * 28, config.hidden_size)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads) for _ in range(config.num_hidden_layers)])
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        self.init_weights()

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.embeddings(pixel_values.view(batch_size, -1))
        hidden_states = embeddings.unsqueeze(1)  # Add sequence dimension

        for layer in self.encoder:
            hidden_states = layer(hidden_states)

        logits = self.classifier(hidden_states.squeeze(1))  # Remove sequence dimension

        return logits


##

#-------- ^^

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, ds, transform=None):
        # @@ ds: Dataset({
        #     features: ['image', 'label'],
        #     num_rows: 60000
        # })
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]['image']
        label = self.ds[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

#-------- $$

def main():

    print('@@ vit arch !!')

    AutoConfig.register("custom_model", CustomConfig)
    AutoModel.register(CustomConfig, CustomModel)

    ##

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    preprocess = lambda pil_img : transform(pil_img)

    mnist = load_dataset("mnist")
    """ ^---- TODO shorten them, cf.
    from datasets import load_dataset  # cifar10
    trainds, testds = load_dataset("cifar10", split=[f"train[:{train_size}]", f"test[:{test_size}]"])

    splits = trainds.train_test_split(test_size=0.1)
    trainds = splits['train']  # 90%
    valds = splits['test']  # 10%
    """

    train_dataset = CustomDataset(mnist["train"], transform=preprocess)
    test_dataset = CustomDataset(mnist["test"], transform=preprocess)

    print('@@ len(train_dataset):', len(train_dataset))  # 60000
    print('@@ len(test_dataset):', len(test_dataset))  # 10000
    print('@@ type(train_dataset[0]):', type(train_dataset[0]))  # <class 'tuple'>

    ##exit()  # !!!!
    #----

    ##

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    ##

    model = CustomModel(CustomConfig())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):  # Train for 3 epochs
        print(f'@@ Epoch: {epoch+1}')
        for batch in train_dataloader:
            pixels, labels = batch
            # print('@@ pixels:', pixels)  # !!!!
            # print('@@ labels:', labels)  # !!!!
            #exit()  # !!!!

            optimizer.zero_grad()
            outputs = model(pixels)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    ##

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            outputs = model(batch["pixel_values"])
            _, predicted = torch.max(outputs.data, 1)
            total += batch["labels"].size(0)
            correct += (predicted == batch["labels"]).sum().item()

    print(f"Accuracy: {100 * correct / total}%")

    ##


if __name__ == "__main__":
    main()
