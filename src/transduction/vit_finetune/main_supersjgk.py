"""
adapted from -- research_mri/transduction/finetune/try_vision_transformers_hugging_face_fine_tuning_cifar10_pytorch.py
"""

# https://huggingface.co/docs/transformers/en/installation
# For CPU-support only, you can conveniently install 🤗 Transformers and a deep learning library in one line. For example, install 🤗 Transformers and PyTorch with:
# pip install 'transformers[torch]'

"""@@
$ pipenv run python3 -m pip install 'transformers[torch]'
Successfully installed accelerate-1.0.0 filelock-3.16.1 fsspec-2024.9.0 huggingface-hub-0.25.1 regex-2024.9.11 safetensors-0.4.5 tokenizers-0.20.0 transformers-4.45.2

$ pipenv run python3 -m pip install datasets
Successfully installed datasets-3.0.1 dill-0.3.8 fsspec-2024.6.1 multiprocess-0.70.16 pyarrow-17.0.0 xxhash-3.5.0

$ pipenv run python3 -m pip install scikit-learn
Successfully installed joblib-1.4.2 scikit-learn-1.5.2 scipy-1.14.1 threadpoolctl-3.5.0
"""


import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#----
from ..plot_if import get_plt, plt_imshow, plt_imshow_tensor  # @@
plt = get_plt()

from torchvision.transforms import ToPILImage, PILToTensor
transform_to_pil = ToPILImage()
transform_to_tensor = PILToTensor()
#----

def load_data(train_size=5000, test_size=1000):
    print('@@ load_data(): ^^')

    """### Loading the Data"""

    trainds, testds = load_dataset("cifar10", split=[f"train[:{train_size}]", f"test[:{test_size}]"])

    splits = trainds.train_test_split(test_size=0.1)
    trainds = splits['train']  # 90%
    valds = splits['test']  # 10%

    if 0:
        print(trainds, valds, testds)
        # Dataset({
        #     features: ['img', 'label'],
        #     num_rows: 4500
        # }) Dataset({
        #     features: ['img', 'label'],
        #     num_rows: 500
        # }) Dataset({
        #     features: ['img', 'label'],
        #     num_rows: 1000
        # })

    ##print(trainds.features, trainds.num_rows, trainds[0])
    # {'img': Image(mode=None, decode=True, id=None), 'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], id=None)} 4500 {'img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=32x32 at 0x14AE707F0>, 'label': 7}
    # horse

    itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
    stoi = dict((v,k) for k,v in enumerate(trainds.features['label'].names))
    ##print(itos, stoi)
    # {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'} {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    if 0:
        img, lab = trainds[0]['img'], itos[trainds[0]['label']]
        ##print(lab)  # truck
        ##print(img.size)  # (32, 32)

        #img  # colab only
        #print(type(img))  # <class 'PIL.PngImagePlugin.PngImageFile'>
        plt_imshow_tensor(plt, transform_to_tensor(img))

    return trainds, valds, testds, itos, stoi


def preprocess_data(processor, trainds, valds, testds):

    """### Preprocessing Data"""

    size = processor.size
    print(size)  # {'height': 224, 'width': 224}
    norm = Normalize(mean=processor.image_mean, std=processor.image_std)

    _transf = Compose([
        Resize(size['height']),
        ToTensor(),
        norm
    ])

    def transf(arg):
        arg['pixels'] = [_transf(image.convert('RGB')) for image in arg['img']]
        return arg

    trainds.set_transform(transf)
    valds.set_transform(transf)
    testds.set_transform(transf)

    if 0:
        print(trainds[0].keys())  # dict_keys(['img', 'label', 'pixels'])
        ex = trainds[0]['pixels']
        print(ex.shape)  # torch.Size([3, 224, 224])

        print(torch.min(ex), torch.max(ex))  # tensor(-0.8745) tensor(1.)
        ex = (ex+1)/2
        print(torch.min(ex), torch.max(ex))  # tensor(0.0627) tensor(1.)

        plt_imshow_tensor(plt, ex)  # ok
        plt_imshow(plt, transform_to_pil(ex))  # ok


def get_finetuned(model_name, itos, stoi):
    print('@@ get_finetuned(): ^^')

    """### Model - Fine Tuning"""

    model_orig = ViTForImageClassification.from_pretrained(model_name)
    print('get_finetuned(): [before] ', model_orig.classifier)
    # The google/vit-base-patch16-224 model is originally fine tuned on imagenet-1K with 1000 output classes

    if 0:
        print(model_orig.config)
        """
        { ...
            "yurt": 915,
            "zebra": 340,
            "zucchini, courgette": 939
          },
          "layer_norm_eps": 1e-12,
          "model_type": "vit",
          "num_attention_heads": 12,
          "num_channels": 3,
          "num_hidden_layers": 12,
          "patch_size": 16,
          "qkv_bias": true,
          "transformers_version": "4.45.2"
        }
        """

    # To use Cifar-10, it needs to be fine tuned again with 10 output classes
    model = ViTForImageClassification.from_pretrained(model_name,
        num_labels=10,
        ignore_mismatched_sizes=True,
        id2label=itos,
        label2id=stoi)
    """
Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224 and are newly initialized because the shapes did not match:
- classifier.bias: found shape torch.Size([1000]) in the checkpoint and torch.Size([10]) in the model instantiated
- classifier.weight: found shape torch.Size([1000, 768]) in the checkpoint and torch.Size([10, 768]) in the model instantiated
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    """
    print('get_finetuned(): [after] ', model.classifier)

    return model


def get_trainer(model, processor, trainds, valds):

    args = TrainingArguments(
        f"test-cifar-10",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir='logs',
        remove_unused_columns=False,
    )
    print(f'@@ get_trainer(): args.per_device_train_batch_size={args.per_device_train_batch_size}')
    print(f'@@ get_trainer(): args.per_device_eval_batch_size={args.per_device_eval_batch_size}')

    def collate_fn(examples):
        pixels = torch.stack([example["pixels"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixels, "labels": labels}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))

    trainer = Trainer(
        model,
        args,
        train_dataset=trainds,
        eval_dataset=valds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    return trainer


def main():
    #trainds, valds, testds, itos, stoi = load_data()
    trainds, valds, testds, itos, stoi = load_data(train_size=5000, test_size=20)

    model_name = "google/vit-base-patch16-224"
    model = get_finetuned(model_name, itos, stoi)
    processor = ViTImageProcessor.from_pretrained(model_name)

    preprocess_data(processor, trainds, valds, testds)

    #@@ ??
    #!pip show accelerate

    trainer = get_trainer(model, processor, trainds, valds)

    """### Training the model for fine tuning"""

    if 0:  #@@
        pass
        # Commented out IPython magic to ensure Python compatibility.
        # %load_ext tensorboard
        # %tensorboard --logdir logs/

##    trainer.train()

    """### Evaluation"""

    outputs = trainer.predict(testds)
    print(outputs.metrics)
    """ 250 <-- 1000 / 4 (test_size / args.per_device_eval_batch_size)
100%|██████████| 250/250 [15:11<00:00,  3.65s/it]
{'test_loss': 2.4680304527282715, 'test_model_preparation_time': 0.0091, 'test_accuracy': 0.075, 'test_runtime': 914.5631, 'test_samples_per_second': 1.093, 'test_steps_per_second': 0.273}
    """


    itos[np.argmax(outputs.predictions[0])], itos[outputs.label_ids[0]]

    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)

    labels = trainds.features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)



if __name__ == "__main__":
    main()
