import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix

from general_bert_model import BertConversationalLanguageClassificationModel


def model_accuracy(model: BertConversationalLanguageClassificationModel, dataloader: DataLoader, device: str):
    """Compute the accuracy of a binary classification model

    Args:
        model (BertConversationalLanguageClassificationModel): a hate speech classification model
        dataloader (DataLoader): a pytorch data loader to test the model with
        device (string): cpu or cuda, the device that the model is on

    Returns:
        float: the accuracy of the model
    """
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:

            pred = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            pred_labels = pred.argmax(dim=1).cpu()

            true_labels = batch["Lang_int"].cpu()

            correct += (pred_labels == true_labels).sum().item()

            total += true_labels.size(0)

            all_preds .extend(pred_labels.cpu().tolist())
            all_labels.extend(true_labels.cpu().tolist())

        print(confusion_matrix(all_labels, all_preds))

        acc = correct / total
        return acc


def get_dataloader(data_split: str, data_path: str = None, batch_size: int = 4, weights="bert-base-uncased"):
    """
    Get a pytorch dataloader for a specific data split

    Args:
        data_split (str): the data split
        data_path (str, optional): a data path if the data is not stored at the default path.
        batch_size (int, optional): the desired batch size. Defaults to 4.

    Returns:
        DataLoader: the pytorch dataloader object
    """

    assert data_split in ["train", "dev", "test"]

    label_map = {"es": 0, "ita":1, "fr":2}

    if data_path is None:
        data = pd.read_csv(f"/home/jgilyard/CS 457 Final Project/{data_split}.tsv", sep="\t", names= ["Sentence", "Language"])

    else:
        data = pd.read_csv(data_path, sep="\t", names=["Sentence", "Language"])
    
    data = data.dropna(subset=["Sentence", "Language"]).reset_index(drop=True) #Drops any NoneType
    data["Lang_int"] = data["Language"].map(label_map)

    #Drops any values that contained a NaN, or where the label_map did not label with a value
    data = data.dropna(subset=["Lang_int"])
    data["Lang_int"] = data["Lang_int"].astype(int)

    dataset = Dataset.from_pandas(data)
    tokenizer = AutoTokenizer.from_pretrained(weights)

    dataset = dataset.map(lambda ex: tokenizer(ex["Sentence"], truncation=True, padding="max_length"), batched=True)
    dataset.set_format(type="torch", columns=["input_ids","attention_mask","Lang_int"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
     
    return dataloader

