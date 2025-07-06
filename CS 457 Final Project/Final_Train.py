
import argparse

import torch 
from torch import cuda, manual_seed, save
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import AdamW
from general_bert_model import BertConversationalLanguageClassificationModel
from tqdm import tqdm
from util import get_dataloader
from util import model_accuracy
from collections import Counter

def train_model(model: BertConversationalLanguageClassificationModel, train_dataloader: DataLoader, 
                dev_dataloader: DataLoader, epochs: int, learning_rate: float) -> float:
    """
    Trains model and prints accuracy on dev data after training

    Arguments:
        model (BertConversationalLanguageClassificationModel): the model to train
        train_dataloader (DataLoader): a pytorch dataloader containing the training data
        dev_dataloader (DataLoader): a pytorch dataloader containing the development data
        epochs (int): the number of epochs to train for (full iterations through the dataset)
        learing_rate (float): the learning rate

    Returns:
        float: the accuracy on the development set
    """
    all_lbls = []
    for batch in train_dataloader:
        all_lbls += batch["Lang_int"].tolist()
        counts = Counter(all_lbls) 
    device = "cuda" if cuda.is_available() else "cpu"

    total       = sum(counts.values())
    num_classes = len(counts)  # e.g. 3
    class_weights = [ total / counts[i] for i in range(num_classes) ]

    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    model.to(device)
    criterion = NLLLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    batch = next(iter(train_dataloader))
    print(type(batch), batch)

#Training
    for epoch in tqdm(range(epochs), desc="Epoch Loop"):
        print("Sucessful epoch: " + str(epoch) + " " +  str(epochs))
        model.train()
        total_loss = 0
        total = 0

        for batch in train_dataloader:
            input_ids= batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["Lang_int"].to(device).long()
            
            model.zero_grad()

            probs = model(input_ids, attention_mask)

            #compute loss
            loss = criterion(probs, label)

            #finds gradient
            loss.backward()
            
            #updates weights
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            total += input_ids.size(0)

            dev = model_accuracy(model, dev_dataloader, device)
            train  = model_accuracy(model, train_dataloader, device)
            avg_loss = total_loss/total
            print(f"Epoch {epoch+1}/{epoch} | Avg loss: {avg_loss:.4f} | Train Accuracy: {train:.4f} | Dev Accuracy: {dev:.4f}")
    return model_accuracy(model, dev_dataloader, device)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, type=int, 
                        help="The number of epochs to train for")
    parser.add_argument("--learning_rate", default=1e-5, type=float, 
                        help="The learning rate")
    parser.add_argument("--batch_size", default=16, type=int, 
                        help="The batch size")
    parser.add_argument("--weights", default= "bert-base-multilingual-cased", type= str, 
                        help="The weight of a given model")
    parser.add_argument("--filename", default='model_weights.pth', type=str, 
                        help="The name of the file path")
    args = parser.parse_args()

    # initialize model and dataloaders
    device = "cuda" if cuda.is_available() else "cpu"

    # seed the model before initializing weights so that your code is deterministic
    manual_seed(457)

    model = BertConversationalLanguageClassificationModel(args.weights).to(device)
    train_dataloader = get_dataloader("train", batch_size=args.batch_size)
    dev_dataloader = get_dataloader("dev", batch_size=args.batch_size)

    train_model(model, train_dataloader, dev_dataloader,
                args.epochs, args.learning_rate)
    
    save(model.state_dict(), args.filename)

if __name__ == "__main__":
    main()
