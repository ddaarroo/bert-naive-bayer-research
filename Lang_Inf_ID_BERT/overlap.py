import pandas as pd

train = pd.read_csv("/home/jgilyard/BERTNaive Bayes/bert-naive-bayer-research/CS 457 Final Project/train_clean.tsv", sep="\t", names=["Sentence", "Language"])
dev = pd.read_csv("/home/jgilyard/BERTNaive Bayes/bert-naive-bayer-research/CS 457 Final Project/dev_clean.tsv", sep="\t", names=["Sentence", "Language"])
test = pd.read_csv("/home/jgilyard/BERTNaive Bayes/bert-naive-bayer-research/CS 457 Final Project/test_clean.tsv", sep="\t", names=["Sentence", "Language"])

train_set = set(train["Sentence"])
dev_set = set(dev["Sentence"])
test_set = set(test["Sentence"])

print("Train ∩ Dev:", len(train_set & dev_set))
print("Train ∩ Test:", len(train_set & test_set))
print("Dev ∩ Test:", len(dev_set & test_set))
