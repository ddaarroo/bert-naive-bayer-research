import pandas as pd
from sklearn.model_selection import train_test_split

# Load all original data
train = pd.read_csv("/home/jgilyard/CS 457 Final Project/train.tsv", sep="\t", names=["Sentence", "Language"])
dev = pd.read_csv("/home/jgilyard/CS 457 Final Project/dev.tsv", sep="\t", names=["Sentence", "Language"])
test = pd.read_csv("/home/jgilyard/CS 457 Final Project/test.tsv", sep="\t", names=["Sentence", "Language"])

# Combine and clean
full_df = pd.concat([train, dev, test])
full_df = full_df.drop_duplicates().dropna().reset_index(drop=True)

# Shuffle
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split: 60% train, 20% dev, 20% test
train_df, temp_df = train_test_split(full_df, test_size=0.4, random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to new files
train_df.to_csv("/home/jgilyard/BERTNaive Bayes/bert-naive-bayer-research/CS 457 Final Project/train_clean.tsv", sep="\t", index=False, header=False)
dev_df.to_csv("/home/jgilyard/BERTNaive Bayes/bert-naive-bayer-research/CS 457 Final Project/dev_clean.tsv", sep="\t", index=False, header=False)
test_df.to_csv("/home/jgilyard/BERTNaive Bayes/bert-naive-bayer-research/CS 457 Final Project/test_clean.tsv", sep="\t", index=False, header=False)

# Summary
print("✅ New clean splits created!")
print(f"Train: {len(train_df)} samples")
print(f"Dev:   {len(dev_df)} samples")
print(f"Test:  {len(test_df)} samples")
