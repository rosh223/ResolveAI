import pandas as pd

df = pd.read_csv('data/processed/uber_support_subsample.csv')

# Print 20 random customer tweets to help us define intents
print("Sample Customer Tweets for Intent Definition:\n")
sample_tweets = df['text_customer'].sample(n=20, random_state=42)

for i, text in enumerate(sample_tweets, 1):
    print(f"{i}. {text}")
    print("-" * 50)
