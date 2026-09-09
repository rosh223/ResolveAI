import pandas as pd
import os

print("Loading dataset...")
df = pd.read_csv('data/raw/twcs/twcs.csv')

# Find all tweets by Uber_Support
brand_tweets = df[df['author_id'] == 'Uber_Support']

# Find all inbound customer tweets that were answered by Uber_Support
# The brand tweet's 'in_response_to_tweet_id' points to the customer's tweet.
customer_tweet_ids = brand_tweets['in_response_to_tweet_id'].dropna().unique()

customer_tweets = df[df['tweet_id'].isin(customer_tweet_ids)]

print(f"Found {len(brand_tweets)} brand tweets and {len(customer_tweets)} inbound customer tweets.")

# Let's create pairs of (Customer Tweet, Brand Reply)
# Merge customer tweets with brand tweets on tweet_id == in_response_to_tweet_id
pairs = pd.merge(
    customer_tweets[['tweet_id', 'text', 'created_at']],
    brand_tweets[['tweet_id', 'in_response_to_tweet_id', 'text', 'created_at']],
    left_on='tweet_id',
    right_on='in_response_to_tweet_id',
    suffixes=('_customer', '_brand')
)

# Subsample 5000 random pairs to create our working dataset
if len(pairs) > 5000:
    subsample = pairs.sample(n=5000, random_state=42)
else:
    subsample = pairs

print(f"Created subsample of {len(subsample)} conversation pairs.")

# Save to processed directory
os.makedirs('data/processed', exist_ok=True)
subsample.to_csv('data/processed/uber_support_subsample.csv', index=False)
print("Saved to data/processed/uber_support_subsample.csv")
