import pandas as pd

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/raw/twcs/twcs.csv')

# Find author_ids that don't look like numbers (brands)
is_brand = ~df['author_id'].str.match(r'^\d+$')
brands = df[is_brand]

# Get the top 15 most active brands
top_brands = brands['author_id'].value_counts().head(15)

print("\nTop 15 Brands by Number of Tweets:")
print(top_brands)
