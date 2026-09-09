import pandas as pd
import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    with open(".env") as f:
        for line in f:
            if line.startswith("GROQ_API_KEY="):
                api_key = line.split("=", 1)[1].strip()

# Initialize OpenAI client to point to Groq's API
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# Intents Definition
INTENT_PROMPT = """You are an expert customer support annotator for Uber.
You need to label the following customer tweet.

Choose exactly ONE intent from this list:
1. "Billing / Refund"
2. "Safety / Driver Behavior"
3. "Account / App Issue"
4. "General Inquiry"
5. "Needs Context"

Also decide if this should be Auto-handled or Escalated to a human.
Escalate if it's a safety issue, a complex billing dispute, or an angry customer. Otherwise, Auto-handle.
Provide a short Reason for your decision.

Customer Tweet: "{tweet}"

Return a JSON object with keys: "intent", "action" (must be "Auto-handle" or "Escalate"), and "reason" (short string).
"""

def get_label(tweet_text):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful JSON-outputting assistant."},
                {"role": "user", "content": INTENT_PROMPT.format(tweet=tweet_text)}
            ],
            response_format={ "type": "json_object" },
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        return {"intent": "Error", "action": "Error", "reason": str(e)}

def main():
    print("Loading subsample...")
    df = pd.read_csv('data/processed/uber_support_subsample.csv')
    
    # Take 150 random samples for our Golden Set
    golden_df = df.sample(n=150, random_state=42).copy()
    
    intents = []
    actions = []
    reasons = []
    
    print("Labeling 150 tweets with OpenAI... (This might take a few minutes)")
    for text in tqdm(golden_df['text_customer']):
        label = get_label(text)
        intents.append(label.get('intent', 'Unknown'))
        actions.append(label.get('action', 'Unknown'))
        reasons.append(label.get('reason', 'Unknown'))
        
        # Free tier rate limit mitigation (typically 3 RPM for new free tiers, but can be higher. 
        # We'll sleep 1 second just to be safe for basic RPM/TPM limits)
        time.sleep(1)
        
    golden_df['intent'] = intents
    golden_df['action'] = actions
    golden_df['escalate_reason'] = reasons
    golden_df['ideal_reply'] = golden_df['text_brand'] # Use the actual brand reply as the ideal reply
    
    golden_df.to_csv('data/processed/golden_set.csv', index=False)
    print("\nDone! Saved to data/processed/golden_set.csv")

if __name__ == "__main__":
    main()
