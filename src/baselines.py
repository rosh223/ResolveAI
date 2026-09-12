import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_baselines():
    print("Loading Golden Set...")
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/processed/golden_set.csv'))
    valid_df = df[df['intent'] != 'Error'].copy()
    
    # --- TRIVIAL BASELINE ---
    print("\nEvaluating Trivial Baseline...")
    # Predict majority class (assume Auto-handle and General Inquiry)
    valid_df['trivial_intent'] = 'General Inquiry'
    valid_df['trivial_action'] = 'Auto-handle'
    valid_df['trivial_reply'] = "We are sorry to hear that. Please DM us your account details so we can help."
    
    intent_acc_t = accuracy_score(valid_df['intent'], valid_df['trivial_intent'])
    action_acc_t = accuracy_score(valid_df['action'], valid_df['trivial_action'])
    print(f"Trivial Intent Accuracy: {intent_acc_t:.2%}")
    print(f"Trivial Action Accuracy: {action_acc_t:.2%}")
    
    
    # --- SIMPLE BASELINE (Zero-Shot LLM) ---
    print("\nEvaluating Simple Baseline (Zero-Shot LLM without RAG)...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        with open(os.path.join(PROJECT_ROOT, '.env')) as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    
    simple_prompt = """You are an Uber customer support agent on Twitter. 
Reply to this customer's tweet concisely and politely. Do not use any historical context.
Customer Tweet: "{tweet}"
"""
    
    simple_replies = []
    for text in tqdm(valid_df['text_customer']):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a helpful Uber support agent."},
                    {"role": "user", "content": simple_prompt.format(tweet=text)}
                ],
                temperature=0.7
            )
            simple_replies.append(response.choices[0].message.content.strip())
        except:
            simple_replies.append("Error")
            
    valid_df['simple_reply'] = simple_replies
    
    # We would run the LLM-as-a-judge on these replies as well, 
    # but for speed during the test, we will just save them and can judge them later.
    valid_df.to_csv(os.path.join(PROJECT_ROOT, 'data/processed/baselines_results.csv'), index=False)
    print("Saved baseline results to data/processed/baselines_results.csv")

if __name__ == "__main__":
    run_baselines()
