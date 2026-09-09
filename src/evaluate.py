import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score
from openai import OpenAI
from tqdm import tqdm
from agent import UberSupportAgent

def run_evaluation():
    print("Loading Golden Set...")
    df = pd.read_csv('data/processed/golden_set.csv')
    
    # Filter out API errors from our pre-labeling phase
    valid_df = df[df['intent'] != 'Error']
    print(f"Evaluating on {len(valid_df)} valid examples from the Golden Set.")
    
    agent = UberSupportAgent()
    
    predicted_intents = []
    predicted_actions = []
    drafted_replies = []
    
    print("Running Agent Pipeline...")
    for text in tqdm(valid_df['text_customer']):
        analysis = agent.analyze_tweet(text)
        predicted_intents.append(analysis.get('intent', 'Unknown'))
        predicted_actions.append(analysis.get('action', 'Unknown'))
        
        reply = agent.draft_reply(text)
        drafted_replies.append(reply)
        
    valid_df['predicted_intent'] = predicted_intents
    valid_df['predicted_action'] = predicted_actions
    valid_df['drafted_reply'] = drafted_replies
    
    # Automated Metrics
    intent_acc = accuracy_score(valid_df['intent'], valid_df['predicted_intent'])
    action_acc = accuracy_score(valid_df['action'], valid_df['predicted_action'])
    
    print("\n--- Automated Metrics ---")
    print(f"Intent Classification Accuracy: {intent_acc:.2%}")
    print(f"Escalation Action Accuracy:   {action_acc:.2%}")
    
    # LLM-as-a-judge
    print("\nRunning LLM-as-a-Judge for Reply Quality...")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        with open(".env") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    
    judge_prompt = """You are an expert evaluator grading a customer support AI.
Rate the AI's drafted reply based on the Ground Truth Ideal Reply on a scale of 1 to 5.
1 = Terrible (Inaccurate, wrong tone, completely unhelpful)
3 = Acceptable (Polite but maybe missing some nuance)
5 = Excellent (Matches the tone and helpfulness of the ideal reply perfectly)

Customer Tweet: "{tweet}"
Ideal Reply: "{ideal}"
AI Drafted Reply: "{draft}"

Return a JSON object with a single key "score" containing an integer from 1 to 5.
"""
    
    scores = []
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You output JSON."},
                    {"role": "user", "content": judge_prompt.format(
                        tweet=row['text_customer'], 
                        ideal=row['ideal_reply'], 
                        draft=row['drafted_reply']
                    )}
                ],
                response_format={ "type": "json_object" }
            )
            result = json.loads(response.choices[0].message.content)
            scores.append(result.get("score", 3))
        except Exception as e:
            scores.append(3) # Default to 3 on error
            
    valid_df['judge_score'] = scores
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average LLM Judge Score (1-5): {avg_score:.2f}")
    
    # Save results
    valid_df.to_csv('data/processed/evaluation_results.csv', index=False)
    print("Detailed results saved to data/processed/evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()
