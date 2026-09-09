import os
import json
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resolve project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class UberSupportAgent:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'uber_support_subsample.csv')
        
        # Load API key explicitly
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            env_path = os.path.join(PROJECT_ROOT, '.env')
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
        
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        
        self.intent_prompt_template = """You are an expert customer support AI for Uber.
Your task is to analyze the following customer tweet and determine two things:
1. The Intent of the message.
2. Whether this message can be safely Auto-handled by an AI, or if it must be Escalated to a human.

Choose exactly ONE intent from this list:
- "Billing / Refund"
- "Safety / Driver Behavior"
- "Account / App Issue"
- "General Inquiry"
- "Needs Context"

Rules for Escalation:
- ALWAYS Escalate if it's a safety issue, accident, or dangerous driver behavior.
- ALWAYS Escalate if the customer is extremely angry, threatening legal action, or using profanity.
- ALWAYS Escalate if the issue is too complex or vague ("Needs Context").
- Auto-handle standard billing questions, app issues, and general inquiries.

Customer Tweet: "{tweet}"

Return a JSON object with:
"intent" (string), "action" ("Auto-handle" or "Escalate"), and "reason" (a brief explanation for the action).
"""

        self.draft_prompt_template = """You are a customer support agent for Uber on Twitter.
Write a helpful, empathetic, and professional reply to the customer's tweet.

Here is the customer's tweet:
"{tweet}"

To help you write the reply, here are some historical examples of how Uber has replied to similar issues:
{context}

Draft the reply. Keep it under 280 characters if possible. Be polite and ask them to DM if necessary.
"""

        # Setup simple RAG (Retrieval-Augmented Generation) using TF-IDF
        self.df = pd.read_csv(data_path).dropna(subset=['text_customer', 'text_brand'])
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text_customer'])


    def analyze_tweet(self, tweet: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a helpful JSON-outputting assistant."},
                    {"role": "user", "content": self.intent_prompt_template.format(tweet=tweet)}
                ],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"intent": "Error", "action": "Escalate", "reason": f"API Error: {str(e)}"}

    def draft_reply(self, tweet: str, top_k=3) -> str:
        # Retrieve similar historical tweets
        tweet_vector = self.vectorizer.transform([tweet])
        similarities = cosine_similarity(tweet_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        context_str = ""
        for idx in top_indices:
            past_tweet = self.df.iloc[idx]['text_customer']
            past_reply = self.df.iloc[idx]['text_brand']
            context_str += f"- Past Issue: {past_tweet}\n  Uber Reply: {past_reply}\n\n"
            
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a helpful Uber support agent."},
                    {"role": "user", "content": self.draft_prompt_template.format(tweet=tweet, context=context_str)}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating reply: {str(e)}"

if __name__ == "__main__":
    # Test the agent's full pipeline
    agent = UberSupportAgent()
    
    test_tweets = [
        "My driver almost hit a pedestrian, this is unacceptable! I want a refund.",
        "How do I apply this promo code to my account?",
        "I got charged $50 for a ride I never took. Help."
    ]
    
    for tweet in test_tweets:
        print(f"Customer Tweet: {tweet}")
        
        analysis = agent.analyze_tweet(tweet)
        print(f"Intent: {analysis.get('intent')}")
        print(f"Action: {analysis.get('action')} (Reason: {analysis.get('reason')})")
        
        reply = agent.draft_reply(tweet)
        print(f"Drafted Reply: {reply}")
        
        print("-" * 50)
