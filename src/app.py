import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify
from agent import UberSupportAgent

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize agent once at startup
print("Initializing Uber Support Agent...")
agent = UberSupportAgent()
print("Agent ready.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    tweet = data.get('tweet', '').strip()
    
    if not tweet:
        return jsonify({'error': 'Please enter a tweet.'}), 400
    
    # Run intent classification + escalation
    analysis = agent.analyze_tweet(tweet)
    
    # Draft a reply
    reply = agent.draft_reply(tweet)
    
    return jsonify({
        'intent': analysis.get('intent', 'Unknown'),
        'action': analysis.get('action', 'Unknown'),
        'reason': analysis.get('reason', ''),
        'reply': reply
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
