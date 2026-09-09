# Uber Support AI Agent

This repository contains an AI customer support agent built for **Uber** (specifically the `@Uber_Support` handle on Twitter) using the Kaggle `thoughtvector/customer-support-on-twitter` dataset.

## 1. Quick Start (Reproducing Results in < 15 mins)

**Prerequisites:**
- Python 3.9+
- A free [Groq API Key](https://console.groq.com/keys)

**Setup Instructions:**
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Add your API key to a `.env` file at the root: `GROQ_API_KEY=your_key_here`
5. Run the full pipeline and evaluation:
   ```bash
   python src/evaluate.py
   ```
*(Note: Data extraction scripts are available in `src/explore.py` and `src/extract_brand.py`)*

---

## 2. Problem Framing

For this project, I chose **Uber_Support**. 
**What "good" means for this brand:** A good support agent for Uber needs to be fast, empathetic, and capable of handling high-stress situations (like safety issues or billing disputes). A good reply is concise (under 280 characters for Twitter) and actively seeks to move complex problems to a secure channel (like DMs or an in-app support ticket) without exposing PII.

**What I chose *not* to build:**
I chose not to build an agent that actually attempts to execute backend actions (e.g., hitting a Stripe API to refund a user). Twitter is a public forum, so the goal of this agent is strictly *triage, de-escalation, and routing*.

I defined 5 core intents for this agent:
1. `Billing / Refund`
2. `Safety / Driver Behavior`
3. `Account / App Issue`
4. `General Inquiry`
5. `Needs Context`

---

## 3. Results vs. Baselines

*(To be populated once the `evaluate.py` script completes)*

| Metric | Trivial Baseline | Simple Baseline (Zero-Shot) | Our Agent (RAG + LLM) |
|--------|------------------|-----------------------------|------------------------|
| Intent Accuracy | ~20% (majority class) | *(pending)* | **47.65%** |
| Escalation Accuracy | ~50% (always Auto-handle) | *(pending)* | **47.65%** |
| LLM Judge Score (1-5) | 1.5 (static reply) | *(pending)* | **3.00** |

---

## 4. Failure Analysis

*(To be populated after reviewing the evaluation results)*

**Top Failure Modes:**
1. **...**
2. **...**
3. **...**
4. **...**
5. **...**

---

## 5. What is misleading about my headline number?

*(To be populated once the final metrics are calculated)*
- **Data Leakage / Ground Truth Quality:** The "ideal reply" in the Golden Set is whatever the human agent actually tweeted. Human agents are not perfect; they sometimes give copy-paste, unhelpful answers. If our LLM gives a *better* answer than the human, the LLM Judge might still penalize it for not matching the ground truth.
- **...**

---

## 6. Decision Log (10-15 non-obvious decisions)

1. **Brand Choice:** Chose Uber over airlines (Delta/AmericanAir) because ride-sharing interactions are often highly standardized but have critical edge cases (safety) that make the Escalate vs. Auto-handle logic interesting.
2. **LLM Provider:** Used Groq's fast inference API with `gpt-oss-20b` instead of OpenAI to ensure high-speed processing and cost-effectiveness (free tier) for grading hundreds of examples.
3. **Escalation Logic Design:** Decided that `Safety / Driver Behavior` and `Needs Context` must be hard-coded to *always* Escalate. An AI should never try to auto-resolve a potential physical safety incident.
4. **Retrieval Mechanism (RAG):** Used a lightweight TF-IDF Vectorizer with Cosine Similarity for the RAG drafting module instead of a heavy embedding model or Vector DB. For a dataset of 5,000 tweets, TF-IDF is incredibly fast, requires no API calls, and runs easily within the 15-minute reproducibility constraint.
5. **Golden Set Size:** Capped the Golden Set at 150 examples to balance statistical significance with the strict rate-limits of the free-tier LLM used for automated judging.
6. **LLM-as-a-Judge Rubric:** Chose a 1-5 scale rather than pass/fail to capture nuances in empathy and tone, which are critical for Uber's brand image.
7. **...** *(Will add more as we finalize the baselines)*

---
*Built for the Hiver SDE Intern Take-Home Assignment.*
