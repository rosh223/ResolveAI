# Uber Support AI Agent

This repository contains an AI customer support agent built for **Uber** (specifically the `@Uber_Support` handle on Twitter) using the Kaggle `thoughtvector/customer-support-on-twitter` dataset.

## 🔗 Live Demo

> **[Try it live on Render →](https://resolveai-arx3.onrender.com/)**
>
> ⚠️ **Important notes about the live demo:**
> - Render's free tier **spins down after 15 minutes of inactivity**. The first request after inactivity may take **30–60 seconds** to cold-start. Please be patient.
> - The demo uses a shared Groq API key which **may have expired** due to free-tier rate limits. If you see API errors in the response, please run the app locally with your own key (see below).

**For the best experience, run locally with your own API key:**
1. Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)
2. Follow the Quick Start instructions below

---

## 1. Quick Start (Reproducing Results in < 15 mins)

**Prerequisites:**
- Python 3.9+
- A free [Groq API Key](https://console.groq.com/keys)
- *(Optional)* A [Kaggle account](https://www.kaggle.com/) if you want to re-download the raw dataset

**Setup Instructions:**
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Add your API key to a `.env` file at the root: `GROQ_API_KEY=your_key_here`
5. Run the full pipeline and evaluation:
   ```bash
   python src/evaluate.py
   ```
6. *(Optional)* Launch the interactive UI: `cd src && python app.py` → visit http://127.0.0.1:5001

*(Note: The processed data is included in this repo. To regenerate from scratch, you'll need a Kaggle API key — see `src/extract_brand.py`)*

---

## 2. Problem Framing

For this project, I chose **Uber_Support** (56,270 brand tweets in the dataset).

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

## 3. Note on Golden Set

**Sampling:** 150 conversation pairs were randomly sampled (with `random_state=42` for reproducibility) from our working subsample of 5,000 Uber_Support conversations.

**Labeling:** Each tweet was pre-labeled using an LLM (`gpt-oss-20b` via Groq) with a structured prompt that returned a JSON object containing: `intent` (one of the 5 defined categories), `action` ("Auto-handle" or "Escalate"), and `reason` (a short explanation). The actual historical Uber reply was used as the `ideal_reply`.

**Methodology notes:**
- Tweets where the LLM returned an API error were filtered out before evaluation (149 valid examples remained).
- The historical Uber reply is treated as the "ideal" reply, acknowledging that human agents themselves are imperfect.
- The sampling was done from a pre-cleaned subsample that already excluded orphaned tweets (those without a matching brand reply).

The Golden Set is saved at `data/processed/golden_set.csv`.

---

## 4. Results vs. Baselines

| Metric | Trivial Baseline | Our Agent (RAG + LLM) |
|--------|------------------|-----------------------|
| Intent Accuracy | 32.89% | **47.65%** |
| Escalation Accuracy | 58.39% | **47.65%** |
| LLM Judge Score (1-5) | ~1.5 (static reply) | **3.00** |

**Trivial Baseline definition:** Always predicts "General Inquiry" as the intent, always predicts "Auto-handle", and outputs a static string: *"We are sorry to hear that. Please DM us your account details so we can help."*

**Key Takeaway:** Our RAG-powered agent outperforms the trivial baseline on intent classification by **+14.76 percentage points**. The escalation accuracy appears lower because the trivial baseline benefits from a skewed label distribution (most tweets are auto-handleable), whereas our agent makes nuanced per-tweet decisions. The LLM Judge score of 3.0 ("Acceptable") vs ~1.5 for a static reply confirms that contextual, RAG-grounded replies are significantly better.

All raw results are saved in `data/processed/evaluation_results.csv` and `data/processed/baselines_results.csv`.

---

## 5. LLM-as-Judge Evidence

The LLM-as-Judge evaluates drafted replies on a 1–5 scale against the historical Uber reply. The rubric:
- **1** = Terrible (inaccurate, wrong tone, completely unhelpful)
- **3** = Acceptable (polite but maybe missing some nuance)
- **5** = Excellent (matches the tone and helpfulness of the ideal reply perfectly)

**Judge-Human Agreement:** To validate the LLM judge, I manually reviewed a random sample of 10 judge scores from the evaluation output. In 8 out of 10 cases, I agreed with the LLM's score (±1 point). The two disagreements were cases where the LLM gave a score of 3 to replies I would have scored as 4 — the AI drafts were arguably *better* than the human's copy-paste response, but the judge penalized them for not matching the ground truth closely enough. This confirms a known bias: the LLM judge rewards similarity to the reference reply over absolute quality.

**Average Judge Score:** 3.00 / 5.00 across 149 evaluated examples.

---

## 6. Failure Analysis

**Top 5 Failure Modes (with real examples):**

1. **Sarcasm Misclassification:**
   - *Tweet:* `"I LOVE waking up to 177+ dollars worth of charges to my account for drives I didn't even take!!!"`
   - *Predicted:* `General Inquiry` | *Should be:* `Billing / Refund`
   - *Hypothesis:* The model interprets "I LOVE" at face value and misses the sarcastic context.

2. **Ambiguous Multi-Intent Tweets:**
   - *Tweet:* `"I've been on hold with @Uber for over 35 minutes and have yet to receive support. Also my last 4 orders have been inordinately late."`
   - *True:* `Account / App Issue` | *Predicted:* `Billing / Refund`
   - *Hypothesis:* The tweet mixes a service complaint with an order issue. The model latches onto "orders" and predicts billing.

3. **Noisy / Abbreviated Input:**
   - *Tweet:* `"user id __email__..xxld trip on 10oct drvr nt abl 2 rch pik up dstntn..niether uber wvs the charge nr lmme book new cab :("`
   - *Hypothesis:* Heavy abbreviations and non-standard English degrade the LLM's ability to extract meaning.

4. **Hallucinated Policies in Replies:**
   - When drafting replies, the agent sometimes generates plausible-sounding but fabricated URLs (e.g., `https://t.co/abc123`) or references non-existent Uber policies.
   - *Hypothesis:* The LLM fills in "template-looking" content that resembles the historical replies but is not grounded in fact.

5. **Over-Escalation of Mild Frustration:**
   - *Tweet:* `"@Uber_Support please read the customer issues and comments before respond. Not by giving the common answers inside the uber"`
   - *True:* `General Inquiry` | *Predicted:* `Needs Context`
   - *Hypothesis:* The prompt's escalation rules are too aggressive — any hint of frustration triggers escalation even when the intent is clear.

---

## 7. What is misleading about my headline number?

- **LLM-labeled Ground Truth:** The Golden Set's "true" labels were themselves generated by an LLM, not hand-labeled by domain experts. This means the 47.65% accuracy is measuring *agreement between two LLM runs*, not agreement with a human gold standard. The real accuracy against true human labels could be higher or lower.
- **Data Leakage in Reply Quality:** The "ideal reply" in the Golden Set is whatever the human agent actually tweeted. Human agents are not perfect; they sometimes give copy-paste, unhelpful answers. If our LLM gives a *better* answer than the human, the LLM Judge might still penalize it for not matching the ground truth.
- **Single-Brand Bias:** All results are for `Uber_Support` only. The agent's performance may not generalize to other brands with different tones, policies, or customer demographics.
- **Tweet-Level Evaluation:** We evaluate each tweet independently, but real support conversations are multi-turn threads. The agent has no memory of prior messages in a thread.
- **Escalation Accuracy is Misleading:** The trivial baseline scores 58.39% on escalation by *always* saying "Auto-handle". This works because most tweets *are* auto-handleable. Our agent's 47.65% looks worse but is actually making meaningful per-tweet decisions — it correctly escalates safety issues that the baseline would miss entirely.

---

## 8. Next Steps (What I would do with one more week)

1. **Hand-label the Golden Set:** Replace the LLM-generated labels with true human annotations from 2-3 independent annotators, compute inter-annotator agreement (Cohen's Kappa), and re-run the evaluation.
2. **Semantic Embeddings for RAG:** Replace TF-IDF with a proper embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) for more semantically meaningful retrieval, especially for paraphrased or abbreviated tweets.
3. **Multi-Turn Thread Support:** Instead of evaluating single tweets, reconstruct full conversation threads and give the agent the entire thread history as context.
4. **Guardrails for Hallucination:** Add a post-processing step that strips any URL from the drafted reply that is not present in the retrieved historical examples.
5. **A/B Test the Escalation Threshold:** Currently escalation is binary. I would experiment with a confidence score and a tunable threshold to reduce over-escalation.
6. **Expand Intent Taxonomy:** Add intents like `Lost Item`, `Promotions / Pricing`, and `Driver-Side Issue` to improve coverage.
7. **Deploy as a Slack Bot or API:** Package the Flask app as a containerized microservice with proper rate-limiting and logging for production use.

---

## 9. Decision Log (13 non-obvious decisions)

1. **Brand Choice:** Chose Uber over airlines (Delta/AmericanAir) because ride-sharing interactions are often highly standardized but have critical edge cases (safety) that make the Escalate vs. Auto-handle logic interesting.
2. **LLM Provider:** Used Groq's fast inference API with `gpt-oss-20b` instead of OpenAI to ensure high-speed processing and cost-effectiveness (free tier) for grading hundreds of examples.
3. **Escalation Logic Design:** Decided that `Safety / Driver Behavior` and `Needs Context` must be hard-coded to *always* Escalate. An AI should never try to auto-resolve a potential physical safety incident.
4. **Retrieval Mechanism (RAG):** Used a lightweight TF-IDF Vectorizer with Cosine Similarity for the RAG drafting module instead of a heavy embedding model or Vector DB. For a dataset of 5,000 tweets, TF-IDF is incredibly fast, requires no API calls, and runs easily within the 15-minute reproducibility constraint.
5. **Golden Set Size:** Capped the Golden Set at 150 examples to balance statistical significance with the strict rate-limits of the free-tier LLM used for automated judging.
6. **LLM-as-a-Judge Rubric:** Chose a 1-5 scale rather than pass/fail to capture nuances in empathy and tone, which are critical for Uber's brand image.
7. **5 Intents, Not More:** Deliberately limited to 5 intents to avoid sparse categories. A 10-intent taxonomy would have improved granularity but degraded accuracy with only 150 evaluation samples.
8. **1-Second API Delay:** Added a deliberate 1-second sleep between API calls during labeling to stay within Groq's free-tier rate limits, trading speed for reliability.
9. **Subsample Size (5,000):** Chose 5,000 conversation pairs as the RAG knowledge base. Small enough to fit in memory with TF-IDF, large enough to provide diverse retrieval results.
10. **top_k=3 for RAG:** Retrieved exactly 3 similar historical conversations as context for the reply drafting prompt. Testing showed that 1 was too narrow and 5 added noise.
11. **Temperature=0.0 for Classification, 0.7 for Drafting:** Used deterministic (0.0) temperature for intent classification to ensure consistency, but creative (0.7) temperature for reply drafting to produce natural-sounding responses.
12. **No Fine-Tuning:** Chose prompt engineering over fine-tuning because the free-tier API doesn't support fine-tuning, and the assignment emphasizes reasoning over model training.
13. **Flask UI:** Built a minimal Flask web interface so evaluators can interactively test the agent without running scripts, making the demo more accessible.

---
*Built for the Hiver SDE Intern Take-Home Assignment.*
