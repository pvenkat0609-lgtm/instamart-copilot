# Instamart Customer Intelligence & Support Copilot

## Overview

Instamart Customer Intelligence & Support Copilot

An end-to-end offline AI-powered customer intelligence and customer support platform built for a grocery delivery business. The project combines Data Engineering, Machine Learning, NLP, Retrieval-Augmented Generation (RAG), and Agentic AI to improve customer retention, automate support operations, and enable intelligent decision-making.

This solution is designed to operate entirely offline without dependence on external LLM APIs, making it cost-effective, reproducible, and suitable for enterprise environments.

🚀 Key Features
*Data Engineering*

-Automated data cleaning and validation
-Missing value handling and standardization
-Referential integrity checks
-SQLite data warehouse creation
-Quality assurance reporting

*Customer Intelligence*
-RFM (Recency, Frequency, Monetary) analysis
-K-Means customer segmentation
-Customer persona generation
-Churn/lapse prediction models
-Retention strategy recommendations

*Support Automation*
-Ticket classification and triage
-Priority detection
-Customer sentiment handling
-PII redaction guardrails
-Automated ticket routing
-Knowledge Assistant (RAG)

*Policy document indexing*
-TF-IDF vector retrieval
-Context-aware question answering
-Source-grounded responses
-Hallucination prevention
-Agentic Refund System

*ReAct-style reasoning and tool usage*
-Automatic refund eligibility checks
-Human escalation workflow
-Checkpoint recovery
-Audit logging

## Reproduce

Run the commands from the repository root.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Generate the supplied synthetic dataset

```powershell
python generate_data.py
```

The supplied `generate_data.py` and `mock_llm.py` are used without modification.


### 3. Run Part A - Data Foundation

```powershell
python src\clean_data.py
python src\load_database.py
python src\run_queries.py
python src\validate_data.py
python src\pandas_analysis.py
```

### 4. Run Part B - Customer Intelligence

```powershell
python src\customer_segments.py
python src\lapse_prediction.py
```

### 5. Run Part C - Support Ticket Triage

```powershell
python src\ticket_triage.py
```

### 6. Run Part D - Policy RAG and ReAct Agent

```powershell
python src\policy_rag.py
python src\agent.py
```

The scripts generate their results in the `outputs/` directory. The agent also creates resumable checkpoints in `state/`.

## Part A - Data Foundation

### Data cleaning and database

- Cleaned customer and order datasets.
- Normalized city and categorical values.
- Loaded the cleaned data into SQLite.
- Created and executed seven SQL analysis queries.
- Ran nine validation checks.

**Database results:**

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders loaded into SQLite | 12,309 |

The validation script completed with `ALL CHECKPOINTS PASSED`.

Completed orders generated revenue of **INR 2,998,668.80**.

### Exploratory data analysis

- Completed orders: 11,110
- Cancelled orders: 1,199
- Mean completed order value: INR 269.91
- Median completed order value: INR 135.00
- Order-value distribution skewness: 4.61
- IQR-based high-value outliers: 897

The correlation between discount percentage and order amount was approximately -0.095. The correlation between quantity and order amount was approximately 0.087. These are weak associations and do not establish causation.

### SQL results to note

**Q2 largest cell.** The largest revenue cell is Bengaluru / Electronics at INR 233,909.45.

**Q6 top customer per city (by completed spend).**

| City | Customer | Total spend (INR) |
|---|---|---:|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

Ties make RANK() differ from ROW_NUMBER() because two customers in a city with exactly the same spend share one RANK, while ROW_NUMBER() still gives them different numbers (1, 2), so the two columns differ after the tie.

### Raw-data audit

The raw datasets were inspected before cleaning. The audit reported:

| Dataset | Shape | Columns with missing values |
|---|---:|---|
| Customers | 1,200 × 5 | age: 61 (5.08%); gender: 30 (2.50%) |
| Products | 40 × 4 | None |
| Orders | 13,084 × 9 | order_amount: 53 (0.41%); delivery_minutes: 2,541 (19.42%) |
| Agent scenarios | 7 × 5 | None |
| Policy evaluation | 14 × 3 | expected_fact: 2 (14.29%) |
| Tickets | 164 × 5 | order_id: 8 (4.88%); text: 4 (2.44%) |

**Column-level raw-data audit**

The following data types and missing-value counts were obtained using Pandas `read_csv()` before cleaning. Missing percentages are calculated against the total rows in each file.

| Dataset | Column | Data type | Missing | Missing % |
|---|---|---|---:|---:|
| Customers | customer_id | str | 0 | 0.00% |
| Customers | city | str | 0 | 0.00% |
| Customers | signup_date | str | 0 | 0.00% |
| Customers | age | float64 | 61 | 5.08% |
| Customers | gender | str | 30 | 2.50% |
| Products | product_id | str | 0 | 0.00% |
| Products | product_name | str | 0 | 0.00% |
| Products | category | str | 0 | 0.00% |
| Products | price_inr | int64 | 0 | 0.00% |
| Orders | order_id | str | 0 | 0.00% |
| Orders | customer_id | str | 0 | 0.00% |
| Orders | product_id | str | 0 | 0.00% |
| Orders | quantity | int64 | 0 | 0.00% |
| Orders | order_ts | str | 0 | 0.00% |
| Orders | status | str | 0 | 0.00% |
| Orders | discount_pct | int64 | 0 | 0.00% |
| Orders | order_amount | str | 53 | 0.41% |
| Orders | delivery_minutes | float64 | 2,541 | 19.42% |
| Agent scenarios | scenario_id | str | 0 | 0.00% |
| Agent scenarios | customer_id | str | 0 | 0.00% |
| Agent scenarios | message | str | 0 | 0.00% |
| Agent scenarios | expected_outcome | str | 0 | 0.00% |
| Agent scenarios | expected_refund_inr | float64 | 0 | 0.00% |
| Policy evaluation | question | str | 0 | 0.00% |
| Policy evaluation | expected_doc | str | 0 | 0.00% |
| Policy evaluation | expected_fact | str | 2 | 14.29% |
| Tickets | ticket_id | str | 0 | 0.00% |
| Tickets | customer_id | str | 0 | 0.00% |
| Tickets | order_id | str | 8 | 4.88% |
| Tickets | text | str | 4 | 2.44% |
| Tickets | true_category | str | 0 | 0.00% |

**Raw order quality checks**

| Check | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts (including missing values) | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |

**Raw customer quality checks**

| Check | Result |
|---|---:|
| Blank ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank genders | 30 |

**Cleaning operations and record counts**

Customer cleaning:

| Operation | Result |
|---|---:|
| Raw customer records | 1,200 |
| Blank ages | 61 |
| Ages outside the accepted 10–100 range | 6 |
| Ages missing before median imputation | 67 |
| Ages filled using city-level median | 67 |
| Missing ages after cleaning | 0 |
| Blank gender values normalized to `unknown` | 30 |
| Cleaned gender categories | female: 610; male: 560; unknown: 30 |
| Distinct raw city spellings | 24 |
| Cleaned city categories | 6 |
| Cleaned age range | 18–60 |

City names were stripped of surrounding whitespace and converted to title case. Gender values were stripped, converted to lowercase and mapped to `female`, `male` or `unknown`. Invalid ages outside 10–100 were treated as missing. Missing ages were imputed using the median age within each city. Signup dates were converted to datetime values.

Order cleaning:

| Operation | Result |
|---|---:|
| Raw order records | 13,084 |
| Exact duplicate rows removed | 119 |
| Rows after exact duplicate removal | 12,965 |
| Rows after retaining the latest record per order ID | 12,309 |
| Additional rows removed by order-ID deduplication | 656 |
| Missing or invalid order amounts filled using product price, quantity and discount | 112 |
| Negative delivery times converted to missing values | 105 |
| Missing completed-order delivery times filled using category-level medians | 774 |
| Missing cancelled-order delivery times retained | 1,199 |
| Missing order amounts after cleaning | 0 |
| Missing completed-order delivery times after cleaning | 0 |

Order timestamps were converted to datetime values, and statuses were standardized to lowercase. Numeric columns were converted to numeric types. Product category and price were joined using `product_id`, with a many-to-one validation. Missing order amounts were calculated as product price × quantity × (1 − discount percentage / 100). Negative delivery times were treated as missing. Missing delivery times were imputed using category-level medians for completed orders; missing delivery times for cancelled orders were retained.

The final cleaned orders dataset contains 12,309 records: 11,110 completed and 1,199 cancelled. Completed-order revenue is INR 2,998,668.80.

**Cleaned customer distributions**

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |

Invalid ages were treated as missing and filled using the median age within each city. The cleaned dataset has no missing ages, with an age range of 18–60.


### SQL query results summary

| Query | Result |
|---|---|
| Q1 | 30 customers met the completed-spend criteria. |
| Q2 | 30 city/category combinations; total revenue INR 2,998,668.80. |
| Q3 | 11 products priced above the average catalogue price. |
| Q4 | 25 customers had no completed order. |
| Q5 | 550 customers spent above the average customer spend of INR 2,552.06. |
| Q6 | 1,175 customers ranked by completed spend within their city; RANK differed from ROW_NUMBER in 8 rows. |
| Q7 | 183 daily revenue rows; final cumulative revenue INR 2,998,668.80. |


### Pandas cross-check and distribution statistics

The Pandas pivot-table grand total matches the SQL revenue total: INR 2,998,668.80.

**City-level completed-order summary**

| City | Total revenue (INR) | Mean order value (INR) | Orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**Completed order-value distribution**

| Statistic | Value (INR) |
|---|---:|
| First quartile (Q1) | 64.80 |
| Third quartile (Q3) | 297.00 |
| Lower outlier fence | -283.50 |
| Upper outlier fence | 645.30 |
| Lower outliers | 0 |
| Upper outliers | 897 |

The distribution is strongly right-skewed. The outlier count is based on the 1.5 × IQR rule; these high-value orders are retained rather than automatically removed.

## Part B - Customer Segmentation and Lapse Prediction

### RFM segmentation

RFM features represent recency, frequency, and monetary value.

- Customers included in RFM analysis: 1,175
- K-means clusters: 4
- Silhouette score: approximately 0.526

**Choosing K.** I tested K from 2 to 8. Inertia falls quickly up to K=4 and more slowly after that (743.1 at K=4, 617.4 at K=5), so the elbow is at K=4. K=4 also has the highest silhouette score (0.526), and its four segments are easy for the business to act on. I chose K=4.

**Why inertia cannot pick K.** Inertia measures how close points are to their cluster centre. It always falls as K grows, so K=8 (366.5) is lower than K=2 (1820.8), because more clusters mean each point sits closer to a centre. A lower number alone therefore does not mean a better model, and we also need the elbow, the silhouette score and business meaning.

**Segment labels.** The cluster numbers (0, 1, 2, 3) are only labels and have no order. The Premium segment is the one with the highest mean monetary value, not a particular number.
- PCA explained variance: approximately 85.2%

**RFM summary (1,175 customers)**

| Measure | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Recency (days) | 0 | 19 | 33.56 | 180 |
| Frequency (orders) | 1 | 5 | 9.46 | 33 |
| Monetary (INR) | 12.00 | 2,286.50 | 2,552.06 | 14,766.55 |

The resulting customer segments were:

| Segment | Customers |
|---|---:|
| Regular | 436 |
| Premium | 138 |
| Loyal | 419 |
| At-Risk | 182 |

**Recommended action per segment**

| Segment | Customers | Recommended action |
|---|---:|---|
| Regular Customers | 436 | Send occasional offers to move them toward more frequent ordering. |
| Loyal Customers | 419 | Reward them with a loyalty programme and early access, so they stay. |
| At-Risk Customers | 182 | Send a win-back offer now, because they have not ordered for about 109 days on average. |
| Premium Customers | 138 | Give them a personal service line and priority support, because they have the highest spend (about INR 7,130). |

The Premium segment contains 138 customers.

### Lapse prediction

The lapse cohort contained 1,156 customers, of whom 443 were classified as lapsed, giving a lapse rate of 38.32%.

The dataset was split into 924 training rows and 232 test rows.

**Leakage controls.** No encoder, scaler or resampler saw test rows: all of them were fitted on the training rows only. No order dated 2025-09-01 or later was used to build the features. The segment labels from the K-means step were not used as features.

**Scaler check.** The scaler learned a mean frequency of 8.1147 from the 924 training rows, while the full-cohort mean is 8.0285. They differ because the scaler was fitted on the training rows only, and the test rows were left out of that calculation. A code check confirmed that the feature columns contain 0 missing values.

| Model | Test accuracy | Lapsed-class F1 |
|---|---:|---:|
| Baseline | 0.6164 | 0.0000 |
| Logistic Regression | 0.7284 | 0.6480 |
| Random Forest | 0.7371 | 0.6554 |
| Oversampled Logistic Regression | 0.7155 | 0.6972 |

**Model choice.** Random Forest was selected. It had the higher cross-validated F1 (0.6811 vs 0.6506) and a slightly higher test F1 on the lapsed class (0.6554 vs 0.6480), Missing a lapsing customer means a lost customer with no retention offer, which costs far more than a wasted discount sent to a loyal customer who would have stayed. Because recall matters more than precision here, F1 on the lapsed class is a better guide than accuracy. The Random Forest also made fewer false alarms (30 vs 32 loyal customers wrongly flagged), so it was the better choice on both counts.

**Top 5 Random Forest features (importance) and business reading**

| Feature | Importance | Business reading |
|---|---:|---|
| Frequency | 0.2796 | How often a customer ordered is the strongest sign of whether they keep ordering. |
| Monetary | 0.2712 | Total spend before the cutoff is almost as important as frequency. |
| Recency | 0.1151 | The longer since the last order, the more likely the customer is drifting away. |
| Average delivery minutes | 0.0613 | Slower deliveries have some link to customers leaving. |
| Cancellation rate | 0.0523 | Customers who cancel more often are somewhat more likely to lapse. |

**Logistic Regression coefficients (sign) and business reading**

| Feature | Coefficient | Sign | Business reading |
|---|---:|---|---|
| Frequency | -1.6982 | Negative | More past orders means a lower chance of lapsing. |
| Favourite category: Electronics | -1.1367 | Negative | Customers who mostly buy electronics are less likely to lapse than the reference category. |
| Monetary | +0.2846 | Positive | Higher spend raises lapse risk only after frequency is accounted for, because spend and frequency move together. This is a model artefact, not a rule that big spenders leave. |

Oversampling increased lapsed-customer recall from approximately 0.6517 to 0.8539 and F1 from approximately 0.6480 to 0.6972, but reduced precision from approximately 0.6444 to 0.5891. This demonstrates the trade-off between identifying more lapsed customers and increasing false positives.

The selected model was saved and reloaded successfully, and its predictions matched after reloading.

**Retention list.** Among the test-set customers in the Premium segment, the 10 with the highest predicted lapse probability were saved to `outputs/retention_list.csv`. Of these 10 customers, 1 truly lapsed (C0525). The Premium flag comes from the whole-window RFM, which includes September, the month that defines the lapse label, so this retention list is illustrative and not a leakage-free evaluation.

**Evaluation limitation:** The Premium segment flag is based on the full-window RFM calculation, including the September period. It should therefore be treated as a descriptive customer segment rather than a strictly leakage-free predictive feature.

### Additional K-means evaluation

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

The K=4 segment profiles were:

| Segment | Customers | Mean recency | Mean frequency | Mean monetary (INR) |
|---|---:|---:|---:|---:|
| Regular | 436 | 28.86 | 3.77 | 1,057.62 |
| Premium | 138 | 24.12 | 6.77 | 7,129.70 |
| Loyal | 419 | 8.77 | 19.40 | 3,410.02 |
| At-Risk | 182 | 109.04 | 2.20 | 685.99 |

PCA component 1 explains 63.4% of variance and component 2 explains 21.8%, for 85.2% combined.

### Lapse-model evaluation details

The GridSearchCV results were:

| Model | Best parameters | Best CV F1 |
|---|---|---:|
| Logistic Regression | `C=100` | 0.6506 |
| Random Forest | `max_depth=8, min_samples_leaf=10` | 0.6811 |

**Logistic Regression test confusion matrix**

| | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 111 | 32 |
| Actual lapsed | 31 | 58 |

Accuracy: 0.7284; precision: 0.6444; recall: 0.6517; lapsed-class F1: 0.6480.

**Random Forest test confusion matrix**

| | Predicted not lapsed | Predicted lapsed |
|---|---:|---:|
| Actual not lapsed | 113 | 30 |
| Actual lapsed | 31 | 58 |

Accuracy: 0.7371; precision: 0.6591; recall: 0.6517; lapsed-class F1: 0.6554.

Both models had the same number of lapsed-customer false negatives (31). Random Forest had two fewer false positives (30 versus 32), which is consistent with its slightly better test F1 and accuracy.

## Part C - Support Ticket Triage

The ticket triage pipeline includes:

- Email, phone number, and order ID redaction.
- Empty-input and long-input guardrails.
- Zero-shot and few-shot classification.
- Retry handling for simulated transient LLM failures.
- JSON validation and keyword-based fallback parsing.
- Invalid-output handling.
- Token-budget enforcement.
- Final routing by category, customer segment, and priority.

The four ticket categories are billing, delivery, technical, and fraud.

### Privacy and input validation

| Check | Result |
|---|---:|
| Email redactions | 55 |
| Phone redactions | 91 |
| Order ID redactions | 132 |
| Total redactions | 278 |
| Residual sensitive patterns | 0 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Eligible tickets | 156 |

All 19 simulated transient errors were recovered through retries.

### Classification results

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy | 85.71% | 100% |
| Average tokens per classified ticket | 60.76 | 138.23 |

Among the 156 eligible tickets:

- Zero-shot output: 135 valid JSON, 12 keyword-recovered, and 9 invalid.
- Few-shot output: 135 valid JSON, 12 keyword-recovered, and 9 invalid.

The zero-shot confusion matrix, using the order billing, delivery, technical, fraud, was:

```text
[[38,  0,  0,  0],
 [ 0, 36,  0,  0],
 [10,  0, 29,  0],
 [11,  0,  0, 23]]
```

**When few-shot is worth the tokens.** The zero-shot prompt used 60.76 tokens per classified ticket and reached 85.71% accuracy (126 of 147). The few-shot prompt used 138.23 tokens per ticket, about 77 tokens more, and reached 100%. Every zero-shot error was a fraud ticket predicted as billing (11) or a technical ticket predicted as billing (10), and a missed fraud ticket is the costly mistake, because fraud must reach the security team quickly. So few-shot is worth the extra tokens when a wrong category has a real cost, and zero-shot is enough when tickets are simple and the budget is tight.

**Which outputs were rule-validated and which went to humans.** In the final few-shot run, 147 tickets passed the output rules and were routed automatically: 135 with valid JSON and 12 recovered by the keyword rule. 17 tickets went to human review: 9 had invalid model output (for example a category outside the allowed list), 4 were empty and 4 were too long, so none of these 8 were sent to the model. No ticket ended with a permanent error, and none was guessed.

### Budget and routing

The zero-shot budget demonstration used a token budget of 6,000. Processing stopped after 99 tickets when token usage reached 6,023, slightly exceeding the budget because the budget was checked before each classification.

**Budget demo result (TOKEN_BUDGET = 6000, zero-shot, saved to `outputs/triage_budget_demo.csv`).** Of the 156 eligible tickets, 99 were classified, 6 had invalid output, and 51 ended as `budget_exhausted`, so the budget did stop the run. Those 51 tickets went to human review without any model call. The final `outputs/triage_results.csv` used a separate few-shot run with TOKEN_BUDGET = 100000, so no ticket was exhausted there.

Final triage output contained 164 rows:

| Route | Tickets |
|---|---:|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

**Routing rules.** Tickets with status skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted go to human_review with priority NORMAL. Otherwise, a fraud ticket goes to security_team with priority HIGH. Otherwise, a ticket from a Premium customer goes to premium_desk with priority HIGH. All other tickets go to support_queue with priority NORMAL. A customer with no row in the segment file is treated as not Premium. This gives 45 HIGH tickets (34 fraud and 11 premium) and 119 NORMAL tickets.

The output includes ticket ID, customer ID, status, category, segment name, priority, route, and tokens used.

## Part D - Policy RAG and ReAct Agent

### Chunking

The five policy files are split into chunks of 40 words with an overlap of 10 words. The first chunk covers words 0 to 39 and the next starts 30 words later. The source file name is stored with every chunk.

| Policy document | Chunks |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

### Retrieval (TF-IDF)

`TfidfVectorizer()` is fitted on the chunk texts. A query is transformed with the same fitted vectorizer, and the chunks are ranked by cosine similarity. `retrieve(query, k=3)` returns a list of dictionaries with `source`, `score` and `text`.

TF (term frequency) measures how often a word appears in a chunk. IDF (inverse document frequency) measures how rare a word is across all chunks, so rare words count for more and common words count for less. A limitation of TF-IDF is that it matches words, not meaning: two questions with the same meaning but different wording can retrieve different chunks.

### Answering with retrieved context

`answer(question)` retrieves the top three chunks and sends two messages to `mock_llm.chat`. The system message contains the words POLICY ASSISTANT and tells the model to answer only from the context, or to say it could not find the information. The user message has the form CONTEXT, then the three chunks each prefixed with `[source file name]`, then QUESTION, then the question. The function returns the answer text and the sources used.

### Retrieval and grounding evaluation

The file `data/raw/policy_eval.csv` has 14 questions: 12 answerable and 2 not covered by any document.

| Setting | Chunks | Expected document in top 3 (of 12) | Answer contains expected fact (of 12) | Fallback "I could not find" (of 2) |
|---|---:|---:|---:|---:|
| 40 words, overlap 10 | 15 | 12 | 12 | 2 |
| 40 words, overlap 0 | 13 | 12 | 11 | 2 |

With overlap 0, one question breaks: "How soon is money returned when a payment fails but is deducted?" (expected fact: 48 hours, in payment_policy.txt). The sentence holding the fact is cut across a chunk boundary, so the retrieved chunks do not contain the full fact. The 10-word overlap keeps those words together in one chunk. Results are saved in `outputs/policy_rag_evaluation.csv` and `outputs/policy_rag_zero_overlap.csv`.

### Tools

The agent has five tools, kept in `TOOLS_SPEC` (name, one-sentence description, parameters) and in the dictionary `TOOL_MAP`, which maps a tool name to a Python function.

| Tool | What it does |
|---|---|
| get_order(order_id) | Reads the order from `data/instamart.db` and returns its status, amount and delivery minutes, or an "order not found" error. |
| search_policy(query) | Returns the retrieval results for the query. |
| calculate_refund(order_amount, delivery_minutes) | Returns the full order amount if delivery took more than 45 minutes, otherwise 0. |
| issue_refund(order_id, refund_inr) | Appends one line to `outputs/refund_ledger.jsonl` and returns the refund amount. |
| escalate_to_human(order_id, refund_inr) | Appends one line to the ledger for a refund above INR 1000 and returns the amount. |

Refund rules: only completed orders can be refunded; delivery in more than 45 minutes gives a 100% refund; delivery in 45 minutes or less gives 0; a refund above INR 1000 is escalated to a human and not issued automatically. All simulated requests are treated as raised inside the 7-day window.

### ReAct loop

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts with a system message containing the word AGENT and the user message. In each iteration the model is called. If the reply has no tool call, its text is the final answer. Otherwise the tool is looked up in `TOOL_MAP` and executed, and its result is added to the messages as a tool message. The function returns `status`, `final_answer`, `steps`, `llm_calls` and `tokens_used`. Every run is appended to `outputs/agent_runs.jsonl`.

### Seven scenario results

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,229 |
| S2 | REFUND | 135.00 | 5 | 1,229 |
| S3 | ESCALATE | 1,999.00 (sent to a human lead, not issued) | 5 | 1,228 |
| S4 | DENIED | 0.00 | 4 | 875 |
| S5 | DENIED | 0.00 | 3 | 539 |
| S6 | NOT_FOUND | 0.00 | 2 | 206 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,327 |

S7 uses a looping request, so it never reaches a final answer and stops after six model calls with the status the brief requires. With the default token limit of 6000 it was not stopped early.

### Stop-safely demonstrations

| Demonstration | Status | Result |
|---|---|---|
| Maximum iterations (S7) | FALLBACK_MAX_ITERATIONS | Stopped after 6 model calls and 3,327 tokens. |
| S1 with token_limit=200 | FALLBACK_TOKEN_LIMIT | Stopped after 2 model calls and 242 tokens; only get_order was executed. |
| S1 with time_limit_seconds=0 | FALLBACK_TIMEOUT | Stopped before the first model call (0 calls, 0 tokens, no tools executed). |

Each fallback returns the steps completed so far and a short hand-off message for a human agent.

### Crash and resume (S1)

After every tool observation the agent saves a checkpoint file in `state/` with the run id, scenario id, full message list, tokens used, step number and the tools already executed. A simulated crash was raised right after the second tool execution, and then `resume_agent(run_id)` loaded the checkpoint and continued. Tools executed across the crash and the resume, in order:

1. get_order
2. search_policy
3. calculate_refund
4. issue_refund

Each tool ran exactly once, and the final status was REFUND with INR 128.25.

### ReAct versus plan-and-execute

ReAct called the model 5 times for S1, 3 for S5 and 2 for S6, using 1,229, 539 and 206 tokens respectively. Each call sends the growing message list again, so cumulative token usage increases with each step.

A plan-and-execute approach would generate a sequence of actions in advance and then execute it. This could reduce model calls for predictable workflows such as S1, but the actual token and call savings were not measured because a separate Plan-Execute baseline was not implemented.

ReAct adapts its next action to each tool result. In S6, the order is not found, so the agent stops after 2 calls with NOT_FOUND. In S5, it reads the cancelled status and denies the request after 3 calls. A fixed plan would need explicit checks or replanning to handle such results safely.

I chose ReAct because each refund decision depends on the previous result: order status, delivery time, and whether the refund exceeds INR 1,000. Iterative checks also support safe stopping and checkpoint-based recovery. These results demonstrate the behaviour of the implemented ReAct agent, not an empirical performance advantage over Plan-Execute.
## Limitations and Next Steps

### Limitations

- The project uses synthetic data, so its customer patterns and model results may not represent real Instamart customers.
- Lapse prediction is evaluated on one time-based cohort and test split. Performance should be revalidated on newer, independent data before operational use.
- The Premium retention list is illustrative: its segment flag uses the full-window RFM calculation, including the September label period. It is not a leakage-free predictive evaluation.
- The supplied `mock_llm.py` provides an offline simulation. Classification, retrieval and agent results do not establish performance with a production language model.
- TF-IDF retrieval matches lexical patterns rather than semantic meaning, so differently worded questions may retrieve less relevant policy chunks.
- Refund, escalation, retry and routing behaviour is simulated and would need integration testing, monitoring and human oversight before deployment.

### Next Steps

- Validate the cleaning and modelling pipeline on representative, anonymised real-world data.
- Repeat lapse evaluation across multiple chronological holdout periods and monitor precision, recall and calibration.
- Build a leakage-free retention workflow using customer segments calculated strictly before the prediction date.
- Evaluate semantic retrieval alongside TF-IDF using a broader set of policy questions.
- Test the agent against additional edge cases and real operational workflows, with audit logs and human approval for sensitive actions.
- If comparing ReAct with plan-and-execute, implement and evaluate both approaches under the same scenarios and token limits.
#   i n s t a m a r t - c o p i l o t 
 
 