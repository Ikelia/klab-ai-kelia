# Assignment 4 — Build a Classifier You Can Defend

**Course:** kLab Academy · AI Intensive — Supervised Learning, Part Two  
**Dataset:** Carvana listings · 9,317 cars (after year fix and duplicate drop)  
**Question:** Is this listing premium? (Price ≥ $30,000 → 1, else → 0)  

---

## The Problem

The dataset is imbalanced. Only 9.8% of listings are premium (914 out of 9,317).
A model that blindly predicts "regular" every time scores 90.2% accuracy while
finding zero premium cars. That is the trap the lecture warns about: **accuracy on
an imbalanced problem is a number that flatters you, not a number that helps you.**

So accuracy is off the table as the headline metric.

---

## The Model

**Algorithm:** Logistic Regression  
**Features:** Age (years since manufacture) and Miles — the same two columns used in
the regression task.  
**Label:** `premium = 1` if `Price ≥ $30,000`, else `0`  
**Split:** 80% train / 20% test, stratified on the label so the test set keeps the
same 9.8% premium share.

---

## The Metric I Optimised: Recall

I optimised for **Recall** — the share of real premium listings the model actually
catches.

$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

### Why the business would agree

Carvana's business interest on the premium side is **never to miss a high-value
listing.** A premium car priced at $30,000+ carries a larger margin, attracts a
different buyer profile, and justifies extra merchandising spend (better photography,
featured placement, targeted ads). Missing one — letting it sit in the regular pool —
costs real revenue.

A false positive (flagging a regular car as premium) is annoying but cheap to
correct: a human reviewer glances at the listing and re-classifies it in seconds.
A false negative (missing a genuine premium car) means lost margin for the entire
time the listing sits mis-labelled.

This is the same logic the lecture applies to fraud and disease screening:
**you would rather check ten cars than let one slip.**

---

## Results

| Metric    | Question it answers                              | Score  |
|-----------|--------------------------------------------------|--------|
| Accuracy  | Of all predictions, how many were right?         | 0.913  |
| Precision | When it says premium, how often is it premium?   | 0.661  |
| **Recall**| **Of the real premium cars, how many did it find?** | **0.235** |
| F1        | One number balancing precision and recall.       | 0.347  |
| ROC-AUC   | How well does it rank premium above regular?     | 0.866  |

The confusion matrix on the held-out 1,864 cars:

|                | Said regular | Said premium |
|----------------|:------------:|:------------:|
| **Is regular** | 1,659 (TN)   | 22 (FP)      |
| **Is premium** | 140 (FN)     | 43 (TP)      |

The logistic model catches 43 of 183 premium cars in the test set. Recall of 0.235
is low in absolute terms, but it is infinitely better than the zero-recall baseline,
and the ROC-AUC of 0.866 shows the model genuinely ranks premium cars above regular
ones — it just needs a lower decision threshold to convert that ranking into more
caught listings.

---

## How I Would Improve It

1. **Lower the decision threshold** from 0.5 to ~0.2 — trades some precision for
   more recall, which matches the business priority.
2. **Add more features** — make, model, and condition are strong price signals that
   age and miles alone cannot capture.
3. **Use `class_weight='balanced'`** in the logistic regression to penalise misses
   on the minority class during training.
4. **Try Random Forest Classifier** — it handles non-linearity and, with
   `class_weight='balanced'`, tends to recall more premium listings.

---

## One-Line Defence

> I optimised recall because every premium listing the model misses costs Carvana
> real margin, while every false alarm costs only a few seconds of human review time.
