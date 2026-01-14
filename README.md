# Confidence Calibration and Overconfidence Analysis of NLP Models

## 1. Introduction
Neural NLP models often output confidence scores alongside predictions.
However, high confidence does not always imply correctness.

This project analyzes the calibration and overconfidence behavior of
a pretrained NLP toxicity classifier using a large public dataset.



## 2. Motivation
In real-world systems, unreliable confidence estimates can be more
harmful than incorrect predictions. Calibration analysis helps assess
whether a model’s confidence aligns with its actual performance.



## 3. Dataset
The study uses the publicly available Jigsaw Toxic Comment
Classification dataset, which contains human-annotated labels for
toxic and non-toxic content.

The dataset is highly imbalanced, making it suitable for studying
confidence reliability under realistic conditions.

Dataset source:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge



## 4. Methodology
1. Sample comments from the dataset for feasibility.
2. Apply a pretrained toxicity classification model.
3. Record predicted labels and confidence scores.
4. Compare predictions against ground truth labels.
5. Compute calibration and overconfidence metrics.



## 5. Metrics
- Accuracy
- Expected Calibration Error (ECE)
- Overconfidence Rate

These metrics provide complementary views of model reliability.



## 6. Results
- Accuracy on the sampled subset was low due to conservative model
  behavior and dataset imbalance.
- Expected Calibration Error remained low, indicating well-aligned
  confidence estimates.
- Overconfidence was rare, suggesting cautious prediction behavior.

Detailed logs and analysis are available in:
experiments folder




## 7. Limitations
- Single model evaluation.
- Subset-based analysis.
- No threshold tuning or retraining.



## 8. Future Work
- Compare calibration across multiple models.
- Apply temperature scaling and post-hoc calibration methods.
- Extend analysis to multi-label toxicity categories.



## 9. Conclusion
This project demonstrates that confidence calibration provides
important insights into NLP model reliability beyond accuracy-based
evaluation.

It highlights the need for confidence-aware assessment in deploying
language models responsibly.

## 10. Dataset Access

The dataset used in this project exceeds GitHub’s file size limits and
is therefore not included in the repository.

To reproduce the experiments, download the dataset from the official
source:

Jigsaw Toxic Comment Classification Dataset:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

Place the downloaded `train.csv` file inside the `data/` directory.



