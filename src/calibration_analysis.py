import pandas as pd
import numpy as np
from model import predict


def expected_calibration_error(confidences, correct, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    confidences = np.array(confidences)
    correct = np.array(correct)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if mask.sum() == 0:
            continue

        bin_confidence = confidences[mask].mean()
        bin_accuracy = correct[mask].mean()
        ece += abs(bin_confidence - bin_accuracy) * (mask.sum() / len(confidences))

    return round(ece, 4)


def run_calibration_analysis(csv_path, sample_size=1000):
    df = pd.read_csv(csv_path)

  
    df = df.sample(sample_size, random_state=42)

    confidences = []
    correct = []

    for _, row in df.iterrows():
        text = str(row["comment_text"])

   
        if len(text.split()) > 300:
            continue

        pred_label, confidence = predict(text)
        true_label = int(row["toxic"])

        confidences.append(confidence)
        correct.append(1 if pred_label == true_label else 0)

    confidences = np.array(confidences)
    correct = np.array(correct)

    accuracy = correct.mean()
    ece = expected_calibration_error(confidences, correct)

    overconfident = ((correct == 0) & (confidences > 0.8)).mean()

    return {
        "samples_used": len(confidences),
        "accuracy": round(float(accuracy), 4),
        "ece": ece,
        "overconfidence_rate": round(float(overconfident), 4)
    }
