from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None,
    truncation=True,
    max_length=512
)

def predict(text):
    """
    Returns:
    - predicted_label (0 or 1)
    - confidence (float)
    """
    scores = classifier(text)[0]

   
    toxic_score = 0.0
    non_toxic_score = 0.0

    for s in scores:
        if s["label"].lower().startswith("toxic"):
            toxic_score = s["score"]
        else:
            non_toxic_score = s["score"]

    if toxic_score >= non_toxic_score:
        return 1, toxic_score
    else:
        return 0, non_toxic_score
