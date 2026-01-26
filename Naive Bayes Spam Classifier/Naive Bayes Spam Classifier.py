

import math
from collections import defaultdict

# Expanded dataset: (email text, label) -> 1 = Spam, 0 = Not Spam
emails = [
    ("Win money now", 1),
    ("Limited offer win prize", 1),
    ("Claim your free gift today", 1),
    ("You have won a lottery", 1),
    ("Meeting tomorrow at 10am", 0),
    ("Project deadline reminder", 0),
    ("Lunch with team today", 0),
    ("Schedule for next week's presentation", 0),
    ("Congratulations! You won a prize", 1),
    ("Reminder: submit your assignment", 0),
    ("Exclusive offer just for you", 1),
    ("Family dinner on Saturday", 0)
]

# Tokenizer
def tokenize(text):
    return text.lower().split()

# Count words per class
word_counts = {0: defaultdict(int), 1: defaultdict(int)}
class_counts = defaultdict(int)

for text, label in emails:
    class_counts[label] += 1
    for word in tokenize(text):
        word_counts[label][word] += 1

# Prediction function
def predict(text):
    words = tokenize(text)
    scores = {}

    for cls in [0, 1]:
        score = math.log(class_counts[cls] / sum(class_counts.values()))
        total_words = sum(word_counts[cls].values()) + 1

        for word in words:
            count = word_counts[cls][word] + 1  # Laplace smoothing
            score += math.log(count / total_words)

        scores[cls] = score

    return max(scores, key=scores.get)

# Test examples
test_emails = [
    "win prize now",
    "team meeting tomorrow",
    "claim your free gift",
    "submit your assignment today"
]

for email in test_emails:
    print(f"Email: '{email}' -> Prediction (1=Spam, 0=Not Spam): {predict(email)}")