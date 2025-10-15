import pandas as pd

# Example data
input_sentences = [
    "How are you?",
    "I love coding.",
    "What is your name?",
    "Good morning!"
]

# Model predictions (replace with your model’s predicted outputs)
predicted_outputs = [
    "तुम कैसे हो?",
    "मुझे कोडिंग पसंद है।",
    "तुम्हारा नाम क्या है?",
    "सुप्रभात!"
]

# Ground-truth / correct translations
expected_outputs = [
    "तुम कैसे हो?",
    "मुझे कोडिंग पसंद है।",
    "तुम्हारा नाम क्या है?",
    "सुप्रभात!"
]

# Evaluate correctness
results = []
for inp, pred, exp in zip(input_sentences, predicted_outputs, expected_outputs):
    correct = "Y" if pred.strip() == exp.strip() else "N"
    results.append([inp, pred, correct])

# Create a table
df = pd.DataFrame(results, columns=["Input Sentence (English)", "Predicted Output (Hindi)", "Correct (Y/N)"])

# Display
print(df.to_string(index=False))

# Optionally save to CSV
df.to_csv("translation_evaluation.csv", index=False, encoding="utf-8-sig")
print("\n✅ Results saved to 'translation_evaluation.csv'")
<img width="629" height="631" alt="Screenshot 2025-10-15 101933" src="https://github.com/user-attachments/assets/eee6719d-4503-4f46-b424-d4244c70b1dc" />
