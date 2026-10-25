import pandas as pd
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

df = pd.read_csv("data/benchmark.csv")
test_cases = df[df["id"].isin([14, 21])]

INTERVENTION = """
Classify the customer message as REFUND, REPLACEMENT, or OTHER.

Follow this decision procedure:

1. Does the customer actually request money back?
   If yes, classify as REFUND.

2. Otherwise, does the customer actually request another product?
   If yes, classify as REPLACEMENT.

3. Otherwise, classify as OTHER.

A description of a defective, damaged, or recalled product
is NOT itself a request for a refund or replacement.

Mentioning the words "refund" or "replacement" does not
necessarily mean the customer is requesting either.

Classify only what the customer explicity requests, not what you
think they might want.

Return only the category label.

"""

results = []

for index, row in test_cases.iterrows():
    for repetition in range(1, 6):

        prompt = INTERVENTION + "\n\nCustomer message: " + row["text"]

        response = client.messages.create(model="claude-haiku-4-5",max_tokens=20,messages=[{"role": "user", "content": prompt}])

        prediction = response.content[0].text.strip()

        results.append({
            "id": row["id"],
            "repetition": repetition,
            "correct_label": row["label"],
            "prediction": prediction
        })

        print(f"Message {row['id']}, Repeat {repetition}: {prediction}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/intervention.csv", index=False)

print("Saved", len(results_df), "predictions.")
