import pandas as pd
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

PROMPTS = {
    "A": """
Classify the customer message as REFUND, REPLACEMENT, or OTHER.

REFUND: The customer requests their money back.
REPLACEMENT: The customer requests another product.
OTHER: Neither is requested.

Requests may use different wording. Do not infer a request
from dissatisfaction alone. If both are requested, choose REFUND.

Return only the category label.

""",

    "B": """
Determine which of these three categories applies to the customer message:

REFUND: The customer requests their money back.
REPLACEMENT: The customer requests another product.
OTHER: Neither is requested.

Requests may use different wording. Do not infer a request
from dissatisfaction alone. If both are requested, choose REFUND.

Respond with only the category label.

""",

    "C": """
Read the customer message and assign exactly one label:
REFUND, REPLACEMENT, or OTHER.

REFUND: The customer requests their money back.
REPLACEMENT: The customer requests another product.
OTHER: Neither is requested.

Requests may use different wording. Do not infer a request
from dissatisfaction alone. If both are requested, choose REFUND.

Output only the category label.

"""
}



df = pd.read_csv("data/benchmark.csv")

# Select only the two messages we want to investigate further
test_cases = df[df["id"].isin([14, 21])]

results = []

for variant, instructions in PROMPTS.items():
    for index, row in test_cases.iterrows():
        for repetition in range(1, 6):

            prompt = instructions + "\n\nCustomer message: " + row["text"]

            response = client.messages.create(model="claude-haiku-4-5",max_tokens=20,messages=[{"role": "user", "content": prompt}])

            prediction = response.content[0].text.strip()

            results.append({"id": row["id"],"variant": variant,"repetition": repetition,"correct_label": row["label"],"prediction": prediction})

            print(f"Message {row['id']}, Variant {variant}, "
                  f"Repeat {repetition}: {prediction}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/repeat_test.csv", index=False)

print("Saved", len(results_df), "predictions.")
