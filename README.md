# llm-prompt-reliability
Evaluating LLM response stability under prompt variations

# LLM Prompt Reliability Evaluation

A small independent Python project investigating how reliably Claude Haiku 4.5 classifies customer-service messages when its instructions are reworded.

## Research question

How sensitive are LLM classifications to semantically equivalent changes in prompt wording?

## Methodology

I constructed a benchmark of 22 fictional customer-service messages, labelled as:

- **REFUND:** The customer requests their money back.
- **REPLACEMENT:** The customer requests another product.
- **OTHER:** Neither request is made.

The classification rules required the model not to infer requests from dissatisfaction alone.

Using Python and the Anthropic API, I evaluated Claude Haiku 4.5 on each message under three differently worded prompts that preserved the same decision rules.

I recorded 66 classifications and measured accuracy against the predefined labels and agreement across prompt variants.

I then investigated two failure cases through 30 additional API calls, repeating each message five times under each prompt variant to examine run-to-run variability.

Finally, I tested an exploratory intervention using a more explicit decision procedure, generating 10 further classifications.

## Results

| Metric                   | Result                 |
| ------------------------ | ---------------------- |
| Benchmark size           | 22 messages            |
| Initial API evaluations  | 66                     |
| Accuracy — Prompt A      | 90.9%                  |
| Accuracy — Prompt B      | 90.9%                  |
| Accuracy — Prompt C      | 95.5%                  |
| Cross-prompt agreement   | 21/22 messages (95.5%) |
| Repeated evaluations     | 30                     |
| Intervention evaluations | 10                     |

Two messages were particularly informative:

**Message 14:** "The replacement you sent is also broken."

The correct label is OTHER. The text does not make an explicit request for a replacement or refund. The model repeatedly incorrectly predicted REFUND or REPLACEMENT. Repeated identical prompts also produced different classifications.

**Message 21:** "This item was recalled by the manufacturer."

Prompts A and B consistently produced REFUND in the repeated experiment, whereas prompt C produced OTHER in four of five repetitions.

These findings suggest that both prompt formulation and run-to-run variability may affect classification behaviour.

### Exploratory intervention

I introduced a decision procedure explicitly distinguishing requests for remedies from descriptions of product problems.

Across the two selected failure cases:

- Original prompts: 4/30 correct classifications (13.3%).
- Intervention: 7/10 correct classifications (70%).

This improvement is descriptive and exploratory, not an independently validated estimate of the intervention's effectiveness.

## Limitations

- The benchmark contains only 22 manually constructed examples.
- Ground-truth labels were assigned by one person.
- Only one model was evaluated.
- The initial experiment used one API call per prompt-message combination.
- The model's generation temperature was not explicitly controlled.
- The intervention was designed after inspecting failures and tested on those same examples.
- The experiment does not establish general LLM reliability or safety.

## Technical implementation

The project uses Python, Pandas and the Anthropic API.

- `data/benchmark.csv` — Benchmark messages and labels.
- `evaluate.py` — Initial prompt-variation evaluation.
- `repeat_test.py` — Repeated evaluations of selected failures.
- `intervention.py` — Exploratory decision-rule intervention.
- `analysis/analysis.py` — Accuracy and stability calculations.
- `results/` — Saved model predictions.

## Reproducing the experiment

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file containing:

```text
ANTHROPIC_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git.

Run the experiments:

```bash
python evaluate.py
python analysis/analysis.py
python repeat_test.py
python intervention.py
```

Running these scripts makes paid Anthropic API calls. Each experimental script writes its results to a CSV file.

## What I learned

This project provided practical experience with API authentication, programmatic LLM requests, structured data collection, benchmark construction, quantitative evaluation and analysis of model failures.

A methodological lesson was that classification accuracy and stability are distinct: a model can consistently produce incorrect answers, and differing outputs across prompt variants cannot automatically be attributed to prompt wording without examining run-to-run variability.

The project also demonstrated the importance of defining ground truth before evaluation and avoiding overconfident conclusions from small, exploratory experiments.
