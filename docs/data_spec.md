# Training Data Contract

## Training unit

A training unit is an `example_id` tied to a `problem_id` and one
`reasoning_style` variant.

Recommended variants:

- `full_reasoning`
- `brief_reasoning`
- `answer_only`
- `candidate_selection`
- `self_correction`

## Required fields

Each JSONL example should include:

- `example_id`
- `problem_id`
- `variant_id`
- `task_family`
- `messages`
- `target`
- `quality`
- `difficulty`
- `provenance`

## Manifest responsibilities

The manifest stores:

- file location and line number
- token counts
- quality and verifier outcomes
- dedup cluster ids
- stage selection flags
- mixture weight

## Split rules

- split by `problem_id`, never by `example_id`
- keep all variants of the same problem in the same split
- do not mix benchmark evaluation examples into training

