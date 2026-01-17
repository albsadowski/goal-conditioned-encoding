# goal-conditioned-encoding

Experimental code for "How Should Multi-Goal Systems Encode Experience?" - a study testing whether parallel goal-dedicated encoding outperforms unified encoding for multi-goal memory systems.

## Overview

This repository contains the code used to run experiments comparing three encoding strategies for multi-goal memory systems:

1. **Unified**: A single encoder stores observations without goal-specific interpretation. Interpretation is deferred to retrieval.
2. **Unified-Interpretive**: A single encoder attempts to capture all goal-perspectives (financial, risk, relationship, competitive) simultaneously in one pass.
3. **Parallel + Arbiter**: Four separate goal-dedicated encoders each encode observations according to their specific goals. An arbiter synthesizes across encodings at retrieval time.

The experiments use LLM-based agents as a model system to test predictions from encoding specificity research (Tulving & Thomson, 1973) and transfer-appropriate processing (Morris et al., 1977) in multi-goal contexts.

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

### API Keys

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
FIREWORKS_API_KEY=your-fireworks-key
```

## Usage

### Running Experiments

Run with a specific model:

```bash
uv run python run.py --scenario scenarios/helix.yaml --model claude-haiku --output results/
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--scenario` | Path to scenario YAML file | required |
| `--model` | LLM for encoding/retrieval (`gpt-5-nano`, `gpt-5-mini`, `claude-haiku`, `claude-sonnet`, `minimax-m2.1`) | `gpt-5-nano` |
| `--judge-model` | LLM for evaluation | `gpt-5-mini` |
| `--buffer-dir` | Directory for perspective memory files | `.buffers` |
| `--output` | Output directory for results CSV | none |
| `--token-limit` | Token budget per perspective | `500` |
| `--n` | Number of runs | `1` |
| `--append` | Append to existing buffers instead of clearing | false |
| `--no-cache` | Disable LLM response caching | false |

### Analyzing Results

Generate statistics from experimental results:

```bash
uv run python stats.py results/
```

## Project Structure

```
goal-conditioned-encoding/
├── src/goal_conditioned_encoding/
│   ├── cli.py          # Main experiment logic
│   └── llm.py          # LLM configuration
├── scenarios/
│   ├── meridian.yaml   # 5 observations, 4 queries
│   └── helix.yaml      # 15 observations, 6 queries
├── results/            # Experiment output CSVs
├── stats.py            # Statistical analysis
├── run.py              # Entry point
└── pyproject.toml
```

## Scenarios

Scenarios are YAML files containing observations and queries:

```yaml
observations:
  - "Client requested 15% discount citing competitor pricing..."
  - "Legal flagged compliance issue in proposed terms..."

queries:
  - text: "What is the financial exposure from recent negotiations?"
    aligned_goal: financial
  - text: "How has the relationship evolved?"
    aligned_goal: relationship
```

## Goal-Perspectives

The system encodes observations through four goal-perspectives:

| Perspective | Focus |
|-------------|-------|
| **Financial** | Margins, costs, revenue, budget variance, P&L impact |
| **Risk** | Liability, precedent, compliance, exposure, mitigation |
| **Relationship** | Trust, commitment, reciprocity, partnership health |
| **Competitive** | Market signals, alternatives, leverage shifts, positioning |

Plus two unified conditions:
- **Unified**: Factual recording without interpretation
- **Unified-Interpretive**: Single-pass multi-perspective encoding

## Evaluation

Responses are evaluated by an LLM judge on four dimensions (1-5 scale):
- **Relevance**: Does it answer what was asked?
- **Completeness**: Are important details included?
- **Accuracy**: Is it correct given the context?
- **Clarity**: Is it clearly expressed?

The composite score is the mean across dimensions.

## License

MIT
