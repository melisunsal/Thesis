# Thesis: Explainability in Transformer-based Predictive Process Mining

This repository contains the implementation for a Master's thesis on **explainability of Transformer-based predictive process mining models**. The project focuses on **next activity prediction** and compares **attention-based explanations with Shapley-based (deletion-only) explanations**, enhanced by **Large Language Model (LLM)** natural language generation.

## Project Overview

This thesis develops an LLM-enhanced explainability framework tailored for Transformer-based Predictive Process Monitoring (PPM). The research objectives include:

- Training a **ProcessTransformer** model for next activity prediction
- Extracting and visualizing **multi-head self-attention weights**
- Computing **SHAP values** using deletion-only perturbations
- Comparing explanation methods in terms of faithfulness and stability
- Integrating **LLMs (GPT-4)** to generate human-readable, process-aware explanations

## Features

- **Transformer-based Prediction**: ProcessTransformer architecture for next activity prediction
- **Attention Visualization**: Multi-head self-attention weight extraction and visualization
- **SHAP Explanations**: Model-agnostic feature importance using Shapley values
- **LLM Integration**: OpenAI GPT-4 powered natural language explanation generation
- **Multiple Datasets**: Support for Helpdesk, BPIC2012, and BPIC2013 event logs

## Repository Structure

```
Thesis/
├── datasets/              # Event log datasets (Helpdesk, BPIC2012, etc.)
├── models/                # Trained ProcessTransformer models
├── explainers/            # Explanation module implementations
├── outputs/               # Generated outputs and results
├── prepareDataset/        # Dataset preparation utilities
├── results/               # Experiment results and metrics
├── visualizeAttention/    # Attention visualization tools
├── data_processing.py     # Data preprocessing pipeline
├── main.py                # Main entry point
├── next_activity.py       # Next activity prediction module
├── get_attention_hooked.py # Attention weight extraction
├── llm_integration.py     # OpenAI API integration
├── explain_with_llm.py    # LLM-based explanation generation
├── shapley_ppm_integration.py           # SHAP integration for PPM
├── shapley_ppm_deletion_only_shap_pkg.py # Deletion-only SHAP
├── shapley_ppm_pad_replace_shap_pkg.py  # Pad-replace SHAP variant
├── shap_experiments_sensible.py         # SHAP experiments
├── shap_quick_compare.py                # Quick SHAP comparison
├── utils_prefix_mapping.py              # Utility functions
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Datasets

The framework supports the following publicly available event logs:

| Dataset | Description | Use Case |
|---------|-------------|----------|
| **Helpdesk** | IT service management tickets | Initial development and testing |
| **BPIC2012** | Loan application process | Scalability validation |

Datasets are located in the `datasets/` directory.

## Prerequisites

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

- **Python**: 3.8 or higher
- **OpenAI API Key**: Required for LLM-based explanation generation

---

## How to Run the Project

Follow these steps to set up and run the explanation generation pipeline.

### Step 1: Clone the Repository

```bash
git clone https://github.com/melisunsal/Thesis.git
cd Thesis
```
- **Git**: Required to clone the repository and install some dependencies
  - Download from [git-scm.com](https://git-scm.com/downloads)
- **OpenAI API Key**: Required for LLM-based explanation generation (Step 5)
- **System packages** (Linux only, for building scipy from source): `gfortran`, `libopenblas-dev`
 
> **Note**: Throughout this guide, if `python` doesn't work on your system, try `python3` instead. Similarly, use `pip3` if `pip` doesn't work.
### Step 2: Create and Activate Virtual Environment

**On Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set the OpenAI API Key

**On Linux/macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**On Windows (Command Prompt):**
```bash
set OPENAI_API_KEY=your-api-key-here
```

**On Windows (PowerShell):**
```bash
$env:OPENAI_API_KEY="your-api-key-here"
```

> Replace `your-api-key-here` with the actual API key.

### Step 5: Run the Program

**Run with default settings (longest prefix):**

```bash
python main.py
```

This will execute the full pipeline including visualization, SHAP computation, and LLM-based explanation generation for the longest prefix in the batch.
 
#### Command-Line Options
 
| Option | Description |
|--------|-------------|
| `--dataset` | Dataset name (default: `BPIC2012-O`) |
| `--prefix_index` | Specific prefix index to explain (default: longest prefix) |
| `--generate_batch` | Generate new batch prefixes before running |

 
> **Note**: The repository includes a pre-generated batch with good prefix index for testing. When generating a new batch with `--generate_batch`, the resulting prefixes may be shorter and less suitable for demonstrating the framework. You may need to run `--generate_batch` multiple times to get prefixes with sufficient length for meaningful explanations.

---

## Reproducibility

All experiments were executed in a Python virtual environment. Exact package versions are provided in `requirements.txt`.

## License

This project is for academic and research purposes.

---

**Keywords**: Predictive Process Monitoring, Transformer, Attention Mechanism, SHAP, Explainability, LLM, GPT-4, Process Mining, Next Activity Prediction