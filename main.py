import subprocess
import sys
from typing import Union
from pathlib import Path


def get_longest_prefix_index_from_txt(path: Union[str, Path]) -> int:
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    prefixes = [ln for ln in lines if ln]  # drop empty lines

    if not prefixes:
        raise ValueError(f"No prefixes found in {path}")

    lengths = [len(p.split()) for p in prefixes]
    return max(range(len(lengths)), key=lengths.__getitem__)


def load_prefix_by_index_from_txt(path: Union[str, Path], idx: int) -> list[str]:
    """Convenience: returns the tokenized prefix at idx."""
    path = Path(path)
    prefixes = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return prefixes[idx].split()


if __name__ == '__main__':

    DATASET = "BPIC2012-O"
    OUTPUTS_DIR = Path("outputs")
    batch_file = OUTPUTS_DIR / DATASET / "batch_prefixes.txt"

    PY = sys.executable  # uses the same venv python running main.py

    # subprocess.run(
    #     [PY, "next_activity.py",
    #      "--dataset", DATASET,
    #      "--epochs", 20],
    #     check=True
    # )
    # -----------------------------
    #  run get_attention_hooked
    # -----------------------------
    subprocess.run(
        [PY, "get_attention_hooked.py", "--dataset", DATASET],
        check=True
    )

    idx = get_longest_prefix_index_from_txt(batch_file)

    subprocess.run(
        [PY, "visualizeAttention/visualization.py",
         "--dataset", DATASET,
         "--prefix_index", str(idx)],
        check=True
    )

    subprocess.run(
        [PY, "shapley_ppm_deletion_only_shap_pkg.py",
         "--dataset", DATASET,
         "--prefix_index", str(idx)],
        check=True
    )

    # # GPT version
    # exp_gpt = explain_prefix(
    #     dataset_name="BPIC2012-O",
    #     out_dir="./outputs",
    #     prefix_index=10,
    #     backend="gpt",
    #     llm_model_name=None,  # will use OPENAI_MODEL_NAME or default
    # )
    # print(exp_gpt)

    # LLMama version
    # exp_llmama = explain_prefix(
    #     dataset_name="BPIC2012-O",
    #     out_dir="./outputs",
    #     prefix_index=10,
    #     backend="llmama",
    #     llm_model_name=None,
    # )
    # print(exp_llmama)
