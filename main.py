import subprocess
import sys
from typing import Union
from pathlib import Path

from explain_with_llm import explain_prefix


def get_longest_prefix_index_from_txt(path: Union[str, Path]) -> int:
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    prefixes = [ln for ln in lines if ln]  # drop empty lines

    if not prefixes:
        raise ValueError(f"No prefixes found in {path}")

    lengths = [len(p.split()) for p in prefixes]
    return max(range(len(lengths)), key=lengths.__getitem__)


def get_nth_longest_prefix_index_from_txt(
    path: Union[str, Path],
    n: int = 0
) -> int:
    """
    n=0 -> longest
    n=1 -> second longest
    n=2 -> third longest
    """
    path = Path(path)
    prefixes = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    if len(prefixes) <= n:
        raise ValueError(f"Not enough prefixes in {path} to get rank {n}")

    lengths = [(i, len(p.split())) for i, p in enumerate(prefixes)]
    lengths.sort(key=lambda x: x[1], reverse=True)

    return lengths[n][0]


def load_prefix_by_index_from_txt(path: Union[str, Path], idx: int) -> list[str]:
    """Convenience: returns the tokenized prefix at idx."""
    path = Path(path)
    prefixes = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return prefixes[idx].split()


if __name__ == '__main__':

    DATASET = "BPIC2012-O"
    OUTPUTS_DIR = Path("outputs")

    PY = sys.executable

    # subprocess.run(
    #     [PY, "next_activity.py",
    #      "--dataset", DATASET,
    #      "--epochs", str(15)],
    #     check=True
    # )
    # -----------------------------
    #  run get_attention_hooked
    # -----------------------------
    # subprocess.run(
    #     [PY, "get_attention_hooked.py", "--dataset", DATASET],
    #     check=True
    # )

    batch_file = OUTPUTS_DIR / DATASET / "batch_prefixes.txt"
    idx = get_longest_prefix_index_from_txt(batch_file)

    # second longest prefix
    idx_second = get_nth_longest_prefix_index_from_txt(batch_file, n=1)

    for i in [idx, idx_second]:

        subprocess.run(
            [PY, "visualizeAttention/visualization.py",
             "--dataset", DATASET,
             "--prefix_index", str(i)],
            check=True
        )

        subprocess.run(
            [PY, "shapley_ppm_deletion_only_shap_pkg.py",
             "--dataset", DATASET,
             "--prefix_index", str(i)],
            check=True
        )
    #
        subprocess.run(
            [PY, "shapley_ppm_pad_replace_shap_pkg.py",
             "--dataset", DATASET,
             "--prefix_index", str(i)],
            check=True
        )

    #-------------------------
    # VERIFICATION


    for i in [idx, idx_second]:
        # subprocess.run(
        #         [PY, "shap_quick_compare.py",
        #          "--dataset", DATASET,
        #          "--prefix_index", str(i)],
        #         check=True
        #     )
        # subprocess.run(
        #     [PY, "shap_experiments.py",
        #      "--dataset", DATASET,
        #      "--prefix_index", str(i),
        #      "--experiment", "all",],
        #     check=True
        # )
        subprocess.run(
            [PY, "shap_experiments_sensible.py",
             "--dataset", DATASET,
             "--prefix_index", str(i),
             "--experiment", "all"],
            check=True
        )

    # -------------------------

    # GPT version
    # exp_gpt = explain_prefix(
    #     dataset_name=DATASET,
    #     out_dir="./outputs",
    #     prefix_index=10,
    #     backend="gpt",
    #     llm_model_name=None,  # will use OPENAI_MODEL_NAME or default
    # )
    # print(exp_gpt.get("explanation"))

    # LLMama version
    # exp_llmama = explain_prefix(
    #     dataset_name="BPIC2012-O",
    #     out_dir="./outputs",
    #     prefix_index=10,
    #     backend="llmama",
    #     llm_model_name=None,
    # )
    # print(exp_llmama)
