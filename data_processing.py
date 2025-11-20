import os
import argparse
import time

from processtransformer import constants
from processtransformer.data.processor import LogsDataProcessor

# --- add: lightweight CSV normalizer (no external deps) ---
import pandas as pd
from pathlib import Path

TARGET_TS_FMT = "%Y-%d-%m %H:%M:%S.%f"  # processor'ın beklediği format (Y-day-month)

def normalize_event_csv(in_path: str,
                        out_path: str = None,
                        ts_col_candidates=("Complete Timestamp", "time:timestamp", "Timestamp", "CompleteTimestamp"),
                        case_col_candidates=("Case ID", "case:concept:name", "case_id", "CaseID"),
                        act_col_candidates=("Activity", "concept:name", "Activity name", "activity"),
                        ):
    """
    - Sütunları beklenen isimlere yeniden adlandırır: Case ID, Activity, Complete Timestamp
    - Timestamp'i esnekçe parse eder ve processor'ın katı formatına çevirir.
    - Virgül/; gibi ayırıcıları otomatik dener.
    """
    in_p = Path(in_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Input file not found: {in_p}")
    if out_path is None:
        out_p = in_p.with_name(in_p.stem + "_normalized.csv")
    else:
        out_p = Path(out_path)

    # 1) CSV'yi oku (ayırıcıyı otomatik dene)
    tried = []
    for sep in (None, ',', ';', '\t'):
        try:
            df = pd.read_csv(in_p, sep=sep)
            tried.append(sep or 'auto')
            if len(df.columns) >= 3:
                break
        except Exception:
            continue
    if 'df' not in locals():
        raise RuntimeError(f"Couldn't read CSV with common separators. Tried: {tried}")

    # 2) Sütunları eşleştir
    def pick(colnames, candidates):
        for c in candidates:
            if c in colnames:
                return c
        return None

    cols = list(df.columns)
    case_col = pick(cols, case_col_candidates)
    act_col  = pick(cols, act_col_candidates)
    ts_col   = pick(cols, ts_col_candidates)

    if not case_col or not act_col or not ts_col:
        raise ValueError(f"Missing required columns. Found: {cols}\n"
                         f"Need something like Case ID / Activity / Complete Timestamp")

    rename_map = {
        case_col: "Case ID",
        act_col:  "Activity",
        ts_col:   "Complete Timestamp"
    }
    df = df.rename(columns=rename_map)

    # 3) Timestamp'i esnekçe parse et → UTC → timezone'suz → hedef formata çevir
    ts = pd.to_datetime(df["Complete Timestamp"], errors="coerce", utc=True)
    bad = ts.isna().sum()
    if bad:
        # istersen drop etmeden önce uyarı basıyoruz
        print(f"[normalize] Warning: {bad} unparsable timestamp(s) will be dropped.")
    keep = ~ts.isna()
    df = df.loc[keep].copy()
    ts = ts.loc[keep].dt.tz_convert("UTC").dt.tz_localize(None)
    df["Complete Timestamp"] = ts.dt.strftime(TARGET_TS_FMT)

    # 4) Kaydet ve yolu döndür
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"[normalize] Saved normalized CSV -> {out_p}")
    return str(out_p)
# --- end add ---

parser = argparse.ArgumentParser(
    description="Process Transformer - Data Processing.")

parser.add_argument("--dataset", 
    type=str, 
    default="BPIC2012-ALL",
    help="dataset name")

parser.add_argument("--dir_path", 
    type=str, 
    default="./datasets", 
    help="path to store processed data")

parser.add_argument("--raw_log_file", 
    type=str, 
    default="./datasets/BPIC2012-ALL/events.csv",
    help="path to raw csv log file")

parser.add_argument("--task", 
    type=constants.Task, 
    default=constants.Task.NEXT_ACTIVITY,
    help="task name")

parser.add_argument("--sort_temporally", 
    type=bool, 
    default=False, 
    help="sort cases by timestamp")

args = parser.parse_args()

if __name__ == "__main__": 
    start = time.time()

    # 1) CSV'yi normalize et (sütun adları + timestamp formatı)
    norm_file = normalize_event_csv(args.raw_log_file)
    use_file = norm_file

    # 2) Processor'ı çalıştır
    data_processor = LogsDataProcessor(
        name=args.dataset,
        filepath=use_file,
        columns=["Case ID", "Activity", "Complete Timestamp"],
        dir_path=args.dir_path,
        pool=1
    )
    data_processor.process_logs(task=args.task, sort_temporally= args.sort_temporally)
    end = time.time()
    print(f"Total processing time: {end - start}")

