from pathlib import Path
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py

# --- Statik yollar ---
INPUT_XES = Path("/Users/Q671967/PycharmProjects/Thesis/prepareDataset/BPI_Challenge_2012.xes.gz")
BASE_OUT = Path("/Users/Q671967/PycharmProjects/Thesis/datasets")
DIR_ALL = BASE_OUT / "BPIC2012-ALL"
DIR_A   = BASE_OUT / "BPIC2012-A"
DIR_O   = BASE_OUT / "BPIC2012-O"
DIR_W   = BASE_OUT / "BPIC2012-W"

def load_bpic12_complete(xes_path: Path) -> pd.DataFrame:
    if not xes_path.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {xes_path}")
    log = xes_importer.apply(str(xes_path))
    df = pm4py.convert_to_dataframe(log)

    # yalnızca 'complete' olaylar (W akımında start/schedule/complete olabilir)
    if "lifecycle:transition" in df.columns:
        m = df["lifecycle:transition"].astype(str).str.lower().eq("complete")
        if m.any():
            df = df.loc[m].copy()

    # kolonları standardize et
    df = df[["case:concept:name", "concept:name", "time:timestamp"]].rename(
        columns={
            "case:concept:name": "Case ID",
            "concept:name": "Activity",
            "time:timestamp": "Complete Timestamp",
        }
    )

    # akım etiketi (A, O, W)
    df["Stream"] = df["Activity"].str.slice(0, 1)

    # vaka içinde sırala
    df = df.sort_values(["Case ID", "Complete Timestamp"]).reset_index(drop=True)
    return df

def write_csvs(df: pd.DataFrame):
    # klasörleri oluştur
    for d in (BASE_OUT, DIR_ALL, DIR_A, DIR_O, DIR_W):
        d.mkdir(parents=True, exist_ok=True)

    # birleşik
    (DIR_ALL / "events.csv").write_text(
        df.to_csv(index=False), encoding="utf-8"
    )  # to_csv string döndürsün diye küçük numara

    # akım bazlı
    for stream, outdir in [("A", DIR_A), ("O", DIR_O), ("W", DIR_W)]:
        sub = df[df["Stream"] == stream].drop(columns=["Stream"])
        (outdir / "events.csv").write_text(
            sub.to_csv(index=False), encoding="utf-8"
        )

    # ilişki özeti kök dizine
    case_stats = (
        df.groupby(["Case ID", "Stream"]).size().unstack(fill_value=0)[["A", "O", "W"]]
        .rename(columns={"A": "n_A", "O": "n_O", "W": "n_W"})
        .reset_index()
    )
    for c in ["A", "O", "W"]:
        case_stats[f"has_{c}"] = (case_stats[f"n_{c}"] > 0).astype(int)
    case_stats.to_csv(BASE_OUT / "BPIC2012_case_streams.csv", index=False)

def main():
    df = load_bpic12_complete(INPUT_XES)
    print(f"[info] events={len(df)} cases={df['Case ID'].nunique()} acts={df['Activity'].nunique()}")
    write_csvs(df)
    print("[done] Yazım tamam. Çıktılar:")
    print(f"  {DIR_ALL / 'events.csv'}")
    print(f"  {DIR_A   / 'events.csv'}")
    print(f"  {DIR_O   / 'events.csv'}")
    print(f"  {DIR_W   / 'events.csv'}")
    print(f"  {BASE_OUT / 'BPIC2012_case_streams.csv'}")

if __name__ == "__main__":
    main()
