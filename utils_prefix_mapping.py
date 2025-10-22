from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

PAD_ID = 0  # xdict'te [PAD] bu olmalı

def _tokenize_prefix(pfx: str, xdict: Dict[str, int]) -> List[int]:
    return [xdict[s] for s in pfx.split()]

def _pad_post(ids: List[int], maxlen: int) -> List[int]:
    ids = ids[:maxlen]
    return ids + [PAD_ID] * (maxlen - len(ids))

def _strip_trailing_pad(seq: List[int]) -> Tuple[int, ...]:
    # sonda ekli PAD(0)’ları kırpıp hashlenebilir tuple döndür
    out = list(seq)
    while out and out[-1] == PAD_ID:
        out.pop()
    return tuple(out)

def _detect_case_col(df: pd.DataFrame) -> str:
    if "caseid" in df.columns:
        return "caseid"
    if "case_id" in df.columns:
        return "case_id"
    # yoksa kullanıcı isterse burayı parametreleştirir
    raise KeyError("Dataset’te case id kolonu bulunamadı (caseid/case_id).")

def build_df_map(
    use_df: pd.DataFrame,
    xdict: Dict[str, int],
    ydict: Dict[str, int],
    maxlen: int,
    case_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    DF tarafı için eşleştirme anahtarları üret:
    key = (prefix tokenlarının PAD’siz hali), yid = next_act id.
    Aynı (key,yid) tekrarı için 'dup' sırası atanır.
    """
    if case_col is None:
        case_col = _detect_case_col(use_df)

    prefixes = use_df["prefix"].values
    next_acts = use_df["next_act"].values

    df_tokens = [_pad_post(_tokenize_prefix(p, xdict), maxlen) for p in prefixes]
    df_keys = [_strip_trailing_pad(t) for t in df_tokens]
    df_yids = [ydict[a] for a in next_acts]

    df_map = pd.DataFrame({
        "key": df_keys,
        "yid": df_yids,
        "orig_index": use_df.index,
        "case_id": use_df[case_col].values,
    })
    # Aynı (key,yid) birden fazla kez varsa sıra ver
    df_map["dup"] = df_map.groupby(["key", "yid"]).cumcount()
    return df_map

def build_xy_map(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Model tarafı için eşleştirme anahtarları üret:
    key = (PAD’siz X satırı), yid = y satırı. 'dup' ile tekrarlı kayıt ayrıştırılır.
    """
    X_keys = [_strip_trailing_pad(list(seq)) for seq in X]
    xy_map = pd.DataFrame({
        "row": np.arange(len(X)),
        "key": X_keys,
        "yid": list(y),
    })
    xy_map["dup"] = xy_map.groupby(["key", "yid"]).cumcount()
    return xy_map

def match_shuffled_to_df(
    X: np.ndarray,
    y: np.ndarray,
    df_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    (key,yid,dup) ile merge edip her X[row] için doğru DF index + case_id verir.
    Dönüş: columns = [row, key, yid, dup, orig_index, case_id]
    """
    xy_map = build_xy_map(X, y)
    matched = xy_map.merge(df_map, on=["key", "yid", "dup"], how="left", validate="one_to_one")

    # Basit kontroller
    missing = matched["orig_index"].isna().sum()
    if missing > 0:
        # Eşleşemeyen kayıt varsa uyarı at
        raise ValueError(f"Eşleşemeyen {missing} satır var. "
                         f"Padding/truncation veya sözlükler uyuşmuyor olabilir.")

    return matched[["row", "orig_index", "case_id", "key", "yid", "dup"]]

def decode_tokens(seq: List[int], inv_xdict: Dict[int, str]) -> List[str]:
    """PAD(0) hariç token id’lerini activity adına çevirir."""
    return [inv_xdict[t] for t in seq if t != PAD_ID]
