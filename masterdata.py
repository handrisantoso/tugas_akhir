import argparse
import ast
import json
import os
import re
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUALITY_SEVERITY = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": 3}

KEEP_COLUMNS = [
    "item_id", "item_name", "item_status", "category_id", "category_name",
    "description",
    "sale", "views", "like", "rating_star", "comment_count",
    "price_min", "price_max", "variant_count", "total_stock", "in_stock",
    "variants_detail",
    "review_count", "followup_review_count", "avg_rating_reviews",
    "positive_review_count", "negative_review_count",
    "quality_level", "quality_tasks",
    "category_mismatch", "violation_status",
]

NUMERIC_COLUMNS = [
    "sale", "views", "like", "rating_star", "comment_count",
    "price_min", "price_max", "variant_count", "total_stock", "in_stock",
    "review_count", "followup_review_count", "avg_rating_reviews",
    "positive_review_count", "negative_review_count",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(data_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} not found — those columns will be empty.")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  [OK]   {filename:<50} {len(df):>6} rows")
    return df


def _load_json(data_dir: str, filename: str) -> list:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_json(val):
    if pd.isna(val) or str(val).strip() in ("", "[]", "{}", "nan"):
        return None
    s = str(val).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def _extract_stock(variant: dict) -> int:
    siv2 = variant.get("stock_info_v2")
    if isinstance(siv2, dict):
        summary = siv2.get("summary_info") or {}
        if "total_available_stock" in summary:
            return int(summary.get("total_available_stock") or 0)
        seller = siv2.get("seller_stock") or []
        return sum(int(s.get("stock") or 0) for s in seller if isinstance(s, dict))
    return 0


def _extract_price(variant: dict) -> float:
    pi = variant.get("price_info") or []
    if isinstance(pi, list) and pi and isinstance(pi[0], dict):
        return float(pi[0].get("current_price") or 0)
    if isinstance(pi, dict):
        return float(pi.get("current_price") or 0)
    return 0.0


def _variant_label(variant: dict) -> str:
    return (variant.get("model_name") or variant.get("model_sku") or "Default").strip() or "Default"


def _norm_cat_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s in ("", "nan", "None", "0"):
        return s if s == "0" else ""
    stripped = s.lstrip("0")
    return stripped if stripped else "0"


# ---------------------------------------------------------------------------
# Shop summary (printed at startup for quick reference)
# ---------------------------------------------------------------------------

def print_shop_summary(data_dir: str) -> None:
    print("\n" + "═" * 60)
    print("  SHOP SUMMARY")
    print("═" * 60)

    # ── Product counts by status (I6) ────────────────────────────────────────
    status_df = _load(data_dir, "bucket_A_item_status_counts.csv")
    if not status_df.empty and "status" in status_df.columns and "count" in status_df.columns:
        print("  Product counts by status:")
        for _, row in status_df.iterrows():
            label = str(row["status"])
            count = str(row["count"])
            print(f"    {label:<12} {count:>5}")
    else:
        print("  Product counts: bucket_A_item_status_counts.csv not found.")
        print("  Re-run data_extraction.py --buckets A to generate it.")

    # ── Shop rating (I2 — all statuses) ──────────────────────────────────────
    rating_df = _load(data_dir, "bucket_A_shop_rating.csv")
    if not rating_df.empty:
        row = rating_df.iloc[0]
        rating  = row.get("computed_rating_star", "—")
        reviews = row.get("computed_total_reviews", "—")
        normal  = row.get("normal_products_reviews", "—")
        try:
            reviews_fmt = f"{int(float(reviews)):,}"
        except (ValueError, TypeError):
            reviews_fmt = str(reviews)
        try:
            normal_fmt = f"{int(float(normal)):,}"
        except (ValueError, TypeError):
            normal_fmt = str(normal)
        print(f"\n  Shop rating : {rating} ⭐")
        print(f"  Total reviews (penilaian toko, all statuses) : {reviews_fmt}")
        print(f"  Total reviews (NORMAL only)                  : {normal_fmt}")
    else:
        print("\n  Shop rating : bucket_A_shop_rating.csv not found.")
        print("  Re-run data_extraction.py --buckets A to generate it.")

    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Per-file processors
# ---------------------------------------------------------------------------

def process_catalog(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = ["item_id", "item_name", "category_id", "item_sku",
            "item_status", "condition", "brand", "description"]
    cols = [c for c in keep if c in df.columns]
    out  = df[cols].copy()
    if "description" in out.columns:
        out["description"] = (
            out["description"].fillna("").str[:300]
            .str.replace(r"\s+", " ", regex=True).str.strip()
        )
    return out


def process_categories(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    lookup: dict[str, str] = {}
    name_col = next(
        (c for c in ["display_category_name", "category_name", "name"] if c in df.columns),
        None,
    )
    if name_col is None:
        print("  [WARN] Categories file has no recognisable name column. "
              "Expected display_category_name.")
        return {}
    for _, row in df.iterrows():
        cat_id = str(row.get("category_id", "")).strip()
        name   = str(row.get(name_col, "")).strip()
        if cat_id and name:
            lookup[cat_id] = name
    print(f"  [Categories] Loaded {len(lookup)} category id → name mappings.")
    return lookup


def process_extra(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in ["item_id", "sale", "views", "likes", "rating_star", "comment_count"]
            if c in df.columns]
    return df[cols].copy()


def process_models(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "item_id", "price_min", "price_max",
            "variant_count", "total_stock", "in_stock_variant_count", "variants_detail",
        ])

    rows = []
    for _, row in df.iterrows():
        item_id = str(row.get("item_id", "")).strip()
        models  = _safe_json(row.get("models"))
        if not isinstance(models, list):
            models = []

        prices  = []
        details = []
        total_stock    = 0
        in_stock_count = 0

        for m in models:
            label = _variant_label(m)
            stock = _extract_stock(m)
            price = _extract_price(m)
            if price > 0:
                prices.append(price)
            total_stock    += stock
            in_stock_count += 1 if stock > 0 else 0
            details.append(f"{label}: Rp {price:,.0f} ({stock} units)")

        rows.append({
            "item_id":               item_id,
            "price_min":             min(prices) if prices else None,
            "price_max":             max(prices) if prices else None,
            "variant_count":         len(models),
            "total_stock":           total_stock,
            "in_stock_variant_count": in_stock_count,
            "variants_detail":       " | ".join(details) if details else "",
        })

    return pd.DataFrame(rows)


def process_comments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "item_id", "review_count", "followup_review_count",
            "avg_rating_reviews",
            "positive_review_count", "negative_review_count",
            "sample_positive_review", "sample_negative_review",
        ])

    df = df.copy()
    df["item_id"]     = df["item_id"].astype(str).str.strip()
    df["rating_star"] = pd.to_numeric(df.get("rating_star", pd.Series(dtype=float)), errors="coerce")
    df["comment"]     = df.get("comment", pd.Series(dtype=str)).fillna("")

    has_fb_flag = "is_fb_cm" in df.columns
    if has_fb_flag:
        df["is_fb_cm"] = pd.to_numeric(df["is_fb_cm"], errors="coerce").fillna(0).astype(int)
        initial_mask  = df["is_fb_cm"] == 0
        followup_mask = df["is_fb_cm"] == 1
    else:
        initial_mask  = pd.Series(True,  index=df.index)
        followup_mask = pd.Series(False, index=df.index)

    rows = []
    for item_id, grp in df.groupby("item_id"):
        initial  = grp[initial_mask.reindex(grp.index, fill_value=True)]
        followup = grp[followup_mask.reindex(grp.index, fill_value=False)]

        ratings = initial["rating_star"].dropna()
        pos     = initial[initial["rating_star"] >= 4]["comment"]
        neg     = initial[initial["rating_star"] <= 2]["comment"]

        rows.append({
            "item_id":                item_id,
            "review_count":           len(initial),
            "followup_review_count":  len(followup),
            "avg_rating_reviews":     round(ratings.mean(), 2) if not ratings.empty else None,
            "positive_review_count":  len(pos),
            "negative_review_count":  len(neg),
            "sample_positive_review": pos.iloc[0][:120].strip() if not pos.empty else "",
            "sample_negative_review": neg.iloc[0][:120].strip() if not neg.empty else "",
        })

    result = pd.DataFrame(rows)
    if has_fb_flag:
        total_followups = result["followup_review_count"].sum()
        print(f"  [Comments] {int(result['review_count'].sum()):,} initial reviews + "
              f"{int(total_followups):,} follow-ups across {len(result)} items.")
    else:
        print("  [Comments] is_fb_cm flag not found — all rows treated as initial reviews. "
              "Re-run data_extraction.py to get the flag (I1 in its header).")
    return result


def process_low_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["item_id", "quality_level", "quality_tasks"])

    out = df.copy()
    out["item_id"] = out["item_id"].astype(str).str.strip()

    label_col = "_quality_level" if "_quality_level" in out.columns else "quality_level"
    task_col  = "unfinished_task" if "unfinished_task" in out.columns else None

    out["_rank"] = out[label_col].map(QUALITY_SEVERITY).fillna(QUALITY_SEVERITY["UNKNOWN"]).astype(int)

    rows = []
    for item_id, grp in out.groupby("item_id"):
        worst_label = grp.loc[grp["_rank"].idxmin(), label_col] if label_col in grp.columns else "UNKNOWN"
        tasks_str = ""
        if task_col and task_col in grp.columns:
            suggestions, seen = [], set()
            for raw in grp[task_col].dropna().unique():
                parsed  = _safe_json(raw)
                entries = parsed if isinstance(parsed, list) else []
                if entries:
                    for entry in entries:
                        s = (entry.get("suggestion", "").strip()
                             if isinstance(entry, dict) else str(entry).strip())
                        if s and s not in seen:
                            seen.add(s); suggestions.append(s)
                elif str(raw).strip() and str(raw).strip() not in seen:
                    seen.add(str(raw).strip()); suggestions.append(str(raw).strip())
            tasks_str = "; ".join(suggestions)
        rows.append({"item_id": item_id, "quality_level": worst_label, "quality_tasks": tasks_str})

    result = pd.DataFrame(rows)
    low_n  = (result["quality_level"] == "LOW").sum()
    med_n  = (result["quality_level"] == "MEDIUM").sum()
    print(f"  [LQ]  {len(out):,} raw rows → {len(result)} unique items  (LOW: {low_n}, MEDIUM: {med_n})")
    return result


def process_violations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["item_id", "deboost", "violation_status"])
    out = df.copy()
    out["item_id"]          = out["item_id"].astype(str).str.strip()
    out["deboost"]          = out.get("deboost", pd.Series(dtype=str)).fillna("")
    out["violation_status"] = out.get("item_status", pd.Series(dtype=str)).fillna("")
    return out[["item_id", "deboost", "violation_status"]]


def process_category_rec(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "item_id", "current_category_id",
            "top_recommended_category", "top_rec_confidence", "category_mismatch",
        ])

    _diag_printed = False
    rows = []
    parse_ok = parse_fail = 0

    for _, row in df.iterrows():
        item_id     = str(row.get("item_id", "")).strip()
        current_cat = str(row.get("current_category_id", "")).strip()
        recs        = _safe_json(row.get("recommended_categories"))
        top_cat = top_conf = ""
        mismatch = False

        if isinstance(recs, list) and recs:
            parse_ok += 1
            top = recs[0]
            if not _diag_printed:
                print(f"  [CatRec] First parsed entry keys: "
                      f"{list(top.keys()) if isinstance(top, dict) else type(top)}")
                _diag_printed = True
            if isinstance(top, dict):
                name_val = (top.get("category_name") or top.get("name")
                            or top.get("display_name") or "")
                id_val   = (top.get("category_id") or top.get("id") or "")
                conf_val = (top.get("confidence") or top.get("score") or "")
                top_cat  = str(name_val or id_val).strip()
                top_conf = str(conf_val).strip()
                rec_norm     = _norm_cat_id(id_val)
                current_norm = _norm_cat_id(current_cat)
                mismatch = bool(rec_norm and current_norm and rec_norm != current_norm)
        else:
            parse_fail += 1
            raw = row.get("recommended_categories")
            if not _diag_printed and raw and str(raw).strip() not in ("", "nan", "[]"):
                print(f"  [CatRec] WARNING: could not parse. Raw (200 chars): {str(raw)[:200]}")
                _diag_printed = True

        rows.append({
            "item_id":                  item_id,
            "current_category_id":      current_cat,
            "top_recommended_category": top_cat,
            "top_rec_confidence":       top_conf,
            "category_mismatch":        mismatch,
        })

    print(f"  [CatRec] Parsed OK: {parse_ok} | Failed/empty: {parse_fail}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def _build(data_dir: str) -> pd.DataFrame | None:
    print(f"\nLoading CSVs from: {data_dir}\n")

    cat      = process_catalog(    _load(data_dir, "bucket_A_catalog.csv"))
    extra    = process_extra(      _load(data_dir, "bucket_A_extra_info.csv"))
    cats_df  = _load(data_dir, "bucket_A_categories.csv")
    cat_lookup = process_categories(cats_df)
    models   = process_models(     _load(data_dir, "bucket_B_models.csv"))
    comments = process_comments(   _load(data_dir, "bucket_C_comments.csv"))
    low_q    = process_low_quality(_load(data_dir, "bucket_C_low_quality_items.csv"))
    viol     = process_violations( _load(data_dir, "bucket_C_violations.csv"))
    cat_rec  = process_category_rec(_load(data_dir, "bucket_C_category_recommend.csv"))

    if cat.empty:
        print("\n[ERROR] bucket_A_catalog.csv is required but missing. Aborting.")
        return None

    cat["item_id"] = cat["item_id"].astype(str).str.strip()

    def ljoin(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        if right.empty:
            return left
        right = right.copy()
        right["item_id"] = right["item_id"].astype(str).str.strip()
        return left.merge(right, on="item_id", how="left")

    master = cat
    master = ljoin(master, extra)
    master = ljoin(master, models)
    master = ljoin(master, comments)

    if not low_q.empty:
        low_q["item_id"] = low_q["item_id"].astype(str).str.strip()
        master = master.merge(low_q, on="item_id", how="left")
        master["quality_level"] = master["quality_level"].fillna("HIGH")
        master["quality_tasks"] = master["quality_tasks"].fillna("")
    else:
        master["quality_level"] = "HIGH"
        master["quality_tasks"] = ""

    master = ljoin(master, viol)
    master = ljoin(master, cat_rec)

    if cat_lookup:
        master["category_name"] = (
            master["category_id"].astype(str).str.strip().map(cat_lookup).fillna("")
        )
        unmapped = (master["category_name"] == "").sum()
        if unmapped:
            print(f"  [WARN] {unmapped} product(s) have a category_id not found in "
                  f"bucket_A_categories. Re-run --buckets A to refresh the category tree.")
    else:
        master["category_name"] = ""
        print("  [WARN] bucket_A_categories.csv missing — category_name column will be empty. "
              "Run data_extraction.py --buckets A to generate it.")

    if "category_id_x" in master.columns:
        master["category_id"] = master["category_id_x"].fillna(master.get("category_id_y", ""))
        master.drop(columns=[c for c in master.columns if re.match(r"category_id_[xy]", c)],
                    inplace=True)

    master = master.rename(columns={
        "likes":                  "like",
        "in_stock_variant_count": "in_stock",
    })

    keep_present = [c for c in KEEP_COLUMNS if c in master.columns]
    missing      = [c for c in KEEP_COLUMNS if c not in master.columns]
    if missing:
        print(f"  [WARN] Columns not found (source file missing?): {missing}")
    master = master[keep_present]

    for col in NUMERIC_COLUMNS:
        if col in master.columns:
            master[col] = pd.to_numeric(master[col], errors="coerce")

    return master


def _default_sort(master: pd.DataFrame) -> pd.DataFrame:
    if "item_id" not in master.columns or master.empty:
        return master
    master = master.copy()
    master["__sort_iid"] = pd.to_numeric(master["item_id"], errors="coerce").fillna(-1)
    master = master.sort_values(by="__sort_iid", ascending=True, na_position="last")
    return master.drop(columns=["__sort_iid"])


def build_masterdata(data_dir: str, out_path: str) -> pd.DataFrame | None:
    print_shop_summary(data_dir)

    master = _build(data_dir)
    if master is None:
        return None
    
    master = _default_sort(master)

    master.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅  masterdata.csv written → {out_path}")
    print(f"   {len(master):,} rows × {len(master.columns)} columns")
    print("   Sorted by item_id ascending (byte-stable output across runs)")

    print()
    print("── Column coverage (non-null %) ─────────────────────────────────────")
    for col in master.columns:
        filled = master[col].apply(
            lambda v: pd.notna(v) and str(v).strip() not in ("", "nan")
        ).sum()
        pct = 100 * filled / len(master) if len(master) else 0
        bar = "█" * int(pct / 5)
        print(f"  {col:<40} {pct:5.1f}%  {bar}")

    return master


# ---------------------------------------------------------------------------
# Verify mode — cross-check the built CSV against raw JSONs
# ---------------------------------------------------------------------------

def _format_pct(num: float, denom: float) -> str:
    if denom == 0:
        return "—"
    return f"{100 * num / denom:5.1f}%"


def verify_masterdata(master: pd.DataFrame, data_dir: str) -> bool:
    print("\n" + "═" * 80)
    print("  VERIFY — cross-checking masterdata.csv vs raw JSONs")
    print("═" * 80)

    passed: list[str] = []
    failed: list[str] = []

    def ok(name: str, detail: str = "") -> None:
        passed.append(name)
        print(f"  ✅  {name:<55} {detail}")

    def fail(name: str, detail: str) -> None:
        failed.append(name)
        print(f"  ❌  {name:<55} {detail}")

    def warn(name: str, detail: str) -> None:
        print(f"  ⚠️   {name:<55} {detail}")

    catalog_raw = _load_json(data_dir, "bucket_A_catalog.json")
    n_catalog   = len(catalog_raw)
    n_master    = len(master)
    if n_master == n_catalog:
        ok("V1 row count == catalog count", f"{n_master:,} rows")
    else:
        fail("V1 row count != catalog count",
             f"master={n_master:,}  catalog={n_catalog:,}  diff={n_master-n_catalog:+d}")

    extra_raw  = _load_json(data_dir, "bucket_A_extra_info.json")
    sum_cc     = sum(int(r.get("comment_count") or 0) for r in extra_raw)
    sum_rc     = int(master["review_count"].fillna(0).sum()) if "review_count" in master else 0
    if sum_rc == sum_cc:
        ok("V2 Σ review_count == Σ comment_count (I1)",
           f"{sum_rc:,} initial reviews reconciled")
    else:
        diff = sum_rc - sum_cc
        pct  = abs(diff) / sum_cc * 100 if sum_cc else 0
        if pct < 1.0:
            warn("V2 Σ review_count ≈ Σ comment_count (I1)",
                 f"master={sum_rc:,}  extra_info={sum_cc:,}  diff={diff:+,} ({pct:.2f}%)")
            passed.append("V2 (within 1% tolerance)")
        else:
            fail("V2 Σ review_count != Σ comment_count (I1)",
                 f"master={sum_rc:,}  extra_info={sum_cc:,}  diff={diff:+,} ({pct:.2f}%)\n"
                 f"      → is_fb_cm filter likely broken; check process_comments()")

    comments_raw = _load_json(data_dir, "bucket_C_comments.json")
    n_rows_total = len(comments_raw)
    sum_fc       = int(master["followup_review_count"].fillna(0).sum()) if "followup_review_count" in master else 0
    accounted    = sum_rc + sum_fc
    if accounted == n_rows_total:
        ok("V3 Σ (initial + followup) == comments rows",
           f"{accounted:,} = {sum_rc:,} initial + {sum_fc:,} followup")
    else:
        fail("V3 comments accounting mismatch",
             f"master={accounted:,}  raw_rows={n_rows_total:,}  diff={accounted-n_rows_total:+,}")

    rating_raw = _load_json(data_dir, "bucket_A_shop_rating.json")
    if rating_raw:
        stored_rating  = float(rating_raw[0].get("computed_rating_star") or 0)
        stored_reviews = int(rating_raw[0].get("computed_total_reviews") or 0)
        normal_reviews_recomputed = sum(
            int(r.get("comment_count") or 0)
            for r in extra_raw
            if float(r.get("rating_star") or 0) > 0 and int(r.get("comment_count") or 0) > 0
        )
        stored_normal = int(rating_raw[0].get("normal_products_reviews") or 0)
        if abs(normal_reviews_recomputed - stored_normal) <= 1:
            ok("V4 shop rating (I2): normal_only reconciles",
               f"stored={stored_normal:,}  recomputed={normal_reviews_recomputed:,}  "
               f"rating={stored_rating} ⭐  total={stored_reviews:,}")
        else:
            fail("V4 shop rating mismatch (I2)",
                 f"stored normal_reviews={stored_normal:,}  "
                 f"recomputed={normal_reviews_recomputed:,}")
    else:
        warn("V4 shop rating (I2): file missing",
             "bucket_A_shop_rating.json not found — re-run --buckets A")
        
    lq_raw = _load_json(data_dir, "bucket_C_low_quality_items.json")
    lq_ids = {str(r.get("item_id", "")) for r in lq_raw if r.get("item_id")}
    if "quality_level" in master.columns:
        master_levels = master["quality_level"].value_counts().to_dict()
        n_low    = int(master_levels.get("LOW", 0))
        n_med    = int(master_levels.get("MEDIUM", 0))
        n_high   = int(master_levels.get("HIGH", 0))
        n_other  = sum(v for k, v in master_levels.items() if k not in ("LOW", "MEDIUM", "HIGH"))
        ids_in_master    = set(master["item_id"].astype(str).tolist())
        lq_in_master     = lq_ids & ids_in_master
        expected_marked  = len(lq_in_master)
        actual_marked    = n_low + n_med
        if actual_marked == expected_marked and n_other == 0:
            ok("V5 quality distribution (I3)",
               f"HIGH={n_high:,}  MEDIUM={n_med:,}  LOW={n_low:,}")
        else:
            fail("V5 quality distribution drift (I3)",
                 f"HIGH={n_high:,}  MEDIUM={n_med:,}  LOW={n_low:,}  other={n_other}\n"
                 f"      → LQ items in catalog: {expected_marked}, marked: {actual_marked}")
    else:
        warn("V5 quality_level column missing", "— skipped")

    if "category_name" in master.columns:
        empty = (master["category_name"].fillna("") == "").sum()
        if empty == 0:
            ok("V6 category_name populated for every row",
               f"{n_master:,} / {n_master:,}")
        else:
            warn("V6 category_name has empty values",
                 f"{empty:,} of {n_master:,} rows ({_format_pct(empty, n_master)}) — "
                 f"refresh bucket_A_categories")
    else:
        fail("V6 category_name missing", "column not built — check process_categories()")

    if "price_min" in master.columns and "price_max" in master.columns:
        bad_price = master[
            master["price_min"].notna() & master["price_max"].notna()
            & (master["price_min"] > master["price_max"])
        ]
        if bad_price.empty:
            ok("V7 price_min ≤ price_max", "all good")
        else:
            fail("V7 price_min > price_max", f"{len(bad_price)} row(s) inverted")
    else:
        warn("V7 price columns missing", "— skipped")

    neg_problems = []
    for col in ["sale", "views", "like", "comment_count", "total_stock",
                "in_stock", "review_count", "followup_review_count",
                "positive_review_count", "negative_review_count"]:
        if col in master.columns:
            n_neg = (master[col] < 0).sum()
            if n_neg > 0:
                neg_problems.append(f"{col}={n_neg}")
    if not neg_problems:
        ok("V8 non-negative counts", "all good")
    else:
        fail("V8 negative values found", "; ".join(neg_problems))

    if "category_mismatch" in master.columns:
        cm = master["category_mismatch"].apply(
            lambda v: bool(v) if not isinstance(v, str) else v.strip().lower() == "true"
        )
        n_mismatch = int(cm.sum())
        pct = 100 * n_mismatch / n_master if n_master else 0
        if pct < 50:
            ok("V9 category_mismatch plausible",
               f"{n_mismatch:,} of {n_master:,} flagged ({pct:.1f}%)")
        else:
            warn("V9 category_mismatch suspiciously high",
                 f"{n_mismatch:,} of {n_master:,} ({pct:.1f}%) — check _norm_cat_id")
    else:
        warn("V9 category_mismatch column missing", "— skipped")

    # Summary
    print("─" * 80)
    n_pass = len(passed)
    n_fail = len(failed)
    if n_fail == 0:
        print(f"  ✅  All checks passed ({n_pass}/{n_pass + n_fail}).")
        print("      You can trust this masterdata.csv for ground-truth generation.")
    else:
        print(f"  ❌  {n_fail} check(s) failed ({n_pass}/{n_pass + n_fail} passed).")
        print(f"      Failures: {failed}")
        print("      Numbers in masterdata.csv should NOT be trusted until these are fixed.")
    print("═" * 80 + "\n")

    return n_fail == 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build masterdata.csv from Shopee data_exports/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data_exports",
                        help="Folder containing the exported CSVs (default: data_exports)")
    parser.add_argument("--out", default="masterdata.csv",
                        help="Output path (default: masterdata.csv)")
    parser.add_argument("--verify", action="store_true",
                        help="After building, cross-check every aggregate against raw JSONs.")
    args = parser.parse_args()

    master = build_masterdata(data_dir=args.data_dir, out_path=args.out)

    if args.verify:
        if master is None:
            print("Cannot verify — masterdata build failed.")
            sys.exit(1)
        ok = verify_masterdata(master, args.data_dir)
        sys.exit(0 if ok else 2)