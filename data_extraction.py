import argparse
import csv
import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ── Credentials ────────────────────────────────────────────────────────────────
load_dotenv()

def _require_env(name: str, *, as_int: bool = False):
    val = os.getenv(name)
    if not val:
        sys.exit(
            f"[Init] ERROR: environment variable {name} is not set. "
            f"Add it to your .env file before running data_extraction.py."
        )
    if as_int:
        try:
            return int(val)
        except ValueError:
            sys.exit(f"[Init] ERROR: {name} must be an integer, got: {val!r}")
    return val


PARTNER_ID   = _require_env("SHOPEE_PARTNER_ID", as_int=True)
PARTNER_KEY  = _require_env("SHOPEE_PARTNER_KEY")
SHOP_ID      = _require_env("SHOPEE_SHOP_ID",    as_int=True)
ACCESS_TOKEN = _require_env("SHOPEE_ACCESS_TOKEN")

HOST = "https://partner.shopeemobile.com"

# ── Sleep constants (seconds) ──────────────────────────────────────────────────
SLEEP_BETWEEN_BATCH_CALLS = 5
SLEEP_BETWEEN_ENDPOINTS   = 5
SLEEP_PER_ITEM            = 5
SLEEP_PAGINATION          = 5
SLEEP_BETWEEN_BUCKETS     = 5
SLEEP_RETRY_INITIAL       = 15.0
SLEEP_RATE_LIMIT_BACKOFF  = 30.0

_RATE_LIMIT_ERRORS = {
    "error_too_many_requests",
    "error_busy",
    "error.too_many_requests",
    "error_system",
}
ALL_ITEM_STATUSES = ["NORMAL", "BANNED", "UNLIST", "REVIEWING"]


# ==============================================================================
# PIPELINE LOG
# ==============================================================================
class PipelineLog:
    OK      = "✅  success"
    SKIP    = "⏭️  skipped"
    ERROR   = "❌  error"
    PARTIAL = "⚠️  partial"

    def __init__(self):
        self._entries: list[dict] = []

    def record(self, api: str, status: str, records: int = 0, note: str = ""):
        self._entries.append({
            "api": api, "status": status, "records": records, "note": note,
        })

    def to_list(self) -> list[dict]:
        return list(self._entries)

    def print_summary(self):
        print("\n" + "═" * 80)
        print("  PIPELINE SUMMARY")
        print("═" * 80)
        print(f"  {'Endpoint':<54} {'Status':<16} {'Records':>7}  Note")
        print("  " + "─" * 76)
        for e in self._entries:
            note = f"  {e['note']}" if e["note"] else ""
            rec  = str(e["records"]) if e["records"] else "—"
            print(f"  {e['api']:<54} {e['status']:<16} {rec:>7}{note}")
        print("═" * 80 + "\n")

LOG = PipelineLog()


# ==============================================================================
# CORE HELPERS
# ==============================================================================
def generate_sign(path: str, timestamp: int) -> str:
    base = f"{PARTNER_ID}{path}{timestamp}{ACCESS_TOKEN}{SHOP_ID}"
    return hmac.new(PARTNER_KEY.encode(), base.encode(), hashlib.sha256).hexdigest()


def shopee_get(
    path: str,
    extra_params: dict = None,
    top_level: bool = False,
    _attempt: int = 0,
    _max_attempts: int = 3,
) -> dict | None:
    timestamp = int(time.time())
    params = {
        "partner_id":   PARTNER_ID,
        "timestamp":    timestamp,
        "access_token": ACCESS_TOKEN,
        "shop_id":      SHOP_ID,
        "sign":         generate_sign(path, timestamp),
    }
    if extra_params:
        params.update(extra_params)

    try:
        resp = requests.get(HOST + path, params=params, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"  [Request Error] {path}: {e}")
        if _attempt < _max_attempts - 1:
            wait = SLEEP_RETRY_INITIAL * (2 ** _attempt)
            print(f"  [Retry] Network error — waiting {wait:.0f}s ({_attempt + 2}/{_max_attempts})...")
            time.sleep(wait)
            return shopee_get(path, extra_params, top_level, _attempt + 1, _max_attempts)
        return None

    err = data.get("error", "")
    if err:
        is_rate_limit = any(rl in err.lower() for rl in _RATE_LIMIT_ERRORS)
        if is_rate_limit and _attempt < _max_attempts - 1:
            wait = SLEEP_RATE_LIMIT_BACKOFF * (2 ** _attempt)
            print(f"  [Rate Limit] '{err}' on {path}. Waiting {wait:.0f}s ({_attempt + 2}/{_max_attempts})...")
            time.sleep(wait)
            return shopee_get(path, extra_params, top_level, _attempt + 1, _max_attempts)
        print(f"  [API Error] {path}: {err} — {data.get('message', '')[:120]}")
        return None

    if top_level:
        ENVELOPE_KEYS = {"error", "message", "request_id", "warning"}
        return {k: v for k, v in data.items() if k not in ENVELOPE_KEYS}

    return data.get("response")


def _sort_records(records: list, key_field: str | None) -> list:
    if not records or not key_field:
        return list(records)
    if not all(isinstance(r, dict) and key_field in r for r in records):
        return list(records)

    def _key(r):
        v = r.get(key_field, "")
        try:
            return (0, int(v))
        except (ValueError, TypeError):
            return (1, str(v))

    return sorted(records, key=_key)


def save(records: list, filename: str, sort_by: str | None = "item_id") -> int:
    if not records:
        print(f"  No data to save for {filename}.")
        return 0

    records = _sort_records(records, sort_by)

    os.makedirs("data_exports", exist_ok=True)

    with open(f"data_exports/{filename}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

    keys: list[str] = []
    for row in records:
        for k in row:
            if k not in keys:
                keys.append(k)

    with open(f"data_exports/{filename}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                for k, v in row.items()
            })

    print(f"  Saved {len(records):,} records → data_exports/{filename}.json + .csv")
    return len(records)


def load_json(filename: str) -> list:
    path = f"data_exports/{filename}.json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fetch_ids_for_status(api: str, status: str) -> list[int]:
    ids: list[int] = []
    offset, has_next = 0, True
    while has_next:
        data = shopee_get(api, {"offset": offset, "page_size": 100, "item_status": status})
        if not data or "item" not in data:
            break
        ids.extend(item["item_id"] for item in data["item"])
        has_next = data.get("has_next_page", False)
        offset   = data.get("next_offset", 0)
        if has_next:
            time.sleep(SLEEP_PAGINATION)
    return ids


def _retry_batch(
    api: str,
    failed_ids: list[int],
    response_key: str,
    batch_size: int = 50,
    label: str = "",
) -> tuple[list[dict], list[int]]:
    if not failed_ids:
        return [], []
    tag = label or api.split("/")[-1]
    print(f"  [Retry] {len(failed_ids)} item(s) failed {tag} — waiting {SLEEP_RETRY_INITIAL:.0f}s...")
    time.sleep(SLEEP_RETRY_INITIAL)
    new_records:  list[dict] = []
    still_failed: list[int]  = []
    for i in range(0, len(failed_ids), batch_size):
        batch     = failed_ids[i : i + batch_size]
        batch_str = ",".join(map(str, batch))
        data      = shopee_get(api, {"item_id_list": batch_str})
        if data and response_key in data:
            fetched = {item["item_id"] for item in data[response_key]}
            new_records.extend(data[response_key])
            still_failed.extend(iid for iid in batch if iid not in fetched)
        else:
            still_failed.extend(batch)
        time.sleep(SLEEP_BETWEEN_BATCH_CALLS)
    return new_records, still_failed


# ==============================================================================
# STEP 0 — MASTER ITEM LIST  (all statuses)
# ==============================================================================
def get_all_item_ids() -> tuple[list[int], dict[str, int], dict[str, list[int]]]:
    API = "/api/v2/product/get_item_list"
    print(f"Fetching item counts by status ({API})...")

    ids_by_status: dict[str, list[int]] = {}
    status_counts: dict[str, int]       = {}
    active_ids:    list[int]            = []

    for status in ALL_ITEM_STATUSES:
        ids = _fetch_ids_for_status(API, status)
        ids = sorted(ids)
        ids_by_status[status] = ids
        status_counts[status] = len(ids)
        if status == "NORMAL":
            active_ids = ids
        print(f"  {status:<12} {len(ids):>4} items")
        time.sleep(SLEEP_PAGINATION)

    total = sum(status_counts.values())
    print(f"  {'TOTAL':<12} {total:>4} items\n")

    save(
        [{"status": k, "count": v} for k, v in status_counts.items()]
        + [{"status": "TOTAL", "count": total}],
        "bucket_A_item_status_counts",
        sort_by=None,
    )
    LOG.record(
        API,
        PipelineLog.OK if active_ids else PipelineLog.ERROR,
        records=total,
        note=(
            f"NORMAL={status_counts.get('NORMAL', 0)}  "
            f"UNLIST={status_counts.get('UNLIST', 0)}  "
            f"BANNED={status_counts.get('BANNED', 0)}  "
            f"REVIEWING={status_counts.get('REVIEWING', 0)}"
        ),
    )
    return active_ids, status_counts, ids_by_status


# ==============================================================================
# BUCKET A — STATIC PUBLIC DATA  (refresh: weekly)
# ==============================================================================

def extract_a_shop_info() -> None:
    API = "/api/v2/shop/get_shop_info"
    print(f"  {API}")
    data = shopee_get(API, top_level=True)
    if not data or "shop_name" not in data:
        LOG.record(API, PipelineLog.ERROR, note="no data — check access token scope")
        return
    n = save([data], "bucket_A_shop_info", sort_by=None)
    LOG.record(API, PipelineLog.OK, records=n)


def extract_a_category_tree() -> None:
    API = "/api/v2/product/get_category"
    print(f"  {API}")
    data = shopee_get(API, {"language": "id"})
    if not data or "category_list" not in data:
        LOG.record(API, PipelineLog.ERROR, note="no category data returned")
        return
    n = save(data["category_list"], "bucket_A_categories", sort_by="category_id")
    LOG.record(API, PipelineLog.OK, records=n,
               note="join via category_id → display_category_name in masterdata")


def _fetch_extra_info_for_non_normal(ids_by_status: dict[str, list[int]]) -> list[dict]:
    API = "/api/v2/product/get_item_extra_info"
    extra: list[dict] = []

    for status, ids in ids_by_status.items():
        if status == "NORMAL" or not ids:
            continue
        print(f"  Fetching extra_info for {len(ids)} {status} item(s) (rating total only)...")
        for i in range(0, len(ids), 50):
            batch = ids[i : i + 50]
            data  = shopee_get(API, {"item_id_list": ",".join(map(str, batch))})
            if data and "item_list" in data:
                extra.extend(data["item_list"])
            time.sleep(SLEEP_BETWEEN_BATCH_CALLS)

    return extra


def extract_a_shop_rating(
    normal_extra_records: list[dict],
    ids_by_status: dict[str, list[int]] | None = None,
) -> None:
    API_LABEL = "computed/shop_rating"

    non_normal_extra: list[dict] = []
    if ids_by_status:
        non_normal_extra = _fetch_extra_info_for_non_normal(ids_by_status)

    all_records = normal_extra_records + non_normal_extra
    print(f"  Computing shop rating from {len(all_records)} product records "
          f"({len(normal_extra_records)} NORMAL + {len(non_normal_extra)} non-NORMAL)...")

    if not all_records:
        LOG.record(API_LABEL, PipelineLog.ERROR, note="no extra_info records available")
        return

    total_weighted       = 0.0
    total_reviews        = 0
    rated_products       = 0
    normal_reviews_total = 0

    normal_ids = {str(r.get("item_id", "")) for r in normal_extra_records}

    for item in all_records:
        rating  = float(item.get("rating_star") or 0)
        count   = int(item.get("comment_count") or 0)
        item_id = str(item.get("item_id", ""))
        if rating > 0 and count > 0:
            total_weighted += rating * count
            total_reviews  += count
            rated_products += 1
        if item_id in normal_ids and count > 0:
            normal_reviews_total += count

    shop_rating = round(total_weighted / total_reviews, 2) if total_reviews > 0 else 0.0

    record = {
        "computed_rating_star":    shop_rating,
        "computed_total_reviews":  total_reviews,
        "products_with_rating":    rated_products,
        "normal_products_reviews": normal_reviews_total,
        "note": (
            "Weighted average of product ratings (rating_star × comment_count) "
            "across ALL statuses (NORMAL + UNLIST + BANNED + REVIEWING). "
        ),
    }
    n = save([record], "bucket_A_shop_rating", sort_by=None)
    print(f"  Shop rating : {shop_rating} ⭐")
    print(f"  Total reviews (all statuses) : {total_reviews:,}")
    print(f"  Total reviews (NORMAL only)  : {normal_reviews_total:,}")
    LOG.record(API_LABEL, PipelineLog.OK, records=n,
               note=f"rating={shop_rating}  total_reviews={total_reviews:,}  "
                    f"normal_only={normal_reviews_total:,}")


def extract_a_catalog(item_ids: list[int]) -> tuple[list[dict], list[dict]]:
 
    API_BASE  = "/api/v2/product/get_item_base_info"
    API_EXTRA = "/api/v2/product/get_item_extra_info"
    print(f"  {API_BASE}")
    print(f"  {API_EXTRA}")

    catalog:          list[dict] = []
    extra:            list[dict] = []
    failed_base_ids:  list[int]  = []
    failed_extra_ids: list[int]  = []

    for i in tqdm(range(0, len(item_ids), 50), desc="  Catalog batches"):
        batch     = item_ids[i : i + 50]
        batch_str = ",".join(map(str, batch))

        base_data = shopee_get(API_BASE, {"item_id_list": batch_str})
        if base_data and "item_list" in base_data:
            fetched_ids = {item["item_id"] for item in base_data["item_list"]}
            catalog.extend(base_data["item_list"])
            failed_base_ids.extend(iid for iid in batch if iid not in fetched_ids)
        else:
            failed_base_ids.extend(batch)

        time.sleep(SLEEP_BETWEEN_ENDPOINTS)

        extra_data = shopee_get(API_EXTRA, {"item_id_list": batch_str})
        if extra_data and "item_list" in extra_data:
            fetched_ids = {item["item_id"] for item in extra_data["item_list"]}
            extra.extend(extra_data["item_list"])
            failed_extra_ids.extend(iid for iid in batch if iid not in fetched_ids)
        else:
            failed_extra_ids.extend(batch)

        if i + 50 < len(item_ids):
            time.sleep(SLEEP_BETWEEN_BATCH_CALLS)

    new_base,  failed_base_ids  = _retry_batch(API_BASE,  failed_base_ids,  "item_list", label="base_info")
    catalog.extend(new_base)
    new_extra, failed_extra_ids = _retry_batch(API_EXTRA, failed_extra_ids, "item_list", label="extra_info")
    extra.extend(new_extra)

    n_base  = save(catalog, "bucket_A_catalog")
    n_extra = save(extra,   "bucket_A_extra_info")

    failed_records: list[dict] = []
    if failed_base_ids:
        print(f"  [WARNING] base_info still missing {len(failed_base_ids)} item(s): {failed_base_ids}")
        failed_records.extend({"item_id": iid, "source": "base_info"} for iid in failed_base_ids)
    if failed_extra_ids:
        print(f"  [WARNING] extra_info still missing {len(failed_extra_ids)} item(s): {failed_extra_ids}")
        failed_records.extend({"item_id": iid, "source": "extra_info"} for iid in failed_extra_ids)
    if failed_records:
        save(failed_records, "failed_item_ids")

    LOG.record(API_BASE,  PipelineLog.OK if not failed_base_ids  else PipelineLog.PARTIAL,
               records=n_base,  note=f"{len(failed_base_ids)} missing"  if failed_base_ids  else "")
    LOG.record(API_EXTRA, PipelineLog.OK if not failed_extra_ids else PipelineLog.PARTIAL,
               records=n_extra, note=f"{len(failed_extra_ids)} missing" if failed_extra_ids else "")

    return catalog, extra


# ==============================================================================
# BUCKET B — VOLATILE PRICE & STOCK  (refresh: every run)
# ==============================================================================

def extract_b_models(item_ids: list[int]) -> None:
    API = "/api/v2/product/get_model_list"
    print(f"  {API}")

    models:     list[dict] = []
    failed_ids: list[int]  = []

    for item_id in tqdm(item_ids, desc="  Models"):
        data = shopee_get(API, {"item_id": item_id})
        if data and "model" in data:
            models.append({
                "item_id":    item_id,
                "variations": data.get("tier_variation", []),
                "models":     data.get("model", []),
            })
        else:
            failed_ids.append(item_id)
        time.sleep(SLEEP_PER_ITEM)

    if failed_ids:
        print(f"  [Retry] {len(failed_ids)} item(s) — waiting {SLEEP_RETRY_INITIAL:.0f}s...")
        time.sleep(SLEEP_RETRY_INITIAL)
        still_failed: list[int] = []
        for item_id in tqdm(failed_ids, desc="  Models retry"):
            data = shopee_get(API, {"item_id": item_id})
            if data and "model" in data:
                models.append({
                    "item_id":    item_id,
                    "variations": data.get("tier_variation", []),
                    "models":     data.get("model", []),
                })
            else:
                still_failed.append(item_id)
            time.sleep(SLEEP_PER_ITEM)
        failed_ids = still_failed

    n = save(models, "bucket_B_models")
    if failed_ids:
        save([{"item_id": iid, "source": "bucket_B_models"} for iid in failed_ids], "failed_model_ids")
        LOG.record(API, PipelineLog.PARTIAL, records=n,
                   note=f"{len(failed_ids)} item(s) still missing after retry")
    else:
        LOG.record(API, PipelineLog.OK if models else PipelineLog.ERROR, records=n)


# ==============================================================================
# BUCKET C-PUBLIC — CUSTOMER REVIEWS  (refresh: daily)
# ==============================================================================

def extract_c_comments(item_ids: list[int]) -> None:
    API = "/api/v2/product/get_comment"
    print(f"  {API}")

    comments:        list[dict] = []
    failed_item_ids: list[int]  = []

    for item_id in tqdm(item_ids, desc="  Comments"):
        cursor, has_more = "", True
        item_ok = True
        while has_more:
            data = shopee_get(API, {"item_id": item_id, "cursor": cursor, "page_size": 100})
            if not data:
                item_ok = False
                break
            comments.extend(data.get("item_comment_list", []))
            has_more = data.get("more", False)
            cursor   = data.get("next_cursor", "")
            if has_more:
                time.sleep(SLEEP_PAGINATION)
        if not item_ok:
            failed_item_ids.append(item_id)
        time.sleep(SLEEP_PER_ITEM)

    if failed_item_ids:
        print(f"  [WARNING] Comments failed for {len(failed_item_ids)} item(s): {failed_item_ids}")

    if comments:
        with_flag = sum(1 for c in comments if "is_fb_cm" in c)
        followups = sum(1 for c in comments if c.get("is_fb_cm") == 1)
        print(f"  [I1 check] {with_flag}/{len(comments)} rows have is_fb_cm; "
              f"{followups} are follow-ups (is_fb_cm=1).")
        if with_flag == 0:
            print("  [WARNING] No is_fb_cm field found on any comment row! "
                  "masterdata.py will treat ALL rows as initial reviews, "
                  "inflating review_count vs Shopee's comment_count.")

    n = save(comments, "bucket_C_comments")
    LOG.record(
        API,
        PipelineLog.OK if not failed_item_ids else PipelineLog.PARTIAL,
        records=n,
        note=f"{len(failed_item_ids)} item(s) skipped" if failed_item_ids else "",
    )


# ==============================================================================
# BUCKET C-PRIVATE — OWNER DIAGNOSTICS & HEALTH  (refresh: daily)
# ==============================================================================

def extract_c_content_diagnosis(item_ids: list[int]) -> None:
    API = "/api/v2/product/get_item_content_diagnosis_result"
    print(f"  {API}")

    diagnostics:    list[dict]      = []
    failed_batches: list[list[int]] = []

    for i in tqdm(range(0, len(item_ids), 48), desc="  Diagnosis batches"):
        batch = item_ids[i : i + 48]
        data  = shopee_get(API, {"item_id_list": ",".join(map(str, batch))})
        if data is None:
            failed_batches.append(batch)
        elif "item_list" in data:
            diagnostics.extend(data["item_list"])
        time.sleep(SLEEP_BETWEEN_BATCH_CALLS)

    if failed_batches:
        print(f"  [Retry] {len(failed_batches)} batch(es) — waiting {SLEEP_RETRY_INITIAL:.0f}s...")
        time.sleep(SLEEP_RETRY_INITIAL)
        still_failed: list[list[int]] = []
        for batch in tqdm(failed_batches, desc="  Diagnosis retry"):
            data = shopee_get(API, {"item_id_list": ",".join(map(str, batch))})
            if data is None:
                still_failed.append(batch)
            elif "item_list" in data:
                diagnostics.extend(data["item_list"])
            time.sleep(SLEEP_BETWEEN_BATCH_CALLS)
        failed_batches = still_failed

    n = save(diagnostics, "bucket_C_diagnostics")
    if failed_batches:
        LOG.record(API, PipelineLog.PARTIAL, records=n,
                   note=f"{len(failed_batches)} batch(es) still missing after retry")
    else:
        note = "no content issues found" if n == 0 else "batch size = 48 (API max)"
        LOG.record(API, PipelineLog.OK, records=n, note=note)


def extract_c_low_quality_items() -> None:
    API = "/api/v2/product/get_item_list_by_content_diagnosis"
    print(f"  {API}")

    QUALITY_LEVELS = {1: "LOW", 2: "MEDIUM"}
    per_level: dict[str, dict[str, dict]] = {"LOW": {}, "MEDIUM": {}}
    api_errors = 0

    for level_int, level_label in QUALITY_LEVELS.items():
        next_offset: str = ""
        page = 0
        MAX_PAGES = 500
        seen_ids: set[str] = set()
        level_dict = per_level[level_label]

        while page < MAX_PAGES:
            params: dict = {"content_quality_level": level_int, "page_size": 48}
            if next_offset:
                params["next_offset"] = next_offset

            data = shopee_get(API, params)
            if data is None:
                api_errors += 1
                break

            items = data.get("item_list", [])
            if not items:
                break

            page_ids = {str(item.get("item_id", "")) for item in items}
            new_ids  = page_ids - seen_ids
            if not new_ids:
                print(f"  [LQ] Cursor loop detected at page {page + 1} for {level_label}. Stopping.")
                break

            for item in items:
                item_id = str(item.get("item_id", ""))
                if item_id and item_id not in seen_ids:
                    item["_quality_level"]     = level_label
                    item["_quality_level_int"] = level_int
                    level_dict[item_id]        = item
                    seen_ids.add(item_id)

            page += 1
            print(f"  [LQ] {level_label} page {page}: +{len(new_ids)} new (total: {len(level_dict)})")

            next_offset = data.get("next_offset", "")
            if not next_offset:
                break
            time.sleep(SLEEP_PAGINATION)

    SEVERITY = {"LOW": 0, "MEDIUM": 1}
    merged: dict[str, dict] = {}
    for level_label, level_dict in per_level.items():
        for item_id, item in level_dict.items():
            existing = merged.get(item_id)
            if existing is None or SEVERITY[level_label] < SEVERITY[existing["_quality_level"]]:
                merged[item_id] = item

    all_records = list(merged.values())
    low_n  = sum(1 for r in all_records if r["_quality_level"] == "LOW")
    med_n  = sum(1 for r in all_records if r["_quality_level"] == "MEDIUM")
    print(f"  [LQ] Final: {len(all_records)} unique items  (LOW: {low_n}, MEDIUM: {med_n})")

    n = save(all_records, "bucket_C_low_quality_items")
    if api_errors > 0:
        LOG.record(API, PipelineLog.ERROR, records=n,
                   note=f"{api_errors} level(s) failed — check API permissions")
    elif all_records:
        LOG.record(API, PipelineLog.OK, records=n, note=f"LOW: {low_n}, MEDIUM: {med_n} unique items")
    else:
        LOG.record(API, PipelineLog.SKIP, records=0, note="all products are HIGH quality")


def extract_c_violations(item_ids: list[int]) -> None:
    API = "/api/v2/product/get_item_violation_info"
    print(f"  {API}")

    violations:     list[dict]      = []
    failed_batches: list[list[int]] = []

    for i in tqdm(range(0, len(item_ids), 50), desc="  Violations"):
        batch = item_ids[i : i + 50]
        data  = shopee_get(API, {"item_id_list": ",".join(map(str, batch))})
        if data and "item_list" in data:
            violations.extend(data["item_list"])
        else:
            failed_batches.append(batch)
        time.sleep(SLEEP_BETWEEN_BATCH_CALLS)

    if failed_batches:
        print(f"  [Retry] {len(failed_batches)} batch(es) — waiting {SLEEP_RETRY_INITIAL:.0f}s...")
        time.sleep(SLEEP_RETRY_INITIAL)
        still_failed: list[list[int]] = []
        for batch in failed_batches:
            data = shopee_get(API, {"item_id_list": ",".join(map(str, batch))})
            if data and "item_list" in data:
                violations.extend(data["item_list"])
            else:
                still_failed.append(batch)
            time.sleep(SLEEP_BETWEEN_BATCH_CALLS)
        failed_batches = still_failed

    n = save(violations, "bucket_C_violations")
    if failed_batches:
        LOG.record(API, PipelineLog.PARTIAL, records=n,
                   note=f"{len(failed_batches)} batch(es) still missing after retry")
    else:
        LOG.record(API, PipelineLog.OK, records=n)


def extract_c_category_recommendations(catalog_records: list[dict]) -> None:
    API = "/api/v2/product/category_recommend"
    print(f"  {API}")

    recommendations: list[dict] = []
    failed_items:    list[int]  = []

    for item in tqdm(catalog_records, desc="  Category recs"):
        item_id   = item.get("item_id")
        item_name = item.get("item_name", "").strip()
        if not item_name:
            continue
        data = shopee_get(API, {"item_name": item_name})
        if data:
            recommendations.append({
                "item_id":                item_id,
                "item_name":              item_name,
                "current_category_id":    item.get("category_id"),
                "recommended_categories": data.get("category_list", []),
            })
        else:
            if item_id:
                failed_items.append(item_id)
        time.sleep(SLEEP_PER_ITEM)

    n = save(recommendations, "bucket_C_category_recommend")
    if failed_items:
        LOG.record(API, PipelineLog.PARTIAL, records=n,
                   note=f"{len(failed_items)} item(s) skipped")
    else:
        LOG.record(API, PipelineLog.OK if recommendations else PipelineLog.ERROR, records=n)


def extract_c_shop_performance() -> None:
    API = "/api/v2/account_health/get_shop_performance"
    print(f"  {API}")
    data = shopee_get(API)
    if not data:
        LOG.record(API, PipelineLog.ERROR, note="no data returned")
        return
    n = save([data], "bucket_C_shop_performance", sort_by=None)
    LOG.record(API, PipelineLog.OK, records=n)


# ==============================================================================
# INTEGRITY CHECK
# ==============================================================================
def run_integrity_check(master_ids: list[int]) -> dict:
    print("\n" + "═" * 80)
    print("  INTEGRITY CHECK")
    print("═" * 80)

    master_set = {str(iid) for iid in master_ids}
    n_master   = len(master_set)
    report     = {"master_item_count": n_master, "buckets": {}}

    def check_bucket(label: str, filename: str, id_field: str = "item_id") -> None:
        records = load_json(filename)
        if not records:
            print(f"  {label:<40} ❌  FILE MISSING or empty")
            report["buckets"][label] = {
                "file": f"{filename}.json", "status": "missing",
                "record_count": 0, "coverage_pct": 0.0,
                "missing_ids": list(master_set),
            }
            return

        found_set: set[str] = set()
        for row in records:
            iid = row.get(id_field)
            if iid is not None:
                found_set.add(str(iid))

        present = master_set & found_set
        missing = master_set - found_set
        extra   = found_set - master_set
        pct     = 100 * len(present) / n_master if n_master else 0.0
        status  = "ok" if not missing else ("partial" if present else "empty")
        icon    = "✅" if not missing else ("⚠️ " if present else "❌")

        print(f"  {label:<40} {icon}  {len(present):>4}/{n_master}  ({pct:5.1f}%)  "
              f"missing={len(missing)}  extra={len(extra)}  rows={len(records):,}")

        report["buckets"][label] = {
            "file": f"{filename}.json", "status": status,
            "record_count": len(records),
            "item_ids_found": len(present),
            "item_ids_missing": len(missing),
            "item_ids_extra": len(extra),
            "coverage_pct": round(pct, 2),
            "missing_ids": sorted(missing, key=lambda x: int(x) if x.isdigit() else 0),
        }

    check_bucket("bucket_A_catalog",            "bucket_A_catalog")
    check_bucket("bucket_A_extra_info",         "bucket_A_extra_info")
    check_bucket("bucket_B_models",             "bucket_B_models")
    check_bucket("bucket_C_comments (items)",   "bucket_C_comments", "item_id")
    check_bucket("bucket_C_diagnostics",        "bucket_C_diagnostics")
    check_bucket("bucket_C_violations",         "bucket_C_violations")
    check_bucket("bucket_C_category_recommend", "bucket_C_category_recommend")

    all_buckets = list(report["buckets"].values())
    any_missing = any(b.get("item_ids_missing", 0) > 0 for b in all_buckets)
    all_ok      = not any_missing and all(b.get("status") == "ok" for b in all_buckets)
    report["overall_status"] = "ok" if all_ok else ("partial" if any_missing else "error")

    print("─" * 80)
    if all_ok:
        print("  ✅  All buckets complete — no missing items detected.")
    else:
        print("  ⚠️   Some items are missing. Rerun with --buckets <letter> to refresh.")
    print("═" * 80 + "\n")

    os.makedirs("data_exports", exist_ok=True)
    with open("data_exports/integrity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print("  Integrity report → data_exports/integrity_report.json\n")
    return report


# ==============================================================================
# MANIFEST
# ==============================================================================
def save_manifest(
    active_ids: list[int],
    status_counts: dict[str, int],
    log_entries: list[dict],
    integrity: dict | None,
) -> None:
    manifest = {
        "run_utc":            datetime.now(timezone.utc).isoformat(),
        "active_item_count":  len(active_ids),
        "item_status_counts": status_counts,
        "total_item_count":   sum(status_counts.values()),
        "endpoints":          log_entries,
        "integrity":          integrity or {},
    }
    os.makedirs("data_exports", exist_ok=True)
    with open("data_exports/pipeline_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    print("  Pipeline manifest → data_exports/pipeline_manifest.json\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shopee data extraction pipeline (writes data_exports/)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--buckets", nargs="+", choices=["A", "B", "C"],
                        default=["A", "B", "C"],
                        help="Buckets to refresh. Example: --buckets A B")
    parser.add_argument("--skip-integrity", action="store_true",
                        help="Skip the post-run integrity check.")
    args = parser.parse_args()

    active_ids, status_counts, ids_by_status = get_all_item_ids()
    if not active_ids:
        print("No active (NORMAL) items found. Aborting.")
        LOG.print_summary()
        return

    catalog_records: list[dict] = []
    extra_records:   list[dict] = []

    # ── Bucket A ─────────────────────────────────────────────────────────────
    if "A" in args.buckets:
        print("── Bucket A: Static Public Data ──────────────────────────────")
        extract_a_shop_info()
        time.sleep(SLEEP_BETWEEN_BUCKETS)

        extract_a_category_tree()
        time.sleep(SLEEP_BETWEEN_BUCKETS)

        catalog_records, extra_records = extract_a_catalog(active_ids)
        time.sleep(SLEEP_BETWEEN_BUCKETS)

        if not extra_records:
            extra_records = load_json("bucket_A_extra_info")
        extract_a_shop_rating(extra_records, ids_by_status)
        print()

    # ── Bucket B ─────────────────────────────────────────────────────────────
    if "B" in args.buckets:
        time.sleep(SLEEP_BETWEEN_BUCKETS)
        print("── Bucket B: Price & Stock (Live) ────────────────────────────")
        extract_b_models(active_ids)
        print()

    # ── Bucket C ─────────────────────────────────────────────────────────────
    if "C" in args.buckets:
        time.sleep(SLEEP_BETWEEN_BUCKETS)
        print("── Bucket C-Public: Customer Reviews ─────────────────────────")
        extract_c_comments(active_ids)
        print()

        time.sleep(SLEEP_BETWEEN_BUCKETS)
        print("── Bucket C-Private: Diagnostics & Health ────────────────────")
        extract_c_content_diagnosis(active_ids)
        time.sleep(SLEEP_BETWEEN_BUCKETS)
        extract_c_low_quality_items()
        time.sleep(SLEEP_BETWEEN_BUCKETS)
        extract_c_violations(active_ids)
        time.sleep(SLEEP_BETWEEN_BUCKETS)
        extract_c_shop_performance()
        time.sleep(SLEEP_BETWEEN_BUCKETS)

        if not catalog_records:
            catalog_records = load_json("bucket_A_catalog")
        if catalog_records:
            extract_c_category_recommendations(catalog_records)
        else:
            print("  Warning: bucket_A_catalog.json not found. Run --buckets A first.")
            LOG.record("/api/v2/product/category_recommend", PipelineLog.SKIP,
                       note="bucket_A_catalog.json missing — run --buckets A first")
        print()

    # ── Post-run checks ───────────────────────────────────────────────────────
    integrity: dict | None = None
    if not args.skip_integrity:
        integrity = run_integrity_check(active_ids)

    LOG.print_summary()
    save_manifest(active_ids, status_counts, LOG.to_list(), integrity)

    print("Pipeline complete. Check the data_exports/ folder.")
    print("Next steps:")
    print("  1. python masterdata.py --verify   # cross-check the final CSV")
    print("  2. rm -rf ./chroma_db && chainlit run rag.py   # rebuild the vector store\n")


if __name__ == "__main__":
    main()