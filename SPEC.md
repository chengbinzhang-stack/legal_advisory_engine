# Permission System Redesign Specification

## Overview

Replace the current 7-parameter permission system with a streamlined 4-parameter system.

## Current State (7 Parameters)

| Parameter | Field Name |
|-----------|------------|
| scraping | `scraping_allowed` |
| manual_collection | `manual_collection_allowed` |
| storing | `storing_allowed` |
| free_display | `display_free_allowed` |
| subscription_display | `display_subscription_allowed` |
| free_redistribute | `redistribute_free_allowed` |
| subscription_redistribute | `redistribute_subscription_allowed` |

## New State (4 Parameters)

| Parameter | Field Name | Description |
|-----------|------------|-------------|
| scrap | `scrap` | Whether automated scraping/crawling is allowed |
| store | `store` | Whether storing/caching data is allowed |
| display_for_free | `display_for_free` | Whether displaying content publicly for free is allowed |
| display_for_commercial | `display_for_commercial` | Whether displaying for commercial purposes is allowed |

## New Bucket Definitions

| Bucket | scrap | store | display_for_free | display_for_commercial |
|--------|-------|-------|------------------|------------------------|
| 1 | N | N | N | N |
| 2 | Y | Y | N | N |
| 3 | Y | Y | Y | N |
| 4 | Y | Y | Y | Y |
| 6 | Y | ? | ? | ? |
| 7 | Y | Y | ? | ? |
| 8 | ? | ? | ? | ? |

Y = allowed, N = not allowed, ? = uncertain (LLM semantic judgment needed)

---

## Changes Required

### 1. `src/classifier/category_buckets.py`

**Remove fields from `CategoryDefinition`:**
- `scraping_allowed` (replaced by `scrap`)
- `manual_collection_allowed` (removed entirely)
- `storing_allowed` (replaced by `store`)
- `display_free_allowed` (replaced by `display_for_free`)
- `display_subscription_allowed` (removed)
- `redistribute_free_allowed` (removed)
- `redistribute_subscription_allowed` (removed)
- `manual_collection_allowed` (removed)

**Add fields to `CategoryDefinition`:**
- `scrap: bool`
- `store: bool`
- `display_for_free: bool`
- `display_for_commercial: bool`

**Update `CATEGORY_DEFINITIONS`:**

```python
@dataclass
class CategoryDefinition:
    bucket_number: int
    name: str
    scrap: bool
    store: bool
    display_for_free: bool
    display_for_commercial: bool

CATEGORY_DEFINITIONS: Dict[WebsiteCategory, CategoryDefinition] = {
    WebsiteCategory.BUCKET_1: CategoryDefinition(
        bucket_number=1,
        name="No Automated Access",
        scrap=False,
        store=False,
        display_for_free=False,
        display_for_commercial=False
    ),
    WebsiteCategory.BUCKET_2: CategoryDefinition(
        bucket_number=2,
        name="Scraping & Storage Only",
        scrap=True,
        store=True,
        display_for_free=False,
        display_for_commercial=False
    ),
    WebsiteCategory.BUCKET_3: CategoryDefinition(
        bucket_number=3,
        name="Free Display Only",
        scrap=True,
        store=True,
        display_for_free=True,
        display_for_commercial=False
    ),
    WebsiteCategory.BUCKET_4: CategoryDefinition(
        bucket_number=4,
        name="Full Commercial Access",
        scrap=True,
        store=True,
        display_for_free=True,
        display_for_commercial=True
    ),
}
```

**Update `PARAM_TO_CATEGORY_FIELD`:**

```python
PARAM_TO_CATEGORY_FIELD = {
    "scrap": "scrap",
    "store": "store",
    "display_for_free": "display_for_free",
    "display_for_commercial": "display_for_commercial",
}
```

### 2. `src/classifier/legal_classifier.py`

**Update `PERMISSION_PARAMS`:**

```python
PERMISSION_PARAMS = [
    "scrap",
    "store",
    "display_for_free",
    "display_for_commercial",
]
```

**Update `SYSTEM_PROMPT`:**
- Replace all references to the 7 old parameters with the 4 new ones
- Remove `manual_collection`, `subscription_display`, `free_redistribute`, `subscription_redistribute` from the prompt
- Update JSON output structure to match new parameters

**Update `UNCERTAIN_RESOLUTION_PROMPT`:**
- Replace the 7 parameter descriptions with the 4 new ones
- Update the output format to reflect new parameters

**Update `_classify_heuristic()`:**
- Replace `param_rules` dictionary with rules for the 4 new parameters

**No changes needed to:**
- `_determine_category()` - uses `PARAM_TO_CATEGORY_FIELD` dynamically
- `_parse_llm_response()` - generic JSON parsing
- `_resolve_uncertain_param()` - generic uncertain resolution

---

## Files to Modify

1. `src/classifier/category_buckets.py` - Bucket definitions
2. `src/classifier/legal_classifier.py` - LLM prompts and parameter handling

## Backward Compatibility

- `LegalAnalysis` model may need updates if it references old permission fields
- Check `src/models/legal_analysis.py` for any hardcoded parameter references

## Testing Considerations

1. Verify bucket scoring still works with 4 parameters
2. Verify uncertain resolution works for buckets 6, 7, 8
3. Verify heuristic fallback works with new parameters
4. Update any integration tests that reference old parameters