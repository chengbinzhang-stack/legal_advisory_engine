"""Category bucket definitions for website classification.

4 permission axes:
  scrap, store, display_for_free, display_for_commercial

Bucket profile matrix:
| Bucket | scrap | store | display_for_free | display_for_commercial |
|--------|-------|-------|------------------|------------------------|
| 1      | N     | N     | N                | N                      |
| 2      | Y     | Y     | N                | N                      |
| 3      | Y     | Y     | Y                | N                      |
| 4      | Y     | Y     | Y                | Y                      |
| 6      | Y     | ?     | ?                | ?                      |
| 7      | Y     | Y     | ?                | ?                      |
| 8      | ?     | ?     | ?                | ?                      |

Buckets 6/7/8 use None (i.e. "wildcard") for the trailing axes. Scoring treats
None as a no-op (does not add or subtract points), so the LLM-driven semantic
judgment for those axes determines whether the bucket is a closer or looser fit.
"""
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class WebsiteCategory(Enum):
    BUCKET_1 = 1
    BUCKET_2 = 2
    BUCKET_3 = 3
    BUCKET_4 = 4
    BUCKET_6 = 6
    BUCKET_7 = 7
    BUCKET_8 = 8


@dataclass
class CategoryDefinition:
    bucket_number: int
    name: str
    # Four permission axes. None == wildcard (does not constrain scoring).
    scrap_allowed: Optional[bool]
    store_allowed: Optional[bool]
    display_free_allowed: Optional[bool]
    display_commercial_allowed: Optional[bool]


CATEGORY_DEFINITIONS: Dict[WebsiteCategory, CategoryDefinition] = {
    WebsiteCategory.BUCKET_1: CategoryDefinition(
        bucket_number=1,
        name="No Access (Manual Reading Only)",
        scrap_allowed=False,
        store_allowed=False,
        display_free_allowed=False,
        display_commercial_allowed=False,
    ),
    WebsiteCategory.BUCKET_2: CategoryDefinition(
        bucket_number=2,
        name="Store Only (No Display)",
        scrap_allowed=True,
        store_allowed=True,
        display_free_allowed=False,
        display_commercial_allowed=False,
    ),
    WebsiteCategory.BUCKET_3: CategoryDefinition(
        bucket_number=3,
        name="Display for Free (No Commercial Display)",
        scrap_allowed=True,
        store_allowed=True,
        display_free_allowed=True,
        display_commercial_allowed=False,
    ),
    WebsiteCategory.BUCKET_4: CategoryDefinition(
        bucket_number=4,
        name="Full Access (Commercial Display Allowed)",
        scrap_allowed=True,
        store_allowed=True,
        display_free_allowed=True,
        display_commercial_allowed=True,
    ),
    WebsiteCategory.BUCKET_6: CategoryDefinition(
        bucket_number=6,
        name="Scrap with Uncertain Display",
        scrap_allowed=True,
        store_allowed=None,   # wildcard — LLM semantic judgment
        display_free_allowed=None,        # wildcard — LLM semantic judgment
        display_commercial_allowed=None,  # wildcard — LLM semantic judgment
    ),
    WebsiteCategory.BUCKET_7: CategoryDefinition(
        bucket_number=7,
        name="Scrap + Store with Uncertain Display",
        scrap_allowed=True,
        store_allowed=True,
        display_free_allowed=None,        # wildcard — LLM semantic judgment
        display_commercial_allowed=None,  # wildcard — LLM semantic judgment
    ),
    WebsiteCategory.BUCKET_8: CategoryDefinition(
        bucket_number=8,
        name="Fully Uncertain (All Axes ?)",
        scrap_allowed=None,
        store_allowed=None,
        display_free_allowed=None,
        display_commercial_allowed=None,
    ),
}


PARAM_TO_CATEGORY_FIELD = {
    "scrap": "scrap_allowed",
    "store": "store_allowed",
    "display_for_free": "display_free_allowed",
    "display_for_commercial": "display_commercial_allowed",
}
