"""Category bucket definitions for website classification."""
from enum import Enum
from typing import Dict
from dataclasses import dataclass

class WebsiteCategory(Enum):
    BUCKET_1 = 1
    BUCKET_2 = 2
    BUCKET_3 = 3
    BUCKET_4 = 4

@dataclass
class CategoryDefinition:
    bucket_number: int
    name: str
    scraping_allowed: bool
    storing_allowed: bool
    display_free_allowed: bool
    display_subscription_allowed: bool
    redistribute_free_allowed: bool
    redistribute_subscription_allowed: bool
    manual_collection_allowed: bool

CATEGORY_DEFINITIONS: Dict[WebsiteCategory, CategoryDefinition] = {
    WebsiteCategory.BUCKET_1: CategoryDefinition(
        bucket_number=1,
        name="Full Access",
        scraping_allowed=True,
        storing_allowed=True,
        display_free_allowed=True,
        display_subscription_allowed=True,
        redistribute_free_allowed=True,
        redistribute_subscription_allowed=True,
        manual_collection_allowed=True
    ),
    WebsiteCategory.BUCKET_2: CategoryDefinition(
        bucket_number=2,
        name="Display Only (No Redistribution)",
        scraping_allowed=True,
        storing_allowed=True,
        display_free_allowed=True,
        display_subscription_allowed=True,
        redistribute_free_allowed=False,
        redistribute_subscription_allowed=False,
        manual_collection_allowed=True
    ),
    WebsiteCategory.BUCKET_3: CategoryDefinition(
        bucket_number=3,
        name="Storage Only (No Display or Redistribution)",
        scraping_allowed=True,
        storing_allowed=True,
        display_free_allowed=False,
        display_subscription_allowed=False,
        redistribute_free_allowed=False,
        redistribute_subscription_allowed=False,
        manual_collection_allowed=True
    ),
    WebsiteCategory.BUCKET_4: CategoryDefinition(
        bucket_number=4,
        name="Manual Collection Only",
        scraping_allowed=False,
        storing_allowed=False,
        display_free_allowed=False,
        display_subscription_allowed=False,
        redistribute_free_allowed=False,
        redistribute_subscription_allowed=False,
        manual_collection_allowed=True
    ),
}

PARAM_TO_CATEGORY_FIELD = {
    "scraping": "scraping_allowed",
    "manual_collection": "manual_collection_allowed",
    "storing": "storing_allowed",
    "free_display": "display_free_allowed",
    "subscription_display": "display_subscription_allowed",
    "free_redistribute": "redistribute_free_allowed",
    "subscription_redistribute": "redistribute_subscription_allowed",
}
