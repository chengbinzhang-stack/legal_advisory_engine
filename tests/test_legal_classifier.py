"""
Comprehensive tests for src/classifier/legal_classifier.py

New 4-permission schema:
  - scrap
  - store
  - display_for_free
  - display_for_commercial

Bucket profile matrix:
  | Bucket | scrap | store | display_for_free | display_for_commercial |
  | 1      | N     | N     | N                | N                      |
  | 2      | Y     | Y     | N                | N                      |
  | 3      | Y     | Y     | Y                | N                      |
  | 4      | Y     | Y     | Y                | Y                      |
  | 6      | Y     | ?     | ?                | ?                      |
  | 7      | Y     | Y     | ?                | ?                      |
  | 8      | ?     | ?     | ?                | ?                      |

Coverage:
  - All 4 permission params correctly classified
  - Buckets 1-4 definite cases
  - Buckets 6/7/8 uncertain with LLM judgment (second-pass)
  - Override logic (reasoning keywords)
  - Edge cases (empty input, malformed JSON, missing fields, etc.)
  - Heuristic fallback path
"""
import json
import pytest
from unittest.mock import MagicMock

from src.classifier.legal_classifier import (
    LegalClassifier,
    PERMISSION_PARAMS,
    SYSTEM_PROMPT,
    UNCERTAIN_RESOLUTION_PROMPT,
)
from src.models.legal_analysis import (
    LegalAnalysis,
    PermissionAnalysis,
    PermissionLevel,
)
# IMPORTANT: category_buckets defines the canonical WebsiteCategory enum
# with BUCKET_1..8 used by legal_classifier. The legal_analysis module
# also defines a WebsiteCategory but the classifier imports from
# category_buckets, so tests must import from there.
from src.classifier.category_buckets import (
    CATEGORY_DEFINITIONS,
    WebsiteCategory,
    PARAM_TO_CATEGORY_FIELD,
)


# =============================================================================
# Helpers
# =============================================================================

def make_param(permission: str, reasoning: str = "Test reasoning", excerpts=None):
    """Helper to build a single parameter block as the LLM would emit."""
    return {
        "permission": permission,
        "reasoning": reasoning,
        "reference_urls": [],
        "relevant_excerpts": excerpts or [],
    }


def make_full_llm_response(perms: dict, reference_urls=None, unique_findings=None):
    """Build a full LLM response covering all 4 params + metadata."""
    payload = {}
    for p in PERMISSION_PARAMS:
        payload[p] = perms.get(p, make_param("uncertain", "No info"))
    payload["reference_urls"] = reference_urls or []
    payload["unique_findings"] = unique_findings or []
    return json.dumps(payload)


def make_mock_llm_client(responses: list):
    """Build a mock LLM client whose .chat() returns successive responses."""
    client = MagicMock()
    client.chat.side_effect = list(responses)
    return client


# =============================================================================
# Fixtures: canned permission sets per bucket profile
# =============================================================================

@pytest.fixture
def bucket1_perms():
    """Bucket 1 profile: all 4 params not_allowed."""
    return {
        "scrap": make_param("not_allowed", "All rights reserved. No automated access."),
        "store": make_param("not_allowed", "Storing our content is prohibited."),
        "display_for_free": make_param("not_allowed", "Public display is not allowed."),
        "display_for_commercial": make_param("not_allowed",
                                             "Subscription display is not permitted."),
    }


@pytest.fixture
def bucket2_perms():
    """Bucket 2 profile: scrap+store allowed, display_for_free+display_for_commercial NOT."""
    return {
        "scrap": make_param("allowed", "You may scrape our pages."),
        "store": make_param("allowed", "You may store the data."),
        "display_for_free": make_param("not_allowed", "Public display is not allowed."),
        "display_for_commercial": make_param("not_allowed",
                                             "Subscription display is not permitted."),
    }


@pytest.fixture
def bucket3_perms():
    """Bucket 3 profile: scrap+store+display_for_free allowed, display_for_commercial NOT."""
    return {
        "scrap": make_param("allowed", "You may scrape."),
        "store": make_param("allowed", "You may cache the data."),
        "display_for_free": make_param("allowed", "You may display for free."),
        "display_for_commercial": make_param("not_allowed",
                                             "Paywalled display is prohibited."),
    }


@pytest.fixture
def bucket4_perms():
    """Bucket 4 profile: all 4 params allowed (Full Access)."""
    return {
        "scrap": make_param("allowed", "You may scrape our public pages for any purpose."),
        "store": make_param("allowed", "You may cache and store the data."),
        "display_for_free": make_param("allowed",
                                       "You may display content publicly for free."),
        "display_for_commercial": make_param("allowed",
                                             "Subscription paywalls are permitted."),
    }


@pytest.fixture
def bucket6_perms():
    """Bucket 6 profile: scrap Y, store + display + commercial uncertain (?).
    Bucket 6 requires scrap to be allowed; everything else must remain UNCERTAIN
    after the keyword-override pass so that the second-pass LLM call decides.
    """
    return {
        "scrap": make_param("allowed", "you may scrape"),
        "store": make_param("uncertain", "ambiguous signal"),
        "display_for_free": make_param("uncertain", "ambiguous signal"),
        "display_for_commercial": make_param("uncertain", "ambiguous signal"),
    }


@pytest.fixture
def bucket7_perms():
    """Bucket 7 profile: scrap Y, store Y, display + commercial uncertain (?)."""
    return {
        "scrap": make_param("allowed", "you may scrape"),
        "store": make_param("allowed", "you may store"),
        "display_for_free": make_param("uncertain", "ambiguous signal"),
        "display_for_commercial": make_param("uncertain", "ambiguous signal"),
    }


@pytest.fixture
def bucket8_perms():
    """Bucket 8 profile: all 4 axes uncertain (?)."""
    return {
        "scrap": make_param("uncertain", "ambiguous"),
        "store": make_param("uncertain", "ambiguous"),
        "display_for_free": make_param("uncertain", "ambiguous"),
        "display_for_commercial": make_param("uncertain", "ambiguous"),
    }


# =============================================================================
# Module-level constants
# =============================================================================

class TestModuleConstants:
    def test_permission_params_has_four_entries(self):
        assert len(PERMISSION_PARAMS) == 4

    def test_permission_params_includes_all_required(self):
        for name in (
            "scrap",
            "store",
            "display_for_free",
            "display_for_commercial",
        ):
            assert name in PERMISSION_PARAMS

    def test_system_prompt_mentions_every_parameter(self):
        for p in PERMISSION_PARAMS:
            assert p in SYSTEM_PROMPT, f"{p} missing from SYSTEM_PROMPT"

    def test_uncertain_resolution_prompt_mentions_every_parameter(self):
        for p in PERMISSION_PARAMS:
            assert p in UNCERTAIN_RESOLUTION_PROMPT, f"{p} missing from UNCERTAIN_RESOLUTION_PROMPT"

    def test_uncertain_resolution_prompt_requires_Y_or_N(self):
        assert "Y" in UNCERTAIN_RESOLUTION_PROMPT
        assert "N" in UNCERTAIN_RESOLUTION_PROMPT
        assert "decision" in UNCERTAIN_RESOLUTION_PROMPT

    def test_param_to_category_field_maps_all_params(self):
        for p in PERMISSION_PARAMS:
            assert p in PARAM_TO_CATEGORY_FIELD
            assert PARAM_TO_CATEGORY_FIELD[p].endswith("_allowed")

    def test_category_definitions_cover_buckets_1_4_6_7_8(self):
        assert WebsiteCategory.BUCKET_1 in CATEGORY_DEFINITIONS
        assert WebsiteCategory.BUCKET_2 in CATEGORY_DEFINITIONS
        assert WebsiteCategory.BUCKET_3 in CATEGORY_DEFINITIONS
        assert WebsiteCategory.BUCKET_4 in CATEGORY_DEFINITIONS
        assert WebsiteCategory.BUCKET_6 in CATEGORY_DEFINITIONS
        assert WebsiteCategory.BUCKET_7 in CATEGORY_DEFINITIONS
        assert WebsiteCategory.BUCKET_8 in CATEGORY_DEFINITIONS

    def test_bucket1_profile_all_false(self):
        d = CATEGORY_DEFINITIONS[WebsiteCategory.BUCKET_1]
        assert d.scrap_allowed is False
        assert d.store_allowed is False
        assert d.display_free_allowed is False
        assert d.display_commercial_allowed is False

    def test_bucket2_profile(self):
        d = CATEGORY_DEFINITIONS[WebsiteCategory.BUCKET_2]
        assert d.scrap_allowed is True
        assert d.store_allowed is True
        assert d.display_free_allowed is False
        assert d.display_commercial_allowed is False

    def test_bucket3_profile(self):
        d = CATEGORY_DEFINITIONS[WebsiteCategory.BUCKET_3]
        assert d.scrap_allowed is True
        assert d.store_allowed is True
        assert d.display_free_allowed is True
        assert d.display_commercial_allowed is False

    def test_bucket4_profile_all_true(self):
        d = CATEGORY_DEFINITIONS[WebsiteCategory.BUCKET_4]
        assert d.scrap_allowed is True
        assert d.store_allowed is True
        assert d.display_free_allowed is True
        assert d.display_commercial_allowed is True

    def test_bucket6_profile_scrap_only_definite(self):
        d = CATEGORY_DEFINITIONS[WebsiteCategory.BUCKET_6]
        assert d.scrap_allowed is True
        assert d.store_allowed is None
        assert d.display_free_allowed is None
        assert d.display_commercial_allowed is None

    def test_bucket7_profile_scrap_and_store_definite(self):
        d = CATEGORY_DEFINITIONS[WebsiteCategory.BUCKET_7]
        assert d.scrap_allowed is True
        assert d.store_allowed is True
        assert d.display_free_allowed is None
        assert d.display_commercial_allowed is None

    def test_bucket8_profile_all_wildcard(self):
        d = CATEGORY_DEFINITIONS[WebsiteCategory.BUCKET_8]
        assert d.scrap_allowed is None
        assert d.store_allowed is None
        assert d.display_free_allowed is None
        assert d.display_commercial_allowed is None


# =============================================================================
# Initialization
# =============================================================================

class TestInitialization:
    def test_default_no_llm_client(self):
        c = LegalClassifier()
        assert c.llm_client is None

    def test_with_minimax_api_key_creates_client(self):
        c = LegalClassifier(api_key="fake-key", base_url="https://example.com")
        assert c.llm_client is not None

    def test_with_gemini_provider_creates_client(self):
        c = LegalClassifier(
            provider="gemini",
            gemini_api_key="fake-key",
            gemini_model="gemini-2.5-flash",
        )
        assert c.llm_client is not None

    def test_gemini_provider_without_api_key_keeps_client_none(self):
        c = LegalClassifier(provider="gemini", gemini_api_key=None)
        assert c.llm_client is None


# =============================================================================
# Bucket 1-4: definite Y/N classifications
# =============================================================================

class TestBucketDefiniteCases:
    def test_bucket1_all_not_allowed(self, bucket1_perms):
        """All 4 params not_allowed -> Bucket 1 (No Access)."""
        client = make_mock_llm_client([make_full_llm_response(bucket1_perms)])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="All rights reserved. No automated access.",
            website_url="https://example.com",
            website_domain="example.com",
        )

        assert analysis.category == WebsiteCategory.BUCKET_1
        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.NOT_ALLOWED

    def test_bucket4_all_allowed(self, bucket4_perms):
        """All 4 params allowed -> Bucket 4 (Full Access)."""
        client = make_mock_llm_client([make_full_llm_response(bucket4_perms)])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="You may use this content freely.",
            website_url="https://example.com",
            website_domain="example.com",
        )

        assert analysis.category == WebsiteCategory.BUCKET_4
        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.ALLOWED

    def test_bucket2_scrap_store_only(self, bucket2_perms):
        """scrap+store allowed, no display -> Bucket 2."""
        client = make_mock_llm_client([make_full_llm_response(bucket2_perms)])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="You may scrape and store but not display.",
            website_url="https://example.com",
            website_domain="example.com",
        )

        assert analysis.category == WebsiteCategory.BUCKET_2
        assert analysis.permissions["scrap"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["store"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["display_for_free"].permission == PermissionLevel.NOT_ALLOWED
        assert analysis.permissions["display_for_commercial"].permission == PermissionLevel.NOT_ALLOWED

    def test_bucket3_free_display_only(self, bucket3_perms):
        """scrap+store+display_for_free allowed, no commercial -> Bucket 3."""
        client = make_mock_llm_client([make_full_llm_response(bucket3_perms)])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="You may display for free but commercial use is prohibited.",
            website_url="https://example.com",
            website_domain="example.com",
        )

        assert analysis.category == WebsiteCategory.BUCKET_3
        assert analysis.permissions["scrap"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["store"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["display_for_free"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["display_for_commercial"].permission == PermissionLevel.NOT_ALLOWED


# =============================================================================
# Permission level mapping (LLM string -> enum)
# =============================================================================

class TestPermissionLevelMapping:
    """The classifier accepts a variety of permission strings; verify normalization."""

    @pytest.mark.parametrize("raw,expected", [
        ("allowed", PermissionLevel.ALLOWED),
        ("not_allowed", PermissionLevel.NOT_ALLOWED),
        ("not permitted", PermissionLevel.NOT_ALLOWED),
        ("prohibited", PermissionLevel.NOT_ALLOWED),
        ("restricted", PermissionLevel.NOT_ALLOWED),
        ("uncertain", PermissionLevel.UNCERTAIN),
        ("unclear", PermissionLevel.UNCERTAIN),
        ("not applicable", PermissionLevel.NOT_APPLICABLE),
        ("n/a", PermissionLevel.NOT_APPLICABLE),
        ("allowed_with_attribution", PermissionLevel.ALLOWED),
        ("allowed_without_attribution", PermissionLevel.ALLOWED),
        ("not_allowed_with_attribution", PermissionLevel.NOT_ALLOWED),
    ])
    def test_perm_string_normalization(self, raw, expected):
        perms = {p: make_param(raw, "Some reasoning") for p in PERMISSION_PARAMS}
        client = make_mock_llm_client([make_full_llm_response(perms)])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == expected, f"failed for raw={raw}"


# =============================================================================
# Override logic: reasoning keywords should override LLM perm value
# =============================================================================

class TestKeywordOverride:
    """When the LLM's reasoning contains an allow/forbid keyword, that keyword
    should win over the LLM's stated permission value."""

    def test_forbid_keyword_overrides_allowed_label(self):
        """LLM says 'allowed' but reasoning contains 'prohibited' (no allow keyword)
        -> should be NOT_ALLOWED."""
        perms = {
            "scrap": make_param(
                "allowed",
                "Per the document, automated mass extraction is strictly "
                "prohibited for our API endpoints.",
            )
        }
        for p in PERMISSION_PARAMS:
            perms.setdefault(p, make_param("uncertain", "no information"))

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert analysis.permissions["scrap"].permission == PermissionLevel.NOT_ALLOWED

    def test_allow_keyword_overrides_not_allowed_label(self):
        """LLM says 'not_allowed' but reasoning contains 'you may' (no forbid keyword)
        -> should be ALLOWED."""
        perms = {
            "store": make_param(
                "not_allowed",
                "The terms note that you may cache and store the public "
                "pages for personal reference use.",
            )
        }
        for p in PERMISSION_PARAMS:
            perms.setdefault(p, make_param("uncertain", "no information"))

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert analysis.permissions["store"].permission == PermissionLevel.ALLOWED

    def test_uncertain_label_with_prohibit_keyword(self):
        """LLM says 'uncertain' and reasoning contains 'prohibited' (no allow keyword)
        -> NOT_ALLOWED via first-pass keyword override. No second-pass needed because
        the override fires before the uncertain-resolution path."""
        perms = {
            "store": make_param(
                "uncertain",
                "Storing content is not allowed; caching is prohibited.",
            )
        }
        for p in PERMISSION_PARAMS:
            if p == "store":
                continue
            perms[p] = make_param("allowed", "you may use this freely.")

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert analysis.permissions["store"].permission == PermissionLevel.NOT_ALLOWED
        # Only one LLM call expected since the keyword override handled store
        assert client.chat.call_count == 1

    def test_uncertain_label_with_allow_keyword(self):
        """LLM says 'uncertain' and reasoning contains 'you may' (no forbid keyword)
        -> ALLOWED without second-pass. Other params are definite (no second pass)."""
        perms = {
            "display_for_free": make_param(
                "uncertain",
                "Generally you may display the content publicly.",
            )
        }
        for p in PERMISSION_PARAMS:
            if p == "display_for_free":
                continue
            perms[p] = make_param("not_allowed", "this is prohibited per the document.")

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert analysis.permissions["display_for_free"].permission == PermissionLevel.ALLOWED
        assert client.chat.call_count == 1

    def test_both_allow_and_forbid_triggers_second_pass(self):
        """When both keyword types are present, the first-pass override is suppressed
        and the param stays UNCERTAIN, triggering the second-pass resolution."""
        perms = {
            "scrap": make_param(
                "uncertain",
                "Generally you may scrape, but mass crawling is prohibited "
                "and disallowed for the API endpoints.",
            )
        }
        for p in PERMISSION_PARAMS:
            if p == "scrap":
                continue
            perms[p] = make_param("allowed", "you may use this freely.")

        # First call: primary. Second call: resolution of scrap.
        second_pass = json.dumps({"decision": "Y", "reasoning": "Because ok."})
        client = make_mock_llm_client([make_full_llm_response(perms), second_pass])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        # LLM said "uncertain", both keywords present -> stays uncertain -> second pass fires
        assert analysis.permissions["scrap"].permission == PermissionLevel.ALLOWED
        assert client.chat.call_count == 2  # second pass invoked exactly once

    @pytest.mark.parametrize("keyword", [
        "explicitly prohibit",
        "expressly prohibit",
        "prohibited",
        "not allowed",
        "strictly forbidden",
        "is not permitted",
        "is prohibited",
        "prohibits",
        "forbidden",
        "disallowed",
        "without our express prior permission",
        "expressly reserved",
        "all rights reserved",
    ])
    def test_each_forbid_keyword_flips_allowed_to_not_allowed(self, keyword):
        # Reasoning must contain the forbid keyword WITHOUT any allow keyword,
        # otherwise the override is suppressed (both-keyword-present branch).
        perms = {p: make_param("allowed", f"Per the document, {keyword} for this use case.")
                 for p in PERMISSION_PARAMS}

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.NOT_ALLOWED, \
                f"keyword '{keyword}' failed to override"

    @pytest.mark.parametrize("keyword", [
        "you may",
        "you can",
        "is permitted",
        "are permitted",
        "is allowed",
        "are allowed",
        "grants the right",
        "hereby grants",
        "freely use",
        "no restriction",
        "open license",
    ])
    def test_each_allow_keyword_flips_not_allowed_to_allowed(self, keyword):
        perms = {p: make_param("not_allowed", f"By default {keyword} use this content for any purpose.")
                 for p in PERMISSION_PARAMS}

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.ALLOWED, \
                f"keyword '{keyword}' failed to override"


# =============================================================================
# Bucket 6/7/8: uncertain cases routed to second LLM pass
# =============================================================================

class TestUncertainResolution:
    """When a param is still UNCERTAIN after the keyword override, the classifier
    makes a second LLM call asking for a Y/N commercial-intent decision."""

    def test_uncertain_triggers_second_pass(self):
        perms = {
            p: make_param("uncertain", "no clear signal in document")
            for p in PERMISSION_PARAMS
        }
        first_response = make_full_llm_response(perms)

        # 4 second-pass decisions, all "Y"
        second_responses = [json.dumps({"decision": "Y", "reasoning": "Because commercial."})] * 4

        client = make_mock_llm_client([first_response] + second_responses)
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        # Should have made 1 + 4 = 5 calls
        assert client.chat.call_count == 5
        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.ALLOWED

    def test_uncertain_second_pass_N(self):
        perms = {
            p: make_param("uncertain", "ambiguous")
            for p in PERMISSION_PARAMS
        }
        first_response = make_full_llm_response(perms)

        second_responses = [json.dumps({"decision": "N", "reasoning": "Because not commercial."})] * 4

        client = make_mock_llm_client([first_response] + second_responses)
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.NOT_ALLOWED

    def test_mixed_first_pass_only_uncertain_routed(self):
        """A second LLM call should only happen for params still UNCERTAIN."""
        perms = {
            "scrap": make_param("allowed", "you may scrape"),
            "store": make_param("not_allowed", "prohibited"),
            # remaining 2 are uncertain
        }
        for p in PERMISSION_PARAMS:
            perms.setdefault(p, make_param("uncertain", "ambiguous"))

        first_response = make_full_llm_response(perms)
        # Only 2 second-pass decisions needed (display_for_free, display_for_commercial)
        second_responses = [json.dumps({"decision": "Y", "reasoning": "Because ok."})] * 2

        client = make_mock_llm_client([first_response] + second_responses)
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        # 1 first pass + 2 second passes
        assert client.chat.call_count == 3
        assert analysis.permissions["scrap"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["store"].permission == PermissionLevel.NOT_ALLOWED

    def test_unparseable_second_pass_keeps_uncertain(self):
        """If the second LLM returns garbage, param stays UNCERTAIN."""
        perms = {
            p: make_param("uncertain", "ambiguous")
            for p in PERMISSION_PARAMS
        }
        first_response = make_full_llm_response(perms)

        # Garbage second-pass responses
        second_responses = ["this is not JSON at all"] * 4

        client = make_mock_llm_client([first_response] + second_responses)
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.UNCERTAIN

    def test_second_pass_exception_keeps_uncertain(self):
        """If the second LLM call raises, param stays UNCERTAIN."""
        perms = {
            p: make_param("uncertain", "ambiguous")
            for p in PERMISSION_PARAMS
        }
        first_response = make_full_llm_response(perms)

        client = MagicMock()
        client.chat.side_effect = [
            first_response,                                       # 1st call: primary analysis
            *[RuntimeError("network down") for _ in range(4)],    # next 4 fail
        ]

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.UNCERTAIN

    def test_second_pass_keyword_override(self):
        """A 'prohibited' in the resolved 'Because ...' reasoning should flip a Y -> NOT_ALLOWED."""
        perms = {
            "scrap": make_param("uncertain", "first pass ambiguous")
        }
        for p in PERMISSION_PARAMS:
            perms.setdefault(p, make_param("uncertain", "no info"))

        first_response = make_full_llm_response(perms)
        # Second pass says Y, but its reasoning contains a prohibit keyword
        second_responses = [json.dumps({
            "decision": "Y",
            "reasoning": "Because scraping is generally permitted but mass extraction is prohibited.",
        })] * 4

        client = make_mock_llm_client([first_response] + second_responses)
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        # The resolved reasoning has a forbid keyword -> flip to NOT_ALLOWED
        assert analysis.permissions["scrap"].permission == PermissionLevel.NOT_ALLOWED

    def test_bucket6_with_second_pass_resolves(self, bucket6_perms):
        """Bucket 6 requires scrap=Y, store/display/commercial uncertain.
        After second-pass N for the wildcards, store should be NOT_ALLOWED which is
        consistent with bucket 4's NOT_ALLOWED profile but scrap/display behavior
        depends on how wildcards resolved. Verify all 4 params got second-pass."""
        first_response = make_full_llm_response(bucket6_perms)
        # 3 second-pass decisions (store, display_for_free, display_for_commercial)
        second_responses = [json.dumps({"decision": "N", "reasoning": "Because restricted."})] * 3

        client = make_mock_llm_client([first_response] + second_responses)
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert client.chat.call_count == 4
        assert analysis.permissions["scrap"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["store"].permission == PermissionLevel.NOT_ALLOWED
        assert analysis.permissions["display_for_free"].permission == PermissionLevel.NOT_ALLOWED
        assert analysis.permissions["display_for_commercial"].permission == PermissionLevel.NOT_ALLOWED

    def test_bucket8_all_uncertain_keeps_uncertain_when_no_llm_resolution(self, bucket8_perms):
        """Bucket 8 stays UNCERTAIN if second-pass also returns garbage."""
        first_response = make_full_llm_response(bucket8_perms)
        # Garbage second-pass
        second_responses = ["not json at all"] * 4

        client = make_mock_llm_client([first_response] + second_responses)
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.UNCERTAIN


# =============================================================================
# Category determination (bucket assignment) using fabricated permissions dicts
# =============================================================================

class TestCategoryDetermination:
    """Direct test of the bucketing algorithm using fabricated permissions dicts."""

    def test_bucket_1_perfect_match(self):
        c = LegalClassifier()
        perms = {
            p: PermissionAnalysis(parameter_name=p, permission=PermissionLevel.NOT_ALLOWED,
                                  reasoning="ok")
            for p in PERMISSION_PARAMS
        }
        cat, _ = c._determine_category(perms)
        assert cat == WebsiteCategory.BUCKET_1

    def test_bucket_2_perfect_match(self):
        c = LegalClassifier()
        perms = {
            "scrap": PermissionAnalysis(parameter_name="scrap",
                                        permission=PermissionLevel.ALLOWED, reasoning="ok"),
            "store": PermissionAnalysis(parameter_name="store",
                                        permission=PermissionLevel.ALLOWED, reasoning="ok"),
            "display_for_free": PermissionAnalysis(parameter_name="display_for_free",
                                                  permission=PermissionLevel.NOT_ALLOWED,
                                                  reasoning="ok"),
            "display_for_commercial": PermissionAnalysis(
                parameter_name="display_for_commercial",
                permission=PermissionLevel.NOT_ALLOWED, reasoning="ok"),
        }
        cat, _ = c._determine_category(perms)
        assert cat == WebsiteCategory.BUCKET_2

    def test_bucket_3_perfect_match(self):
        c = LegalClassifier()
        perms = {
            "scrap": PermissionAnalysis(parameter_name="scrap",
                                        permission=PermissionLevel.ALLOWED, reasoning="ok"),
            "store": PermissionAnalysis(parameter_name="store",
                                        permission=PermissionLevel.ALLOWED, reasoning="ok"),
            "display_for_free": PermissionAnalysis(parameter_name="display_for_free",
                                                  permission=PermissionLevel.ALLOWED,
                                                  reasoning="ok"),
            "display_for_commercial": PermissionAnalysis(
                parameter_name="display_for_commercial",
                permission=PermissionLevel.NOT_ALLOWED, reasoning="ok"),
        }
        cat, _ = c._determine_category(perms)
        assert cat == WebsiteCategory.BUCKET_3

    def test_bucket_4_perfect_match(self):
        c = LegalClassifier()
        perms = {
            p: PermissionAnalysis(parameter_name=p, permission=PermissionLevel.ALLOWED,
                                  reasoning="ok")
            for p in PERMISSION_PARAMS
        }
        cat, _ = c._determine_category(perms)
        assert cat == WebsiteCategory.BUCKET_4

    def test_bucket_6_uncertain_axes_surface_when_no_resolution(self):
        """If only scrap is allowed and the rest are UNCERTAIN, bucket 6 (most wildcards
        among definite-match candidates) should win because of the wildcard tie-break."""
        c = LegalClassifier()
        perms = {
            "scrap": PermissionAnalysis(parameter_name="scrap",
                                        permission=PermissionLevel.ALLOWED, reasoning="ok"),
            "store": PermissionAnalysis(parameter_name="store",
                                        permission=PermissionLevel.UNCERTAIN, reasoning="ok"),
            "display_for_free": PermissionAnalysis(parameter_name="display_for_free",
                                                  permission=PermissionLevel.UNCERTAIN,
                                                  reasoning="ok"),
            "display_for_commercial": PermissionAnalysis(
                parameter_name="display_for_commercial",
                permission=PermissionLevel.UNCERTAIN, reasoning="ok"),
        }
        cat, _ = c._determine_category(perms)
        # Bucket 6 has scrap=Y + 3 wildcards. Bucket 7 has scrap=Y, store=expected Y (mismatch)
        # so B6 wins the score-and-wildcard tie-break.
        assert cat == WebsiteCategory.BUCKET_6

    def test_bucket_7_uncertain_axes_surface_when_no_resolution(self):
        """If scrap+store are allowed and the display axes are UNCERTAIN, bucket 7 wins."""
        c = LegalClassifier()
        perms = {
            "scrap": PermissionAnalysis(parameter_name="scrap",
                                        permission=PermissionLevel.ALLOWED, reasoning="ok"),
            "store": PermissionAnalysis(parameter_name="store",
                                        permission=PermissionLevel.ALLOWED, reasoning="ok"),
            "display_for_free": PermissionAnalysis(parameter_name="display_for_free",
                                                  permission=PermissionLevel.UNCERTAIN,
                                                  reasoning="ok"),
            "display_for_commercial": PermissionAnalysis(
                parameter_name="display_for_commercial",
                permission=PermissionLevel.UNCERTAIN, reasoning="ok"),
        }
        cat, _ = c._determine_category(perms)
        assert cat == WebsiteCategory.BUCKET_7

    def test_bucket_8_all_uncertain(self):
        """All 4 axes UNCERTAIN -> bucket 8 (all wildcards = highest wildcards)."""
        c = LegalClassifier()
        perms = {
            p: PermissionAnalysis(parameter_name=p, permission=PermissionLevel.UNCERTAIN,
                                  reasoning="ok")
            for p in PERMISSION_PARAMS
        }
        cat, _ = c._determine_category(perms)
        # All wildcards tie on score (0) but bucket 8 has the most wildcards (4) -> wins
        assert cat == WebsiteCategory.BUCKET_8

    def test_missing_param_does_not_crash(self):
        c = LegalClassifier()
        perms = {
            p: PermissionAnalysis(parameter_name=p, permission=PermissionLevel.ALLOWED,
                                  reasoning="ok")
            for p in PERMISSION_PARAMS
            if p != "scrap"
        }
        cat, _ = c._determine_category(perms)
        # Just check it doesn't throw
        assert cat in WebsiteCategory

    def test_tie_break_prefers_more_wildcards(self):
        """Score tie -> bucket with more wildcards wins."""
        c = LegalClassifier()
        perms = {
            "scrap": PermissionAnalysis(parameter_name="scrap",
                                        permission=PermissionLevel.UNCERTAIN, reasoning="ok"),
            "store": PermissionAnalysis(parameter_name="store",
                                        permission=PermissionLevel.UNCERTAIN, reasoning="ok"),
            "display_for_free": PermissionAnalysis(parameter_name="display_for_free",
                                                  permission=PermissionLevel.UNCERTAIN,
                                                  reasoning="ok"),
            "display_for_commercial": PermissionAnalysis(
                parameter_name="display_for_commercial",
                permission=PermissionLevel.UNCERTAIN, reasoning="ok"),
        }
        cat, _ = c._determine_category(perms)
        # All buckets score 0 on UNCERTAIN, but bucket 8 has 4 wildcards > others -> B8 wins
        assert cat == WebsiteCategory.BUCKET_8


# =============================================================================
# Edge cases: empty / missing inputs
# =============================================================================

class TestEdgeCases:
    def test_no_llm_client_falls_back_to_heuristic(self):
        c = LegalClassifier()
        assert c.llm_client is None

        analysis = c.classify_permissions(
            text="You may scrape this content. Redistribution is prohibited.",
            website_url="https://example.com",
            website_domain="example.com",
        )

        # Heuristic should still return a valid LegalAnalysis with all 4 params
        assert isinstance(analysis, LegalAnalysis)
        assert set(analysis.permissions.keys()) == set(PERMISSION_PARAMS)
        assert len(analysis.permissions) == 4

    def test_empty_text(self):
        perms = {p: make_param("uncertain", "no info") for p in PERMISSION_PARAMS}
        client = make_mock_llm_client([make_full_llm_response(perms)])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="", website_url="https://e.com", website_domain="e.com"
        )

        assert isinstance(analysis, LegalAnalysis)
        assert len(analysis.permissions) == 4

    def test_missing_param_in_llm_response_defaults_to_uncertain(self):
        """If the LLM omits a param, the classifier must still produce an entry for it."""
        partial = json.dumps({
            "scrap": make_param("allowed", "you may scrape"),
            "reference_urls": [],
            "unique_findings": [],
        })

        client = make_mock_llm_client([partial])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        # All 4 should still be present
        assert len(analysis.permissions) == 4
        for p in PERMISSION_PARAMS:
            assert p in analysis.permissions

    def test_malformed_json_response_falls_back(self):
        """Garbage LLM response should trigger heuristic fallback."""
        client = make_mock_llm_client(["this is completely not JSON {{{"])

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        # Heuristic fallback path returns a LegalAnalysis
        assert isinstance(analysis, LegalAnalysis)
        assert len(analysis.permissions) == 4

    def test_llm_exception_falls_back_to_heuristic(self):
        client = MagicMock()
        client.chat.side_effect = RuntimeError("API error")

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="You may scrape and store this content. Redistribution is prohibited.",
            website_url="https://e.com",
            website_domain="e.com",
        )

        assert isinstance(analysis, LegalAnalysis)
        assert len(analysis.permissions) == 4
        # The reasoning should mention the LLM failure
        assert any("LLM unavailable" in p.reasoning for p in analysis.permissions.values())

    def test_document_urls_default_to_empty_dict(self):
        perms = {p: make_param("uncertain", "no info") for p in PERMISSION_PARAMS}
        client = make_mock_llm_client([make_full_llm_response(perms)])

        c = LegalClassifier()
        c.llm_client = client

        # Call WITHOUT document_urls argument
        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )
        assert isinstance(analysis, LegalAnalysis)

    def test_relevant_excerpts_handled_when_dict(self):
        perms = {
            "scrap": make_param("uncertain", "no info", excerpts=[
                {"text": "you may scrape", "source": "https://e.com/terms"}
            ])
        }
        for p in PERMISSION_PARAMS:
            perms.setdefault(p, make_param("uncertain", "no info"))

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert len(analysis.permissions["scrap"].relevant_excerpts) == 1
        assert analysis.permissions["scrap"].relevant_excerpts[0]["text"] == "you may scrape"
        assert analysis.permissions["scrap"].relevant_excerpts[0]["source"] == "https://e.com/terms"

    def test_relevant_excerpts_handled_when_string(self):
        perms = {
            "scrap": make_param("uncertain", "no info",
                                excerpts=["plain string excerpt"])
        }
        for p in PERMISSION_PARAMS:
            perms.setdefault(p, make_param("uncertain", "no info"))

        client = make_mock_llm_client([make_full_llm_response(perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert len(analysis.permissions["scrap"].relevant_excerpts) == 1
        assert analysis.permissions["scrap"].relevant_excerpts[0]["text"] == "plain string excerpt"
        assert analysis.permissions["scrap"].relevant_excerpts[0]["source"] == ""

    def test_all_four_params_present_after_classification(self, bucket4_perms):
        """Sanity: every classification result must expose all 4 permission params."""
        client = make_mock_llm_client([make_full_llm_response(bucket4_perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert "scrap" in analysis.permissions
        assert "store" in analysis.permissions
        assert "display_for_free" in analysis.permissions
        assert "display_for_commercial" in analysis.permissions


# =============================================================================
# JSON parsing edge cases
# =============================================================================

class TestParseLLMResponse:
    def test_direct_json(self):
        c = LegalClassifier()
        out = c._parse_llm_response('{"decision": "Y", "reasoning": "ok"}')
        assert out == {"decision": "Y", "reasoning": "ok"}

    def test_markdown_wrapped_json(self):
        c = LegalClassifier()
        out = c._parse_llm_response('```json\n{"decision": "N"}\n```')
        assert out == {"decision": "N"}

    def test_json_embedded_in_prose(self):
        c = LegalClassifier()
        out = c._parse_llm_response('Here is the result: {"decision": "Y"} thanks!')
        assert out == {"decision": "Y"}

    def test_unparseable_returns_empty(self):
        c = LegalClassifier()
        out = c._parse_llm_response("not json at all")
        assert out == {}


# =============================================================================
# Confidence score logic
# =============================================================================

class TestConfidenceScore:
    def test_allowed_has_high_confidence(self, bucket4_perms):
        client = make_mock_llm_client([make_full_llm_response(bucket4_perms)])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )
        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].confidence_score >= 0.9

    def test_uncertain_has_low_confidence(self):
        perms = {p: make_param("uncertain", "no info") for p in PERMISSION_PARAMS}
        # 1 first pass + 4 second passes that all return unparseable
        responses = [make_full_llm_response(perms)] + ["not json" for _ in range(4)]
        client = make_mock_llm_client(responses)

        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )
        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.UNCERTAIN
            assert analysis.permissions[p].confidence_score <= 0.5


# =============================================================================
# Unique findings
# =============================================================================

class TestUniqueFindings:
    def test_unique_findings_propagated(self):
        perms = {p: make_param("uncertain", "no info") for p in PERMISSION_PARAMS}
        response = make_full_llm_response(
            perms,
            unique_findings=["Found indicators of API Access",
                             "Found indicators of DMCA Protected"],
        )
        client = make_mock_llm_client([response])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert "Found indicators of API Access" in analysis.unique_findings
        assert "Found indicators of DMCA Protected" in analysis.unique_findings

    def test_reference_urls_collected(self):
        perms = {p: make_param("uncertain", "no info") for p in PERMISSION_PARAMS}
        response = make_full_llm_response(
            perms,
            reference_urls=["https://e.com/terms", "https://e.com/privacy"],
        )
        client = make_mock_llm_client([response])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://e.com", website_domain="e.com"
        )

        assert "https://e.com/terms" in analysis.summary_text
        assert "https://e.com/privacy" in analysis.summary_text


# =============================================================================
# Heuristic fallback path (no LLM)
# =============================================================================

class TestHeuristicFallback:
    def test_heuristic_detects_prohibition_keywords(self):
        c = LegalClassifier()
        # display_for_free forbidden list: ["cannot display", "display prohibited",
        # "no public display"]. Allowed: ["display", "show", "view", "public", "free use"].
        # The negation-context detector fires on "shall not display" — the "display"
        # allow keyword is preceded by "shall not" within 80 chars, so it counts as
        # a negation hit. Net: forbid_count += negation_count, allow_count -= negation_count.
        text = "You shall not display our content publicly. Users shall not display this material."
        analysis = c._classify_heuristic(
            text=text,
            website_url="https://e.com",
            website_domain="e.com",
        )

        assert analysis.permissions["display_for_free"].permission == PermissionLevel.NOT_ALLOWED

    def test_heuristic_detects_store_prohibition(self):
        c = LegalClassifier()
        # store forbidden list contains "no storage" / "no caching"
        text = "no storage or archiving is permitted on our endpoints."
        analysis = c._classify_heuristic(
            text=text,
            website_url="https://e.com",
            website_domain="e.com",
        )

        assert analysis.permissions["store"].permission == PermissionLevel.NOT_ALLOWED

    def test_heuristic_detects_permission_keywords(self):
        c = LegalClassifier()
        text = "You may scrape and store. Display is allowed. Redistribution permitted."
        analysis = c._classify_heuristic(
            text=text,
            website_url="https://e.com",
            website_domain="e.com",
        )

        assert analysis.permissions["scrap"].permission == PermissionLevel.ALLOWED
        assert analysis.permissions["store"].permission == PermissionLevel.ALLOWED

    def test_heuristic_detects_negation_context(self):
        """The heuristic must recognize 'shall not X' patterns as prohibition via
        the negation-context detector (count_negation_contexts).
        The detector scans 80 chars BEFORE each allow-keyword hit for negation patterns
        such as 'shall not <word>'. The allow keyword itself must come AFTER 'not'."""
        c = LegalClassifier()
        # 'publicly' is an allow keyword. 'shall not' precedes it within 80 chars.
        # The detector matches r'shall\s+not\s+\w+' against the 80-char context
        # preceding 'publicly', which contains '...shall not...'.
        text = "These terms shall not be construed as granting any right to display our content publicly."
        analysis = c._classify_heuristic(
            text=text,
            website_url="https://e.com",
            website_domain="e.com",
        )

        # 'publicly' preceded by 'shall not' -> negation hit -> flips to NOT_ALLOWED
        assert analysis.permissions["display_for_free"].permission == PermissionLevel.NOT_ALLOWED

    def test_heuristic_no_signal_marks_uncertain(self):
        c = LegalClassifier()
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        analysis = c._classify_heuristic(
            text=text,
            website_url="https://e.com",
            website_domain="e.com",
        )

        # No keywords -> UNCERTAIN for all 4
        for p in PERMISSION_PARAMS:
            assert analysis.permissions[p].permission == PermissionLevel.UNCERTAIN

    def test_heuristic_returns_valid_legal_analysis(self):
        c = LegalClassifier()
        analysis = c._classify_heuristic(
            text="You may scrape. Display prohibited.",
            website_url="https://e.com",
            website_domain="e.com",
        )

        assert isinstance(analysis, LegalAnalysis)
        assert len(analysis.permissions) == 4
        # Must return a category that is one of the 7 buckets (1-4, 6-8)
        assert analysis.category in (
            WebsiteCategory.BUCKET_1,
            WebsiteCategory.BUCKET_2,
            WebsiteCategory.BUCKET_3,
            WebsiteCategory.BUCKET_4,
            WebsiteCategory.BUCKET_6,
            WebsiteCategory.BUCKET_7,
            WebsiteCategory.BUCKET_8,
        )


# =============================================================================
# Integration: end-to-end pipeline on a fabricated doc
# =============================================================================

class TestEndToEndIntegration:
    def test_full_pipeline_with_realistic_doc(self):
        """Simulate an LLM output for a permissive site with mixed signals."""
        perms = {
            "scrap": make_param("allowed", "you may scrape public pages"),
            "store": make_param("allowed", "you can cache the content"),
            "display_for_free": make_param("allowed", "freely use under open license"),
            "display_for_commercial": make_param(
                "not_allowed",
                "the document does not address paywalled display; "
                "redistribution for commercial resale is prohibited",
            ),
        }
        response = make_full_llm_response(perms)
        client = make_mock_llm_client([response])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="Wikipedia-like terms of service",
            website_url="https://wikipedia.org",
            website_domain="wikipedia.org",
        )

        # Verify structure
        assert analysis.website_domain == "wikipedia.org"
        assert analysis.category is not None
        # All 4 params populated
        assert len(analysis.permissions) == 4
        # Override flipped display_for_commercial to NOT_ALLOWED because reasoning has 'prohibited'
        assert analysis.permissions["display_for_commercial"].permission == PermissionLevel.NOT_ALLOWED
        # Allowed params stayed allowed
        assert analysis.permissions["scrap"].permission == PermissionLevel.ALLOWED
        # Category should be one of the 7 buckets
        assert analysis.category in (
            WebsiteCategory.BUCKET_1,
            WebsiteCategory.BUCKET_2,
            WebsiteCategory.BUCKET_3,
            WebsiteCategory.BUCKET_4,
            WebsiteCategory.BUCKET_6,
            WebsiteCategory.BUCKET_7,
            WebsiteCategory.BUCKET_8,
        )

    def test_summary_text_built(self):
        perms = {p: make_param("uncertain", "no info") for p in PERMISSION_PARAMS}
        response = make_full_llm_response(perms)
        client = make_mock_llm_client([response])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://example.com", website_domain="example.com"
        )

        assert "example.com" in analysis.summary_text
        assert "Category: Bucket" in analysis.summary_text

    def test_category_reasoning_includes_bucket_name(self, bucket4_perms):
        perms_response = make_full_llm_response(bucket4_perms)
        client = make_mock_llm_client([perms_response])
        c = LegalClassifier()
        c.llm_client = client

        analysis = c.classify_permissions(
            text="x", website_url="https://example.com", website_domain="example.com"
        )

        # The reasoning should mention the bucket's name
        assert CATEGORY_DEFINITIONS[analysis.category].name in analysis.category_reasoning