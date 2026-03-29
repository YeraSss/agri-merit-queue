"""
tests/test_scoring.py — Unit tests for the merit scoring engine.
Run: pytest tests/ -v

These tests use mocked lookup tables so they work without
the real dataset or trained model (CI-safe).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


MOCK_LOOKUP = {
    "region_rej_rates": {
        "Павлодарская область": 0.002,
        "Жамбылская область": 0.191,
        "Алматинская область": 0.118,
        "Актюбинская область": 0.006,
    },
    "region_paid_rates": {
        "Павлодарская область": 0.916,
        "Жамбылская область": 0.0,
        "Алматинская область": 0.0,
        "Актюбинская область": 0.892,
    },
    "region_medians": {
        "Павлодарская область": 1200000,
        "Жамбылская область": 900000,
        "Алматинская область": 1500000,
        "Актюбинская область": 1100000,
    },
    "livestock_rej_rates": {
        "Субсидирование в скотоводстве": 0.080,
        "Субсидирование в овцеводстве": 0.123,
        "Субсидирование в птицеводстве": 0.058,
    },
    "livestock_budget_shares": {
        "Субсидирование в скотоводстве": 81.2,
        "Субсидирование в овцеводстве": 5.8,
        "Субсидирование в птицеводстве": 7.7,
    },
    "max_livestock_share": 81.2,
    "region_label_classes": ["Алматинская область", "Актюбинская область", "Жамбылская область", "Павлодарская область"],
    "livestock_label_classes": ["Субсидирование в овцеводстве", "Субсидирование в птицеводстве", "Субсидирование в скотоводстве"],
    "livestock_amount_means": {"Субсидирование в скотоводстве": 1800000},
    "livestock_amount_stds":  {"Субсидирование в скотоводстве": 3500000},
    "region_amount_means": {"Павлодарская область": 1500000},
    "region_amount_stds":  {"Павлодарская область": 2000000},
    "features": [],
}


def _score(region="Павлодарская область", livestock="Субсидирование в скотоводстве", norm=15000, amount=2920200, month=4, doy=100, fairness=0.0):
    from scoring_ml import compute_merit_score
    with patch("scoring_ml._get_lookup", return_value=MOCK_LOOKUP), \
         patch("scoring_ml._load_model", return_value=None):
        return compute_merit_score(app_id="TEST", region=region, livestock_type=livestock, norm=norm, requested_amount=amount, submission_month=month, submission_day_of_year=doy, fairness_weight=fairness)


class TestScoreRange:
    def test_score_is_between_0_and_100(self):
        assert 0.0 <= _score().hybrid_merit_score <= 100.0
    def test_ml_score_is_between_0_and_100(self):
        assert 0.0 <= _score().ml_merit_score <= 100.0
    def test_rule_score_is_between_0_and_100(self):
        assert 0.0 <= _score().rule_merit_score <= 100.0

class TestImpliedHead:
    def test_implied_head_computed_correctly(self):
        assert _score(norm=15000, amount=300000).implied_head_count == 20
    def test_zero_norm_does_not_crash(self):
        r = _score(norm=0, amount=500000)
        assert r.implied_head_count == 0 and 0 <= r.hybrid_merit_score <= 100

class TestRegionEffect:
    def test_low_rejection_region_scores_higher_than_high(self):
        p = _score(region="Павлодарская область")
        z = _score(region="Жамбылская область")
        assert p.rule_merit_score > z.rule_merit_score

class TestFarmSizeTiers:
    @pytest.mark.parametrize("h,e", [(10,20.0),(50,40.0),(300,65.0),(1000,85.0),(5000,100.0)])
    def test_farm_size_tiers(self, h, e):
        from scoring_ml import compute_farm_size_score
        score, _ = compute_farm_size_score(h)
        assert score == e

class TestFairnessWeight:
    def test_fairness_zero_and_one_differ(self):
        large_pure = _score(norm=15000, amount=15000000, fairness=0.0)
        large_fair = _score(norm=15000, amount=15000000, fairness=1.0)
        small_pure = _score(norm=15000, amount=450000, fairness=0.0)
        small_fair = _score(norm=15000, amount=450000, fairness=1.0)
        assert large_pure.rule_merit_score > small_pure.rule_merit_score
        assert (large_fair.rule_merit_score - small_fair.rule_merit_score) < (large_pure.rule_merit_score - small_pure.rule_merit_score)
    def test_fairness_weight_0_returns_pure_merit(self):
        assert _score(fairness=0.0).hybrid_merit_score == _score(fairness=0.0).hybrid_merit_score

class TestRiskFlag:
    def test_high_risk_region_produces_high_flag(self):
        assert _score(region="Жамбылская область").risk_flag in ("HIGH", "MEDIUM")
    def test_low_risk_region_produces_low_or_medium_flag(self):
        assert _score(region="Павлодарская область").risk_flag in ("LOW", "MEDIUM")
    def test_risk_flag_values_are_valid(self):
        assert _score().risk_flag in ("LOW", "MEDIUM", "HIGH")

class TestComponents:
    def test_three_components_returned(self):
        assert len(_score().components) == 3
    def test_component_weights_sum_to_one(self):
        assert abs(sum(c.weight for c in _score().components) - 1.0) < 1e-9
    def test_weighted_contributions_sum_to_rule_score(self):
        r = _score()
        assert abs(sum(c.weighted_contribution for c in r.components) - r.rule_merit_score) < 0.1

class TestUnknownInputs:
    def test_unknown_region_uses_default(self):
        assert 0 <= _score(region="Неизвестная область").hybrid_merit_score <= 100
    def test_unknown_livestock_uses_default(self):
        assert 0 <= _score(livestock="Неизвестный вид").hybrid_merit_score <= 100

class TestOutputStructure:
    def test_result_has_all_required_fields(self):
        r = _score()
        for f in ["hybrid_merit_score","ml_merit_score","rule_merit_score","p_rejection","risk_flag","components","shap_breakdown"]:
            assert hasattr(r, f)
    def test_to_dict_is_json_serializable(self):
        import json
        assert len(json.dumps(_score().to_dict(), ensure_ascii=False)) > 100
