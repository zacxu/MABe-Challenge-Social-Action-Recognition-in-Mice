"""Single public train/inference feature-builder entry point."""

from __future__ import annotations

from mabe.features.common import CategoryRegistry, FeatureConfig, FeatureTable, MetadataContext
from mabe.features.pair import build_pair_features
from mabe.features.solo import build_solo_features
from mabe.pose import CleanPose


def build_feature_tables(
    pose: CleanPose,
    context: MetadataContext,
    config: FeatureConfig,
    categories: CategoryRegistry,
) -> tuple[FeatureTable, FeatureTable]:
    """Build solo and ordered-pair features without labels or split-specific branches."""
    solo = build_solo_features(pose, context, config, categories)
    pair = build_pair_features(pose, context, config, categories)
    return solo, pair

