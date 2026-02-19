"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_features_json():
    """A realistic features_json dict for testing."""
    return {
        "scientific_name": "Amanita muscaria",
        "common_names": ["Fly agaric"],
        "family": "Amanitaceae",
        "genus": "Amanita",
        "cap": {
            "shape": "convex to flat",
            "colors": ["red", "orange-red", "yellow-orange"],
            "surface_texture": "smooth with white wart-like scales",
            "scales_or_warts": "white universal veil remnants as patches",
            "diameter_min_cm": 8.0,
            "diameter_max_cm": 20.0,
        },
        "gills": {
            "hymenium_type": "gills",
            "attachment": "free",
            "spacing": "crowded",
            "color": "white",
            "color_young": "white",
        },
        "stem": {
            "height_min_cm": 8.0,
            "height_max_cm": 20.0,
            "diameter_min_cm": 1.0,
            "diameter_max_cm": 3.0,
            "color": "white",
            "surface_texture": "smooth to slightly fibrous",
            "consistency": "solid becoming hollow",
        },
        "veil": {
            "present": True,
            "type": "ring-like",
            "shape": "pendant, skirt-like",
            "mobility": "fixed",
            "color": "white",
            "brittleness": "persistent, membranous",
        },
        "volva": {
            "present": True,
            "type": "friable remnants",
            "shape": "rings of scales around bulbous base",
            "color": "white",
        },
        "flesh": {
            "color": "white",
            "bruising_color": None,
            "odor": "not distinctive",
        },
        "spore_print_color": "white",
        "overall_size_class": "large",
        "ecology": {
            "habitat_types": ["deciduous forest", "coniferous forest", "mixed woodland"],
            "substrate": "soil",
            "associated_trees": ["birch", "pine", "spruce"],
            "fruiting_seasons": ["late summer", "autumn"],
            "geographic_regions": ["Europe", "North America", "Asia"],
            "altitude_notes": "sea level to montane",
            "growth_pattern": "solitary or scattered",
            "growth_position": "ground level",
        },
        "edibility": "toxic",
        "known_toxins": ["ibotenic acid", "muscimol"],
        "known_lookalikes": ["Amanita caesarea"],
    }
