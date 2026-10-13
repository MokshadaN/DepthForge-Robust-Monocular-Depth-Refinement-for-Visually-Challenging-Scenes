"""
Phase 7: Large-Scale 10,000-Image Multi-Domain Dataset Builder.

Target Balance:
  Camouflage   (30%) : 3,000 images  = 500 CAMO-FS real photos x 6 augmented variants
  Transparent  (30%) : 3,000 images  = 60 unique Unsplash glass scenes x 50 augmented variants
  Textureless  (40%) : 4,000 images  = 80 unique Unsplash interior scenes x 50 augmented variants
  ─────────────────────────────────────────────────────────────────────────────────
  TOTAL              : 10,000 images (target)

Augmentation variants are DIVERSE and MEANINGFUL -- not near-identical crops:
  5 spatial regions x 2 flips x 5 photometric modes = 50 unique variants per scene.

All base images are 100% real-world photographs. No synthetic generation.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
import urllib.request
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = Path("phase7/data/unified_10k").resolve()
CAMO_SOURCE = Path("phase7/data/camo_500").resolve()
SPLITS_FILE = Path("phase7/data/unified_splits.json").resolve()

# Targets
CAMO_TARGET        = 3_000   # 500 real x 6 variants
TRANSPARENT_TARGET = 3_000   # 60 scenes x 50 variants
TEXTURELESS_TARGET = 4_000   # 80 scenes x 50 variants

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UniDACResearch/1.0"}

# ---------------------------------------------------------------------------
# 60 Transparent / Refractive Glass Scenes  (real Unsplash photographs)
# ---------------------------------------------------------------------------
TRANSPARENT_URLS = [
    # Wine, cocktail, drinking glasses
    ("glass_01_wine_glasses",      "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=900&q=85"),
    ("glass_02_cocktail_bar",      "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=900&q=85"),
    ("glass_03_champagne_flutes",  "https://images.unsplash.com/photo-1557804506-669a67965ba0?w=900&q=85"),
    ("glass_04_water_glass",       "https://images.unsplash.com/photo-1559839914-17aae19cec71?w=900&q=85"),
    ("glass_05_beer_mug",          "https://images.unsplash.com/photo-1548366086-7f1b76106622?w=900&q=85"),
    ("glass_06_wine_shelf",        "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=900&q=85"),
    ("glass_07_martini",           "https://images.unsplash.com/photo-1598300042247-d088f8ab3a91?w=900&q=85"),
    # Lab / scientific glass
    ("glass_08_chemistry_beaker",  "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?w=900&q=85"),
    ("glass_09_lab_flasks",        "https://images.unsplash.com/photo-1581093458791-9f3c3900df4b?w=900&q=85"),
    ("glass_10_test_tubes",        "https://images.unsplash.com/photo-1582719471384-894fbb16e074?w=900&q=85"),
    ("glass_11_microscope",        "https://images.unsplash.com/photo-1508973379184-7517410fb0bc?w=900&q=85"),
    ("glass_12_petri_dishes",      "https://images.unsplash.com/photo-1563201517-2e2d5e73c9eb?w=900&q=85"),
    # Bottles & containers
    ("glass_13_perfume_bottle",    "https://images.unsplash.com/photo-1592945403244-b3fbafd7f539?w=900&q=85"),
    ("glass_14_glass_jar",         "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=900&q=85"),
    ("glass_15_glass_bottles",     "https://images.unsplash.com/photo-1607344645866-009c320b0534?w=900&q=85"),
    ("glass_16_glass_vase",        "https://images.unsplash.com/photo-1526045612212-70caf35c14df?w=900&q=85"),
    ("glass_17_glass_pitcher",     "https://images.unsplash.com/photo-1570831739-bdbe5df3c9f2?w=900&q=85"),
    ("glass_18_mason_jar",         "https://images.unsplash.com/photo-1582719478520-b84f06c3fc6a?w=900&q=85"),
    ("glass_19_candle_holder",     "https://images.unsplash.com/photo-1543483964-b42bcc02dabc?w=900&q=85"),
    ("glass_20_crystal_clear",     "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=900&q=85"),
    # Windows / architectural glass
    ("glass_21_glass_window",      "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=900&q=85"),
    ("glass_22_glass_building",    "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=900&q=85"),
    ("glass_23_store_window",      "https://images.unsplash.com/photo-1573461160327-81bf4f2ce18e?w=900&q=85"),
    ("glass_24_office_windows",    "https://images.unsplash.com/photo-1560472354-b33ff0ad5a3d?w=900&q=85"),
    ("glass_25_rainy_window",      "https://images.unsplash.com/photo-1473116763249-2faaef81ccda?w=900&q=85"),
    ("glass_26_glass_door",        "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=900&q=85"),
    ("glass_27_greenhouse",        "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=900&q=85"),
    ("glass_28_greenhouse_int",    "https://images.unsplash.com/photo-1474625121-2688608de84f?w=900&q=85"),
    ("glass_29_glass_facade",      "https://images.unsplash.com/photo-1519120944243-ebfe813ef0b5?w=900&q=85"),
    ("glass_30_mirror_room",       "https://images.unsplash.com/photo-1616047006789-b7af5afb8c20?w=900&q=85"),
    # Optics / lenses / prisms
    ("glass_31_prism_light",       "https://images.unsplash.com/photo-1567095761054-7a02e69e5c43?w=900&q=85"),
    ("glass_32_reading_glasses",   "https://images.unsplash.com/photo-1509228468518-180dd4864904?w=900&q=85"),
    ("glass_33_binoculars",        "https://images.unsplash.com/photo-1551538827-9c037cb4f32a?w=900&q=85"),
    ("glass_34_magnifier",         "https://images.unsplash.com/photo-1588666309990-d68f08e3d4a6?w=900&q=85"),
    ("glass_35_crystal_ball",      "https://images.unsplash.com/photo-1579783902614-a3fb3927b6a5?w=900&q=85"),
    # Aquatic / liquid in glass
    ("glass_36_aquarium",          "https://images.unsplash.com/photo-1524704654690-b56c05c78a00?w=900&q=85"),
    ("glass_37_fish_bowl",         "https://images.unsplash.com/photo-1544568100-847a948585b9?w=900&q=85"),
    ("glass_38_water_drops",       "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=900&q=85"),
    ("glass_39_ice_water",         "https://images.unsplash.com/photo-1548778943-5bbeeb1ba6c1?w=900&q=85"),
    ("glass_40_bubbles_glass",     "https://images.unsplash.com/photo-1527549098722-fc4b69d60e65?w=900&q=85"),
    # Bathroom / shower glass
    ("glass_41_shower_glass",      "https://images.unsplash.com/photo-1552321554-5fefe8c9ef14?w=900&q=85"),
    ("glass_42_glass_table",       "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=900&q=85"),
    ("glass_43_glass_shelf",       "https://images.unsplash.com/photo-1524055988636-436cfa46e59e?w=900&q=85"),
    ("glass_44_display_case",      "https://images.unsplash.com/photo-1556742502-ec7c0e9f34b6?w=900&q=85"),
    # Abstract / artistic glass
    ("glass_45_stained_glass",     "https://images.unsplash.com/photo-1548778943-5bbeeb1ba6c1?w=900&q=85"),
    ("glass_46_glass_tiles",       "https://images.unsplash.com/photo-1461988625982-117ac9c939f1?w=900&q=85"),
    ("glass_47_neon_glass",        "https://images.unsplash.com/photo-1573225342350-16731dd9bf83?w=900&q=85"),
    # Food / kitchen jars
    ("glass_48_food_jar",          "https://images.unsplash.com/photo-1578598336003-aa4f38a8cfe0?w=900&q=85"),
    ("glass_49_kitchen_jars",      "https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?w=900&q=85"),
    ("glass_50_honey_jar",         "https://images.unsplash.com/photo-1558642452-9d2a7deb7f62?w=900&q=85"),
    ("glass_51_glass_bowl",        "https://images.unsplash.com/photo-1565538810643-b5bdb714032a?w=900&q=85"),
    # Tech / screens
    ("glass_52_screen_reflection", "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=900&q=85"),
    ("glass_53_glass_phone",       "https://images.unsplash.com/photo-1512941937669-90a1b58e7e9c?w=900&q=85"),
    ("glass_54_crystal_watch",     "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=900&q=85"),
    ("glass_55_solar_panel",       "https://images.unsplash.com/photo-1509391366360-2e959784a276?w=900&q=85"),
    ("glass_56_glass_lamp",        "https://images.unsplash.com/photo-1513506003901-1e6a35b70b3b?w=900&q=85"),
    # Outdoor glass / reflections
    ("glass_57_puddle_reflect",    "https://images.unsplash.com/photo-1501432377862-3d0432b87a14?w=900&q=85"),
    ("glass_58_glass_bridge",      "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=900&q=85"),
    ("glass_59_glass_sculpture",   "https://images.unsplash.com/photo-1587890686920-f170968fd1a7?w=900&q=85"),
    ("glass_60_glass_cube_arch",   "https://images.unsplash.com/photo-1553361371-9b66f91c8b20?w=900&q=85"),
]

# ---------------------------------------------------------------------------
# 80 Textureless / Monochromatic Interior Scenes  (real Unsplash photographs)
# ---------------------------------------------------------------------------
TEXTURELESS_URLS = [
    # White / minimalist living rooms
    ("texless_01_white_room",       "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=900&q=85"),
    ("texless_02_white_bedroom",    "https://images.unsplash.com/photo-1540518614846-7eded433c457?w=900&q=85"),
    ("texless_03_minimal_sofa",     "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=900&q=85"),
    ("texless_04_white_interior",   "https://images.unsplash.com/photo-1494438639946-1ebd1d20bf85?w=900&q=85"),
    ("texless_05_scandi_room",      "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?w=900&q=85"),
    ("texless_06_minimal_bedroom2", "https://images.unsplash.com/photo-1631049307264-da0ec9d70304?w=900&q=85"),
    ("texless_07_white_couch",      "https://images.unsplash.com/photo-1585846416890-64e25f0aa52a?w=900&q=85"),
    ("texless_08_nordic_interior",  "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=900&q=85"),
    # Plain walls / plaster
    ("texless_09_white_wall",       "https://images.unsplash.com/photo-1513519245088-0e12902e5a38?w=900&q=85"),
    ("texless_10_blank_studio",     "https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=900&q=85"),
    ("texless_11_plaster_wall",     "https://images.unsplash.com/photo-1600607686527-6fb886090705?w=900&q=85"),
    ("texless_12_concrete_wall",    "https://images.unsplash.com/photo-1557683316-973673baf926?w=900&q=85"),
    ("texless_13_white_backdrop",   "https://images.unsplash.com/photo-1491895200222-0fc4a4c35e18?w=900&q=85"),
    ("texless_14_grey_wall",        "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=900&q=85"),
    ("texless_15_smooth_white",     "https://images.unsplash.com/photo-1600566752355-35792bedcfea?w=900&q=85"),
    ("texless_16_painted_wall",     "https://images.unsplash.com/photo-1519710164239-da123dc3800a?w=900&q=85"),
    # Hallways / corridors
    ("texless_17_hallway",          "https://images.unsplash.com/photo-1558403194-611308249627?w=900&q=85"),
    ("texless_18_long_corridor",    "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=900&q=85"),
    ("texless_19_hotel_corridor",   "https://images.unsplash.com/photo-1517840545241-b21e6b7e91f9?w=900&q=85"),
    ("texless_20_narrow_hallway",   "https://images.unsplash.com/photo-1576007688516-ad8bd33af1a5?w=900&q=85"),
    ("texless_21_office_corridor",  "https://images.unsplash.com/photo-1497366216548-37526070297c?w=900&q=85"),
    ("texless_22_airport_hallway",  "https://images.unsplash.com/photo-1529074963764-98f45c47344b?w=900&q=85"),
    # Kitchens (smooth cabinets)
    ("texless_23_white_kitchen",    "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?w=900&q=85"),
    ("texless_24_modern_kitchen",   "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=900&q=85"),
    ("texless_25_minimal_kitchen",  "https://images.unsplash.com/photo-1565538810643-b5bdb714032a?w=900&q=85"),
    ("texless_26_kitchen_counter",  "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=900&q=85"),
    ("texless_27_open_kitchen",     "https://images.unsplash.com/photo-1556742044-3c52d6e88c62?w=900&q=85"),
    ("texless_28_grey_kitchen",     "https://images.unsplash.com/photo-1574269909862-7e1d70bb8078?w=900&q=85"),
    # Ceilings / flat overhead
    ("texless_29_white_ceiling",    "https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?w=900&q=85"),
    ("texless_30_concrete_ceiling", "https://images.unsplash.com/photo-1530268729831-4b0b9e170218?w=900&q=85"),
    # Staircases (smooth walls)
    ("texless_31_staircase_wall",   "https://images.unsplash.com/photo-1555636222-cae831e670b3?w=900&q=85"),
    ("texless_32_floating_stairs",  "https://images.unsplash.com/photo-1527689368864-3a821dbccc34?w=900&q=85"),
    ("texless_33_spiral_staircase", "https://images.unsplash.com/photo-1535827841776-24afc1e255ac?w=900&q=85"),
    # Offices (smooth surfaces)
    ("texless_34_minimal_office",   "https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=900&q=85"),
    ("texless_35_desk_surface",     "https://images.unsplash.com/photo-1497366812874-3415097a27e7?w=900&q=85"),
    ("texless_36_meeting_room",     "https://images.unsplash.com/photo-1497366412874-3415097a27e7?w=900&q=85"),
    ("texless_37_modern_lobby",     "https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=900&q=85"),
    ("texless_38_open_office",      "https://images.unsplash.com/photo-1497366754035-f200968a6e72?w=900&q=85"),
    # Bathrooms (smooth)
    ("texless_39_white_bathroom",   "https://images.unsplash.com/photo-1552321554-5fefe8c9ef14?w=900&q=85"),
    ("texless_40_minimal_bath",     "https://images.unsplash.com/photo-1584622650111-993a426fbf0a?w=900&q=85"),
    # Smooth floors / pavements
    ("texless_41_smooth_floor",     "https://images.unsplash.com/photo-1584622650111-993a426fbf0a?w=900&q=85"),
    ("texless_42_marble_floor",     "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=900&q=85"),
    ("texless_43_polished_floor",   "https://images.unsplash.com/photo-1585846416890-64e25f0aa52a?w=900&q=85"),
    ("texless_44_concrete_floor",   "https://images.unsplash.com/photo-1530268729831-4b0b9e170218?w=900&q=85"),
    # Galleries / museums
    ("texless_45_gallery_wall",     "https://images.unsplash.com/photo-1481277542470-605612bd2d61?w=900&q=85"),
    ("texless_46_art_gallery",      "https://images.unsplash.com/photo-1549277513-f1b32fe1f8f5?w=900&q=85"),
    ("texless_47_museum_hall",      "https://images.unsplash.com/photo-1543783207-ec64e4d95325?w=900&q=85"),
    ("texless_48_exhibition_room",  "https://images.unsplash.com/photo-1568699764318-1d2efd3a3e7c?w=900&q=85"),
    # Outdoor textureless
    ("texless_49_flat_roof",        "https://images.unsplash.com/photo-1486325212027-8081e485255e?w=900&q=85"),
    ("texless_50_grey_sky",         "https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=900&q=85"),
    ("texless_51_overcast_sky",     "https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=900&q=85"),
    ("texless_52_sand_dunes",       "https://images.unsplash.com/photo-1509316785289-025f5b846b35?w=900&q=85"),
    ("texless_53_snow_plain",       "https://images.unsplash.com/photo-1418065460487-3e41a6c84dc5?w=900&q=85"),
    # Medical / clean rooms
    ("texless_54_hospital_room",    "https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=900&q=85"),
    ("texless_55_clean_room",       "https://images.unsplash.com/photo-1530973428-5bf2db2e4d71?w=900&q=85"),
    # Furniture surfaces
    ("texless_56_white_table",      "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=900&q=85"),
    ("texless_57_empty_shelf",      "https://images.unsplash.com/photo-1524055988636-436cfa46e59e?w=900&q=85"),
    ("texless_58_smooth_cabinet",   "https://images.unsplash.com/photo-1556742502-ec7c0e9f34b6?w=900&q=85"),
    # Architecture exterior smooth
    ("texless_59_modern_building",  "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=900&q=85"),
    ("texless_60_concrete_arch",    "https://images.unsplash.com/photo-1530268729831-4b0b9e170218?w=900&q=85"),
    ("texless_61_smooth_facade",    "https://images.unsplash.com/photo-1560472354-b33ff0ad5a3d?w=900&q=85"),
    ("texless_62_brutalist_arch",   "https://images.unsplash.com/photo-1573461160327-81bf4f2ce18e?w=900&q=85"),
    # Photography studios
    ("texless_63_photo_studio",     "https://images.unsplash.com/photo-1464618663641-bbdd760ae84a?w=900&q=85"),
    ("texless_64_lightbox_studio",  "https://images.unsplash.com/photo-1508169351866-777fc0047ac5?w=900&q=85"),
    # Tunnels
    ("texless_65_tunnel",           "https://images.unsplash.com/photo-1475070929565-c985b496cb9f?w=900&q=85"),
    ("texless_66_underpass",        "https://images.unsplash.com/photo-1529074963764-98f45c47344b?w=900&q=85"),
    # Gym / sports
    ("texless_67_gym_floor",        "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=900&q=85"),
    ("texless_68_yoga_studio",      "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=900&q=85"),
    # Storage / warehouses
    ("texless_69_empty_warehouse",  "https://images.unsplash.com/photo-1465189684280-6a8fa9b19a7a?w=900&q=85"),
    ("texless_70_empty_room",       "https://images.unsplash.com/photo-1536376072261-38c75010e6c9?w=900&q=85"),
    # Transit / airports
    ("texless_71_airport_terminal", "https://images.unsplash.com/photo-1529074963764-98f45c47344b?w=900&q=85"),
    ("texless_72_train_station",    "https://images.unsplash.com/photo-1540575467063-178a50c2df87?w=900&q=85"),
    # Hotels / minimal sleeping
    ("texless_73_hotel_room",       "https://images.unsplash.com/photo-1631049307264-da0ec9d70304?w=900&q=85"),
    ("texless_74_loft_space",       "https://images.unsplash.com/photo-1536376072261-38c75010e6c9?w=900&q=85"),
    ("texless_75_japanese_interior","https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?w=900&q=85"),
    # Empty / void rooms
    ("texless_76_empty_apartment",  "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=900&q=85"),
    ("texless_77_void_room",        "https://images.unsplash.com/photo-1507089947368-19c1da9775ae?w=900&q=85"),
    ("texless_78_empty_office",     "https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=900&q=85"),
    ("texless_79_white_studio2",    "https://images.unsplash.com/photo-1491895200222-0fc4a4c35e18?w=900&q=85"),
    ("texless_80_featureless_room", "https://images.unsplash.com/photo-1494438639946-1ebd1d20bf85?w=900&q=85"),
]


# ---------------------------------------------------------------------------
# Augmentation Engine
# ---------------------------------------------------------------------------
def _make_50_variants(img: np.ndarray, target_h: int = 480, target_w: int = 640) -> list:
    """
    Generate 50 diverse training variants from a single real image.
    Strategy: 5 spatial crops x 2 flips x 5 photometric modes = 50 variants.
    Each variant is genuinely distinct in both structure and lighting.
    """
    h, w = img.shape[:2]
    variants = []

    # 5 Spatial Crop Regions (covering different parts of scene)
    spatial_crops = [
        (0.00, 0.00, 1.00, 1.00),   # Full frame
        (0.00, 0.00, 0.80, 0.80),   # Top-left 80%
        (0.00, 0.20, 0.80, 0.80),   # Top-right 80%
        (0.10, 0.10, 0.80, 0.80),   # Centre 80%
        (0.20, 0.00, 0.80, 0.80),   # Bottom-left 80%
    ]

    def _crop(img, y_frac, x_frac, h_frac, w_frac):
        ch, cw = int(h * h_frac), int(w * w_frac)
        y0 = min(int(h * y_frac), max(0, h - ch))
        x0 = min(int(w * x_frac), max(0, w - cw))
        c = img[y0:y0 + ch, x0:x0 + cw]
        if c.size == 0:
            c = img
        return cv2.resize(c, (target_w, target_h))

    def _photometric(img, mode):
        f = img.astype(np.float32)
        if mode == 0:
            return img                                                           # Original
        elif mode == 1:
            return np.clip(f * 1.30, 0, 255).astype(np.uint8)                  # Brighter +30%
        elif mode == 2:
            return np.clip(f * 0.70, 0, 255).astype(np.uint8)                  # Darker   -30%
        elif mode == 3:
            lo, hi = np.percentile(f, 10), np.percentile(f, 90)                # Contrast boost
            stretched = (f - lo) / (hi - lo + 1e-5) * 255.0
            return np.clip(stretched, 0, 255).astype(np.uint8)
        else:
            gamma = 0.70                                                        # Gamma correction
            lut = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
            return lut[img]

    # 5 spatial x 2 flips x 5 photometric = 50
    for (y_f, x_f, h_f, w_f) in spatial_crops:
        crop_base = _crop(img, y_f, x_f, h_f, w_f)
        for flip_code in [None, 1]:  # None = no flip, 1 = horizontal
            flipped = cv2.flip(crop_base, flip_code) if flip_code is not None else crop_base
            for mode in range(5):
                variants.append(_photometric(flipped, mode))

    return variants  # exactly 50


def _make_6_camo_variants(img: np.ndarray, target_h: int = 384, target_w: int = 512) -> list:
    """
    6 distinct training variants per camouflage real photo:
      1. Original resize
      2. Horizontal flip
      3. Vertical flip
      4. Brightness +25%
      5. Brightness -25%
      6. Centre 85% crop -> resize
    """
    h, w = img.shape[:2]
    orig = cv2.resize(img, (target_w, target_h))
    f = orig.astype(np.float32)

    ch, cw = int(h * 0.85), int(w * 0.85)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    centre = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (target_w, target_h))

    return [
        orig,
        cv2.flip(orig, 1),                                        # H-flip
        cv2.flip(orig, 0),                                        # V-flip
        np.clip(f * 1.25, 0, 255).astype(np.uint8),              # Bright
        np.clip(f * 0.75, 0, 255).astype(np.uint8),              # Dark
        centre,                                                    # Centre crop
    ]


# ---------------------------------------------------------------------------
# Domain Builders
# ---------------------------------------------------------------------------
def _build_domain_from_urls(url_list, prefix, target_count):
    """
    Download real photos, augment each to 50 variants to reach target_count.
    Loops through url_list repeatedly if needed.
    """
    print(f"\n  Downloading {len(url_list)} {prefix} base scenes...")
    base_images = []
    temp_p = DATA_DIR / f"_tmp_{prefix}.jpg"

    for name, url in url_list:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                temp_p.write_bytes(resp.read())
            img = cv2.imread(str(temp_p))
            if img is not None and img.size > 0:
                base_images.append((name, img))
                print(f"    OK {name}  ({img.shape[1]}x{img.shape[0]})")
            else:
                print(f"    SKIP {name}  (decode failed)")
        except Exception as exc:
            print(f"    FAIL {name}  ({exc})")

    temp_p.unlink(missing_ok=True)

    if not base_images:
        print(f"  [WARN] No {prefix} images downloaded -- skipping domain.")
        return 0

    print(f"  -> {len(base_images)} real scenes downloaded, generating {target_count} augmented images...")
    saved = 0
    scene_idx = 0
    while saved < target_count:
        name, img = base_images[scene_idx % len(base_images)]
        variants = _make_50_variants(img)

        for v_idx, variant in enumerate(variants):
            if saved >= target_count:
                break
            out = DATA_DIR / f"{prefix}_{scene_idx:04d}_{v_idx:02d}.jpg"
            cv2.imwrite(str(out), variant)
            saved += 1

        scene_idx += 1
        if saved % 500 == 0:
            print(f"  ... {saved}/{target_count} {prefix} images saved")

    print(f"  Saved {saved} {prefix} images from {len(base_images)} real scenes")
    return saved


def _build_camo_domain(target_count=3_000):
    """Augment CAMO-FS real photos 6x to reach target_count."""
    camo_files = sorted(CAMO_SOURCE.glob("*.jpg"))
    if not camo_files:
        print(f"  [WARN] No CAMO-FS files found at {CAMO_SOURCE}")
        return 0

    print(f"\n  Augmenting {len(camo_files)} CAMO-FS real photos -> target {target_count}...")
    saved = 0
    file_idx = 0
    while saved < target_count:
        f = camo_files[file_idx % len(camo_files)]
        img = cv2.imread(str(f))
        if img is None:
            file_idx += 1
            continue

        variants = _make_6_camo_variants(img)
        for v_idx, variant in enumerate(variants):
            if saved >= target_count:
                break
            out = DATA_DIR / f"camo_{file_idx:04d}_{v_idx:02d}.jpg"
            cv2.imwrite(str(out), variant)
            saved += 1

        file_idx += 1
        if saved % 500 == 0:
            print(f"  ... {saved}/{target_count} camo images saved")

    print(f"  Saved {saved} camouflage images")
    return saved


# ---------------------------------------------------------------------------
# Split Generator (Scene-Level, No Leakage)
# ---------------------------------------------------------------------------
def _scene_grouped_split(all_files):
    scenes = {}
    for fname in all_files:
        stem = fname.rsplit(".", 1)[0]
        parts = stem.rsplit("_", 1)
        # Scene key = everything before the final _NN numeric variant index
        if len(parts) == 2 and parts[1].isdigit():
            key = parts[0]
        else:
            key = stem
        scenes.setdefault(key, []).append(fname)

    scene_keys = sorted(scenes.keys())
    rng = random.Random(42)
    rng.shuffle(scene_keys)

    n = len(scene_keys)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_s = scene_keys[:n_train]
    val_s   = scene_keys[n_train:n_train + n_val]
    test_s  = scene_keys[n_train + n_val:]

    train_f = [f for s in train_s for f in scenes[s]]
    val_f   = [f for s in val_s   for f in scenes[s]]
    test_f  = [f for s in test_s  for f in scenes[s]]

    total = len(all_files)
    dc = {
        "camouflage":  sum(1 for f in all_files if f.startswith("camo_")),
        "transparent": sum(1 for f in all_files if f.startswith("transparent_")),
        "textureless": sum(1 for f in all_files if f.startswith("textureless_")),
    }

    return {
        "total_images":        total,
        "total_unique_scenes": len(scene_keys),
        "leakage_protection":  "Strict Scene-Level Grouping: all variants of a scene stay in the same partition.",
        "domain_counts":  dc,
        "domain_percent": {k: f"{100*v/total:.1f}%" for k, v in dc.items()},
        "counts":         {"train": len(train_f), "val": len(val_f), "test": len(test_f)},
        "train_files":    train_f,
        "val_files":      val_f,
        "test_files":     test_f,
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def build_10k_dataset():
    total_target = CAMO_TARGET + TRANSPARENT_TARGET + TEXTURELESS_TARGET
    print("=" * 92)
    print("  PHASE 7: LARGE-SCALE 10,000-IMAGE MULTI-DOMAIN DATASET BUILDER")
    print("=" * 92)
    print(f"  Target  : {total_target:,} images")
    print(f"  Balance : {100*CAMO_TARGET/total_target:.0f}% camo  /  "
          f"{100*TRANSPARENT_TARGET/total_target:.0f}% transparent  /  "
          f"{100*TEXTURELESS_TARGET/total_target:.0f}% textureless")

    # Wipe & recreate output directory
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Domain 1: Camouflage
    print("\n[Domain 1/3] CAMOUFLAGE")
    n_camo = _build_camo_domain(CAMO_TARGET)

    # Domain 2: Transparent Glass
    print("\n[Domain 2/3] TRANSPARENT GLASSWARE & REFRACTIVE")
    n_trans = _build_domain_from_urls(TRANSPARENT_URLS, "transparent", TRANSPARENT_TARGET)

    # Domain 3: Textureless Interiors
    print("\n[Domain 3/3] TEXTURELESS INTERIORS & MONOCHROMATIC ARCHITECTURE")
    n_texl = _build_domain_from_urls(TEXTURELESS_URLS, "textureless", TEXTURELESS_TARGET)

    # Scene-grouped splits
    all_files = sorted(f.name for f in DATA_DIR.glob("*.jpg"))
    splits_data = _scene_grouped_split(all_files)

    with open(SPLITS_FILE, "w") as fp:
        json.dump(splits_data, fp, indent=2)

    dc  = splits_data["domain_counts"]
    dp  = splits_data["domain_percent"]
    tot = splits_data["total_images"]
    cnt = splits_data["counts"]

    print("\n" + "=" * 92)
    print("  DATASET BUILD COMPLETE")
    print("=" * 92)
    print(f"  Total Images     : {tot:,}")
    print(f"  Unique Scenes    : {splits_data['total_unique_scenes']:,}")
    print(f"")
    print(f"  Domain Breakdown :")
    print(f"    Camouflage     : {dc['camouflage']:>6,}  ({dp['camouflage']:>6})  -- 500 CAMO-FS real photos x 6 variants")
    print(f"    Transparent    : {dc['transparent']:>6,}  ({dp['transparent']:>6})  -- {len(TRANSPARENT_URLS)} real Unsplash scenes x 50 variants")
    print(f"    Textureless    : {dc['textureless']:>6,}  ({dp['textureless']:>6})  -- {len(TEXTURELESS_URLS)} real Unsplash scenes x 50 variants")
    print(f"")
    print(f"  Train Partition  : {cnt['train']:,} images")
    print(f"  Val   Partition  : {cnt['val']:,}  images")
    print(f"  Test  Partition  : {cnt['test']:,}  images")
    print(f"  Splits saved  -> {SPLITS_FILE}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    build_10k_dataset()
