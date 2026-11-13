"""Define nominal camera models for Collect 2."""

# Collect 2 records describe seven Basler stations with nominal ranges;
# site 4 is the recapture station. Reported video dimensions are
# 1920 x 1080 at 30 Hz, while example frames are also labeled 2056 x 1004.
# Whether the latter are capture sizes or processed examples is unverified.
# Full camera model names below are inferred from abbreviated collection records.
#
# Manufacturer specifications (sensor model, pixel pitch, pixel bit depths):
# https://www.baslerweb.com/en-us/shop/aca2040-120um/
# https://www.baslerweb.com/en-us/shop/aca2040-120uc/
# https://www.baslerweb.com/en-us/shop/aca2440-75uc/
# https://www.baslerweb.com/en-us/shop/aca2040-55uc/
# The 120um/120uc/55uc default sensor resolution is 2048 x 1536; the 75uc
# default is 2448 x 2048. Their respective rates at default settings are
# 120/120/55/75 Hz; these are not necessarily maximum rates at other ROIs.
# Basler advertises 10- and 12-bit pixel depths, but this does not establish
# the BRC2 capture setting, output format, or actual ADC operating setting.
# Basler's EMVA data for the 120um gives 2.3 e- temporal dark noise and
# 10,500 e- saturation capacity. Using dark noise for read_noise is a proxy;
# the 120um values are also proxies for the color 120uc. For the color 75uc
# and 55uc, respectively, use EMVA values from the monochrome IMX250 and
# IMX265 counterparts (2.4 e-/10,700 e- and 2.2 e-/10,400 e-). These are
# model-matched proxies, not measurements of the BRC2 color cameras:
# https://www.baslerweb.com/en-us/shop/aca2440-75um/
# https://www.baslerweb.com/en-us/shop/aca2040-55um/
#
# The following are modeling assumptions, not measured BRC2 capture settings:
# exposure (BRC1's 30 Hz value), 12-bit model depth, jitter, ground-camera
# altitude, haze and turbulence (BRC1 proxies); aperture, distortion, and
# wavefront error (lens-family or focal-length proxies). A listed focal length
# and lift height may themselves be approximate. The collection records do not
# identify the exact lenses or aperture settings, so D is not a measured value.
# In particular, the 160m station's Tamron is approximated as f/5 at 150 mm; the 156m recapture station uses the
# existing 400 mm Sigma estimate. Teleconverter focal lengths are calculated;
# the estimated entrance-pupil diameter D remains that of the base lens.
# _sensor_model is metadata: these configs do not model Bayer sampling,
# demosaicing, sensor-specific spectral QE, or optical-filter transmission.
# turbulence_eval loads this file when the brc2 profile is selected.
# interp=True enables pyBSM interpolation at the modeled ranges and altitudes.
CAMERA_CONFIGS: dict[str, dict] = {
    "250m": {  # 250m ground monochrome station; ground, monochrome
        "f": 0.220,          # Sigma @ ~220 mm
        "D": 0.0393,         # estimated f/5.6
        "p_x": 3.45e-6,
        "int_time": 0.0318,
        "read_noise": 2.3,   # proxy: 120um EMVA typical temporal dark noise, e-
        "max_n": 10500,      # 120um EMVA typical saturation capacity, e-
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 250,
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 250,
        "_sensor_model": "Sony IMX252LLR-C (mono, Basler acA2040-120um)",
        "_source_lens": "250m ground monochrome station: Basler 120um + Sigma @ ~220mm",
        "_distortion_k": [-0.012, 0.0, 0.0],
        "_pv": 1.0,
    },
    "400m": {  # 400m ground color station; ground, color
        "f": 0.400,          # Sigma @ ~400 mm
        "D": 0.0635,         # estimated f/6.3
        "p_x": 3.45e-6,
        "int_time": 0.0318,
        "read_noise": 2.3,
        "max_n": 10500,
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 400,
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 400,
        "_sensor_model": "Sony IMX252LQR-C (color Bayer, Basler acA2040-120uc)",
        "_source_lens": "400m ground color station: Basler 120uc + Sigma @ ~400mm",
        "_distortion_k": [-0.018, 0.0, 0.0],
        "_pv": 1.6,
    },
    "500m": {  # 500m ground color station; ground, color
        "f": 0.840,          # Sigma @ 600 mm x 1.4
        "D": 0.0952,         # estimated 600 mm / f/6.3; unchanged by 1.4x
        "p_x": 3.45e-6,
        "int_time": 0.0318,
        "read_noise": 2.3,
        "max_n": 10500,
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 500,
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 500,
        "_sensor_model": "Sony IMX252LQR-C (color Bayer, Basler acA2040-120uc)",
        "_source_lens": "500m ground color station: Basler 120uc + Sigma @ 600mm x 1.4",
        "_distortion_k": [-0.025, 0.0, 0.0],
        "_pv": 2.4,
    },
    "1000m": {  # 1000m ground color station; ground, color
        "f": 1.120,          # 800 mm x 1.4
        "D": 0.100,          # estimated 800 mm / f/8; base lens unspecified
        "p_x": 3.45e-6,
        "int_time": 0.0318,
        "read_noise": 2.3,
        "max_n": 10500,
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 1000,
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 1000,
        "_sensor_model": "Sony IMX252LQR-C (color Bayer, Basler acA2040-120uc)",
        "_source_lens": "1000m ground color station: Basler 120uc + 800mm x 1.4",
        "_distortion_k": [-0.035, 0.0, 0.0],
        "_pv": 3.2,
    },
    "250m_lift": {  # 250m elevated color station; ~3 m above ground, color; platform unknown
        "f": 0.300,          # Sigma @ 300 mm
        "D": 0.050,          # estimated f/6.0
        "p_x": 3.45e-6,
        "int_time": 0.0318,
        "read_noise": 2.3,
        "max_n": 10500,
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 3.0,      # camera height used as scenario-altitude proxy
        "ground_range": 250,
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 250,
        "_sensor_model": "Sony IMX252LQR-C (color Bayer, Basler acA2040-120uc)",
        "_source_lens": "250m elevated color station: Basler 120uc + Sigma @ 300mm (~3m high)",
        "_distortion_k": [-0.015, 0.0, 0.0],
        "_pv": 1.2,
    },
    "160m": {  # 160m ground color station; ground, color
        "f": 0.150,          # Tamron @ ~150 mm
        "D": 0.030,          # estimated f/5; actual Tamron model unknown
        "p_x": 3.45e-6,
        "int_time": 0.0318,
        "read_noise": 2.4,   # 75um mono counterpart dark-noise proxy, e-
        "max_n": 10700,      # 75um mono counterpart saturation proxy, e-
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 160,
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 160,
        "_sensor_model": "Sony IMX250LQR-C (color Bayer, Basler acA2440-75uc)",
        "_source_lens": "160m ground color station: Basler 75uc + Tamron @ ~150mm",
        "_distortion_k": [-0.01, 0.0, 0.0],  # BRC1 ~150 mm proxy
        "_pv": 0.8,                             # BRC1 ~150 mm proxy
    },
    "156m": {  # 156m recapture color station; ground, recapture, color
        "f": 0.400,          # Sigma @ ~400 mm
        "D": 0.0635,         # estimated f/6.3; 400m ground color station proxy
        "p_x": 3.45e-6,
        "int_time": 0.0318,
        "read_noise": 2.2,   # 55um mono counterpart dark-noise proxy, e-
        "max_n": 10400,      # 55um mono counterpart saturation proxy, e-
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 156,
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 156,
        "_sensor_model": "Sony IMX265LQR-C (color Bayer, Basler acA2040-55uc)",
        "_source_lens": "156m recapture color station: Basler 55uc + Sigma @ ~400mm (recapture)",
        "_distortion_k": [-0.018, 0.0, 0.0],  # 400m ground color station proxy
        "_pv": 1.6,                               # 400m ground color station proxy
    },
}
