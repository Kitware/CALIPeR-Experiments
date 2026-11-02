"""Define Collect 1 camera models and measured pixel scales."""

CAMERA_CONFIGS: dict[str, dict] = {
    "260m": {
        "f": 0.15203,        # Effective focal length: range × pixel pitch / metres-per-pixel.
        "D": 0.03018,        # f / 5.04 (Sigma zoom ~f/5.04 at 152mm)
        "p_x": 3.45e-6,      # Sony IMX252LLR-C (Basler acA2040-120um datasheet)
        "int_time": 0.0318,  # Integration time: 31.8 ms.
        "read_noise": 2.3,   # 2.3 e- RMS (Basler EMVA 1288 acA2040-120um)
        "max_n": 10500,      # 10.5 ke- saturation (Basler EMVA 1288 acA2040-120um)
        "bit_depth": 12.0,   # 12-bit ADC (Basler ace; 10/12 selectable)
        "s_x": 2.0e-6,       # ~2 µrad RMS platform jitter (tripod + wind)
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 260,  # MODTRAN atmosphere interpolated between 100m and 500m.
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,  # horizontal boundary-layer Cn2 for late-afternoon
                                # conditions over mixed terrain (asphalt + grass)
        # Camera metadata and additional optical parameters used by the adapter.
        "_camera_distance_m": 260,
        "_sensor_model": "Sony IMX252LLR-C (mono, Basler acA2040-120um)",
        "_source_lens": "Basler 120um + Sigma F5-6.3DG",  # Effective f≈152mm.
        # Radial distortion follows radiometric conversion; pv is phase variance.
        "_distortion_k": [-0.01, 0.0, 0.0],   # mild pincushion at ~152mm
        "_pv": 0.8,   # Wavefront error phase variance (rad²).
    },
    "380m": {
        "f": 0.17480,        # Effective focal length from the subject-plane pixel scale.
        "D": 0.03207,        # f / 5.45 (Sigma zoom ~f/5.45 at 175mm)
        "p_x": 3.45e-6,      # Sony IMX250LQR-C (Basler acA2440-75uc datasheet)
        "int_time": 0.0318,  # Integration time: 31.8 ms.
        "read_noise": 2.3,   # Pregius Gen-2 family (matches Basler EMVA 1288)
        "max_n": 10500,      # Pregius Gen-2 family saturation capacity
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 380,  # MODTRAN atmosphere interpolated between 100m and 500m.
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 380,
        "_sensor_model": "Sony IMX250LQR-C (color Bayer, Basler acA2440-75uc)",
        "_source_lens": "Basler 75uc + Sigma @ ~220mm",  # Nominal lens setting; effective f≈175mm.
        "_distortion_k": [-0.015, 0.0, 0.0],
        "_pv": 1.2,
    },
    "500m": {
        "f": 0.25000,        # Effective focal length from the subject-plane pixel scale.
        "D": 0.04316,        # f / 5.79 (Sigma zoom ~f/5.79 at 250mm)
        "p_x": 3.45e-6,      # Sony IMX265LQR-C (Basler acA2040-55uc datasheet)
        "int_time": 0.0318,  # Integration time: 31.8 ms.
        "read_noise": 2.3,   # Pregius Gen-2 family
        "max_n": 10500,      # Pregius Gen-2 family
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 500,  # exact DB entry; interp=True is a no-op here.
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 500,
        "_sensor_model": "Sony IMX265LQR-C (color Bayer, Basler acA2040-55uc)",
        "_source_lens": "Basler 55uc + Sigma @ ~320mm",  # Nominal lens setting; effective f≈250mm.
        "_distortion_k": [-0.02, 0.0, 0.0],
        "_pv": 1.8,
    },
    "1000m": {
        "f": 0.32547,        # Effective focal length from the subject-plane pixel scale.
        "D": 0.05419,        # f / 6.01 (Sigma zoom ~f/6.01 at 325mm)
        "p_x": 3.45e-6,      # Sony IMX264LQR-C (Basler acA2440-35uc datasheet)
        "int_time": 0.0318,  # Integration time: 31.8 ms.
        "read_noise": 2.3,   # Pregius Gen-2 family
        "max_n": 10500,      # Pregius Gen-2 family
        "bit_depth": 12.0,
        "s_x": 2.0e-6,
        "s_y": 2.0e-6,
        "altitude": 2.0,
        "ground_range": 1000,  # exact DB entry; interp=True is a no-op here.
        "interp": True,
        "ihaze": 1,
        "cn2_at_1m": 5.0e-13,
        "_camera_distance_m": 1000,
        "_sensor_model": "Sony IMX264LQR-C (color Bayer, Basler acA2440-35uc)",
        "_source_lens": "Basler 35uc + Sigma APO EX IF HSM",  # Effective f≈325mm.
        "_distortion_k": [-0.03, 0.0, 0.0],
        "_pv": 2.5,
    },
}

# Subject-plane metres per pixel, inferred using a 1.70 m person height.
M_PER_PIXEL: dict[str, float] = {
    "260m": 0.0059,
    "380m": 0.0075,
    "500m": 0.0069,
    "1000m": 0.0106,
}
