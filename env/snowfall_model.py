"""
snowfall_model.py
=================
Stochastic snowfall model fitted to NOAA Buffalo historical data.

When real NOAA data is available (data/noaa_buffalo_snowfall.csv),
the Gamma distribution parameters are fitted automatically.

When data is unavailable, empirically estimated parameters from
Buffalo's historical record (avg ~94 in/year) are used as defaults.

Buffalo, NY Station: GHCND:USW00014733
Data: https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00014733/detail

NOAA CSV column used: SNOW  (daily snowfall in tenths of mm)
Convert: inches = tenths_mm / 254
"""

import os
import numpy as np
import warnings

try:
    import pandas as pd
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Default Gamma parameters estimated from Buffalo's historical record
# (shape, loc, scale) fitted on non-zero snowfall days
_DEFAULT_PARAMS = {
    "shape": 0.72,
    "loc":   0.0,
    "scale": 2.8,   # inches
}

# Intensity multipliers
INTENSITY_MULTIPLIERS = {
    "none":   0.0,
    "low":    0.4,
    "medium": 1.0,
    "high":   1.9,
    "blizzard": 3.5,
}

# Probability of snowfall on any given timestep by intensity
SNOWFALL_PROBABILITY = {
    "none":    0.0,
    "low":     0.25,
    "medium":  0.45,
    "high":    0.70,
    "blizzard":0.95,
}


class SnowfallModel:
    """
    Gamma-distributed daily snowfall sampler.

    Parameters
    ----------
    data_path : str or None
        Path to NOAA CSV file. If None or file not found, uses default params.
    intensity : str
        One of 'none', 'low', 'medium', 'high', 'blizzard'.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, data_path=None, intensity="medium", seed=None):
        self.intensity = intensity
        self.rng = np.random.default_rng(seed)
        self._fit(data_path)

    def _fit(self, data_path):
        """Fit Gamma distribution to NOAA data or use defaults."""
        if data_path and os.path.exists(data_path) and SCIPY_AVAILABLE:
            try:
                df = pd.read_csv(data_path)
                # NOAA SNOW column is in tenths of mm
                col = "SNOW" if "SNOW" in df.columns else df.columns[0]
                raw = pd.to_numeric(df[col], errors="coerce").dropna().values
                inches = raw / 254.0
                non_zero = inches[inches > 0]

                if len(non_zero) > 30:
                    shape, loc, scale = stats.gamma.fit(non_zero, floc=0)
                    self.shape = shape
                    self.loc   = loc
                    self.scale = scale
                    print(f"[snowfall] Fitted Gamma: shape={shape:.3f}, scale={scale:.3f} "
                          f"from {len(non_zero)} non-zero observations")
                    return
            except Exception as e:
                warnings.warn(f"[snowfall] Could not fit from data: {e}. Using defaults.")

        self.shape = _DEFAULT_PARAMS["shape"]
        self.loc   = _DEFAULT_PARAMS["loc"]
        self.scale = _DEFAULT_PARAMS["scale"]
        print(f"[snowfall] Using default Gamma params: "
              f"shape={self.shape}, scale={self.scale}")

    def sample(self, intensity=None):
        """
        Sample snowfall (inches) for one timestep.

        Returns 0.0 if no snowfall event occurs (Bernoulli gate),
        otherwise samples from the fitted Gamma distribution.
        """
        intensity = intensity or self.intensity
        prob  = SNOWFALL_PROBABILITY.get(intensity, 0.45)
        mult  = INTENSITY_MULTIPLIERS.get(intensity, 1.0)

        if self.rng.random() > prob:
            return 0.0   # No snow this timestep

        raw = float(self.rng.gamma(self.shape, self.scale))
        return max(0.0, raw * mult)

    def sample_batch(self, n, intensity=None):
        """Sample n snowfall values."""
        return np.array([self.sample(intensity) for _ in range(n)])

    def set_intensity(self, intensity):
        assert intensity in INTENSITY_MULTIPLIERS, \
            f"intensity must be one of {list(INTENSITY_MULTIPLIERS.keys())}"
        self.intensity = intensity

    @property
    def expected_daily_inches(self):
        """Expected snowfall per day (accounting for zero-inflation)."""
        p = SNOWFALL_PROBABILITY[self.intensity]
        m = INTENSITY_MULTIPLIERS[self.intensity]
        return p * self.shape * self.scale * m

    def __repr__(self):
        return (f"SnowfallModel(intensity={self.intensity}, "
                f"shape={self.shape:.3f}, scale={self.scale:.3f})")


# Module-level default instance
_default_model = None


def get_model(intensity="medium", data_path=None, seed=None):
    """Return a cached SnowfallModel instance."""
    global _default_model
    if _default_model is None or _default_model.intensity != intensity:
        _default_model = SnowfallModel(
            data_path=data_path,
            intensity=intensity,
            seed=seed
        )
    return _default_model


if __name__ == "__main__":
    model = SnowfallModel(intensity="medium", seed=42)
    samples = model.sample_batch(365)
    nonzero = samples[samples > 0]
    print(f"Annual snowfall estimate: {samples.sum():.1f} inches")
    print(f"Snow days: {len(nonzero)} / 365")
    print(f"Mean on snow day: {nonzero.mean():.2f} inches")
    print(f"Max single day: {samples.max():.2f} inches")
