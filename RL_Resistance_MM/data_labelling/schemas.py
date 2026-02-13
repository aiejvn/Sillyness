from dataclasses import dataclass


@dataclass
class RegionConfig:
    """Pixel coordinates for an OCR region of interest."""
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple:
        """Return (left, upper, right, lower) for PIL cropping."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


# The time-burn popup region: shifted right to skip +/- sign, narrowed to exclude "Sec."
TIME_BURN_POPUP_REGION = RegionConfig(x=1146, y=68, width=100, height=81)


# Bio energy counter region: spec says (122, 938, 64, 76)
BIO_ENERGY_REGION = RegionConfig(x=122, y=938, width=64, height=76)


# Survivor health bar regions: left edge strip (10 pixels wide) for red/yellow/green classification
SURVIVOR_HEALTH_BAR_REGIONS = {
    1: RegionConfig(x=114, y=227, width=10, height=53),
    2: RegionConfig(x=114, y=293, width=10, height=53),
    3: RegionConfig(x=114, y=361, width=10, height=53),
    4: RegionConfig(x=114, y=427, width=10, height=53),
}

# Survivor full icon regions: entire portrait (62x53) for infection (purple overlay) detection
SURVIVOR_FULL_ICON_REGIONS = {
    1: RegionConfig(x=114, y=227, width=62, height=53),
    2: RegionConfig(x=114, y=293, width=62, height=53),
    3: RegionConfig(x=114, y=361, width=62, height=53),
    4: RegionConfig(x=114, y=427, width=62, height=53),
}


@dataclass
class SurvivorStatusReading:
    """Color proportions for a survivor status icon at a specific frame.

    Health status (red/yellow/green) and infection (purple) are independent dimensions.
    """
    frame_number: int
    survivor_id: int  # 1-4
    # Health dimension (mutually exclusive)
    red: float  # proportion 0.0-1.0: damaged/downed
    yellow: float  # proportion: debuffed
    green: float  # proportion: healthy
    # Infection dimension (independent)
    purple: float  # proportion: infected level
    # Derived classifications
    health_status: str  # dominant health color: "red", "yellow", or "green"
    infection_level: str  # infection severity: "none", "low", "medium", "high"

    @property
    def health_proportions(self) -> dict[str, float]:
        return {"red": self.red, "yellow": self.yellow, "green": self.green}

    @property
    def all_proportions(self) -> dict[str, float]:
        return {"red": self.red, "purple": self.purple, "yellow": self.yellow, "green": self.green}


@dataclass
class BioEnergyReading:
    """A sampled bio energy value at a specific frame."""
    frame_number: int
    value: int  # current bio energy (positive integer)
    raw_text: str  # the raw OCR'd text for debugging


@dataclass
class TimeBurnEvent:
    """A detected time-burn or time-gain popup event."""
    frame_number: int
    delta: int  # negative = time burned (good), positive = time gained (bad)
    raw_text: str  # the raw OCR'd text for debugging
