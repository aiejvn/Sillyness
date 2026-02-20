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


# Main game clock region: "MM SS" countdown at top-center of screen
MAIN_CLOCK_REGION = RegionConfig(x=813, y=56, width=288, height=104)

# Camera icon region: for detecting camera uptime status
CAMERA_ICON_REGION = RegionConfig(x=1745, y=81, width=61, height=41)


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


@dataclass
class ClockReading:
    """Main game clock value at a specific frame."""
    frame_number: int
    clock_seconds: int  # total seconds remaining (MM*60 + SS)
    raw_text: str  # raw OCR'd text for debugging


@dataclass
class ClockTimeBurnEvent:
    """Time burn/gain event detected by comparing consecutive clock readings.

    Uses frame gap to distinguish real burns from normal ticks.
    At 60 FPS, 1 real second = ~60 frames. Expected clock change = -elapsed_seconds.
    anomaly = clock_change + elapsed_seconds (positive=gain, negative=burn).
    """
    frame_number: int
    clock_seconds: int  # clock value at this frame
    delta: int  # raw clock change from previous reading
    frame_gap: int  # frames elapsed since previous reading
    elapsed_seconds: float  # frame_gap / fps — real time elapsed
    anomaly: float  # delta + elapsed_seconds: <0 = burn, >0 = gain


@dataclass
class CameraStatusReading:
    """Camera uptime status at a specific frame."""
    frame_number: int
    red: float  # proportion 0.0-1.0: disabled camera indicator
    white: float  # proportion: active camera indicator (white pixels)
    camera_status: str  # "online", "disabled", or "neutral"
