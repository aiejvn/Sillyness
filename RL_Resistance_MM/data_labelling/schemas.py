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
