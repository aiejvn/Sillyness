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


# The time-burn popup region: narrowed to exclude "Sec." text
TIME_BURN_POPUP_REGION = RegionConfig(x=1146, y=68, width=110, height=81)


@dataclass
class TimeBurnEvent:
    """A detected time-burn or time-gain popup event."""
    frame_number: int
    delta: int  # negative = time burned (good), positive = time gained (bad)
    raw_text: str  # the raw OCR'd text for debugging
