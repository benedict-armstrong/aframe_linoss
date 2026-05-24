from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

from .cbc import CBCGenerator


class RegressionCBCGenerator(CBCGenerator):
    """CBCGenerator subclass for regression: adds right_pad and window_offset.

    right_pad: seconds of post-coalescence silence in the generated waveform
               (sets coalescence position: coal_time = duration - right_pad).
    window_offset: seconds before coalescence to anchor the model window
                   (e.g. window_offset=3.0 with kernel_length=1.0 → [59s, 60s]).
    """

    def __init__(
        self,
        *args,
        right_pad: float = 0.0,
        window_offset: float = 0.0,
        duration: float,
        f_min: float,
        f_ref: float,
        **kwargs,
    ):
        super().__init__(*args, duration=duration, f_min=f_min, f_ref=f_ref, **kwargs)
        self.right_pad = right_pad
        self.window_offset = window_offset
        # Recreate waveform_generator with the correct right_pad
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            self.approximant,
            self.sample_rate,
            duration,
            f_min,
            self.f_ref,
            right_pad,
        )
