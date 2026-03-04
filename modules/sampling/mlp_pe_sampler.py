import math
import torch

from model.modules import ModelState
from model.modules.sampling.pe_sampler import PositionalEncodingSampler
from torch import nn

class PositionalEncodingSamplerMLP(PositionalEncodingSampler):
    """
    PE Sampler with learnable MLP decoder for more expressive reconstruction.
    """

    def __init__(self, state: ModelState, num_frequencies: int = 6, **kwargs):
        # Don't call parent __init__ yet - we need to set up decoder first
        nn.Module.__init__(self)

        self.state = state
        self.device = state.device
        self.num_frequencies = num_frequencies
        self.include_input = kwargs.get('include_input', True)
        self.log_sampling = kwargs.get('log_sampling', True)

        self.evaluate_mode = kwargs.get('evaluate_mode', False)
        self.should_optimize = state.opt.optimize_intervals and not self.evaluate_mode
        self.num_channels = kwargs.get('num_channels', 1)
        self.mode = kwargs.get('mode', 'single' if self.num_channels == 1 else 'multi')

        # Encoding dimension
        self.encoding_dim = 2 * num_frequencies + (1 if self.include_input else 0)

        # Frequency bands
        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies)
        self.register_buffer('freq_bands', freq_bands * math.pi)

        # Learnable decoder MLP
        # Small network:  PE dim -> hidden -> 1
        hidden_dim = max(16, self.encoding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Initialize decoder to approximate identity at init
        self._init_decoder_identity()

        # Initialize intervals
        Us, Vs = state.Us, state.Vs
        self._init_intervals(Us, Vs, kwargs)

        self.vis_probs = torch.zeros((self.num_channels, Us, Vs, 2), device=self.device)
        self.uv_viewpoint = {}
        self.active_uid = None

    def _init_decoder_identity(self):
        """Initialize decoder to approximate identity mapping at initialization."""
        # For the decoder to output the DC component (first element if include_input)
        # we initialize first layer to heavily weight the DC component
        with torch.no_grad():
            # First linear layer
            self.decoder[0].weight.zero_()
            self.decoder[0].bias.zero_()

            if self.include_input:
                # Strongly connect DC input to first hidden unit
                self.decoder[0].weight[0, 0] = 5.0

            # Last layer bias to 0. 5 (middle of sigmoid)
            self.decoder[-2].bias.fill_(0.0)

    def _decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode using learnable MLP."""
        # encoded: [... , encoding_dim]
        original_shape = encoded.shape[:-1]
        flat = encoded.reshape(-1, self.encoding_dim)

        decoded = self.decoder(flat)  # [..., 1]
        decoded = decoded.reshape(original_shape)

        # Small epsilon to avoid exact 0 or 1
        return decoded.clamp(1e-6, 1 - 1e-6)