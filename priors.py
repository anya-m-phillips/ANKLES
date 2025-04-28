### needs to be imported into ANKLES script
import torch
from torch.distributions import Distribution, constraints
from sklearn.neighbors import KernelDensity

### make priors based on y training data.
class KDEPrior(Distribution):
    def __init__(self, samples, bandwidth=0.2):
        super().__init__(validate_args=False)
        self.device = samples.device
        self.samples = samples
        self.kde = KernelDensity(bandwidth=bandwidth)
        self.kde.fit(samples.cpu().numpy())

    def log_prob(self, x):
        x_np = x.cpu().numpy()
        log_probs = self.kde.score_samples(x_np)
        return torch.tensor(log_probs, device=self.device, dtype=torch.float32)

    def sample(self, sample_shape=torch.Size()):
        indices = torch.randint(0, len(self.samples), sample_shape)
        return self.samples[indices]

    @property
    def support(self):
        return constraints.real  # or `real_vector` if you want more specific
    
    @property
    def mean(self):
        return self.samples.mean(dim=0)

    @property
    def stddev(self):
        return self.samples.std(dim=0)