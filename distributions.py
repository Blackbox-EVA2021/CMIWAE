import torch
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


if hasattr(torch, "broadcast_shapes"):
    my_broadcast_shapes = torch.broadcast_shapes
else:
    import numpy

    if hasattr(numpy, "broadcast_shapes"):
        my_broadcast_shapes = numpy.broadcast_shapes
    else:

        def my_broadcast_shapes(*shapes):
            return broadcast_all(*(torch.empty(*s) for s in shapes))[0].shape



class Binned(Distribution):
    """Binned distribution that takes into account
    that both CNT and BA have to be zero at the same time."""

    arg_constraints = {}
    has_enumerate_support = False
    has_rsample = False

    def __init__(self, logits):
        validate_args = False
        assert logits.shape[-3] == 57

        self.logit = logits[..., -1:, :, :]

        # 28 logits for non-zero values
        CNT_non_zero_logits = logits[..., 0:28, :, :]
        # 28 logits for non-zero values
        BA_non_zero_logits = logits[..., 28:-1, :, :]
        self.non_zero_logits = torch.stack(
            (CNT_non_zero_logits, BA_non_zero_logits), dim=-4)  # [K, bs, 2, 28, h, w]

        self.CNT_non_zero_thresholds = torch.tensor(
            [
                # -np.inf,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                1e6  # +np.inf,
            ]
        ).to(logits.device)

        self.BA_non_zero_thresholds = torch.tensor(
            [
                # -np.inf,
                0,
                1,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
                1000,
                1500,
                2000,
                5000,
                10000,
                20000,
                30000,
                40000,
                50000,
                100000,
                1e6  # +np.inf,
            ]
        ).to(
            logits.device
        )  # shape = [29]

        CNT_bin_sizes = self.CNT_non_zero_thresholds[1:] - \
            self.CNT_non_zero_thresholds[:-1]
        BA_bin_sizes = self.BA_non_zero_thresholds[1:] - \
            self.BA_non_zero_thresholds[:-1]

        self.non_zero_bin_sizes = torch.stack(
            (CNT_bin_sizes, BA_bin_sizes), dim=0
        )  # [2, 28], last bin goes to +inf

        super().__init__(
            (*self.logit.shape[:2], 1, *self.logit.shape[3:]), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    def sample(self, sample_shape=torch.Size()):
        return NotImplementedError

    def log_prob(self, value):
        """returns tensor of shape with shape[-3]==1"""
        assert value.shape[-3] == 2

        non_zero = value != 0
        any_non_zero = non_zero.any(dim=-3, keepdim=True)
        both_zero = any_non_zero.logical_not()

        # calculate log prob for non-zero case

        CNT = value[..., 0:1, :, :]  # [bs, 1, h, w]
        CNT_idx = torch.bucketize(
            CNT, self.CNT_non_zero_thresholds
        )  # indices 1...28, shape [bs, 1, h, w]

        BA = value[..., 1:2, :, :]  # [bs, 1, h, w]
        BA_idx = torch.bucketize(
            BA, self.BA_non_zero_thresholds
        )  # indices 1...28, shape [bs, 1, h, w]

        value_idx = (
            torch.cat((CNT_idx, BA_idx), dim=-3)[None, ..., None, :, :] - 1
        )  # indices 0...27, shape [1, bs, 2, 1, h, w]
        value_idx = value_idx.expand(
            self.non_zero_logits.shape[0], -1, -1, -1, -1, -1
        )  # shape [K, bs, 2, 1, h, w]
        # fix out-of-range indices where value==0 (ignored later anyway)
        value_idx = non_zero[..., None, :, :] * value_idx

        non_zero_bucket_log_probs = torch.nn.LogSoftmax(dim=-3)(
            self.non_zero_logits
        )  # [K, bs, 2, 28, h, w]
        non_zero_bucket_log_prob_densities = (
            non_zero_bucket_log_probs -
            self.non_zero_bin_sizes.log()[..., None, None]
        )  # [K, bs, 2, 28, h, w]

        non_zero_log_prob_densities = torch.gather(
            input=non_zero_bucket_log_prob_densities, index=value_idx, dim=-3
        )  # [K, bs, 2, 1, h, w]
        non_zero_log_prob_densities = non_zero_log_prob_densities.squeeze(
            dim=-3)  # [K, bs, 2, h, w]

        out = both_zero * torch.nn.functional.logsigmoid(self.logit)
        out += any_non_zero * torch.nn.functional.logsigmoid(-self.logit)
        out += (non_zero * non_zero_log_prob_densities).sum(dim=-3, keepdim=True)

        return out

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        zero_bucket_prob = self.logit.sigmoid()  # [K, bs, 1, h, w]

        zero_bucket_probs = zero_bucket_prob.unsqueeze(
            -4).expand(-1, -1, 2, -1, -1, -1)  # [K, bs, 2, 1, h, w]

        non_zero_bucket_probs = torch.softmax(
            self.non_zero_logits, dim=-3)  # [K, bs, 2, 28, h, w]

        bucket_probs = torch.cat((zero_bucket_probs,
                                 (1 - zero_bucket_probs) * non_zero_bucket_probs), dim=-3)   # [K, bs, 2, 29, h, w]

        cdf = torch.cumsum(bucket_probs, dim=-3)

        return torch.movedim(cdf, -3, 0)[:-1]  # [28, K, bs, 2, h, w]


class ZeroModifiedLogNormal(Distribution):
    """Takes into account that both CNT and BA have to be zero at the same time."""

    arg_constraints = {}
    has_enumerate_support = False
    has_rsample = True

    def __init__(self, logit, meanlog, sdlog):
        validate_args = False
        assert logit.shape[-3] == 1
        assert meanlog.shape[-3] == 2
        assert sdlog.shape[-3] == 2
        self.logit = logit
        self.lognormal = torch.distributions.LogNormal(
            meanlog, sdlog, validate_args=validate_args
        )
        batch_shape = broadcast_all(
            self.logit, torch.empty(*self.lognormal.batch_shape)
        )[0].shape
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return torch.bernoulli(
                1 - torch.sigmoid(self.logit.expand(*
                                  sample_shape, *self.logit.shape))
            ) * self.lognormal.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size(), temperature=0.1):
        shape = self._extended_shape(sample_shape)
        logit = self.logit.expand(*shape)
        logits = torch.stack(
            (logit / 2, -logit / 2), dim=-1
        )  # division by 2 accounts for normalization in gumbel_softmax
        return torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=True)[
            ..., 1
        ] * self.lognormal.rsample(sample_shape)

    def log_prob(self, value):
        """returns tensor of shape with shape[-3]==1"""
        assert value.shape[-3] == 2

        non_zero = value != 0
        any_non_zero = non_zero.any(dim=-3, keepdim=True)
        both_zero = any_non_zero.logical_not()

        out = both_zero * torch.nn.functional.logsigmoid(self.logit)
        out += any_non_zero * torch.nn.functional.logsigmoid(-self.logit)
        out += (non_zero * self.lognormal.log_prob(torch.where(non_zero,
                value, 1e-15 * torch.ones_like(value))).type(value.dtype)).sum(dim=-3, keepdim=True)

        return out

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        p0 = self.logit.sigmoid()
        return p0 + (1 - p0) * self.lognormal.cdf(value)