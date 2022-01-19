import torch
from fastai.callback.core import Callback
from fastai.callback.tracker import SaveModelCallback

# Define training callbacks

class TestScoreCallback(Callback):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.score = torch.tensor([0.0, 0.0])
        self.min_score = torch.tensor([1000000.0, 1000000.0])

        CNT_bins = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18,
                                    20, 22, 24, 26, 28, 30, 40, 50, 60, 70, 80, 90, 100]).float()
        BA_bins = torch.tensor([0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                150, 200, 250, 300, 400, 500, 1000, 1500, 2000,
                                5000, 10000, 20000, 30000, 40000, 50000, 100000]).float()

        omega_CNT = 1 - (1 + (CNT_bins + 1)**2 / 1000)**(-1/4)
        omega_CNT = omega_CNT / omega_CNT[-1]

        omega_BA = 1 - (1 + (BA_bins + 1) / 1000)**(-1/4)
        omega_BA = omega_BA / omega_BA[-1]

        self.bins = (
            torch.stack((CNT_bins, BA_bins), dim=-1)
            .view(28, 1, 1, 2, 1, 1)
            .float()
            .to(device)
        )
        self.omega = (
            torch.stack((omega_CNT, omega_BA), dim=-1)
            .view(28, 1, 1, 2, 1, 1)
            .float()
            .to(device)
        )

    def after_train(self):
        score = 0
        N = 0
        with torch.no_grad():
            for *xb, yb in self.dls[2]:
                with torch.no_grad():
                    qz_x, px_z, pz, z = self.model(*xb)

                x_target, mask, eval_mask = yb

                # if BA is 0, assume CNT is 0 and vice versa
                known_zeros = (mask != 0) & (xb[0][:, :2] == 0)
                CNT_known_zeros = known_zeros[:, 0:1]  # [bs, 1, h, w]
                BA_known_zeros = known_zeros[:, 1:2]

                # if BA is not 0, assume CNT is not 0 and vice versa
                known_non_zeros = (mask != 0) & (xb[0][:, :2] != 0)
                CNT_known_non_zeros = known_non_zeros[:, 0:1]
                BA_known_non_zeros = known_non_zeros[:, 1:2]

                lqz_x = qz_x.log_prob(z).sum(-1)

                single_channel_mask = mask.bool().any(
                    dim=-3, keepdim=True)  # [bs, 1, h, w]
                lpx_z = px_z.log_prob(x_target).mul(single_channel_mask).view(
                    *px_z.batch_shape[:2], -1).sum(-1)

                lpz = pz.log_prob(z).sum(-1)

                lw = lpz + lpx_z - lqz_x  # lw.shape == [K, bs]

                w = torch.nn.functional.softmax(lw, dim=0)

                # fix known zeros
                # px_z.logit.shape == [K, bs, 1, h, w]
                px_z.logit[..., CNT_known_zeros] = 1e6
                px_z.logit[..., BA_known_zeros] = 1e6

                # fix known non-zeros
                px_z.logit[..., CNT_known_non_zeros] = -1e6
                px_z.logit[..., BA_known_non_zeros] = -1e6

                # [28(bins), K, bs, 2(CNT+BA), h, w]
                cdf = px_z.cdf(self.bins)

                # [28(bins), 1, bs, 2(CNT+BA), h, w]
                pred = (w.view(*w.shape, 1, 1, 1) *
                        cdf).sum(dim=1, keepdim=True)

                i_u = (self.bins >= x_target).float()

                # do not sum over channels
                score += (eval_mask * self.omega * (i_u - pred)
                            ** 2).sum(dim=(0, 1, 2, 4, 5))

                # do not sum over channels
                N += eval_mask.sum(dim=(0, 2, 3))

        score = score / N * 80000
        score /= 1000  # scale score for ergonomy

        self.score = score
        if self.score.sum() < self.min_score.sum():
            self.min_score = self.score

    def total_score(self):
        return self.score.sum()

    def CNT_score(self):
        return self.score[0]

    def BA_score(self):
        return self.score[1]

    
class MySaveModelCallback(SaveModelCallback):
    def after_fit(self, **kwargs):
        pass  # do not load best model after fit when using DistributedLearner