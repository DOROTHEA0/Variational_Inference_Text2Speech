import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vits_modules.attention import Encoder
from vits_modules.commons import sequence_mask, init_weights, rand_slice_segments, get_padding
from vits_modules.modules import WN, ResidualCouplingLayer, Flip, ResBlock1, ResBlock2, LRELU_SLOPE
from utils import re_parameterize
import monotonic_align


# 先验编码器 文本编码器
class PriorEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=hidden_channels)
        self.encoding = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.projection = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        nn.init.normal_(self.embedding.weight, 0.0, hidden_channels ** -0.5)

    def forward(self, x, x_lengths):
        x = self.embedding(x) * math.sqrt(self.hidden_channels)  # (batch, token, hidden_channels)
        x = x.permute(0, 2, 1)  # (batch, hidden_channels, token)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = x * x_mask
        x = self.encoding(x, x_mask)
        vae_out = self.projection(x) * x_mask
        mu, log_var = torch.split(vae_out, self.out_channels, dim=1)
        return x, mu, log_var, x_mask


# 后验编码器 频谱编码器 s=多说话人
class PosteriorEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        # 频谱图高度看作通道数，转换成embedding dim (batch, h, w) -> (batch, embedding_dim, w), 长度越长意味着文本越长
        self.pre_layer = nn.Conv1d(in_channels, hidden_channels, 1)
        self.encoder = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.projection = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, speaker=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre_layer(x) * x_mask
        x = self.encoder(x, x_mask, g=speaker)
        vae_out = self.projection(x) * x_mask
        mu, log_var = torch.split(vae_out, self.out_channels, dim=1)
        z = re_parameterize(mu, log_var) * x_mask
        return z, mu, log_var, x_mask


class Decoder(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.pre_layer = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                   k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, speaker=None):
        x = self.pre_layer(x)
        if speaker is not None:
            x = x + self.cond(speaker)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class ResidualCouplingFlow(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                                    gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, s=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=s, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=s, reverse=reverse)
        return x


class StochasticDurationPredictor(nn.Module):
    pass


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = nn.utils.weight_norm if not use_spectral_norm else nn.utils.spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = nn.utils.weight_norm if not use_spectral_norm else nn.utils.spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# 判别器
class Discriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(Discriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs.extend([DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods])
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# 整个生成网络
class Generator(nn.Module):
    def __init__(self, n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 **kwargs):
        super().__init__()

        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        # 先验编码器 文本输入
        self.prior_encoder = PriorEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers,
                                          kernel_size, p_dropout)
        # 后验编码器 频谱图输入
        self.posterior_encoder = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                                  gin_channels=gin_channels)
        # 解码器 波形生成器
        self.decoder = Decoder(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                               upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        # 隐变量转换流
        self.flow = ResidualCouplingFlow(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
        # 时长预测器
        if self.use_sdp:  # 使用随机时长预测器
            self.duration_predictor = None
        else:  # 不使用随机时长预测器
            self.duration_predictor = None
        if n_speakers > 1:
            self.speaker_embedding = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_len, y, y_len, speaker_id=None):
        # 文本编码
        x, p_mu, p_log_var, x_mask = self.prior_encoder(x, x_len)
        s_embedding = self.speaker_embedding(speaker_id) if self.n_speakers > 1 else None
        # 频谱编码
        q_z, q_mu, q_log_var, y_mask = self.posterior_encoder(y, y_len, s_embedding)
        # 流转换为p分布
        p_z = self.flow(q_z, y_mask)

        # 文本频谱对齐MAS
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * p_log_var)  # [b, d, t]
            #  ([b, 1, t_s]) + ([b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]) + ([b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]) + ([b, 1, t_s])
            log_N = torch.sum(-0.5 * math.log(2 * math.pi) - p_log_var, [1], keepdim=True) \
                    + torch.matmul(-0.5 * (p_z ** 2).transpose(1, 2), s_p_sq_r) \
                    + torch.matmul(p_z.transpose(1, 2), (p_mu * s_p_sq_r)) \
                    + torch.sum(-0.5 * (p_mu ** 2) * s_p_sq_r, [1], keepdim=True)

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(log_N, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # 扩充: 将token维度扩充到频谱维度 [b, e_dim, t] => [b, e_dim, s]
        # attn: [b, 1, s, t], p_pu = [b, e_dim, t]
        # attn.squeeze(1): [b, s, t], p_mu.transpose(1, 2): [b, t, e_dim]
        # attn.squeeze(1) @ p_mu.transpose(1, 2): [b, s, e_dim] => transpose: [b, e_dim, s]
        p_mu, p_log_var = torch.matmul(attn.squeeze(1), p_mu.transpose(1, 2)).transpose(1, 2), \
            torch.matmul(attn.squeeze(1), p_log_var.transpose(1, 2)).transpose(1, 2)

        w = attn.sum(2)
        # 时长预测

        z_slice, ids_slice = rand_slice_segments(q_z, y_len, self.segment_size)
        target_wav = self.decoder(z_slice, speaker=s_embedding)
        return (target_wav, ids_slice), (x_mask, y_mask), (q_z, p_z, p_mu, p_log_var, q_mu, q_log_var)
