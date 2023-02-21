from config import DataConfig
import torchaudio
import torch
torchaudio.set_audio_backend(DataConfig.torchaudio_backend)


def load_wav_to_torch(wav_path):
    wav, sr = torchaudio.load(wav_path)
    if sr != DataConfig.sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=DataConfig.sample_rate)(wav)
        sr = DataConfig.sample_rate
    return wav, sr


def load_annotate_text(text_path, split_char='|'):
    data_segment = []
    with open(text_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data_segment.append(tuple(line.split(split_char)))
    return data_segment


def re_parameterize(mu, log_var):
    return torch.randn_like(mu) * torch.exp(log_var) + mu
