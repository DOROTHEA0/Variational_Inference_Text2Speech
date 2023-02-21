from dataclasses import dataclass

@dataclass
class DataConfig:
    torchaudio_backend: str = 'soundfile'
    language_model: str = 'english'
    train_file_path: str = 'filelists/ljs_audio_text_train_filelist.txt.cleaned'
    val_file_path: str = 'filelists/ljs_audio_text_val_filelist.txt.cleaned'
    test_file_path: str = 'filelists/ljs_audio_text_test_filelist.txt.cleaned'
    training_annotate_text: str = ''
    cleaned_text: bool = True
    add_blank: bool = False
    sample_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_speakers: int = 1
    min_text_len: int = 1
    max_text_len: int = 200
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    save_spect: bool = True




@dataclass
class TrainingConfig:
    log_interval: int = 200
    eval_interval: int = 1000
    seed: int = 1234
    epochs: int = 20000
    learning_rate: float = 2e-4
    betas: tuple = (0.8, 0.99)
    eps: int = 1e-9
    batch_size: int = 64
    fp16_run: bool = True
    lr_decay: float = 0.999875
    segment_size: int = 8192
    init_lr_ratio: int = 1
    warmup_epochs: int = 0
    c_mel: float = 45
    c_kl: float = 1.0


@dataclass
class ModelConfig:
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = '1'
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: tuple = (8, 8, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: tuple = (16, 16, 4, 4)
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 0

