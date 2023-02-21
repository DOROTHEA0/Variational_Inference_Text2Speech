import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from config import TrainingConfig, DataConfig, ModelConfig
from text.symbols import symbols, symbols_zh
from dataset import TextAudioDataset, single_speaker_collate_fn, SortedSampler


def main():
    tokens = symbols if DataConfig.language_model == 'english' else symbols_zh

    train_dataset = TextAudioDataset(data_type='train')
    train_sampler = SortedSampler(dataset=train_dataset, boundaries=[32,300,400,500,600,700,800,900,1000],
                                  batch_size=TrainingConfig.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=8,
                              collate_fn=single_speaker_collate_fn, shuffle=False, pin_memory=True)


    net_g = Generator(len(tokens), DataConfig.filter_length // 2 + 1, TrainingConfig.segment_size // DataConfig.hop_length, **vars(ModelConfig())).cuda()
    net_d = Discriminator(ModelConfig.use_spectral_norm).cuda()

    optim_g = torch.optim.AdamW(net_g.parameters(), TrainingConfig.learning_rate, betas=TrainingConfig.betas, eps=TrainingConfig.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), TrainingConfig.learning_rate, betas=TrainingConfig.betas, eps=TrainingConfig.eps)






if __name__ == '__main__':
    main()