import os
import torch
from vits_modules import commons
from vits_modules.mel_processing import spectrogram_torch
from torch.utils.data import Dataset, DataLoader, Sampler
from utils import load_annotate_text, load_wav_to_torch
from text import text_to_sequence, cleaned_text_to_sequence
from config import DataConfig


class TextAudioDataset(Dataset):
    def __init__(self, data_type='train'):
        self.cleaned_text = DataConfig.cleaned_text
        self.add_blank = DataConfig.add_blank
        self.sampling_rate = DataConfig.sample_rate
        self.filter_length = DataConfig.filter_length
        self.hop_length = DataConfig.hop_length
        self.win_length = DataConfig.win_length
        self.min_text_len = DataConfig.min_text_len
        self.max_text_len = DataConfig.max_text_len
        if data_type == 'val':
            file_path = DataConfig.val_file_path
        elif data_type == 'test':
            file_path = DataConfig.test_file_path
        else:
            file_path = DataConfig.train_file_path

        self.wav_text_path = load_annotate_text(file_path)
        self.spec_lens = []
        self._filter()

    def __getitem__(self, item):
        wav_text_line = self.wav_text_path[item]
        wav_path, text = wav_text_line
        spect, wav = self.get_spect_wav(wav_path)
        text = self.get_text(text)
        return text, spect, wav

    def __len__(self):
        return len(self.wav_text_path)

    def _filter(self):
        audiopaths_and_text_new = []
        for audiopath, text in self.wav_text_path:
            if self.min_text_len <= len(text) <= self.max_text_len:
                audiopaths_and_text_new.append((audiopath, text))
                self.spec_lens.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.wav_text_path = audiopaths_and_text_new

    def get_spect_wav(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio, self.filter_length, self.hop_length, self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            if DataConfig.save_spect:
                torch.save(spec, spec_filename)
        return spec, audio

    def get_text(self, text):
        text = text.replace('\n', '')
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm


class SortedSampler(Sampler):
    def __init__(self, dataset, batch_size, boundaries, shuffle=True):
        super().__init__(dataset)
        self.dataset= dataset
        self.spec_lens = dataset.spec_lens
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.shuffle = shuffle
        self.epoch = 0

        self.buckets, self.num_samples_per_bucket = self._create_buckets()

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.spec_lens)):
            idx = self.binary_search(self.spec_lens[i])
            if idx != -1:
                buckets[idx].append(i)
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            num_samples_per_bucket.append(len(buckets[i]))
        return buckets, num_samples_per_bucket

    def binary_search(self, x, left=0, right=None):
        if right is None:
            right = len(self.boundaries) - 1

        if right > left:
            mid = (right + left) >> 1
            if self.boundaries[mid] < x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self.binary_search(x, left, mid)
            else:
                return self.binary_search(x, mid + 1, right)
        else:
            return -1

    def set_epochs(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            ids_bucket = indices[i]
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size: (j + 1) * self.batch_size]]
                batches.append(batch)
            if len(ids_bucket) % self.batch_size != 0:
                rem = len(ids_bucket) % self.batch_size
                batches.append(ids_bucket[len(ids_bucket) - rem: len(ids_bucket)])

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        return iter(batches)

    def __len__(self):
        return len(self.dataset)

class TextAudioSpeakerDataset(Dataset):
    pass




def single_speaker_collate_fn(batch):
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[1].size(1) for x in batch]),
        dim=0, descending=True)

    max_text_len = max([len(x[0]) for x in batch])
    max_spec_len = max([x[1].size(1) for x in batch])
    max_wav_len = max([x[2].size(1) for x in batch])

    text_lengths = torch.LongTensor(len(batch))
    spec_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))

    text_padded = torch.LongTensor(len(batch), max_text_len)
    spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
    wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
    text_padded.zero_()
    spec_padded.zero_()
    wav_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]

        text = row[0]
        text_padded[i, :text.size(0)] = text
        text_lengths[i] = text.size(0)

        spec = row[1]
        spec_padded[i, :, :spec.size(1)] = spec
        spec_lengths[i] = spec.size(1)

        wav = row[2]
        wav_padded[i, :, :wav.size(1)] = wav
        wav_lengths[i] = wav.size(1)

    return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


