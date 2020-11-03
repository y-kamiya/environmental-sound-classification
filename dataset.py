import torch
import torchaudio
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pandas as pd
import os
import random
import math

class BaseDataset(Dataset):
    def __init__(self, config, csv_path, audio_dir, folderList):
        super(BaseDataset, self).__init__()
        torchaudio.set_audio_backend('sox_io')
        self.config = config

        csvData = pd.read_csv(csv_path)

        self.filenames = []
        self.labels = []

        for i in range(0,len(csvData)):
            if csvData.iloc[i, 1] in folderList:
                self.filenames.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 2])

        self.audio_dir = audio_dir

    def n_files(self):
        return len(self.filenames)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.filenames)

    def data_filename(self, folderList):
        class_name = self.__class__.__name__
        folder_str = ''.join([str(n) for n in folderList])
        if self.config.segmented:
            return '{}{}.segmented.pth'.format(class_name, folder_str)
        return '{}{}.pth'.format(class_name, folder_str)


class LogmelDataset(BaseDataset):
    def __init__(self, config, csv_path, audio_dir, folderList, apply_augment=True):
        super(LogmelDataset, self).__init__(config, csv_path, audio_dir, folderList)
        self.apply_augment = apply_augment

        data_cache_path = os.path.join(self.config.dataroot, self.data_filename(folderList))
        if not os.path.exists(data_cache_path):
            frame_size = 512
            window_size = 1024
            frame_per_segment = 41
            segment_size = frame_size * frame_per_segment
            step_size = segment_size // 2

            transforms_mel = transforms.Compose([
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=22050, win_length=window_size, n_fft=window_size, hop_length=frame_size, n_mels=60, normalized=True),
                torchaudio.transforms.AmplitudeToDB(top_db=80.0),
            ])

            self.data = torch.empty(0)
            self.segment_labels = []
            self.file_ids = []
            for index, file in enumerate(self.filenames):
                path = os.path.join(self.audio_dir, file)
                tensor, _ = torchaudio.load(path)
                if not self.config.segmented:
                    data, label = self.__create_data(index, tensor, transforms_mel)
                    if data is not None:
                        self.data = torch.cat((self.data, data.unsqueeze(0)))
                        self.segment_labels.append(label)
                        self.file_ids.append(index)
                    continue

                start = 0
                clip = tensor[:, start:(start+segment_size-1)]
                while clip.shape[1] == segment_size-1:
                    data, label = self.__create_data(index, clip, transforms_mel)
                    if data is not None:
                        self.data = torch.cat((self.data, data.unsqueeze(0)))
                        self.segment_labels.append(label)
                        self.file_ids.append(index)
                    start += step_size
                    clip = tensor[:, start:(start+segment_size-1)]

            torch.save({
                'data': self.data,
                'label': self.segment_labels,
                'file_ids': self.file_ids,
            }, data_cache_path)

        loaded = torch.load(data_cache_path, map_location=torch.device(self.config.device_name))
        self.data = loaded['data']
        self.segment_labels = loaded['label']
        self.file_ids = loaded['file_ids']

        mean = self.data.mean()
        std = self.data.std()

        self.transforms_norm = transforms.Compose([
            transforms.Normalize(mean, std),
        ])

    def __create_data(self, index, wave, transforms):
        mel = transforms(wave)
        if torch.mean(mel) < -70.0:
            return None, None
        return mel, self.labels[index]

    def __augment(self, data):
        if not self.apply_augment:
            return data

        _, n_mel, n_time = data.shape

        mel_width = random.randint(0, self.config.augment_mel_width_max)
        mel_start = random.randint(0, n_mel - mel_width)
        mel_end = mel_start + mel_width

        time_width = random.randint(0, self.config.augment_time_width_max)
        time_start = random.randint(0, n_time - time_width)
        time_end = time_start + time_width

        data[0][mel_start:mel_end, :] = 0
        data[0][:, time_start:time_end] = 0
        return data

    def __getitem__(self, index):
        data = self.data[index]
        data = self.transforms_norm(data)

        data = self.__augment(data)
        label = self.segment_labels[index]

        deltas = torchaudio.functional.compute_deltas(data)
        return torch.cat((data, deltas), dim=0), label, self.file_ids[index]

    def __len__(self):
        return len(self.data)

class WaveDataset(BaseDataset):
    def __getitem__(self, index):
        path = os.path.join(self.audio_dir, self.filenames[index])
        sound = torchaudio.load(path, out = None, normalization = True)
        soundData = sound[0].permute(1, 0)

        tempData = torch.zeros([160000, 1])
        if soundData.numel() < 160000:
            tempData[:soundData.numel()] = soundData[:]
        else:
            tempData[:] = soundData[:160000]

        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5] #take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index], index

class EnvNetDataset(BaseDataset):
    SAMPLING_RATE = 16000
    INPUT_SEC = 1.5

    def __init__(self, config, csv_path, audio_dir, folderList):
        super(EnvNetDataset, self).__init__(config, csv_path, audio_dir, folderList)

        n_padding = int(self.SAMPLING_RATE * self.INPUT_SEC / 2)

        self.segment_size = int(self.SAMPLING_RATE * self.INPUT_SEC)

        data_cache_path = os.path.join(self.config.dataroot, self.data_filename(folderList))
        if not os.path.exists(data_cache_path):
            trans = transforms.Compose([
                torchaudio.transforms.Resample(44100, self.SAMPLING_RATE)
            ])
            self.sounds = []

            for i, file in enumerate(self.filenames):
                # if i > 30:
                #     break
                path = os.path.join(self.audio_dir, file)
                sound = torchaudio.load(path, normalize=True)
                resampled = trans(sound[0].squeeze())
                sound = F.pad(resampled, (n_padding, n_padding), 'constant', 0)
                self.sounds.append(sound)

            torch.save({
                'sounds': self.sounds,
            }, data_cache_path)

        loaded = torch.load(data_cache_path, map_location=torch.device('cpu'))
        self.sounds = loaded['sounds']


    # def __len__(self):
    #     return 30

    def is_enough_amplitude(self, data):
        return self.config.amplitude_threshold < torch.max(torch.abs(data))

    def random_crop(self, sound):
        max_iter = 10000
        for i in range(max_iter):
            start = random.randint(0, len(sound) - self.segment_size)
            data = sound[start : start + self.segment_size]
            if self.is_enough_amplitude(data):
                break

        if i == max_iter - 1:
            self.config.logger.warning("valid section is not found: index {}".format(index))

        return data

    def __getitem__(self, index):
        data = self.random_crop(self.sounds[index])
        return data.unsqueeze(0), self.labels[index], index

class EnvNetEvalDataset(EnvNetDataset):
    def __init__(self, config, csv_path, audio_dir, folderList):
        super(EnvNetEvalDataset, self).__init__(config, csv_path, audio_dir, folderList)

        step_size = self.step_size(self.sounds[0])
        self.sounds_segmented = []
        self.labels_segmented = []
        self.file_ids = []

        for index, sound in enumerate(self.sounds):
            start = 0
            clip = sound[start:(start+self.segment_size)]
            while clip.shape[0] == self.segment_size:
                if self.is_enough_amplitude(clip):
                    self.sounds_segmented.append(clip.unsqueeze(0))
                    self.labels_segmented.append(self.labels[index])
                    self.file_ids.append(index)

                start += step_size
                clip = sound[start:(start+self.segment_size)]

    def step_size(self, _):
        return int(self.SAMPLING_RATE * 0.2)

    def __getitem__(self, index):
        return self.sounds_segmented[index], self.labels_segmented[index], self.file_ids[index]

    def __len__(self):
        return len(self.sounds_segmented)

class BcLearningDataset(EnvNetDataset):
    def __init__(self, config, csv_path, audio_dir, folderList):
        super(BcLearningDataset, self).__init__(config, csv_path, audio_dir, folderList)

    def __getitem__(self, index):
        if not self.config.use_augment:
            sound = self.sounds[index]
            return self.random_crop(sound), self.labels[index], index

        while (True):
            rand1 = random.randint(0, len(self.sounds)) - 1
            rand2 = random.randint(0, len(self.sounds)) - 1
            label1 = self.labels[rand1]
            label2 = self.labels[rand2]
            if label1 != label2:
                sound1 = self.random_crop(self.sounds[rand1])
                sound2 = self.random_crop(self.sounds[rand2])
                break

        G1 = self.__compute_gain_max(sound1)
        G2 = self.__compute_gain_max(sound2)

        r = random.random()
        p = 1 / (1 + 10 ** (G1 - G2) / 20 * (1 - r) / r)
        sound = (p * sound1 + (1 - p) * sound2) / ((p ** 2 + (1 - p) ** 2) ** 0.5)

        target = torch.zeros(self.config.n_class)
        target[label1] = r
        target[label2] = 1 - r

        return sound.unsqueeze(0), target, index
    
    def __compute_gain_max(self, x, n_fft=2048, min_db=-80):
       stride = n_fft // 2
       gains = torch.empty(0)
       for start in range(0, len(x) - n_fft + 1, stride):
           end = start + n_fft
           gains = torch.cat((gains, torch.mean(x[start:end] ** 2).unsqueeze(0)))

       gain_max = max(torch.max(gains).item(), 10 ** (min_db / 10))
       return 10 * math.log10(gain_max)


class BcLearningEvalDataset(EnvNetEvalDataset):
    N_CROPS = 10

    def __init__(self, config, csv_path, audio_dir, folderList):
        super(BcLearningEvalDataset, self).__init__(config, csv_path, audio_dir, folderList)

    def step_size(self, sound):
        return (len(sound) - self.segment_size) // (self.N_CROPS - 1)



