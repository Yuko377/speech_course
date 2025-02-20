from functools import partial

import librosa
import numpy as np
import scipy


class Sequential:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, inp: np.ndarray):
        res = inp
        for transform in self.transforms:
            res = transform(res)
        return res


class Windowing:
    def __init__(self, window_size=1024, hop_length=None):
        self.window_size = window_size
        self.hop_length = hop_length if hop_length else self.window_size // 2
    
    def __call__(self, waveform):
        # Your code here
        new_waveform = np.concatenate((np.zeros(self.window_size // 2), waveform, np.zeros(self.window_size // 2)))
        windows = []
        num_of_windows = (len(waveform) - self.window_size % 2) // self.hop_length + 1
        for i in range(num_of_windows):
            windows.append(new_waveform[i * self.hop_length: i * self.hop_length + self.window_size])
        return np.array(windows)
    

class Hann:
    def __init__(self, window_size=1024):
        self.value = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, (window_size - 1) / window_size, window_size)))

    
    def __call__(self, windows):
        return windows * self.value



class DFT:
    def __init__(self, n_freqs=None):
        self.n_freqs = n_freqs

    def __call__(self, windows):
        # Your code here
        exps = np.exp(-2 * np.pi * 1j * np.arange(0, windows.shape[1]) / windows.shape[1])
        
        tmp_len = None
        if windows.shape[1] % 2:
            tmp_len = (windows.shape[1] + windows.shape[1] % 2) // 2
        else:
            tmp_len = windows.shape[1] // 2 + 1

        result_len = min(self.n_freqs, tmp_len) if self.n_freqs else tmp_len
        curr_exps = np.ones(windows.shape[1], dtype=np.complex128)
        spec = []
        for _ in range(result_len):
            spec.append(np.abs(np.dot(windows, curr_exps)))
            curr_exps *= exps

        return np.array(spec).T


class Square:
    def __call__(self, array):
        return np.square(array)


class Mel:
    def __init__(self, n_fft, n_mels=80, sample_rate=22050):
        # Your code here
        self.mel_matrix = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=1, fmax=8192)
        self.mel_pinv = np.linalg.pinv(self.mel_matrix)
        # ^^^^^^^^^^^^^^


    def __call__(self, spec):
        # Your code here
        # ^^^^^^^^^^^^^^

        return spec @ self.mel_matrix.T

    def restore(self, mel):
        # Your code here
        # ^^^^^^^^^^^^^^

        return mel @ self.mel_pinv.T


class GriffinLim:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.griffin_lim = partial(
            librosa.griffinlim,
            n_iter=32,
            hop_length=hop_length,
            win_length=window_size,
            n_fft=window_size,
            window='hann'
        )

    def __call__(self, spec):
        return self.griffin_lim(spec.T)


class Wav2Spectrogram:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None):
        self.windowing = Windowing(window_size=window_size, hop_length=hop_length)
        self.hann = Hann(window_size=window_size)
        self.fft = DFT(n_freqs=n_freqs)
        # self.square = Square()
        self.griffin_lim = GriffinLim(window_size=window_size, hop_length=hop_length, n_freqs=n_freqs)

    def __call__(self, waveform):
        return self.fft(self.hann(self.windowing(waveform)))

    def restore(self, spec):
        return self.griffin_lim(spec)


class Wav2Mel:
    def __init__(self, window_size=1024, hop_length=None, n_freqs=None, n_mels=80, sample_rate=22050):
        self.wav_to_spec = Wav2Spectrogram(
            window_size=window_size,
            hop_length=hop_length,
            n_freqs=n_freqs)
        self.spec_to_mel = Mel(
            n_fft=window_size,
            n_mels=n_mels,
            sample_rate=sample_rate)

    def __call__(self, waveform):
        return self.spec_to_mel(self.wav_to_spec(waveform))

    def restore(self, mel):
        return self.wav_to_spec.restore(self.spec_to_mel.restore(mel))


class TimeReverse:
    def __call__(self, mel):
        # Your code here
        return mel[::-1,]
        # ^^^^^^^^^^^^^^



class Loudness:
    def __init__(self, loudness_factor):
        # Your code here
        self.lf = loudness_factor
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        return mel * self.lf
        # ^^^^^^^^^^^^^^




class PitchUp:
    def __init__(self, num_mels_up):
        # Your code here
        self.nmu = num_mels_up
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        new_mel = np.roll(mel, self.nmu, axis=1)
        new_mel[:, :self.nmu] = 0
        return new_mel
        # ^^^^^^^^^^^^^^



class PitchDown:
    def __init__(self, num_mels_down):
        # Your code here
        self.nmd = num_mels_down
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        new_mel = np.roll(mel, -1 * self.nmd, axis=1)
        new_mel[:, -1 * self.nmd:] = 0
        return new_mel
        # ^^^^^^^^^^^^^^




class SpeedUpDown:
    def __init__(self, speed_up_factor=1.0):
        # <SOLUTION_START>
        self.speed_up_factor = speed_up_factor
        # <SOLUTION_END>

    def __call__(self, mel):
        # <SOLUTION_START>
        indices = np.linspace(0, mel.shape[0] - 1, int(self.speed_up_factor * mel.shape[0]))
        indices = np.round(indices).astype("int")
        return mel[indices, :]
        # <SOLUTION_END>





class FrequenciesSwap:
    def __call__(self, mel):
        # Your code here
        return mel[:,::-1]

        # ^^^^^^^^^^^^^^



class WeakFrequenciesRemoval:
    def __init__(self, quantile=0.05):
        # Your code here
        self.q = quantile
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        mq = np.quantile(mel, self.q)
        new_mel = mel
        new_mel[mel < mq] = 0
        return new_mel
        # ^^^^^^^^^^^^^^



class Cringe1:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^



class Cringe2:
    def __init__(self):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^


    def __call__(self, mel):
        # Your code here
        raise NotImplementedError("TODO: assignment")
        # ^^^^^^^^^^^^^^

