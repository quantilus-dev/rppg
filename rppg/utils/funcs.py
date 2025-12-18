import torch
import numpy as np
from scipy import signal
from scipy.sparse import spdiags
from rppg.utils.visualization import hrv_comparison_plot, hr_comparison_bpf

from matplotlib import pyplot as plt
# Lazy import of neurokit2 to avoid Python 3.10+ syntax issues in older versions
_nk = None

def _get_nk():
    """Lazy import of neurokit2 to avoid import-time errors."""
    global _nk
    if _nk is None:
        import neurokit2 as nk
        _nk = nk
    return _nk


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This  is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = len(signal)

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def detrend_torch(signals, Lambda=100):
    """
    Detrend 1D signals with diagonal matrix D, using torch batch matrix multiplication

    :param signals: Singals with linear trend
    :param Lambda:
    :return:
    """
    test_n, length = signals.shape

    H = torch.eye(length)
    ones = torch.ones(length - 2)

    diag1 = torch.cat((torch.diag(ones), torch.zeros((length - 2, 2))), dim=-1)
    diag2 = torch.cat((torch.zeros((length - 2, 1)), torch.diag(-2 * ones), torch.zeros((length - 2, 1))), dim=-1)
    diag3 = torch.cat((torch.zeros((length - 2, 2)), torch.diag(ones)), dim=-1)
    D = diag1 + diag2 + diag3

    detrended_signal = torch.bmm(signals.unsqueeze(1),
                                 (H - torch.linalg.inv(H + (Lambda ** 2) * torch.t(D) @ D)).to('cuda').expand(test_n,
                                                                                                              -1,
                                                                                                              -1)).squeeze()
    return detrended_signal


def BPF(input_val, fs=30, low=0.75, high=2.5):
    low = low / (0.5 * fs)
    high = high / (0.5 * fs)
    [b_pulse, a_pulse] = signal.butter(6, [low, high], btype='bandpass')
    if type(input_val) == torch.Tensor:
        return signal.filtfilt(b_pulse, a_pulse, np.double(input_val.cpu().numpy()))
    else:
        return signal.filtfilt(b_pulse, a_pulse, np.double(input_val))


def plot_graph(start_point, length, target, inference):
    plt.rcParams["figure.figsize"] = (14, 5)
    plt.plot(range(len(target[start_point:start_point + length])), target[start_point:start_point + length],
             label='target')
    plt.plot(range(len(inference[start_point:start_point + length])), inference[start_point:start_point + length],
             label='inference')
    plt.legend(fontsize='x-large')
    plt.show()


def normalize(input_val):
    return (input_val - np.mean(input_val)) / np.std(input_val)


def normalize_torch(input_val):
    if type(input_val) != torch.Tensor:
        input_val = torch.from_numpy(input_val.copy())
    min = torch.min(input_val, dim=-1, keepdim=True)[0]
    max = torch.max(input_val, dim=-1, keepdim=True)[0]
    return (input_val - min) / (max - min)


def _nearest_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def get_hrv(ppg_signal, fs=30.):
    nk = _get_nk()  # Lazy import
    ppg_peaks = nk.ppg_findpeaks(ppg_signal, sampling_rate=fs)['PPG_Peaks']
    hrv = nk.signal_rate(ppg_peaks, sampling_rate=fs, desired_length=len(ppg_signal))
    return hrv


def calc_hr_torch(calc_type, ppg_signals, fs=30.):
    test_n, sig_length = ppg_signals.shape
    hr_list = torch.empty(test_n)
    if calc_type == "FFT":
        ppg_signals = ppg_signals - torch.mean(ppg_signals, dim=-1, keepdim=True)
        N = sig_length
        k = torch.arange(N)
        T = N / fs
        freq = k / T
        amplitude = torch.abs(torch.fft.rfft(ppg_signals, n=N, dim=-1)) / N

        hr_list = freq[torch.argmax(amplitude, dim=-1)] * 60

        return hr_list
    else:  # calc_type == "Peak"
        hrv_list = -torch.ones((test_n, sig_length // fs * 10))
        index_list = -torch.ones((test_n, sig_length // fs * 10))
        width = 11  # odd / physnet(5), diff (11)
        window_maxima = torch.nn.functional.max_pool1d(ppg_signals, width, 1, padding=width // 2, return_indices=True)[
            1].squeeze()

        for i in range(test_n):
            candidate = window_maxima[i].unique()
            nice_peaks = candidate[window_maxima[i][candidate] == candidate]
            nice_peaks = nice_peaks[
                ppg_signals[i][nice_peaks] > torch.mean(ppg_signals[i][nice_peaks] / 2)]  # peak thresholding
            beat_interval = torch.diff(nice_peaks)  # sample
            hrv = beat_interval / fs  # second
            hr = torch.mean(60 / hrv)
            hr_list[i] = hr
            hrv_list[i, :len(hrv)] = hrv * 1000  # milli second
            index_list[i, :len(nice_peaks)] = nice_peaks

        hrv_list = hrv_list[:, :torch.max(torch.sum(hrv_list > 0, dim=-1))]
        index_list = index_list[:, :torch.max(torch.sum(index_list > 0, dim=-1))]

        return hr_list, hrv_list, index_list


def calculate_hr(cal_type, ppg_signal, fs=60., low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    if cal_type == "FFT":
        ppg_signal = np.expand_dims(ppg_signal, 0)
        N = _nearest_power_of_2(ppg_signal.shape[1])
        f_ppg, pxx_ppg = signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60

    else:
        hrv = get_hrv(ppg_signal, fs=fs)
        hr = np.mean(hrv, dtype=np.float32)
        # ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
        # hr = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr


def mag2db(magnitude):
    return 20. * np.log10(magnitude)


def get_hr(rppg, bvp, model_type, vital_type='HR', cal_type='FFT', fs=30, bpf=None):
    if cal_type == 'FFT' and vital_type == 'HRV':
        raise ValueError("'FFT' cannot calculate HRV. To calculate HRV, use 'PEAK' method instead.")
    if cal_type not in ['FFT', 'PEAK']:
        raise NotImplementedError("cal_type must be 'FFT' or 'PEAK'.")

    if model_type == "DIFF":
        bvp = detrend_torch(torch.cumsum(bvp, dim=-1))
        rppg = detrend_torch(torch.cumsum(rppg, dim=-1))
    else:
        bvp = detrend_torch(bvp)
        rppg = detrend_torch(rppg)

    if bpf != 'None':
        low, high = bpf
        bvp = normalize_torch(BPF(bvp, fs, low, high))
        rppg = normalize_torch(BPF(rppg, fs, low, high))
    else:
        bvp = normalize_torch(bvp)
        rppg = normalize_torch(rppg)

    # TODO: torch bpf
    hr_pred = calc_hr_torch(cal_type, rppg, fs)
    hr_target = calc_hr_torch(cal_type, bvp, fs)

    if cal_type == 'PEAK':
        hr_pred, hrv_pred, index_pred = hr_pred
        hr_target, hrv_target, index_target = hr_target
        if vital_type == 'HRV':
            return hrv_pred, hrv_target

    return hr_pred, hr_target


def get_rr(rppg, bvp, model_type, vital_type='RR', cal_type='FFT', fs=30):
    """
    Calculate Respiration Rate (RR) from rPPG and BVP signals.
    Standard Respiration Band: 0.15 Hz - 0.4 Hz (9 - 24 breaths/min)
    """
    if cal_type == 'FFT' and vital_type == 'HRV':
        raise ValueError("'FFT' cannot calculate HRV. To calculate HRV, use 'PEAK' method instead.")
    if cal_type not in ['FFT', 'PEAK']:
        raise NotImplementedError("cal_type must be 'FFT' or 'PEAK'.")

    # Detrend logic similar to Heart Rate
    if model_type == "DIFF":
        bvp = detrend_torch(torch.cumsum(bvp, dim=-1))
        rppg = detrend_torch(torch.cumsum(rppg, dim=-1))
    else:
        bvp = detrend_torch(bvp)
        rppg = detrend_torch(rppg)

    # Use Respiration Band: 0.15 Hz - 0.4 Hz
    low, high = 0.15, 0.4
    bvp = normalize_torch(BPF(bvp, fs, low, high))
    rppg = normalize_torch(BPF(rppg, fs, low, high))

    # Reuse calc_hr_torch because the math (FFT/Peak) is identical, just different frequency band
    rr_pred = calc_hr_torch(cal_type, rppg, fs)
    rr_target = calc_hr_torch(cal_type, bvp, fs)

    return rr_pred, rr_target


def get_bp(preds, targets):
    """
    Extract SBP and DBP from predictions and targets.
    Input shape: (N, T, 2) where channel 0 is SBP, channel 1 is DBP
    Returns: SBP_pred, SBP_target, DBP_pred, DBP_target (all 1D arrays of means)
    """
    # preds, targets: (N, 2) or (N, T, 2)
    # We take the mean over time T if it exists
    if len(preds.shape) == 3:
        # (N, T, 2) -> Mean over T -> (N, 2)
        sbp_pred = torch.mean(preds[:, :, 0], dim=1)
        dbp_pred = torch.mean(preds[:, :, 1], dim=1)
        sbp_target = torch.mean(targets[:, :, 0], dim=1)
        dbp_target = torch.mean(targets[:, :, 1], dim=1)
    else:
        sbp_pred = preds[:, 0]
        dbp_pred = preds[:, 1]
        sbp_target = targets[:, 0]
        dbp_target = targets[:, 1]
        
    return sbp_pred, sbp_target, dbp_pred, dbp_target


    return rr_pred, rr_target


def calculate_spo2(video_tensor, sample_rate=30):
    """
    Calculate SpO2 using Ratio-of-Ratios (RoR) from raw RGB video tensor.
    video_tensor: (T, H, W, C) or (C, T, H, W). Assumed RGB.
    Note: Requires RAW video (non-normalized) to compute DC component correctly.
    Formula: SpO2 = A - B * ( (AC_r/DC_r) / (AC_g/DC_g) )
    Standard Approx: A=110, B=25
    """
    if torch.is_tensor(video_tensor):
        video_tensor = video_tensor.detach().cpu().numpy()
    
    # Ensure (T, H, W, C) layout for easy mean calc
    if video_tensor.shape[0] == 3: # C, T, H, W
         # Permute to T, H, W, C
         video_tensor = np.transpose(video_tensor, (1, 2, 3, 0))
    elif video_tensor.shape[1] == 3 and len(video_tensor.shape) == 5: # B, C, T, H, W
         # Take first item in batch or mean over batch
         video_tensor = np.transpose(video_tensor[0], (1, 2, 3, 0))

    # 1. Spatial Averaging -> (T, C)
    # Mean over H(1), W(2)
    rgb_signals = np.mean(video_tensor, axis=(1, 2)) # Shape (T, 3)

    # 2. Extract AC/DC components
    # DC = Mean of signal
    dc_r = np.mean(rgb_signals[:, 0]) # Red
    dc_g = np.mean(rgb_signals[:, 1]) # Green
    dc_b = np.mean(rgb_signals[:, 2]) # Blue (often used as reference instead of Green)

    # AC = Std Dev of signal (or Peak-to-Peak)
    ac_r = np.std(rgb_signals[:, 0])
    ac_g = np.std(rgb_signals[:, 1])
    ac_b = np.std(rgb_signals[:, 2])
    
    # Avoid div by zero
    if dc_r == 0 or dc_g == 0:
        return 0.0

    # 3. Calculate Ratio of Ratios
    # RoR = (AC_r / DC_r) / (AC_g / DC_g)
    # Note: Some papers use Red/IR. Here we use Red/Green as proxy for standard camera.
    ratio = (ac_r / dc_r) / (ac_g / dc_g)
    
    # 4. Calibration (Approximation)
    # SpO2 = 110 - 25 * Ratio
    spo2 = 110 - 25 * ratio
    
    # Clip to realistic range
    return np.clip(spo2, 80, 100)


def MAE(pred, label):
    return np.mean(np.abs(pred - label))


def RMSE(pred, label):
    return np.sqrt(np.mean((pred - label) ** 2))


def MAPE(pred, label):
    return np.mean(np.abs((pred - label) / label)) * 100


def corr(pred, label):
    return np.corrcoef(pred, label)


def SD(pred, label):
    return np.std(pred - label)

class IrrelevantPowerRatio(torch.nn.Module):
    # we reuse the code in Gideon2021 to get irrelevant power ratio
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    def __init__(self, Fs, high_pass, low_pass):
        super(IrrelevantPowerRatio, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds):
        # Get PSD
        X_real = torch.view_as_real(torch.fft.rfft(preds, dim=-1, norm='forward'))

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X_real.shape[-2])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = torch.sum(torch.linalg.norm(X_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = torch.sum(torch.linalg.norm(X_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = torch.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = zero_energy[ii] / denom[ii]
        return energy_ratio


def sinc_impulse_response(cutoff: torch.Tensor, window_size: int = 513, high_pass: bool = False):
    # https://github.com/pytorch/audio/blob/main/torchaudio/prototype/functional/_dsp.py
    """Create windowed-sinc impulse response for given cutoff frequencies.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        cutoff (Tensor): Cutoff frequencies for low-pass sinc filter.

        window_size (int, optional): Size of the Hamming window to apply. Must be odd.
        (Default: 513)

        high_pass (bool, optional):
            If ``True``, convert the resulting filter to high-pass.
            Otherwise low-pass filter is returned. Default: ``False``.

    Returns:
        Tensor: A series of impulse responses. Shape: `(..., window_size)`.
    """
    if window_size % 2 == 0:
        raise ValueError(f"`window_size` must be odd. Given: {window_size}")

    half = window_size // 2
    device, dtype = cutoff.device, cutoff.dtype
    idx = torch.linspace(-half, half, window_size, device=device, dtype=dtype)

    filt = torch.special.sinc(cutoff.unsqueeze(-1) * idx.unsqueeze(0))
    filt = filt * torch.hamming_window(window_size, device=device, dtype=dtype, periodic=False).unsqueeze(0)
    filt = filt / filt.sum(dim=-1, keepdim=True).abs()

    # High pass IR is obtained by subtracting low_pass IR from delta function.
    # https://courses.engr.illinois.edu/ece401/fa2020/slides/lec10.pdf
    if high_pass:
        filt = -filt
        filt[..., half] = 1.0 + filt[..., half]
    return filt
