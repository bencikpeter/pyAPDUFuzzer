from scipy import signal

class Filter:
    @staticmethod
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    @staticmethod
    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = Filter.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

    @staticmethod
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = Filter.butter_lowpass(cutoff, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

