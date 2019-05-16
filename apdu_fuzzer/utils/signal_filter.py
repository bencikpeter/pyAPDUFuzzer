from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import medfilt
import numpy as np
import pandas as pd
from numba.decorators import jit
from matplotlib.mlab import find
from copy import copy, deepcopy
from fastdtw import fastdtw, dtw
from statistics import median

card_levels = [-793, 200, 340, 500, 1707, 2207]# [1900, 2075, 2200]
trimming_threshold = 250


class Filter:
    @staticmethod
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    @staticmethod
    def butter_highpass_filter(data, cutoff, fs=500000, order=5):
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
    def butter_lowpass_filter(data, cutoff, fs=500000, order=5):
        b, a = Filter.butter_lowpass(cutoff, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

# -----------------------------------------------------------
    @staticmethod
    def freq_zero_crossing(
            sig):
        """
        Frequency estimation from zero crossing method
        sig - input signal
        fs - sampling rate

        return:
        dominant period

        code credit: http://qingkaikong.blogspot.com/2017/01/signal-processing-finding-periodic.html#
        """
        # Find the indices where there's a crossing
        indices = find((sig[1:] >= 0) & (sig[:-1] < 0))

        # Let's calculate the real crossings by interpolate
        crossings = [i - sig[i] / (sig[i + 1] - sig[i]) for i in indices]

        # Let's get the time between each crossing
        # the diff function will get how many samples between each crossing
        # we divide the sampling rate to get the time between them
        delta_t = np.diff(crossings)

        # Get the mean value for the period
        period = np.mean(delta_t)

        return period, delta_t

    @staticmethod
    def find_period(clock_trace):
        mean = np.mean(clock_trace)
        estimated_frequency, delta_crossings = Filter.freq_zero_crossing(clock_trace - mean)
        return int(round(estimated_frequency))

    @staticmethod
    def find_deltas(clock_trace):
        mean = np.mean(clock_trace)
        estimated_frequency, delta_crossings = Filter.freq_zero_crossing(clock_trace - mean)
        return delta_crossings

    @staticmethod
    def clocknoise_filter_static(traces, period=2389): # magic number for 500000Hz sampling frequency - setup specific
        # TODO: make dependent on sampling frequency?
        # TODO: what if pre-trigger data in not long enough? trigger new mesurement? discard data?
        clean = []
        for i in range(len(traces)):
            clock_signal = traces[i][0:period]

            while len(clock_signal) < len(traces[i]):
                clock_signal = np.hstack([clock_signal, clock_signal])

            clock_signal = clock_signal[0:len(traces[i])]

            clean.append(traces[i] - clock_signal)
        return clean

    @staticmethod
    def clocknoise_remover_dynamic(trace, clock_trace):
        period = Filter.find_period(clock_trace)
        clock_signal = trace[0:period]

        while len(clock_signal) < len(trace):
            clock_signal = np.hstack([clock_signal, clock_signal])

        clock_signal = clock_signal[0:len(trace)]
        clean = trace - clock_signal

        return clean

    @staticmethod
    def clocknoise_remover_variable(trace, clock_trace):
        deltas = np.around(Filter.find_deltas(clock_trace))
        deltas = deltas.astype("int")
        max_delta = max(deltas)
        signal_sample = trace[0:max_delta]

        # alternatively use cycles and average - did not provide any significant benefit
        # signal_sample = [sum(x) / len(x) for x in zip(*[trace[0:max_delta], trace[max_delta:2*max_delta]])]

        clock_signal = signal_sample[0:deltas[0]]
        for d in deltas[1:]:
            clock_signal = np.hstack([clock_signal, signal_sample[0:d]])

        trace = trace[0:len(clock_signal)]

        clean = trace - clock_signal
        return clean

    @staticmethod
    def clocknoise_filter_variable(traces, clock_traces):
        assert len(traces) == len(clock_traces)

        clean = []
        for i in range(len(traces)):
            clean.append(Filter.clocknoise_remover_variable(traces[i], clock_traces[i]))
        return clean

    @staticmethod
    def clocknoise_filter_dynamic(traces, clock_traces):
        assert len(traces) == len(clock_traces)

        clean = []
        for i in range(len(traces)):
            clean.append(Filter.clocknoise_remover_dynamic(traces[i], clock_traces[i]))
        return clean


# -----------------------------------------------------------

    @staticmethod
    def align(traces, fast=True, radius=1): # TODO: code credit Jan J.
        reference = traces[0]
        result = [deepcopy(traces[0])]

        for trace in traces[1:]:
            if fast:
                dist, path = fastdtw(reference, trace, radius=radius)
            else:
                dist, path = dtw(reference, trace)
            result_samples = np.zeros(len(trace), dtype=trace.dtype)
            scale = np.ones(len(trace), dtype=trace.dtype)
            # print(path)
            for x, y in path:
                if x >= len(result_samples) or y >= len(trace):
                    break
                result_samples[x] = trace[y]
                scale[x] += 0
            result_samples //= scale
            del scale
            result.append(result_samples)
        return result

# -----------------------------------------------------------

    @staticmethod
    def combine_traces_avg(trace_list):
        return [sum(x) / len(x) for x in zip(*trace_list)]

    @staticmethod
    def combine_traces_median(trace_list):
        return [median(x) for x in zip(*trace_list)]

    @staticmethod
    def combine_traces_wighted_avg_median_distance(trace_list):
        res_list = []
        for x in zip(*trace_list):
            med = median(x)
            data = list(x)
            weights = [1/(abs(i-med)+1) for i in data]  # +1 solves the problem of infinitesimally small (or zero) distances
            values = [data[i]*weights[i] for i in range(len(data))]
            res_list.append(sum(values)/sum(weights))
        return res_list

    @staticmethod
    def combine_traces_wighted_avg_mean_distance(trace_list):
        res_list = []
        for x in zip(*trace_list):
            avg = sum(x)/len(x)
            data = list(x)
            weights = [1/(abs(i-avg)+1) for i in x]  # +1 solves the problem of infinitesimally small (or zero) distances
            values = [data[i] * weights[i] for i in range(len(data))]
            res_list.append(sum(values) / sum(weights))
        return res_list

    @staticmethod
    def combine_traces_wighted_avg_mean_distance_to_others(trace_list):
        res_list = []
        for x in zip(*trace_list):
            avg = sum(x)/len(x)
            data = list(x)
            weights = []
            for i in range(len(data)):
                weights.append(1/(sum([abs(data[i]-data[q]) for q in range(len(data))])/len(data)))
            values = [data[i] * weights[i] for i in range(len(data))]
            res_list.append(sum(values) / sum(weights))
        return res_list

    @staticmethod
    def combine_traces_wighted_avg_median_distance_to_others(trace_list):
        res_list = []
        for x in zip(*trace_list):
            avg = sum(x) / len(x)
            data = list(x)
            weights = []
            for i in range(len(data)):
                weights.append(1 / median([abs(data[i] - data[q]) for q in range(len(data))]))
            values = [data[i] * weights[i] for i in range(len(data))]
            res_list.append(sum(values) / sum(weights))
        return res_list


# -----------------------------------------------------------
    @staticmethod
    @jit
    def thresholding_algo(y, lag=10, threshold=30, influence=0.25):
        signals = np.zeros(len(y))
        filteredY = np.array(y)
        avgFilter = np.zeros(len(y))
        stdFilter = np.zeros(len(y))
        avgFilter[lag - 1] = np.mean(y[0:lag])
        stdFilter[lag - 1] = np.std(y[0:lag])
        for i in range(lag, len(y) - 1):
            if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
                if y[i] > avgFilter[i - 1]:
                    signals[i] = 1
                else:
                    signals[i] = -1

                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
                avgFilter[i] = np.mean(filteredY[(i - lag):i])
                stdFilter[i] = np.std(filteredY[(i - lag):i])
            else:
                signals[i] = 0
                filteredY[i] = y[i]
                avgFilter[i] = np.mean(filteredY[(i - lag):i])
                stdFilter[i] = np.std(filteredY[(i - lag):i])

        return dict(signals=np.asarray(signals),
                    avgFilter=np.asarray(avgFilter),
                    stdFilter=np.asarray(stdFilter))

# -----------------------------------------------------------

    @staticmethod
    def create_bins(delimiters): # []
        delimiters.sort()

        bins = [(0, (delimiters[0] + delimiters[1]) / 2)]

        for i in range(1, len(delimiters)-1, 1):
            bins.append(((delimiters[i]+delimiters[i-1])/2+1, (delimiters[i]+delimiters[i+1])/2))

        bins.append(((delimiters[len(delimiters)-1]+delimiters[len(delimiters)-2])/2+1, 10000))

        return bins, delimiters

    @staticmethod
    def discretize(trace):
        bins, centers = Filter.create_bins(copy(card_levels))

        discretized = []
        for i in range(len(trace)):
            for j in range(len(bins)):
                if bins[j][0] <= trace[i] <= bins[j][1]:
                    discretized.append(j)

        return discretized


# -----------------------------------------------------------

    @staticmethod
    def quantify(power_data, frequency, pre_samples):  # power_data = {"power":[], "timing":[]}

        ripple_crop = 500  # first few samples cropped due the initial ripple after low-pass filter
        power_list = copy([i[0] for i in power_data["power"]])
        clock_list = copy([i[1] for i in power_data["power"]])

        # 1. Filter the noise
        for i in range(len(power_list)):
            power_list[i] = Filter.butter_lowpass_filter(power_list[i], 2000, frequency)[ripple_crop:]
            clock_list[i] = Filter.butter_lowpass_filter(clock_list[i], 2000, frequency)[ripple_crop:]

        power_list = Filter.clocknoise_filter_variable(power_list, clock_list)

        trimmed_list = []
        for i in range(len(power_list)):
            try:
                first = next(i for i, v in enumerate(power_list[i]) if v >= trimming_threshold)
                last = next(i for i, v in reversed(list(enumerate(power_list[i]))) if v >= trimming_threshold)

                trimmed_list.append(power_list[i][first:last])
            except StopIteration as e: #probably trigger misfire, mesurement is lost
                pass

        if len(trimmed_list) == 0:
            return []
        power_list = trimmed_list

        power = Filter.combine_traces_median(power_list)

        power = medfilt(power, 501)

        power = Filter.discretize([int(x) for x in power])
        power = np.interp(np.arange(0, len(power), 100), np.arange(0, len(power)), power)

        return power
