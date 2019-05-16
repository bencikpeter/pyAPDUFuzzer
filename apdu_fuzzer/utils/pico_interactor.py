import ctypes
import copy
from picosdk.ps4000 import ps4000 as ps
from picosdk.functions import adc2mV, assert_pico_ok
from .signal_filter import Filter
from time import sleep

trigger_value = 4000 #ADC


class PicoInteractor:
    """
    A class used to contain functionality needed to control PicoScope 4224

    """

    def __init__(self, sampling_frequency=500000, sampling_duration=30): # duration in ms, frequency in Hz
        """

        :param sampling_frequency: sampling frequency for the oscilloscope
        :param sampling_duration:  sampling time for the oscilloscope
        """

        self.status = {}
        self.pico_handle = ctypes.c_int16()

        # Define some useful constants
        self.range_channel_A = self.range_channel_B = ps.PS4000_RANGE["PS4000_50MV"]
        self.range_channel_B = ps.PS4000_RANGE["PS4000_10V"]
        self.channel_A = ps.PS4000_CHANNEL["PS4000_CHANNEL_A"]
        self.channel_B = ps.PS4000_CHANNEL["PS4000_CHANNEL_B"]
        self.DC_coupling = ps.PICO_COUPLING["DC"]
        self.enabled = 1

        self.rising_trigger = 2
        self.oversample_size = 4
        self.downsample_ratio = 0
        self.agregation_mode = 0
        self.ps4000max_values = ctypes.c_int16(32764)

        self.timebase = None
        self.samples = 0
        self.frequency = sampling_frequency
        self.sampling_time = sampling_duration

        # Connect to the pico
        self.status["openunit"] = ps.ps4000OpenUnit(ctypes.byref(self.pico_handle), None)
        assert_pico_ok(self.status["openunit"])

        # open both channels
        self.status["setChA"] = ps.ps4000SetChannel(self.pico_handle, self.channel_A, self.enabled,
                                                    self.DC_coupling, self.range_channel_A)
        assert_pico_ok(self.status["setChA"])
        self.status["setChB"] = ps.ps4000SetChannel(self.pico_handle, self.channel_B, self.enabled,
                                                    self.DC_coupling, self.range_channel_B)
        assert_pico_ok(self.status["setChB"])

        # Find sampling frequency
        self.__set_timebase(sampling_frequency, sampling_duration)

        trigger_timeout = 3000  # milliseconds
        trigger_delay = 0  # sample periods
        trigger_milivolt_threshold = 6
        trigger_adc_threshold = trigger_value  # ADC counts TODO actually calculate this somehow - this value si +/- allright
        self.status["trigger"] = ps.ps4000SetSimpleTrigger(self.pico_handle, self.enabled, self.channel_A,
                                                           trigger_adc_threshold, self.rising_trigger, trigger_delay,
                                                           trigger_timeout)
        assert_pico_ok(self.status["trigger"])

        self.__set_buffers()

    def start_measurement(self, sampling_time=0): # here just start pico measure
        """
        Starts the oscilloscope measurement

        :param sampling_time: the time to be measured, if not specified time or shorter than previous
                from previous call will be used,
        :return: approximate time the measurment will take
        """
        if sampling_time > self.sampling_time:
            self.__set_timebase(self.frequency, sampling_time)
            self.__set_buffers()

        measuring_time = ctypes.c_int16()  # milliseconds
        self.status["runBlock"] = ps.ps4000RunBlock(self.pico_handle, self.pre_samples, self.post_samples,
                                                    self.timebase, self.oversample_size,
                                                    ctypes.byref(measuring_time), 0, None, None)
        assert_pico_ok(self.status["runBlock"])
        sleep(0.05)  # important for pre-trigger sample collection - at least 5ms of idle data needed
        return measuring_time

    def get_measured_data(self):  # here wait for measured data
        """
        Waits for the oscilloscope to finish measurement and provides the resulting data

        :return: raw measured data from both channels
        """
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status["isReady"] = ps.ps4000IsReady(self.pico_handle, ctypes.byref(ready))

        # create overflow loaction
        overflow = ctypes.c_int16()
        # create converted type maxSamples
        cmax_samples = ctypes.c_int32(self.samples)

        # TODO: should I zero the buffers? what will be the eventual overhead?

        # get measured data
        self.status["getValues"] = ps.ps4000GetValues(self.pico_handle, 0, ctypes.byref(cmax_samples),
                                                      self.downsample_ratio, self.agregation_mode, 0,
                                                      ctypes.byref(overflow))
        assert_pico_ok(self.status["getValues"])

        channelA_mV_values = self.buffer_A # adc2mV(self.buffer_A, self.range_channel_A, self.ps4000max_values) - useful but takes too much time when processng
        channelB_mV_values = self.buffer_B # adc2mV(self.buffer_B, self.range_channel_B, self.ps4000max_values)

        return copy.copy(channelA_mV_values), copy.copy(channelB_mV_values)

    def get_quantified_data(self, timing=30):
        """ Deprecated function, used as a first draft of trace quantification """

        ch_a, ch_b = self.get_measured_data() # channel B currently not used

        power = Filter.butter_lowpass_filter(ch_a, 5000, self.frequency)
        power = Filter.butter_highpass_filter(power, 100, self.frequency)
        n = 100
        samples = timing // self.period_s
        power = power[self.pre_samples:int(self.pre_samples + samples)]
        power = [int(sum(power[i:i + n]) // n) for i in range(0, len(power), n)]
        power = [1000 + i for i in power]
        ratio = 255 / max(power)
        power = [int(i * ratio) for i in power]

        return power

    def quantify(self, power_data):
        """
        Expects the collection of raw data from the oscilloscope for quantification
        :param power_data: {"power":[], "timing":[]}
            Power data to be quantified
        :return: power trace vector consumable by AFL
        """
        # TODO: check if self.frequency and self.pre_samples could not be changed during measurement of the trace_set
        return Filter.quantify(power_data, self.frequency, self.pre_samples)

    def __set_buffers(self):
        """
        Allocates and binds the buffers used to store measurement
        """
        self.buffer_A = (ctypes.c_int16 * self.samples)()
        self.buffer_B = (ctypes.c_int16 * self.samples)()

        self.status["setBufferA"] = ps.ps4000SetDataBuffer(self.pico_handle, self.channel_A,
                                                           ctypes.byref(self.buffer_A),
                                                           self.samples)
        assert_pico_ok(self.status["setBufferA"])
        self.status["setBufferB"] = ps.ps4000SetDataBuffer(self.pico_handle, self.channel_B,
                                                           ctypes.byref(self.buffer_B),
                                                           self.samples)
        assert_pico_ok(self.status["setBufferB"])

    def __set_timebase(self, frequency, duration): # frequency in Hz, duration in ms
        """
        Calculates and sets the internal parameter of the oscilloscope based on sampling frequency and duration of
        the measurement

        :param frequency: sampling frequency specified for the measurement in Hz
        :param duration: duration of the measurement in milliseconds
        """
        self.period_s = 1/frequency
        period_ns = 1000000000*self.period_s # perion in nanoseconds
        duration_ns = 1000000*duration

        if period_ns <= 50:
            raise NotImplementedError  # this is quite a high frequency, maybe not worth implementing for this usecase
        else:
            self.timebase = round((self.period_s*20000000)+1)

        self.samples = round(duration_ns/period_ns) + 20000

        # whats diff between samples and maximum_samples?? max can be lower if internal mem of oscilo is not enough

        sampling_interval = ctypes.c_float()
        maximum_samples = ctypes.c_int32()
        self.status["getTimebase2"] = ps.ps4000GetTimebase2(self.pico_handle, self.timebase, self.samples,
                                                            ctypes.byref(sampling_interval), self.oversample_size,
                                                            ctypes.byref(maximum_samples), 0)
        assert_pico_ok(self.status["getTimebase2"])

        # TODO: assert self.samples vs max samples and period vs sampling interval

        if maximum_samples.value < self.samples:
            self.samples = maximum_samples.value

        self.pre_samples = 20000
        self.post_samples = self.samples - self.pre_samples

    def __del__(self):
        """
        Closes all open connections
        """
        self.status["stop"] = ps.ps4000Stop(self.pico_handle)
        assert_pico_ok(self.status["stop"])

        ps.ps4000CloseUnit(self.pico_handle)
