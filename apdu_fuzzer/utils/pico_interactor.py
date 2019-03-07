import ctypes
from picosdk.ps4000 import ps4000 as ps
from picosdk.functions import adc2mV, assert_pico_ok


class PicoInteractor:

    def __init__(self, sampling_frequency=50000, sampling_duration=30): # duration in ms, frequency in Hz

        self.status = {}
        self.pico_handle = ctypes.c_int16()

        # Define some useful constants
        self.range_channel_A = self.range_channel_B = ps.PS4000_RANGE["PS4000_50MV"]
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

        # should a program have a trigger??
        # TODO: play with triggering - should some even be here?

        # Find sampling frequency
        self.__set_timebase(sampling_frequency, sampling_duration)

        trigger_timeout = 300  # milliseconds
        trigger_delay = 0  # sample periods
        trigger_milivolt_threshold = 6
        trigger_adc_threshold = 1700  # ADC counts TODO actually calculate this somehow - this value si +/- allright
        self.status["trigger"] = ps.ps4000SetSimpleTrigger(self.pico_handle, self.enabled, self.channel_A,
                                                           trigger_adc_threshold, self.rising_trigger, trigger_delay,
                                                           trigger_timeout)
        assert_pico_ok(self.status["trigger"])


        # set buffers
        # note - might be a performance bottleneck - maybe move allocation to __init__?
        self.buffer_A = (ctypes.c_int16 * self.samples)()
        self.buffer_B = (ctypes.c_int16 * self.samples)()

        self.status["setBufferA"] = ps.ps4000SetDataBuffer(self.pico_handle, self.channel_A, ctypes.byref(self.buffer_A),
                                                           self.samples)
        assert_pico_ok(self.status["setBufferA"])
        self.status["setBufferB"] = ps.ps4000SetDataBuffer(self.pico_handle, self.channel_B, ctypes.byref(self.buffer_B),
                                                           self.samples)
        assert_pico_ok(self.status["setBufferB"])

    def start_measurement(self): # here just start pico measure

        measuring_time = ctypes.c_int16()  # milliseconds
        self.status["runBlock"] = ps.ps4000RunBlock(self.pico_handle, (self.samples//8), self.samples - (self.samples//8),
                                                    self.timebase, self.oversample_size,
                                                    ctypes.byref(measuring_time), 0, None, None)
        assert_pico_ok(self.status["runBlock"])
        return measuring_time

    def get_measured_data(self):  # here wait for measured data
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

        channelA_mV_values = self.buffer_A # adc2mV(self.buffer_A, self.range_channel_A, self.ps4000max_values)
        channelB_mV_values = self.buffer_B # adc2mV(self.buffer_B, self.range_channel_B, self.ps4000max_values)

        return channelA_mV_values, channelB_mV_values

    def __quantify(self):
        pass

    def __set_timebase(self, frequency, duration): # frequency in Hz, duration in ms
        period_s = 1/frequency
        period_ns = 1000000000*period_s # perion in nanoseconds
        duration_ns = 1000000*duration

        if period_ns <= 50:
            raise NotImplementedError  # this is quite a high frequency, maybe not worth implementing for this usecase
        else:
            self.timebase = round((period_s*20000000)+1)

        self.samples = round(duration_ns/period_ns)

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

    def __del__(self):
        self.status["stop"] = ps.ps4000Stop(self.pico_handle)
        assert_pico_ok(self.status["stop"])

        ps.ps4000CloseUnit(self.pico_handle)
