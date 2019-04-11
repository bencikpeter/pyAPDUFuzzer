import smartcard
from llsmartcard.card import CAC
from smartcard.System import readers
from smartcard.sw.SWExceptions import SWException
import time
import sys

from ..config import BLACKLIST
from .logging import debug, info, warning, error
from .util import raise_critical_error
from .pico_interactor import PicoInteractor
from .signal_filter import Filter

class CardCrashedException(Exception):
    pass


class CardInteractor:

    def __init__(self, card_reader_id):
        self.card_reader_id = card_reader_id
        self.card_connection = None
        self.card = self.get_card()
        self.frequency = 500000
        self.pico = PicoInteractor(sampling_frequency=self.frequency)

    def get_card(self):
        if self.card_connection:
            self.card_connection.disconnect()
        reader_list = readers()
        card_reader = reader_list[self.card_reader_id]

        try:
            connection = card_reader.createConnection()
            self.card_connection = connection
            connection.connect()
            return CAC(connection)

        except smartcard.Exceptions.NoCardException as ex:
            raise_critical_error("card.interactor", ex)

    @staticmethod
    def _is_blacklisted(element):
        for blcklst in BLACKLIST:
            ok = False
            for key in blcklst:
                if element.inp[key] not in blcklst[key]:
                    ok = True
            if not ok:
                warning('fuzzer', 'Hit a blacklisted packet {}'.format(element.get_inp_data()))
                return True

        return False

    def send_element(self, element):
        if self._is_blacklisted(element):
            element.misc['error_status'] = 1
            return element
        try:
            res = self.send_apdu(element.get_inp_data())
            element.set_output(res[0], res[1], res[2], res[3], res[4])
        except CardCrashedException:
            element.misc['error_status'] = 1
        return element

    # noinspection PyProtectedMember
    def send_apdu(self, input_data):
        power = None
        timing = -1
        sw1 = 0x00
        sw2 = 0x00
        stri = "Trying : ", [hex(i) for i in input_data]
        debug("card.interactor", stri)
        reps = 3  # how many measurements for any operation
        power_data = {"timing": [], "power": []}
        try:
            # empty run to len osciloscope know for how long to mesure
            start = time.time()
            (data, sw1, sw2) = self.card._send_apdu(input_data)
            end = time.time()
            calibration = end-start

            for i in range(reps):
                self.pico.start_measurement(calibration)
                start = time.time()
                self.card._send_apdu(input_data)
                end = time.time()
                power_data["timing"].append(end - start)
                power_data["power"].append(self.pico.get_measured_data())

        except SWException as e:
            # Did we get an unsuccessful attempt?
            info("card.interactor", e)
        except KeyboardInterrupt:
            sys.exit()
        except smartcard.Exceptions.CardConnectionException as ex:
            warning("card.interactor", "Reconnecting the card because of {} while processing {}".format(ex, str(input_data)))
            self.card = self.get_card()
            raise CardCrashedException
        except Exception as e:
            warning("card.interactor", "{}:{}".format(type(e), e))
            (data, sw1, sw2) = ([], 0xFF, 0xFF)

        power = self.pico.quantify(power_data)
        timing = sum(power_data["timing"])/float(len(power_data["timing"])) # average timing of three operations
        stri = "Got : ", data, hex(sw1), hex(sw2)
        debug("card.interactor", stri)
        # TODO: Somewhere here should be some quantification of powertrace data added
        return sw1, sw2, data, timing, power
