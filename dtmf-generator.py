import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


class DtmfGenerator:
    DTMF_TABLE = {
        "1": np.array([1209, 697]),
        "2": np.array([1336, 697]),
        "3": np.array([1477, 697]),
        "A": np.array([1633, 697]),
        "4": np.array([1209, 770]),
        "5": np.array([1336, 770]),
        "6": np.array([1477, 770]),
        "B": np.array([1633, 770]),
        "7": np.array([1209, 852]),
        "8": np.array([1336, 852]),
        "9": np.array([1477, 852]),
        "C": np.array([1633, 852]),
        "*": np.array([1209, 941]),
        "0": np.array([1336, 941]),
        "#": np.array([1477, 941]),
        "D": np.array([1633, 941]),
    }

    def __init__(
        self, phone_number: str, Fs: np.float, time: np.float, delay: np.float, debug=False: bool
    ):
        self.phone_number = phone_number
        self.Fs = Fs
        self.time = time
        self.delay = delay

        self.signal = self.compose(self.phone_number, self.Fs, self.time, self.delay)

    def __dtmf_function(
        self, number: str, Fs: np.float, time: np.float, delay: np.float
    ) -> np.array:
        """
        Function which generate DTMF tone (samples) to one specific character and its delay

        :number: Represents the character to be converted to DTMF tone
        :Fs: Sample frequency used to generate the signal in Hz
        :time: Duration of each tone in seconds
        :delay: Duration of delay between each tone in seconds

        :return: Array with samples to the DTMF tone and delay
        """

        time_tone = np.arange(0, time, (1 / Fs))
        time_delay = np.arange(0, delay, (1 / Fs))

        tone_samples = np.sin(
            2 * np.pi * self.DTMF_TABLE[number][0] * time_tone
        ) + np.sin(2 * np.pi * self.DTMF_TABLE[number][1] * time_tone)
        delay_samples = np.sin(2 * np.pi * 0 * time_delay)

        return (
            np.append(tone_samples, delay_samples)
        ) / 2  # divide by 2 to normalize between -1 and 1

    def compose(
        self, phone_number: str, Fs: np.float, time: np.float, delay: np.float
    ) -> np.array:
        """
        Function which generate DTMF tones (samples) to the phone number

        :number: Represents the number to be converted to DTMF tone
        :Fs: Sample frequency used to generate the signal in Hz
        :time: Duration of each tone in seconds
        :delay: Duration of delay between each tone in seconds

        :return: Array with samples to the DTMF tone and delay
        """
        signal = np.array([])

        for number in phone_number:
            tone_delay_signal = self.__dtmf_function(number, Fs, time, delay)
            signal = np.append(signal, tone_delay_signal)

            if

        return signal

    def test_signal(self, filename: str, Fs: np.float, time: np.float, delay: np.float):
        rate, signal = wav.read(filename)
        for i in np.arange(0, 6):
            self.test_tone(signal[(i * Fs) : (i + 1) * Fs], Fs, time, delay)

    def test_tone(
        self, signal: np.array, Fs: np.float, time: np.float, delay: np.float
    ):
        tone_fft = np.fft.fft(signal, Fs)
        time_tone = np.arange(0, time + (1 / Fs), (1 / Fs))
        freq = np.fft.fftfreq(len(tone_fft), time_tone[1] - time_tone[0])

        # plt.figure()
        # plt.plot(freq, np.abs(tone_fft))
        # plt.xlim([600, 2000])
        # plt.show()


def main():
    dtmf = DtmfGenerator("91987651279", 8000, 0.08, 0.08)
    wav.write("file.wav", dtmf.Fs, dtmf.signal)
    dtmf.test_signal("file.wav", dtmf.Fs, dtmf.time, dtmf.delay)


if __name__ == "__main__":
    main()
