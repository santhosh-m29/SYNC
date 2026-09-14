"""Single PortAudio stream and a bounded single-producer/single-consumer ring."""
import numpy as np


class AudioRing:
    def __init__(self, capacity, channels=2):
        self.data = np.zeros((capacity, channels), dtype=np.float32)
        self.capacity = capacity
        self.read_frame = self.write_frame = 0
        self.underruns = 0

    @property
    def available(self):
        return self.write_frame - self.read_frame

    def write(self, samples):
        count = len(samples)
        if count > self.capacity - self.available:
            return False
        start = self.write_frame % self.capacity
        first = min(count, self.capacity - start)
        self.data[start:start + first] = samples[:first]
        self.data[:count - first] = samples[first:]
        self.write_frame += count  # publish only after samples are copied
        return True

    def read_into(self, output):
        count = min(len(output), self.available)
        start = self.read_frame % self.capacity
        first = min(count, self.capacity - start)
        output[:first] = self.data[start:start + first]
        output[first:count] = self.data[:count - first]
        if count < len(output):
            output[count:].fill(0)
            self.underruns += 1
        self.read_frame += count


class OutputDevice:
    def __init__(self, ring, sample_rate, blocksize=512, device=None):
        import sounddevice as sd
        self.ring = ring
        self.status_count = 0
        self.underflows = 0
        try:
            self.stream = sd.OutputStream(samplerate=sample_rate, channels=2, dtype="float32",
                blocksize=blocksize, latency="high", device=device, callback=self._callback)
        except sd.PortAudioError as error:
            native = sd.query_devices(device, "output")["default_samplerate"]
            raise RuntimeError(f"Cannot open audio output at {sample_rate} Hz; device native rate is {native:g} Hz: {error}") from error

    def _callback(self, outdata, frames, time_info, status):
        # No decoding, DSP, file I/O, queue locks, logging, or future waits here.
        if status:
            self.status_count += 1
            if status.output_underflow:
                self.underflows += 1
        self.ring.read_into(outdata)

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()
