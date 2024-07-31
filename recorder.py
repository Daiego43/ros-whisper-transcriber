import os
import pyaudio
import wave
import time
import torch
import pathlib
import threading
from faster_whisper_transcriptor import Transcriber
from options import MicOptionsSelector


class MicRecorder:
    def __init__(self,
                 device_index=0,
                 rate=16000,
                 chunk=16000,
                 channels=1,
                 format=pyaudio.paInt16,
                 audio_dir="data/recordings",
                 transcriptions_dir="data/transcriptions",
                 model_size="medium",
                 device='cuda' if torch.cuda.is_available() else 'cpu'
                 ):

        self.model_size = model_size
        self.device = device
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.format = format
        self.device_index = device_index
        self.audio_dir = pathlib.Path(audio_dir)
        self.transcriptions_dir = pathlib.Path(transcriptions_dir)
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.transcriber = Transcriber(model_size=self.model_size, device=self.device,
                                       transcription_dir=self.transcriptions_dir)

        if not self.audio_dir.exists():
            self.audio_dir.mkdir(parents=True)
        if not self.transcriptions_dir.exists():
            self.transcriptions_dir.mkdir(parents=True)

    def start_recording(self):
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  input_device_index=self.device_index,
                                  frames_per_buffer=self.chunk)

        self.frames = []
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        print("Recording started. Press Enter to stop.")
        input()
        return self.stop_recording()

    def _record_audio(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            except IOError as e:
                print(f"Error recording: {e}")
                self.is_recording = False

    def stop_recording(self):
        self.is_recording = False
        self.recording_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        filepath = self._save_audio()
        return filepath

    def _save_audio(self):
        timestamp = int(time.time())
        filepath = str(self.audio_dir / f"recording_{timestamp}.wav")
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        return filepath

    def _validate_audio_file(self, filepath):
        try:
            with wave.open(filepath, 'rb') as wf:
                return wf.getnframes() > 0
        except wave.Error:
            return False

    def transcribe_audio(self, filepath):
        if self._validate_audio_file(filepath):
            info = self.transcriber.get_transcription_info(filepath, int(time.time()))
            return info
        else:
            print("Invalid audio file. Skipping transcription.")
            return None

    def record_and_transcribe_chunk(self, audio_time_size=5):
        self.start_recording()
        time.sleep(audio_time_size)
        filepath = self.stop_recording()
        self.transcribe_audio(filepath)

    def __str__(self):
        return f"MicRecorder(device_index={self.device_index}, rate={self.rate})"


def mic_factory():
    option_selection = MicOptionsSelector()
    option_selection.enriched_audio_device_selection_process()
    index, rate, device_info, model_size = option_selection.get_selection()
    print(f"Selected device: {device_info}")
    mic_recorder = MicRecorder(device_index=device_info[0], rate=rate, chunk=rate, model_size=model_size, device='cpu')
    return mic_recorder
