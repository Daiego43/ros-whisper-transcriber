#!/usr/bin/env python3
import wave
import numpy as np
import simpleaudio as sa
from recorder import mic_factory, MicRecorder


def play_audio(filepath, volume=10):
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        audio_data = wf.readframes(num_frames)

    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = audio_array * volume
    audio_array = np.clip(audio_array, -32768, 32767)
    audio_data = audio_array.astype(np.int16).tobytes()

    wave_obj = sa.WaveObject(audio_data, num_channels, sample_width, sample_rate)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def main_manual(recorder):
    while True:
        input("Press Enter to start recording...")
        print("Recording...")
        filepath = recorder.start_recording()
        print(filepath)
        play_audio(filepath)
        print("Transcribing...")
        transcription_info = recorder.transcribe_audio(filepath)
        recorder.transcriber.print_info(transcription_info)


if __name__ == "__main__":
    mic_recorder = MicRecorder(device_index=10, rate=48000, chunk=1024, device='cpu', model_size='tiny')
    # mic_recorder = mic_factory()
    try:
        main_manual(mic_recorder)
    finally:
        print("Closing recorder")
