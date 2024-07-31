import pathlib
import time
from faster_whisper import WhisperModel
import yaml


class Transcriber:
    def __init__(self, model_size="tiny", device="cpu", transcription_dir="data/transcriptions"):
        self.model = WhisperModel(model_size, device=device, compute_type="float32")
        self.transcription_dir = pathlib.Path(transcription_dir)

    def get_transcription_info(self, audio_path, timestamp):
        start_time = time.time()
        segments, info = self.model.transcribe(audio_path, word_timestamps=True)
        inference_time = time.time() - start_time
        transcription = ''
        words_with_timestamps = []

        for segment in segments:
            transcription += segment.text + ' '
            for word in segment.words:
                words_with_timestamps.append({
                    'word': str(word.word),
                    'start': float(word.start),
                    'end': float(word.end)
                })

        transcription = transcription.strip()
        transcription_info = {
            'detected_language': info.language,
            'language_probability': info.language_probability,
            'inference_time': inference_time,
            'transcription': transcription,
            'words': words_with_timestamps
        }

        output_filename = self.transcription_dir / f"transcription_info_{timestamp}.yaml"
        with open(output_filename, 'w') as yaml_file:
            yaml.dump(transcription_info, yaml_file, default_flow_style=False, allow_unicode=True)

        return transcription_info

    def print_info(self, transcription_info, words=False):
        print("Detected language:", transcription_info['detected_language'])
        print("Language probability:", transcription_info['language_probability'])
        print("Inference time:", transcription_info['inference_time'])
        print("Transcription:", transcription_info['transcription'])
        if words:
            for word in transcription_info['words']:
                print(f"[{word['start']} -> {word['end']}]: {word['word']}")


def example():
    transcriber = Transcriber(model_size="small", device="cpu")
    recordings_dir = pathlib.Path("data/recordings")
    for filename in recordings_dir.iterdir():
        audio_path = recordings_dir / str(filename)
        timestamp = int(str(filename).split('.')[0].split('_')[-1])
        transcription_info = transcriber.get_transcription_info(
            audio_path=audio_path,
            timestamp=timestamp
        )
        transcriber.print_info(transcription_info)


# Ejemplo de uso
if __name__ == '__main__':
    example()
