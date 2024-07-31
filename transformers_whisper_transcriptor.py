import pathlib
import time
from transformers import pipeline
import yaml
import torch


class Transcriber:
    def __init__(self, model_size="medium",
                 task="automatic-speech-recognition",
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 transcription_dir="data/transcriptions"):
        self.task = task
        self.model_name = f'openai/whisper-{model_size}'
        self.device = 0 if device == 'cuda' else -1
        self.transcription_pipeline = pipeline(task=self.task, model=self.model_name, device=self.device)
        self.transcription_dir = pathlib.Path(transcription_dir)
        self.transcription_dir.mkdir(parents=True, exist_ok=True)

    def get_transcription_info(self, audio_path, timestamp):
        start_time = time.time()
        result = self.transcription_pipeline(audio_path, return_timestamps='word')
        inference_time = time.time() - start_time
        transcription = result['text']
        words_with_timestamps = []

        for segment in result['chunks']:
            words_with_timestamps.append({
                'word': segment['text'],
                'start': segment['timestamp'][0],
                'end': segment['timestamp'][1]
            })

        transcription_info = {
            'detected_language': result.get('language', 'unknown'),
            'language_probability': result.get('language_probability', 0.0),
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


# Ejemplo de uso
if __name__ == '__main__':
    transcriber = Transcriber(model_size="medium", task="automatic-speech-recognition")
    audio_path = 'data/recordings/recording_1722193596.wav'
    timestamp = int(time.time())
    info = transcriber.get_transcription_info(audio_path, timestamp)
    transcriber.print_info(info)
