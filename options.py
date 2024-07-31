import pyaudio
from rich.console import Console
from rich.prompt import Prompt


class MicOptionsSelector:
    def __init__(self):
        self.device_index = None
        self.rate = None
        self.device_info = None
        self.model_size = None

    def get_selection(self):
        return self.device_index, self.rate, self.device_info, self.model_size

    def audio_device_selector(self):
        p = pyaudio.PyAudio()
        available_devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            supported_rates = self.get_mic_rate(i)
            if supported_rates:
                available_devices.append((i, info['name'], supported_rates))
        p.terminate()
        return available_devices

    def get_mic_rate(self, device_index):
        p = pyaudio.PyAudio()
        info = p.get_device_info_by_index(device_index)
        supported_rates = []
        for rate in [8000, 16000, 22050, 44100, 48000, 96000, 192000]:
            try:
                if p.is_format_supported(rate, input_device=info["index"], input_channels=1,
                                         input_format=pyaudio.paInt16):
                    supported_rates.append(rate)
            except ValueError:
                pass
        p.terminate()
        return supported_rates

    def enriched_audio_device_selector(self):
        devices = self.audio_device_selector()
        console = Console()
        device_dict = dict(zip([i for i in range(len(devices))], devices))
        for key, value in device_dict.items():
            console.print(f"{key}: {value[1]}")
            console.print("Supported sample rates:")
            for rate in value[2]:
                console.print(f"  {rate} Hz")
        selection = Prompt.ask("Select a device", choices=[str(i) for i in range(len(device_dict.keys()))])
        selection = int(selection)
        return selection, device_dict[selection]

    def enriched_audio_device_selection_process(self):
        device_index, device_info = self.enriched_audio_device_selector()
        selected_rate = Prompt.ask(f"Select available rates for {device_info[1]}",
                                   choices=[str(rate) for rate in device_info[2]])
        self.device_index = device_index
        self.rate = int(selected_rate)
        self.device_info = device_info
        whisper_size = Prompt.ask("Select model size", choices=["tiny", "small", "medium", "large-v2"])
        self.model_size = whisper_size
