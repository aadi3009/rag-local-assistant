from tts import Synthesizer
import sounddevice as sd
import numpy as np





# Path to your ONNX model file
model_path = 'glados.onnx'

# Initialize the Synthesizer with the model path. Change use_cuda to False if not using an NVIDIA GPU.
synth = Synthesizer(model_path, use_cuda=True)

while True:
    text_to_speak = input("Enter text to synthesize (type 'exit' to quit): ")
    if text_to_speak.lower() == 'exit':
        break

    # Generate speech
    audio_data = synth.generate_speech_audio(text_to_speak)

    # Play the audio directly
    sd.play(audio_data, 22050)  # 22050 is the sample rate based on the JSON configuration of the model
    sd.wait()  # Wait until audio is finished playing
    print('Audio has been played.')

print("Exiting the program.")
