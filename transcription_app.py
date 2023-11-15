import streamlit as st
from transformers import pipeline
import torchaudio
from io import BytesIO

# Streamlit page configuration
st.set_page_config(
    page_title="Audio Transcription with Whisper",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("Audio Transcription with Whisper")
st.write("This app transcribes audio files using the Whisper model from Hugging Face.")

# Load transcription model
whisper_transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# File upload
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    audio_data_bytes = BytesIO(uploaded_file.getvalue())
    
    # Load the audio file as a waveform
    waveform, sample_rate = torchaudio.load(audio_data_bytes)
    
    # Convert the waveform tensor to a NumPy array
    audio_np = waveform.numpy()
    
    with st.spinner('Transcribing...'):
        # Perform transcription
        transcription_result = whisper_transcriber(audio_np, sampling_rate=sample_rate.item())
        transcription_text = transcription_result["text"]

    # Show the transcription
    st.subheader("Transcript")
    st.write(transcription_text)
else:
    st.warning("Please upload an audio file to transcribe.")