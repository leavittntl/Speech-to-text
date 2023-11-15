import streamlit as st
from transformers import pipeline
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
whisper_transcriber = pipeline("automatic-speech-recognition", model="distil-whisper/tinier-v2")

# File upload
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    audio_data = BytesIO(uploaded_file.getvalue())
    with st.spinner('Transcribing...'):
        # Perform transcription
        transcription_result = whisper_transcriber(audio_data)
        transcription_text = transcription_result["text"]

    # Show the transcription
    st.subheader("Transcript")
    st.write(transcription_text)
else:
    st.warning("Please upload an audio file to transcribe.")