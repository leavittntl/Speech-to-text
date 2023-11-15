import streamlit as st
import openai

# Placeholder for API key input
api_key_placeholder = "<Enter your OpenAI API key>"

# Streamlit app
def main():
    st.title('Audio Transcription with OpenAI Whisper')

    # API key input
    api_key = st.text_input("OpenAI API Key", value=api_key_placeholder)
        
    if api_key == api_key_placeholder or not api_key:
        st.warning("Please enter a valid OpenAI API key to proceed.")
        st.stop()
    else:
        openai.api_key = api_key  # Set the API key for OpenAI

    # Audio file uploader
    audio_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'ogg', 'webm'])

    if audio_file is not None:
        # Display an audio player widget
        st.audio(audio_file, format='audio.mp3', start_time=0)

        # Confirm before transcribing
        if st.button('Transcribe Audio'):
            try:
                # Show a message while the transcription is being processed
                with st.spinner('Transcribing...'):
                    # Call the transcribe function
                    transcript_response = openai.Audio.transcribe("whisper-1", audio_file)

                    # Check the response for transcription result
                    if transcript_response:
                        transcript = transcript_response.get('text', 'No transcription found.')
                        # Display the transcription
                        st.success('Transcription completed:')
                        st.write(transcript)
                    else:
                        st.error('No transcription response was received.')
            except Exception as e:
                # If an error occurs, display the error message
                st.error(f'An error occurred during transcription: {e}')

if __name__ == "__main__":
    main()