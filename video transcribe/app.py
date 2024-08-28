from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from pytube import YouTube
from deep_translator import GoogleTranslator
import os

app = Flask(__name__)

def extract_audio_chunks(video_file_path, chunk_duration=5):
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    duration = video_clip.duration
    chunks = []

    start_time = 0
    while start_time < duration:
        end_time = min(start_time + chunk_duration, duration)
        audio_chunk = audio_clip.subclip(start_time, end_time)
        chunk_file_path = f'audio_chunk_{int(start_time)}_{int(end_time)}.wav'
        audio_chunk.write_audiofile(chunk_file_path, codec='pcm_s16le')
        chunks.append((chunk_file_path, start_time, end_time))
        start_time = end_time

    audio_clip.close()
    video_clip.close()
    return chunks

def transcribe_audio_chunk(audio_chunk_path):
    recognizer = sr.Recognizer()
    transcription_data = []

    with sr.AudioFile(audio_chunk_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, show_all=True)
            if 'alternative' in text:
                for result in text['alternative']:
                    transcription_data.append(result.get('transcript', ''))
            else:
                transcription_data.append("No transcriptions found.")
        except sr.UnknownValueError:
            transcription_data.append("Audio is not clear.")
        except sr.RequestError as e:
            transcription_data.append(f"API request error: {e}")

    return transcription_data

def translate_subtitles(subtitles, target_language):
    translator = GoogleTranslator(source='auto', target=target_language)
    for subtitle in subtitles:
        translated_text = translator.translate(subtitle['transcript'])
        subtitle['transcript'] = translated_text
    return subtitles

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    video_file_path = None
    youtube_url = request.form.get('youtube_url')
    target_language = request.form.get('language', 'en')

    try:
        if youtube_url:
            # Download video from YouTube
            yt = YouTube(youtube_url)
            video_stream = yt.streams.filter(only_audio=True).first()
            video_file_path = video_stream.download(filename='youtube_video.mp4')
        else:
            # Handle uploaded video
            video_file = request.files['video']
            video_file_path = 'uploaded_video.mp4'
            video_file.save(video_file_path)
        
        # Extract audio in chunks
        chunks = extract_audio_chunks(video_file_path)

        # Transcribe each chunk
        subtitle_data = []
        for chunk_file_path, start_time, end_time in chunks:
            chunk_transcriptions = transcribe_audio_chunk(chunk_file_path)
            for transcript in chunk_transcriptions:
                subtitle_data.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'transcript': transcript
                })

        # Translate subtitles if necessary
        if target_language != 'en':
            subtitle_data = translate_subtitles(subtitle_data, target_language)

    except Exception as e:
        subtitle_data.append({"error": str(e)})
    finally:
        # Clean up the temporary files
        if video_file_path and os.path.exists(video_file_path):
            os.remove(video_file_path)
        for chunk_file_path, _, _ in chunks:
            if os.path.exists(chunk_file_path):
                os.remove(chunk_file_path)

    return jsonify(subtitle_data)

if __name__ == '__main__':
    app.run(debug=True)
