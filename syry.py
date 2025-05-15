#!/usr/bin/env python3

import click
import requests
import urllib3
import os
import platform
import sys
import time
import tempfile
import wave
import subprocess
from dotenv import load_dotenv


# These will be conditionally imported when needed
# to avoid errors if not installed
REQUIRED_PACKAGES = {
    "faster_whisper": "faster-whisper",
    "pyaudio": "pyaudio",
    "keyboard": "keyboard"
}


load_dotenv()
PHONE_IP=os.getenv('PHONE_IP')
PHONE_USER=os.getenv('PHONE_USER')
PHONE_PASSWORD=os.getenv('PHONE_PASSWORD')


# disable insecure-HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def ensure_package(package_name):
    """
    Ensures a package is imported, prompts for installation if not available
    """
    try:
        return __import__(package_name)
    except ImportError:
        pip_name = REQUIRED_PACKAGES.get(package_name, package_name)
        click.echo(f"Required package '{pip_name}' is not installed.")
        if click.confirm(f"Do you want to install {pip_name} now?", default=True):
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return __import__(package_name)
        else:
            click.echo(f"Cannot continue without {pip_name}.")
            sys.exit(1)

@click.group()
def cli():
    """Syry - Phone and audio utility tool"""
    pass

@cli.command()
@click.argument('number')
@click.option('--verbose', is_flag=True, help='Show full HTTP response')
def call(number, verbose):
    """
    Trigger a Yealink T31P outgoing call via its HTTP action URI.
    """
    number=number.replace('+', '')
    number=number.replace(' ', '')
    url = (
        f"https://{PHONE_IP}/servlet?key={number}"
    )
    if verbose:
        click.echo(f'Calling {number} via {url}')
    # default HTTP Basic Auth credentials
    auth = (os.getenv("PHONE_USER"), os.getenv('PHONE_PASSWORD'))
    resp = requests.get(url, auth=auth, verify=False)

    if verbose:
        click.echo(f'HTTP {resp.status_code}\n\n{resp.text}')
    else:
        click.echo(resp.status_code)

@cli.command('listen')
@click.option('--model', default='turbo', help='Whisper model to use for transcription')
def listen_for_contact(model):
    """
    If platform is macOS: listen to the microphone until a key is pressed,
    then transcribe with faster-whisper turbo model.
    """
    if platform.system() != 'Darwin':
        click.echo("This function is only available on macOS.")
        return
    
    try:
        # Dynamically import required packages
        pyaudio = ensure_package("pyaudio")
        fw = ensure_package("faster_whisper")
        
        # For macOS, we'll use a different approach for key detection
        # that works better with the Core Audio framework
        import threading
        stop_recording = threading.Event()
        
        def wait_for_keypress():
            input("Press Enter to stop recording...\n")
            stop_recording.set()
        
        # Set up audio parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        stream = None
        frames = []
        
        try:
            # Start the keypress detection thread
            keypress_thread = threading.Thread(target=wait_for_keypress)
            keypress_thread.daemon = True
            keypress_thread.start()
            
            # Open stream with error handling
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            click.echo("Starting audio recording. Press Enter to stop recording...")
            
            # Start recording with proper error handling
            while not stop_recording.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:
                    click.echo(f"Warning: {e}")
                    continue
                except Exception as e:
                    click.echo(f"Error: {e}")
                    break
                    
                time.sleep(0.01)  # Small sleep to reduce CPU usage
        
        except Exception as e:
            click.echo(f"Error during recording: {e}")
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            return
            
        finally:
            # Stop and close the stream
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    click.echo(f"Error closing stream: {e}")
            
            if audio:
                audio.terminate()
        
        if not frames:
            click.echo("No audio data recorded.")
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            return
        
        try:
            click.echo("Recording stopped. Saving audio...")
            
            # Save the recorded audio to WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            
            click.echo(f"Audio saved. Transcribing with faster-whisper {model} model...")
            
            # Load the model and transcribe
            model_instance = fw.WhisperModel(model, device="cpu", compute_type="int8")
            segments, info = model_instance.transcribe(temp_filename, beam_size=5)
            
            # Print transcription
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            click.echo("\nTranscription:")
            click.echo(transcription.strip())
            
            # Return the transcription
            return transcription.strip()
            
        except Exception as e:
            click.echo(f"Error processing audio: {e}")
            return None
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                try:
                    os.unlink(temp_filename)
                except Exception as e:
                    click.echo(f"Error removing temporary file: {e}")
    except Exception as e:
        click.echo(f"Error in listen_for_contact: {e}")
        return None

if __name__ == '__main__':
    cli()
