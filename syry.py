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
import llm as llm_module
import faster_whisper as fw
from dotenv import load_dotenv


# These will be conditionally imported when needed
# to avoid errors if not installed
REQUIRED_PACKAGES = {
    "pyaudio": "pyaudio",
    "keyboard": "keyboard",
    "llm": "llm",
    "lxml": "lxml"
}


load_dotenv()
PHONE_IP=os.getenv('PHONE_IP')
PHONE_USER=os.getenv('PHONE_USER')
PHONE_PASSWORD=os.getenv('PHONE_PASSWORD')
KURT_IP=os.getenv('KURT_IP')


# disable insecure-HTTPS warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
    #remove all non numeric characters
    re.sub(r'[^0-9+]', '', number)
    url = f"https://{PHONE_IP}/servlet?key={number}"
    if verbose:
        click.echo(f'Calling {number} via {url}')
    # default HTTP Basic Auth credentials
    auth = (os.getenv("PHONE_USER"), os.getenv('PHONE_PASSWORD'))
    resp = requests.get(url, auth=auth, verify=False)
    if verbose:
        click.echo(f'HTTP {resp.status_code}\n\n{resp.text}')

@cli.command('listen')
@click.option('--model', default='turbo', help='Whisper model to use for transcription')
def listen_for_contact(model):
    if platform.system() == 'Darwin':
        transcription=listen_macos(model)
    elif platform.system() == 'Linux':
        transcription=listen_linux(model)
    addressbook=get_addressbook()
    number=select_number(transcription, addressbook)
    if number:
        click.echo(f'Calling {number}...')
        call(str(number))
    else:
        click.echo('No number found.')
    
def get_addressbook():
    """
    Get the address book from Kurt.
    """
    url=f"http://{KURT_IP}:8000/phonebook.xml"
    try:
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            return response.text
        else:
            click.echo(f"Error fetching address book: {response.status_code}")
            return None
    except requests.RequestException as e:
        click.echo(f"Error fetching address book: {e}")
        return None
    
def select_number(transcription, addressbook):
    """
    Use Gemma 2 2B model to analyze the transcription and select a phone number from the address book.
    
    Args:
        transcription (str): The transcribed text from the user's speech
        addressbook (str): XML string containing the phonebook data
        
    Returns:
        str: Selected phone number or None if no match is found
    """
    if not transcription or not addressbook:
        click.echo("Missing transcription or address book data.")
        return None
    
    try:
        # Import required packages
        
        from lxml import etree
        
        click.echo("Parsing address book...")
        # Parse the XML address book
        try:
            root = etree.fromstring(addressbook.encode('utf-8'))
            entries = []
            
            # Extract entries from XML
            for entry in root.findall(".//DirectoryEntry"):
                name_elem = entry.find("Name")
                tel_elems = entry.findall("Telephone")
                
                if name_elem is not None and name_elem.text and tel_elems:
                    name = name_elem.text.strip()
                    phones = [tel.text.strip() for tel in tel_elems if tel.text]
                    if name and phones:
                        entries.append({"name": name, "phones": phones})
        except Exception as e:
            click.echo(f"Error parsing address book XML: {e}")
        
        if not entries:
            click.echo("No valid entries found in the address book.")
            return None
            
        click.echo(f"Found {len(entries)} contacts in the address book.")
        
        # Prepare the prompt for the LLM
        prompt = f"""
You are an AI assistant helping to find a contact in a phone directory based on a spoken request.

The user said: "{transcription}"

Available contacts:
"""
        for i, entry in enumerate(entries):
            phones_str = ", ".join(entry["phones"])
            prompt += f"{i+1}. {entry['name']} - {phones_str}\n"
            
        prompt += """
Based on the user's request, which contact should be called? If multiple phone numbers are available for a contact, select the most appropriate one (mobile preferred for personal contacts, office/main number for businesses).

Return ONLY the phone number to call, including country code. If no match is found or the request is unclear, return "NONE".
"""
        
        click.echo("Querying Gemma 2 2B model...")
        
        # Call the Gemma 2 2B model using the llm package
        try:
            response = llm_module.get("gemma2:2b").prompt(prompt)
            result = str(response).strip()
            click.echo(f"Model response: {result}")
            
            # Process the response to extract a valid phone number
            import re
            phone_match = re.search(r'[\+]?[0-9]{10,}', result)
            
            if phone_match:
                selected_number = phone_match.group(0)
                click.echo(f"Selected number: {selected_number}")
                return selected_number
            elif "NONE" in result.upper():
                click.echo("No matching contact found by the model.")
                return None
            else:
                # Try to match the response with phone numbers in entries
                for entry in entries:
                    for phone in entry["phones"]:
                        if phone in result:
                            click.echo(f"Found matching number in response: {phone}")
                            return phone
                
                click.echo("Could not extract a valid phone number from the model response.")
                return None
                
        except Exception as e:
            click.echo(f"Error querying the Gemma model: {e}")
            
            # Fallback: Simple keyword matching as a backup method
            transcription_lower = transcription.lower()
            best_match = None
            best_score = 0
            
            for entry in entries:
                name_lower = entry["name"].lower()
                # Simple keyword matching score (number of words from name found in transcription)
                name_words = name_lower.split()
                score = sum(1 for word in name_words if word in transcription_lower)
                
                if score > best_score and entry["phones"]:
                    best_score = score
                    # Prefer first number (usually main number)
                    best_match = entry["phones"][0]
            
            if best_match:
                click.echo(f"Fallback method found number: {best_match}")
                return best_match
            
            return None
            
    except Exception as e:
        click.echo(f"Error in select_number: {e}")
        return None
    
    
def listen_macos(model):
    """
    If platform is macOS: listen to the microphone until a key is pressed,
    then transcribe with faster-whisper turbo model.
    """
    try:
        # Dynamically import required packages
        import pyaudio
        
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

def listen_linux(model):
    """
    If platform is Linux: listen to the microphone until a key is pressed,
    then transcribe with faster-whisper model.
    """
    try:
        import keyboard
        # Dynamically import required packages
        import pyaudio
        
        click.echo("Starting audio recording. Press any key to stop recording...")
        
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
            # Open stream with error handling
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            # Start a thread to detect key presses
            stop_recording = [False]
            
            def key_listener():
                keyboard.wait()  # Any key
                stop_recording[0] = True
            
            import threading
            key_thread = threading.Thread(target=key_listener)
            key_thread.daemon = True
            key_thread.start()
            
            # Start recording with proper error handling
            while not stop_recording[0]:
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
            return None
            
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
            return None
        
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
        click.echo(f"Error in listen_linux: {e}")
        return None

if __name__ == '__main__':
    cli()
