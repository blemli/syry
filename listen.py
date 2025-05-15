# Required imports
import os
import time
import tempfile
import wave
import click
import faster_whisper as fw

# Global dictionary to store loaded models by name
_loaded_whisper_models = {}

def load_model_async(model_name, device=None, compute_type=None):
    """
    Load the model asynchronously in a separate thread.
    This can be called at program startup to have the model ready when needed.
    
    Args:
        model_name (str): The name of the whisper model to load
        device (str): Device to use for inference ('cpu', 'cuda', 'auto')
        compute_type (str): Compute type for inference
    """
    import threading
    
    def _load_in_background():
        load_model(model_name, device=device, compute_type=compute_type)
        click.echo("Model preloaded and ready for use!")
    
    thread = threading.Thread(target=_load_in_background)
    thread.daemon = True
    thread.start()
    return thread

def load_model(model_name, device=None, compute_type=None, cpu_threads=None, num_workers=2):
    """
    Preload the Whisper model to make subsequent transcriptions faster.
    
    Args:
        model_name (str): The name of the whisper model to load (e.g., 'tiny', 'base', 'small', 'medium', 'large')
        device (str): Device to use for inference ('cpu', 'cuda', 'auto'). Default is auto-detected.
        compute_type (str): Compute type for inference ('float16', 'int8', 'int8_float16'). Default is based on device.
        cpu_threads (int): Number of CPU threads to use (None = automatic)
        num_workers (int): Number of workers for processing (more can be faster, default=2)
        
    Returns:
        The loaded WhisperModel instance
    """
    global _loaded_whisper_models
    
    # Auto-detect best device if not specified
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    
    # Choose optimal compute type based on device if not specified
    if compute_type is None:
        if device == "cuda":
            compute_type = "float16"  # Better performance on GPU
        else:
            compute_type = "int8"  # Better performance on CPU
    
    # Generate a key for this specific model configuration
    model_key = f"{model_name}_{device}_{compute_type}"
    
    if model_key not in _loaded_whisper_models:
        click.echo(f"Loading faster-whisper {model_name} model on {device} using {compute_type}...")
        start_time = time.time()
        
        model = fw.WhisperModel(
            model_name, 
            device=device, 
            compute_type=compute_type, 
            cpu_threads=8,
            num_workers=num_workers
        )
        
        load_time = time.time() - start_time
        click.echo(f"Model loaded successfully in {load_time:.2f} seconds!")
        _loaded_whisper_models[model_key] = model
    
    return _loaded_whisper_models[model_key]

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
            
            # Use the preloaded model for transcription with optimized parameters
            model_instance = load_model(model)
            segments, info = model_instance.transcribe(
                temp_filename, 
                beam_size=3,  # Lower beam size for faster processing
                vad_filter=True,  # Filter out non-speech
                vad_parameters=dict(min_silence_duration_ms=500)  # Optimize VAD
            )
            
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
            
            # Use the preloaded model for transcription with optimized parameters
            model_instance = load_model(model)
            segments, info = model_instance.transcribe(
                temp_filename, 
                beam_size=3,  # Lower beam size for faster processing
                vad_filter=True,  # Filter out non-speech
                vad_parameters=dict(min_silence_duration_ms=500)  # Optimize VAD
            )
            
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
