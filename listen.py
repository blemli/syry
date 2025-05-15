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