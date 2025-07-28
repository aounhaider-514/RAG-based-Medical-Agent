import speech_recognition as sr
import pyttsx3
import time
import threading
import re
from medbot_rag import medical_qa
from multimodal_router import process_image
import cv2

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def capture_image(filename="captured_image.jpg"):
    """Capture an image from the default camera"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
    cap.release()
    return ret

def listen(timeout=5):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"\nListening for {timeout} seconds...")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=timeout)
            query = r.recognize_google(audio)
            print(f"YOU SAID: {query}")
            return query
        except sr.WaitTimeoutError:
            return ""
        except Exception:
            return ""

def speak(text):
    print(f"ASSISTANT: {text}")
    engine.say(text)
    engine.runAndWait()

def parse_reminder(command):
    try:
        if "remind me to" in command:
            parts = command.split("remind me to")[1].split(" at ")
        elif "set a reminder for" in command:
            parts = command.split("set a reminder for")[1].split(" at ")
        else:
            return None, None
        
        message = parts[0].strip()
        time_str = parts[1].strip()
        
        if re.match(r'^\d{1,2}:\d{2}$', time_str):
            return message, time_str
        return None, None
    except:
        return None, None

if __name__ == "__main__":
    speak("Medical assistant initializing. Please wait...")
    
    speak("Loading medical knowledge base...")
    medical_qa("Initialization query")
    
    speak("Medical assistant ready. How can I help with your health today?")
    
    while True:
        try:
            query = listen(timeout=9)
            if not query:
                continue
                
            if "exit" in query.lower() or "goodbye" in query.lower():
                speak("Goodbye! Remember to take care of your health.")
                break
                
            # Wound analysis command
            if "analyze my wound" in query.lower():
                speak("Please show your wound to the camera")
                if capture_image("captured_wound.jpg"):
                    result = process_image("captured_wound.jpg", "wound")
                    response = (f"Detected {result['wound_type']} wound. "
                               f"Severity: {result['severity']}. "
                               f"First aid: {result['first_aid']}")
                    speak(response)
                else:
                    speak("Could not access camera")
                continue
                    
            # X-ray analysis command
            elif "analyze my x-ray" in query.lower() or "analyze my xray" in query.lower():
                speak("Please show the X-ray to the camera")
                if capture_image("captured_xray.jpg"):
                    result = process_image("captured_xray.jpg", "xray")
                    response = (f"X-ray shows {result['condition']} with "
                               f"{result['confidence']*100:.1f}% confidence. "
                               f"Explanation: {result['explanation']}")
                    speak(response)
                else:
                    speak("Could not access camera")
                continue
                
            # Reminder command
            if "remind" in query.lower():
                message, time_str = parse_reminder(query.lower())
                if message and time_str:
                    speak(f"Reminder set for {message} at {time_str}")
                else:
                    speak("Please say: 'Remind me to [action] at [time]'")
            else:
                speak("Processing your medical query...")
                response = medical_qa(query)
                speak(response)
                
        except KeyboardInterrupt:
            speak("Medical assistant shutting down")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            speak("I encountered an issue. Please try again.")