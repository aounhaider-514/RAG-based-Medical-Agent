@echo off
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages...
pip install ollama langchain-text-splitters langchain-chroma langchain-ollama speechrecognition pyttsx3 lxml pyaudio pytesseract pillow opencv-python tensorflow

echo Downloading Ollama model...
ollama pull mistral

echo Starting Medical Assistant...
python app.py

if %errorlevel% neq 0 (
    echo Error running the application. Check app.py and dependencies.
    pause
)
pause