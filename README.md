# RAG-based-Medical-Agent
**MediMate - Medical Assistant Application**


**MediMate is a comprehensive medical assistant application that combines AI-powered medical knowledge with specialized image analysis capabilities. It provides instant answers to medical questions, analyzes wound and X-ray images, and manages medication reminders through an intuitive voice-enabled interface.**

**Key Features**
🧠 Medical Knowledge Base (RAG System)
Answers medical questions using Retrieval-Augmented Generation
Powered by Mistral AI model with medical dataset
Provides citations and safety disclaimers

🩹 **Wound Analysis**
Classifies 10 types of wounds using CNN model
Determines severity (Mild/Moderate/Severe)
Provides first aid instruction
90%+ accuracy on test dataset

🩻 **X-ray Analysis**
Detects pneumonia in chest X-rays
Provides confidence levels for predictions
Explains findings in simple terms
Uses transfer learning with DenseNet121

🎙️ **Voice Interface**
Natural voice commands for all features
Voice responses with medical explanations
Hands-free operation support
Multi-language support (via Google Speech Recognition)

📱 **Modern UI**
Dark theme with high-contrast interface
Tabbed organization of features
Real-time status updates
Responsive design for different screen sizes

**Installation Guide**
System Requirements
OS: Windows 10/11 (Linux/macOS supported with minor adjustments)
Processor: 1.8 GHz dual-core CPU or better
RAM: 8GB minimum (16GB recommended)
Storage: 2GB free space
Camera: Required for image capture features
Microphone: Required for voice commands

**Quick Start (Windows)**
Download the MediMate package
Extract the zip file
Double-click RUN.bat
Wait for dependencies to install (5-15 minutes)
The application will launch automatically

**MANUAL INSTALLATION:**
# Create virtual environment (recommended)
python -m venv medienv
source medienv/bin/activate  # Linux/macOS
medienv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Download AI models
ollama pull mistral

# Launch application
python app.py
