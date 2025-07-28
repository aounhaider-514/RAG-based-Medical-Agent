import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
from datetime import datetime
from medbot_rag import medical_qa
from multimodal_router import process_image
from voice import listen, speak, parse_reminder
from reminder_service import ReminderService
import re
import cv2
#from wound_detection import classify_wound
class MedicalAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Assistant")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Color scheme
        self.bg_color = "#050d1a"            # Almost black, deep royal blue
        self.fg_color = "#ffffff"            # White text for strong contrast
        self.accent_color = "#00ff88"        # Royal blue (used in borders/frames)
        self.secondary_color = "#0a1326"     # Slightly lighter than bg (for containers)
        self.highlight_color = "#00ff88"     # Bright green (for tabs like "Medical QA")

        # Apply theme
        self.root.configure(bg=self.bg_color)
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('.', background=self.bg_color, foreground=self.fg_color)
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color, font=('Arial', 10))
        style.configure('TButton', background=self.secondary_color, foreground=self.fg_color, 
                        font=('Arial', 10, 'bold'), borderwidth=1)
        style.map('TButton', 
                 background=[('active', self.highlight_color), ('pressed', self.accent_color)],
                 foreground=[('active', self.fg_color), ('pressed', self.fg_color)])
        style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.secondary_color, foreground=self.fg_color,
                        padding=[10, 5], font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab', 
                 background=[('selected', self.accent_color), ('active', self.highlight_color)],
                 foreground=[('selected', '#000000'), ('active', self.fg_color)])
        style.configure('TEntry', fieldbackground=self.secondary_color, foreground=self.fg_color)
        style.configure('TCombobox', fieldbackground=self.secondary_color, foreground=self.fg_color)
        style.configure('Treeview', background=self.secondary_color, foreground=self.fg_color,
                        fieldbackground=self.secondary_color, rowheight=25)
        style.configure('Treeview.Heading', background=self.accent_color, foreground="#000000",
                        font=('Arial', 10, 'bold'))
        style.map('Treeview', background=[('selected', self.highlight_color)])
        
        # Initialize services
        self.reminder_service = ReminderService()
        self.voice_active = False
        self.initialized = False
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.qa_frame = ttk.Frame(self.notebook)
        self.reminder_frame = ttk.Frame(self.notebook)
        self.image_frame = ttk.Frame(self.notebook)  # New image analysis tab
        self.help_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.qa_frame, text="Medical QA")
        self.notebook.add(self.reminder_frame, text="Reminders")
        self.notebook.add(self.image_frame, text="Image Analysis")  # Add new tab
        self.notebook.add(self.help_frame, text="Help")
        
        # Build UI components
        self.create_qa_tab()
        self.create_reminder_tab()
        self.create_image_tab()  # Create image analysis UI
        self.create_help_tab()
        
        # Start initialization in background
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing medical knowledge base...")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                                  anchor=tk.W, bg="#222222", fg=self.accent_color, font=('Arial', 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        threading.Thread(target=self.initialize_system, daemon=True).start()
    
    def initialize_system(self):
        """Initialize the medical knowledge base in the background"""
        try:
            medical_qa("Initialization query")
            self.initialized = True
            self.root.after(0, lambda: self.status_var.set("Ready. Ask a medical question or set a reminder."))
            self.root.after(0, lambda: speak("Medical assistant ready"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Initialization error: {str(e)}"))
    
    def create_qa_tab(self):
        """Create the Medical QA tab with themed colors"""
        # Input area
        input_frame = ttk.LabelFrame(self.qa_frame, text="Ask a Medical Question")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.question_entry = tk.Text(input_frame, height=4, wrap=tk.WORD, 
                                     bg=self.secondary_color, fg=self.fg_color, 
                                     insertbackground=self.fg_color)
        self.question_entry.pack(fill=tk.X, padx=5, pady=5)
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ask_btn = ttk.Button(btn_frame, text="Ask", command=self.ask_question)
        ask_btn.pack(side=tk.LEFT, padx=5)
        
        voice_btn = ttk.Button(btn_frame, text="🎤 Voice Input", command=self.start_voice_input)
        voice_btn.pack(side=tk.LEFT, padx=5)
        
        # Response area
        response_frame = ttk.LabelFrame(self.qa_frame, text="Response")
        response_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.response_text = scrolledtext.ScrolledText(
            response_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED,
            bg=self.secondary_color,
            fg=self.fg_color,
            insertbackground=self.fg_color
        )
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_image_tab(self):
        """Create the Image Analysis tab"""
        frame = ttk.Frame(self.image_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image selection
        select_frame = ttk.LabelFrame(frame, text="Select Image")
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.image_path_var = tk.StringVar()
        ttk.Entry(select_frame, textvariable=self.image_path_var, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(select_frame, text="Browse", command=self.browse_image).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Image type
        type_frame = ttk.LabelFrame(frame, text="Image Type")
        type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.image_type = tk.StringVar(value="wound")
        ttk.Radiobutton(type_frame, text="Wound", variable=self.image_type, value="wound").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(type_frame, text="X-ray", variable=self.image_type, value="xray").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(type_frame, text="Camera Capture", variable=self.image_type, value="camera").pack(side=tk.LEFT, padx=10)
        
        # Process button
        ttk.Button(frame, text="Analyze Image", command=self.process_image).pack(pady=10)
        
        # Result display
        result_frame = ttk.LabelFrame(frame, text="Analysis Result")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_result = scrolledtext.ScrolledText(
            result_frame, 
            wrap=tk.WORD, 
            state=tk.DISABLED,
            bg=self.secondary_color,
            fg=self.fg_color,
            insertbackground=self.fg_color
        )
        self.image_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path_var.set(file_path)
    
    def process_image(self):
        image_type = self.image_type.get()
        
        if image_type == "camera":
            self.capture_and_process()
            return
            
        image_path = self.image_path_var.get()
        if not image_path:
            messagebox.showwarning("Input Error", "Please select an image file")
            return
        
        self.status_var.set(f"Processing {image_type} image...")
        
        # Process in background
        threading.Thread(target=self._process_image_background, args=(image_path, image_type), daemon=True).start()
    
    def capture_and_process(self):
        """Capture image from camera and process it"""
        image_type = self.image_type.get()
        image_path = f"captured_{image_type}.jpg"
        
        self.status_var.set("Capturing image from camera...")
        self.root.update()
        
        if self.capture_image(image_path):
            self.status_var.set(f"Processing {image_type} image...")
            threading.Thread(target=self._process_image_background, args=(image_path, image_type), daemon=True).start()
        else:
            self.status_var.set("Could not access camera")
            speak("Could not access camera")
    
    def capture_image(self, filename):
        """Capture an image from the default camera"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(filename, frame)
        cap.release()
        return ret
    
    # MODIFY THIS METHOD:
    def _process_image_background(self, image_path, image_type):
        try:
            result = process_image(image_path, image_type)  # Directly use process_image
        
            # Format the result for display
            if image_type == 'wound':
                text = f"Wound Type: {result['wound_type']}\n"
                text += f"Severity: {result['severity']} (Confidence: {result['confidence']*100:.1f}%)\n"
                text += f"First Aid: {result['first_aid']}\n"
            elif image_type == 'xray':
                text = f"Condition: {result['condition']} (Confidence: {result['confidence']*100:.1f}%)\n"
                text += f"Explanation: {result['explanation']}\n"
        
            # Update the UI
            self.root.after(0, self._show_image_result, text)
            # Also speak the result
            speak(text)
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))
    def _show_image_result(self, text):
        self.image_result.config(state=tk.NORMAL)
        self.image_result.delete(1.0, tk.END)
        self.image_result.insert(tk.END, text)
        self.image_result.config(state=tk.DISABLED)
    
    def create_reminder_tab(self):
        """Create the Reminders tab with themed colors"""
        # Current reminders list
        list_frame = ttk.LabelFrame(self.reminder_frame, text="Current Reminders")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ("time", "message")
        self.reminder_tree = ttk.Treeview(
            list_frame, 
            columns=columns, 
            show="headings", 
            selectmode="browse"
        )
        
        self.reminder_tree.heading("time", text="Time")
        self.reminder_tree.heading("message", text="Message")
        self.reminder_tree.column("time", width=100, anchor=tk.CENTER)
        self.reminder_tree.column("message", width=400, anchor=tk.W)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.reminder_tree.yview)
        self.reminder_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.reminder_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add reminder form
        form_frame = ttk.LabelFrame(self.reminder_frame, text="Add New Reminder")
        form_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(form_frame, text="Message:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.reminder_msg = ttk.Entry(form_frame, width=40)
        self.reminder_msg.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(form_frame, text="Time (HH:MM):").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.reminder_time = ttk.Entry(form_frame, width=10)
        self.reminder_time.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        btn_frame = ttk.Frame(form_frame)
        btn_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        add_btn = ttk.Button(btn_frame, text="Add Reminder", command=self.add_reminder)
        add_btn.pack(side=tk.LEFT, padx=5)
        
        voice_reminder_btn = ttk.Button(btn_frame, text="🎤 Voice Add", command=self.start_reminder_voice)
        voice_reminder_btn.pack(side=tk.LEFT, padx=5)
        
        del_btn = ttk.Button(btn_frame, text="Delete Selected", command=self.delete_reminder)
        del_btn.pack(side=tk.LEFT, padx=5)
        
        # Refresh button
        refresh_btn = ttk.Button(self.reminder_frame, text="Refresh List", command=self.refresh_reminders)
        refresh_btn.pack(pady=5)
        
        # Initial load
        self.refresh_reminders()
    
    def create_help_tab(self):
        """Create the Help tab"""
        help_text = """
        Medical Assistant Application
        
        Features:
        1. Medical QA: Ask medical questions and get AI-powered responses
        2. Reminders: Set medication reminders that show popup notifications
        3. Image Analysis: Analyze wounds and X-rays using AI
        
        Usage:
        
        Medical QA Tab:
        - Type your question in the text box and click "Ask"
        - Or click "Voice Input" to ask with your microphone
        
        Reminders Tab:
        - Add new reminders with message and time (HH:MM)
        - Delete existing reminders by selecting and clicking "Delete Selected"
        - Use "Voice Add" to set reminders by voice
        
        Image Analysis Tab:
        - Select an image file or capture from camera
        - Choose between wound or X-ray analysis
        - Click "Analyze Image" to get results
        
        Voice Commands:
        - For questions: Just ask naturally (e.g., "What is diabetes?")
        - For reminders: "Remind me to [message] at [time]" 
          (e.g., "Remind me to take insulin at 14:30")
        - For images: "Analyze my wound" or "Analyze my X-ray"
        
        Requirements:
        - Internet connection for AI responses
        - Microphone for voice input
        - Camera for image capture
        - Ollama running in the background
        
        Note: This application is for informational purposes only and 
        does not replace professional medical advice.
        """
        
        help_label = tk.Label(
            self.help_frame, 
            text=help_text,
            justify=tk.LEFT,
            padx=20,
            pady=20,
            bg=self.bg_color,
            fg=self.fg_color
        )
        help_label.pack(fill=tk.BOTH, expand=True)
    
    def ask_question(self):
        """Process a medical question"""
        if not self.initialized:
            messagebox.showinfo("Initializing", "Medical knowledge base is still loading. Please wait...")
            return
            
        question = self.question_entry.get("1.0", tk.END).strip()
        if not question:
            return
            
        self.question_entry.delete("1.0", tk.END)
        self.status_var.set("Processing your question...")
        
        # Process in background to keep UI responsive
        threading.Thread(target=self.process_question, args=(question,), daemon=True).start()
    
    def process_question(self, question):
        """Background processing of medical question"""
        try:
            response = medical_qa(question)
            self.root.after(0, self.show_response, response)
            speak(response)
        except Exception as e:
            self.root.after(0, self.show_response, f"Error: {str(e)}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def show_response(self, response):
        """Display response in the UI"""
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, response)
        self.response_text.config(state=tk.DISABLED)
    
    def start_voice_input(self):
        """Start voice input for question"""
        if self.voice_active:
            return
            
        self.voice_active = True
        self.status_var.set("Listening... Speak your question")
        
        threading.Thread(target=self.process_voice_input, daemon=True).start()
    
    def process_voice_input(self):
        """Background processing of voice input"""
        try:
            query = listen(timeout=9)
            if query:
                if "analyze my wound" in query.lower():
                    self.root.after(0, lambda: self.image_type.set("wound"))
                    self.root.after(0, lambda: self.image_type.set("camera"))
                    self.root.after(0, self.capture_and_process)
                elif "analyze my x-ray" in query.lower() or "analyze my xray" in query.lower():
                    self.root.after(0, lambda: self.image_type.set("xray"))
                    self.root.after(0, lambda: self.image_type.set("camera"))
                    self.root.after(0, self.capture_and_process)
                else:
                    self.root.after(0, self.question_entry.delete, "1.0", tk.END)
                    self.root.after(0, self.question_entry.insert, tk.END, query)
                    self.root.after(0, self.ask_question)
        finally:
            self.voice_active = False
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def add_reminder(self):
        """Add a new reminder"""
        message = self.reminder_msg.get().strip()
        time_str = self.reminder_time.get().strip()
        
        if not message or not time_str:
            messagebox.showwarning("Input Error", "Both message and time are required")
            return
            
        if not re.match(r'^\d{1,2}:\d{2}$', time_str):
            messagebox.showwarning("Format Error", "Time must be in HH:MM format (e.g., 14:30)")
            return
            
        self.reminder_service.add_reminder(message, time_str)
        self.reminder_msg.delete(0, tk.END)
        self.reminder_time.delete(0, tk.END)
        self.refresh_reminders()
        speak(f"Reminder set for {message} at {time_str}")
    
    def start_reminder_voice(self):
        """Start voice input for reminder"""
        if self.voice_active:
            return
            
        self.voice_active = True
        self.status_var.set("Listening... Say: 'Remind me to [action] at [time]'")
        
        threading.Thread(target=self.process_reminder_voice, daemon=True).start()
    
    def process_reminder_voice(self):
        """Background processing of reminder voice input"""
        try:
            query = listen(timeout=9)
            if query:
                message, time_str = parse_reminder(query.lower())
                if message and time_str:
                    self.root.after(0, self.reminder_msg.delete, 0, tk.END)
                    self.root.after(0, self.reminder_msg.insert, 0, message)
                    self.root.after(0, self.reminder_time.delete, 0, tk.END)
                    self.root.after(0, self.reminder_time.insert, 0, time_str)
                    self.root.after(0, self.add_reminder)
                else:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Voice Command", 
                        "Please say: 'Remind me to [action] at [time]'"
                    ))
        finally:
            self.voice_active = False
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def delete_reminder(self):
        """Delete selected reminder"""
        selected = self.reminder_tree.selection()
        if not selected:
            return
            
        # Get reminder details
        item = self.reminder_tree.item(selected[0])
        values = item["values"]
        
        # Find and remove reminder
        for i, reminder in enumerate(self.reminder_service.reminders):
            if reminder["message"] == values[1] and reminder["time"] == values[0]:
                del self.reminder_service.reminders[i]
                self.reminder_service.save_reminders()
                self.refresh_reminders()
                speak("Reminder deleted")
                break
    
    def refresh_reminders(self):
        """Refresh the reminders list"""
        self.reminder_tree.delete(*self.reminder_tree.get_children())
        for reminder in self.reminder_service.reminders:
            self.reminder_tree.insert("", tk.END, values=(reminder["time"], reminder["message"]))

if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalAssistantApp(root)
    root.mainloop()