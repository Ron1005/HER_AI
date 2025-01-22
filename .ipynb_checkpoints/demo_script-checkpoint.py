import pygame
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from advanced_ai import AdvancedConversationalAI, VoiceSystem
import json

class AIUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HER AI Assistant")
        
        # Initialize AI
        self.ai = AdvancedConversationalAI()
        self.voice_system = VoiceSystem()
        self.voice_enabled = False
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, width=60, height=20)
        self.chat_display.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        
        # Input field
        self.input_field = ttk.Entry(self.main_frame, width=50)
        self.input_field.grid(row=1, column=0, padx=5, pady=5)
        
        # Send button
        self.send_button = ttk.Button(self.main_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=5, pady=5)
        
        # Voice toggle button
        self.voice_button = ttk.Button(self.main_frame, text="Enable Voice", command=self.toggle_voice)
        self.voice_button.grid(row=1, column=2, padx=5, pady=5)
        
        # Emotional state display
        self.emotion_frame = ttk.LabelFrame(self.main_frame, text="AI Emotional State", padding="5")
        self.emotion_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Initialize with welcome message
        self.display_message("AI: Hello! I'm Samantha. How can I help you today?")
        
        # Bind Enter key to send message
        self.input_field.bind("<Return>", lambda e: self.send_message())
        
    def display_message(self, message):
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.see(tk.END)
        
    def send_message(self):
        user_input = self.input_field.get()
        if user_input.strip():
            # Display user message
            self.display_message(f"You: {user_input}")
            
            # Process through AI
            response = self.ai.process_input(user_input)
            
            # Display AI response
            self.display_message(f"AI: {response}")
            
            # Speak response if voice is enabled
            if self.voice_enabled:
                threading.Thread(target=self.voice_system.speak, args=(response,)).start()
            
            # Update emotional state display
            self.update_emotional_display()
            
            # Clear input field
            self.input_field.delete(0, tk.END)
    
    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        button_text = "Disable Voice" if self.voice_enabled else "Enable Voice"
        self.voice_button.config(text=button_text)
        
        if self.voice_enabled:
            self.display_message("AI: Voice interaction enabled")
        else:
            self.display_message("AI: Voice interaction disabled")
    
    def update_emotional_display(self):
        # Clear previous emotional state display
        for widget in self.emotion_frame.winfo_children():
            widget.destroy()
        
        # Display current emotional states
        for emotion, value in self.ai.emotional_state.emotions.items():
            ttk.Label(self.emotion_frame, text=f"{emotion.capitalize()}: ").grid(row=0, column=list(self.ai.emotional_state.emotions.keys()).index(emotion)*2)
            progress = ttk.Progressbar(self.emotion_frame, length=50, maximum=1.0)
            progress.grid(row=0, column=list(self.ai.emotional_state.emotions.keys()).index(emotion)*2+1)
            progress['value'] = value

def main():
    root = tk.Tk()
    app = AIUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
