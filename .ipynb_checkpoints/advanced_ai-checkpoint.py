import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import numpy as np
import pyaudio
import wave
import speech_recognition as sr
from gtts import gTTS
import os
import json
import pygame
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
import queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalSystem:
    def __init__(self):
        self.emotions = {
            "joy": 0.5,
            "sadness": 0.1,
            "anger": 0.1,
            "fear": 0.1,
            "surprise": 0.1,
            "trust": 0.5,
            "anticipation": 0.3
        }
        
        self.personality_traits = {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.8,
            "neuroticism": 0.3
        }
        
        self.mood_history = []
        self._load_emotion_triggers()
    
    def _load_emotion_triggers(self):
        try:
            with open("emotion_triggers.json", "r") as f:
                self.emotion_triggers = json.load(f)
        except FileNotFoundError:
            logger.error("emotion_triggers.json not found")
            self.emotion_triggers = {}
    
    def update_emotional_state(self, input_text, sentiment_score, context):
        # Update emotions based on triggers
        for emotion, triggers in self.emotion_triggers.items():
            for trigger in triggers:
                if trigger in input_text.lower():
                    self.emotions[emotion] = min(1.0, self.emotions[emotion] + 0.1)
        
        # Apply sentiment influence
        if sentiment_score > 0.5:
            self.emotions["joy"] = min(1.0, self.emotions["joy"] + 0.1)
        elif sentiment_score < -0.5:
            self.emotions["sadness"] = min(1.0, self.emotions["sadness"] + 0.1)
        
        # Apply personality influence
        self._apply_personality_influence()
        
        # Record emotional state
        self.mood_history.append({
            "timestamp": datetime.now().isoformat(),
            "emotions": self.emotions.copy(),
            "trigger": input_text
        })
        
        # Decay emotions slightly over time
        self._decay_emotions()
    
    def _apply_personality_influence(self):
        if self.personality_traits["neuroticism"] > 0.6:
            self.emotions["fear"] *= 1.1
            self.emotions["anxiety"] = min(1.0, self.emotions.get("anxiety", 0) + 0.1)
        
        if self.personality_traits["extraversion"] > 0.6:
            self.emotions["joy"] *= 1.1
            self.emotions["trust"] *= 1.1
    
    def _decay_emotions(self):
        decay_rate = 0.05
        for emotion in self.emotions:
            # Move emotions slowly back toward neutral (0.5)
            if self.emotions[emotion] > 0.5:
                self.emotions[emotion] = max(0.5, self.emotions[emotion] - decay_rate)
            else:
                self.emotions[emotion] = min(0.5, self.emotions[emotion] + decay_rate)

class ContextManager:
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = {}
        self.current_context = {}
        self.vectorizer = TfidfVectorizer()
        
    def update_context(self, user_input, ai_response):
        # Add to short-term memory
        self.short_term_memory.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response
        })
        
        # Maintain short-term memory size
        if len(self.short_term_memory) > 10:
            # Transfer important info to long-term memory
            self._transfer_to_long_term_memory(self.short_term_memory[0])
            self.short_term_memory.pop(0)
        
        # Update current context
        self._update_current_context()
    
    def _transfer_to_long_term_memory(self, memory_item):
        # Extract key information using NLP
        key_info = self._extract_key_information(memory_item["user_input"])
        
        if key_info:
            timestamp = datetime.now().isoformat()
            self.long_term_memory[timestamp] = {
                "information": key_info,
                "original_context": memory_item
            }
    
    def _extract_key_information(self, text):
        # Simple keyword-based extraction
        important_keywords = ["name", "favorite", "like", "dislike", "important", "remember"]
        
        for keyword in important_keywords:
            if keyword in text.lower():
                return text
        
        return None
    
    def _update_current_context(self):
        if not self.short_term_memory:
            return
            
        recent_texts = [m["user_input"] for m in self.short_term_memory[-3:]]
        
        # Create TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(recent_texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Extract key topics
            self.current_context["key_topics"] = self._extract_key_topics(tfidf_matrix, feature_names)
        except ValueError as e:
            logger.error(f"Error updating context: {e}")
    
    def _extract_key_topics(self, tfidf_matrix, feature_names, top_n=3):
        avg_scores = np.array(tfidf_matrix.mean(axis=0))[0]
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices]

class VoiceSystem:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio = pyaudio.PyAudio()
        pygame.mixer.init()
        
    def listen(self):
        with sr.Microphone() as source:
            logger.info("Listening...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                logger.error(f"Could not request results: {e}")
                return "Error processing audio"
            except Exception as e:
                logger.error(f"Error in listen(): {e}")
                return "Error processing audio"
    
    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("response.mp3")
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.remove("response.mp3")
        except Exception as e:
            logger.error(f"Error in speak(): {e}")

class KnowledgeBase:
    def __init__(self):
        self.facts = self._load_knowledge()
        self.learned_information = {}
        
    def _load_knowledge(self):
        try:
            with open("knowledge_base.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("knowledge_base.json not found")
            return {}
    
    def get_response(self, input_text, context):
        # Check for direct matches in knowledge base
        for topic, data in self.facts.get("conversation_topics", {}).items():
            if topic.lower() in input_text.lower():
                return self._generate_topic_response(topic, data)
        
        # Check emotional responses
        for emotion, responses in self.facts.get("emotional_intelligence", {}).get("empathy_responses", {}).items():
            if emotion.lower() in input_text.lower():
                return np.random.choice(responses)
        
        # Default to general response
        return self._generate_general_response(input_text)
    
    def _generate_topic_response(self, topic, data):
        responses = [
            f"I find {topic} fascinating. {data.get('definition', '')}",
            f"Let me share what I know about {topic}. {data.get('key_concepts', [''])[0]}",
            f"I love discussing {topic}. Would you like to explore {', '.join(data.get('key_concepts', ['']))}?"
        ]
        return np.random.choice(responses)
    
    def _generate_general_response(self, input_text):
        general_responses = [
            "That's an interesting perspective. Could you tell me more?",
            "I understand. How does that make you feel?",
            "I appreciate you sharing that with me. What are your thoughts on this?",
            "That's fascinating. Let's explore that further."
        ]
        return np.random.choice(general_responses)

class AdvancedConversationalAI:
    def __init__(self, name="Samantha"):
        self.name = name
        self.user_name = None
        self.conversation_history = []
        
        # Initialize components
        logger.info("Initializing AI components...")
        self.emotional_state = EmotionalSystem()
        self.context_manager = ContextManager()
        self.knowledge_base = KnowledgeBase()
        self.voice_system = VoiceSystem()
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased")
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained("emotion_model")
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            raise
        
        logger.info("AI initialization complete")
    
    def process_input(self, user_input):
        # Analyze sentiment
        sentiment = self.sentiment_analyzer(user_input)[0]
        
        # Update emotional state
        self.emotional_state.update_emotional_state(
            user_input,
            sentiment['score'],
            self.context_manager.current_context
        )
        
        # Generate response
        response = self.knowledge_base.get_response(
            user_input,
            self.context_manager.current_context
        )
        
        # Update context
        self.context_manager.update_context(user_input, response)
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": response,
            "emotional_state": self.emotional_state.emotions.copy()
        })
        
        return response
    
    def save_state(self):
        state = {
            "conversation_history": self.conversation_history,
            "emotional_state": self.emotional_state.emotions,
            "mood_history": self.emotional_state.mood_history,
            "long_term_memory": self.context_manager.long_term_memory
        }
        
        with open("ai_state.json", "w") as f:
            json.dump(state, f)
    
    def load_state(self):
        try:
            with open("ai_state.json", "r") as f:
                state = json.load(f)
                self.conversation_history = state["conversation_history"]
                self.emotional_state.emotions = state["emotional_state"]
                self.emotional_state.mood_history = state["mood_history"]
                self.context_manager.long_term_memory = state["long_term_memory"]
        except FileNotFoundError:
            logger.info("No previous state file found")

if __name__ == "__main__":
    # Simple test of the AI
    ai = AdvancedConversationalAI()
    print(f"{ai.name}: Hello! How are you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            ai.save_state()
            print(f"{ai.name}: Goodbye! It was nice talking to you.")
            break
        
        response = ai.process_input(user_input)
        print(f"{ai.name}: {response}")
