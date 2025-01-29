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
from collections import defaultdict
from textblob import TextBlob
import numpy as np
from datetime import datetime, timedelta
import spacy
from collections import defaultdict, deque
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Set


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNLPPipeline:
    def __init__(self):
        # Multiple NLP models for comprehensive analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.intent_classifier = pipeline("text-classification")
        self.named_entity_recognizer = pipeline("ner")
        self.question_answerer = pipeline("question-answering")
    
    def comprehensive_analysis(self, text):
        return {
            "sentiment": self.sentiment_analyzer(text)[0],
            "intent": self.intent_classifier(text)[0],
            "entities": self.named_entity_recognizer(text),
            "key_information": self._extract_key_info(text)
        }
    
    def _extract_key_info(self, text):
        # Advanced information extraction
        key_phrases = []
        for entity in self.named_entity_recognizer(text):
            if entity['score'] > 0.7:
                key_phrases.append(entity['word'])
        return key_phrases

class ContinuousLearningSystem:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.interaction_memory = []
        self.learning_threshold = 5  # Interactions needed to update knowledge
    
    def record_interaction(self, user_input, ai_response):
        self.interaction_memory.append({
            "input": user_input,
            "response": ai_response,
            "timestamp": datetime.now()
        })
        
        # Trigger learning if threshold is met
        if len(self.interaction_memory) >= self.learning_threshold:
            self.update_knowledge_base()
    
    def update_knowledge_base(self):
        # Extract common themes and update knowledge
        topics = self._identify_recurring_topics()
        for topic, count in topics.items():
            if count > 2:  # Significant topic
                self._enhance_topic_knowledge(topic)
        
        # Reset interaction memory
        self.interaction_memory = []
    
    def _identify_recurring_topics(self):
        # Use NLP to identify recurring conversation topics
        topic_counter = {}
        for interaction in self.interaction_memory:
            # Extract key topics from interactions
            topics = self.extract_topics(interaction['input'])
            for topic in topics:
                topic_counter[topic] = topic_counter.get(topic, 0) + 1
        return topic_counter
    
    def _enhance_topic_knowledge(self, topic):
        # Dynamically update knowledge base with learned information
        if topic not in self.knowledge_base.facts['conversation_topics']:
            self.knowledge_base.facts['conversation_topics'][topic] = {
                "learning_source": "continuous_interaction",
                "mentions": len(self.interaction_memory)
            }

class EmotionalSystem:
    # def __init__(self):
    #     self.emotions = {
    #         "joy": 0.5,
    #         "sadness": 0.1,
    #         "anger": 0.1,
    #         "fear": 0.1,
    #         "surprise": 0.1,
    #         "trust": 0.5,
    #         "anticipation": 0.3
    #     }
        
    #     self.personality_traits = {
    #         "openness": 0.8,
    #         "conscientiousness": 0.7,
    #         "extraversion": 0.6,
    #         "agreeableness": 0.8,
    #         "neuroticism": 0.3
    #     }
        
    #     self.mood_history = []
    #     self._load_emotion_triggers()
    
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
        #self._apply_personality_influence(blob.sentiment.subjectivity)
        
        # Record emotional state
        self.mood_history.append({
            "timestamp": datetime.now().isoformat(),
            "emotions": self.emotions.copy(),
            "trigger": input_text
        })
        
        # Decay emotions slightly over time
        #self._decay_emotions()

    def _apply_personality_influence(self, subjectivity):
        # Modify personality influence using subjectivity
        if self.personality_traits["neuroticism"] > 0.6:
            self.emotions["fear"] *= (1 + subjectivity)
            self.emotions["anxiety"] = min(1.0, self.emotions.get("anxiety", 0) + subjectivity)
        
        if self.personality_traits["extraversion"] > 0.6:
            self.emotions["joy"] *= (1 + subjectivity)
            self.emotions["trust"] *= (1 + subjectivity)
    
    # def _decay_emotions(self):
    #     decay_rate = 0.05
    #     for emotion in self.emotions:
    #         # Move emotions slowly back toward neutral (0.5)
    #         if self.emotions[emotion] > 0.5:
    #             self.emotions[emotion] = max(0.5, self.emotions[emotion] - decay_rate)
    #         else:
    #             self.emotions[emotion] = min(0.5, self.emotions[emotion] + decay_rate)
    def __init__(self):
        # Use defaultdict to handle new emotions gracefully
        self.emotions = defaultdict(float, {
            "joy": 0.5,
            "sadness": 0.1,
            "anger": 0.1,
            "fear": 0.1,
            "surprise": 0.1,
            "trust": 0.5,
            "anticipation": 0.3,
            "interest": 0.5,
            "curiosity": 0.6,
            "empathy": 0.7
        })
        
        # Enhanced personality system with more traits
        self.personality_traits = {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.8,
            "neuroticism": 0.3,
            "adaptability": 0.7,
            "creativity": 0.8,
            "emotional_depth": 0.9
        }
        
        # Enhanced memory systems
        self.mood_history = []
        self.emotional_memory = defaultdict(lambda: defaultdict(float))
        self.topic_emotions = defaultdict(lambda: defaultdict(float))
        self.user_emotional_patterns = defaultdict(lambda: defaultdict(float))
        
        # Emotional constants
        self.EMOTION_DECAY_RATE = 0.05
        self.MEMORY_INFLUENCE_RATE = 0.3
        self.MAX_MEMORY_AGE = timedelta(days=30)
        
        self._load_emotion_triggers()
    
    def _load_emotion_triggers(self):
        try:
            with open("emotion_triggers.json", "r") as f:
                base_triggers = json.load(f)
                self.emotion_triggers = self._expand_emotion_triggers(base_triggers)
        except FileNotFoundError:
            logger.error("emotion_triggers.json not found")
            self.emotion_triggers = {}
    
    def _expand_emotion_triggers(self, base_triggers):
        """Expand basic emotion triggers with variations and combinations"""
        expanded = defaultdict(list)
        intensity_modifiers = ["slightly", "somewhat", "very", "extremely", "incredibly"]
        temporal_modifiers = ["currently", "lately", "always", "sometimes"]
        
        for emotion, triggers in base_triggers.items():
            expanded[emotion] = triggers.copy()
            
            for trigger in triggers:
                # Add intensity variations
                expanded[emotion].extend([f"{mod} {trigger}" for mod in intensity_modifiers])
                
                # Add temporal variations
                expanded[emotion].extend([f"{mod} {trigger}" for mod in temporal_modifiers])
                
                # Add feeling variations
                expanded[emotion].extend([
                    f"feeling {trigger}",
                    f"felt {trigger}",
                    f"makes me {trigger}",
                    f"becoming {trigger}"
                ])
        
        return expanded

    def update_emotional_state(self, input_text, sentiment_score, context):
        """Update emotional state based on input and context"""
        # Analyze text sentiment using TextBlob for more nuanced analysis
        blob = TextBlob(input_text)
        
        # Calculate emotional changes
        emotion_changes = self._calculate_emotion_changes(
            input_text, 
            blob.sentiment.polarity,
            blob.sentiment.subjectivity,
            context
        )
        
        # Apply emotional changes
        self._apply_emotion_changes(emotion_changes)
        
        # Update emotional memories
        self._update_emotional_memory(input_text, context)
        
        # Apply personality influence
        self._apply_personality_influence(blob.sentiment.subjectivity)
        
        # Record current state
       # self._record_emotional_state(input_text, context)
        
        # Natural decay
       # self._decay_emotions()

    def _calculate_emotion_changes(self, text, polarity, subjectivity, context):
        """Calculate how emotions should change based on input"""
        changes = defaultdict(float)
        
        # Check for emotion triggers
        for emotion, triggers in self.emotion_triggers.items():
            for trigger in triggers:
                if trigger.lower() in text.lower():
                    # Base change
                    change = 0.1
                    
                    # Modify based on subjectivity
                    change *= (1 + subjectivity)
                    
                    # Modify based on context
                    if trigger in context.get("key_topics", []):
                        change *= 1.5
                    
                    changes[emotion] += change
        
        # Consider emotional memory influence
        self._add_memory_influence(changes, context)
        
        return changes

    def _add_memory_influence(self, changes, context):
        """Add influence from emotional memories"""
        for topic in context.get("key_topics", []):
            if topic in self.emotional_memory:
                for emotion, strength in self.emotional_memory[topic].items():
                    changes[emotion] += strength * self.MEMORY_INFLUENCE_RATE

    def _apply_emotion_changes(self, changes):
        """Apply emotional changes while maintaining balance"""
        for emotion, change in changes.items():
            self.emotions[emotion] = min(1.0, max(0.0, self.emotions[emotion] + change))
        
        # Normalize emotions to maintain overall balance
        total = sum(self.emotions.values())
        if total > 0:
            for emotion in self.emotions:
                self.emotions[emotion] /= total

    def _update_emotional_memory(self, input_text, context):
        """Update emotional associations with topics"""
        current_time = datetime.now()
        
        # Update topic emotions
        for topic in context.get("key_topics", []):
            for emotion, value in self.emotions.items():
                # Exponential moving average
                self.topic_emotions[topic][emotion] = (
                    0.8 * self.topic_emotions[topic][emotion] +
                    0.2 * value
                )
        
        # Clean old memories
        self._clean_old_memories(current_time)

    def _clean_old_memories(self, current_time):
        """Remove memories older than MAX_MEMORY_AGE"""
        self.mood_history = [
            m for m in self.mood_history 
            if current_time - datetime.fromisoformat(m["timestamp"]) <= self.MAX_MEMORY_AGE
        ]

    def get_dominant_emotion(self):
        """Return the currently dominant emotion"""
        return max(self.emotions.items(), key=lambda x: x[1])

    def get_emotional_response_modifier(self):
        """Get a modifier for responses based on emotional state"""
        dominant_emotion, strength = self.get_dominant_emotion()
        
        modifiers = {
            "joy": ["happily", "enthusiastically", "cheerfully"],
            "sadness": ["gently", "softly", "carefully"],
            "empathy": ["understandingly", "compassionately", "thoughtfully"]
        }
        
        if dominant_emotion in modifiers and strength > 0.6:
            return random.choice(modifiers[dominant_emotion])
        return ""

class Memory:
    def __init__(self, content: str, timestamp: datetime, topics: Set[str], 
                 importance: float = 0.5, context: Dict = None):
        self.content = content
        self.timestamp = timestamp
        self.topics = topics
        self.importance = importance
        self.context = context or {}
        self.access_count = 0
        self.last_accessed = timestamp

    def access(self):
        self.access_count += 1
        self.last_accessed = datetime.now()
        # Increase importance based on access frequency
        self.importance = min(1.0, self.importance + 0.1)



class ContextManager:
    # def __init__(self):
    #     self.short_term_memory = []
    #     self.long_term_memory = {}
    #     self.current_context = {}
    #     self.vectorizer = TfidfVectorizer()
        
    # def update_context(self, user_input, ai_response):
    #     # Add to short-term memory
    #     self.short_term_memory.append({
    #         "timestamp": datetime.now().isoformat(),
    #         "user_input": user_input,
    #         "ai_response": ai_response
    #     })
        
    #     # Maintain short-term memory size
    #     if len(self.short_term_memory) > 10:
    #         # Transfer important info to long-term memory
    #         self._transfer_to_long_term_memory(self.short_term_memory[0])
    #         self.short_term_memory.pop(0)
        
    #     # Update current context
    #     self._update_current_context()
    
    # def _transfer_to_long_term_memory(self, memory_item):
    #     # Extract key information using NLP
    #     key_info = self._extract_key_information(memory_item["user_input"])
        
    #     if key_info:
    #         timestamp = datetime.now().isoformat()
    #         self.long_term_memory[timestamp] = {
    #             "information": key_info,
    #             "original_context": memory_item
    #         }
    
    # def _extract_key_information(self, text):
    #     # Simple keyword-based extraction
    #     important_keywords = ["name", "favorite", "like", "dislike", "important", "remember"]
        
    #     for keyword in important_keywords:
    #         if keyword in text.lower():
    #             return text
        
    #     return None
    
    # def _update_current_context(self):
    #     if not self.short_term_memory:
    #         return
            
    #     recent_texts = [m["user_input"] for m in self.short_term_memory[-3:]]
        
    #     # Create TF-IDF matrix
    #     try:
    #         tfidf_matrix = self.vectorizer.fit_transform(recent_texts)
    #         feature_names = self.vectorizer.get_feature_names_out()
            
    #         # Extract key topics
    #         self.current_context["key_topics"] = self._extract_key_topics(tfidf_matrix, feature_names)
    #     except ValueError as e:
    #         logger.error(f"Error updating context: {e}")
    
    # def _extract_key_topics(self, tfidf_matrix, feature_names, top_n=3):
    #     avg_scores = np.array(tfidf_matrix.mean(axis=0))[0]
    #     top_indices = avg_scores.argsort()[-top_n:][::-1]
    #     return [feature_names[i] for i in top_indices]
    def __init__(self):
        # Initialize core components
        self.short_term_memory = deque(maxlen=10)  # Last 10 interactions
        self.long_term_memory: Dict[str, Memory] = {}
        self.working_memory: Dict[str, Any] = {}  # Current active context
        self.topic_graph = defaultdict(dict)  # Topic relationships
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )
        except Exception as e:
            logger.error(f"Error loading NLP components: {e}")
            raise

        # Configuration
        self.config = {
            'memory_importance_threshold': 0.6,
            'topic_similarity_threshold': 0.3,
            'max_topic_associations': 5,
            'memory_decay_rate': 0.1,
            'long_term_memory_cleanup_days': 30
        }

    def update_context(self, user_input: str, ai_response: str) -> Dict:
        """
        Update context with new interaction and return current context
        """
        try:
            # Process input
            doc = self.nlp(user_input)
            
            # Extract key information
            current_time = datetime.now()
            analysis = self._analyze_input(doc)
            
            # Update memories
            self._update_memories(user_input, ai_response, analysis, current_time)
            
            # Update topic relationships
            self._update_topic_relationships(analysis['topics'])
            
            # Update working memory
            self._update_working_memory(analysis, current_time)
            
            # Perform maintenance
            self._maintain_memory_system()
            
            return self.get_current_context()
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return self.working_memory

    def _analyze_input(self, doc) -> Dict:
        """
        Perform comprehensive input analysis
        """
        return {
            'topics': self._extract_topics(doc),
            'entities': self._extract_entities(doc),
            'intent': self._determine_intent(doc),
            'sentiment': self._analyze_sentiment(doc),
            'key_phrases': self._extract_key_phrases(doc),
            'user_preferences': self._extract_preferences(doc),
            'importance': self._calculate_importance(doc)
        }

    def _extract_topics(self, doc) -> Set[str]:
        """
        Extract topics using multiple methods for better coverage
        """
        topics = set()
        
        # Extract from noun phrases
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and len(chunk.text.strip()) > 2:
                topics.add(chunk.text.lower())
        
        # Extract from named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']:
                topics.add(ent.text.lower())
        
        # Extract from important keywords
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                not token.is_stop and 
                len(token.text.strip()) > 2):
                topics.add(token.text.lower())
        
        return topics

    def _determine_intent(self, doc) -> str:
        """
        Determine user intent through multiple signals
        """
        # Check for questions
        if any(token.text.endswith('?') for token in doc):
            return 'question'
            
        # Check for commands/requests
        command_verbs = {'tell', 'show', 'help', 'find', 'give', 'explain'}
        if any(token.lemma_ in command_verbs for token in doc):
            return 'command'
            
        # Check for statements
        statement_indicators = {'think', 'believe', 'feel', 'know'}
        if any(token.lemma_ in statement_indicators for token in doc):
            return 'statement'
            
        # Check for emotional content
        emotion_words = {'love', 'hate', 'like', 'dislike', 'angry', 'happy'}
        if any(token.lemma_ in emotion_words for token in doc):
            return 'emotional'
            
        return 'general'

    def _update_memories(self, user_input: str, ai_response: str, 
                        analysis: Dict, timestamp: datetime):
        """
        Update both short-term and long-term memories
        """
        # Create memory object
        memory = Memory(
            content=user_input,
            timestamp=timestamp,
            topics=analysis['topics'],
            importance=analysis['importance'],
            context={
                'ai_response': ai_response,
                'analysis': analysis
            }
        )
        
        # Update short-term memory
        self.short_term_memory.append(memory)
        
        # Check if memory should be stored long-term
        if memory.importance >= self.config['memory_importance_threshold']:
            memory_key = f"{timestamp.isoformat()}-{hash(user_input)}"
            self.long_term_memory[memory_key] = memory

    def _update_topic_relationships(self, current_topics: Set[str]):
        """
        Update topic relationships based on co-occurrence
        """
        if len(current_topics) < 2:
            return
            
        # Convert topics to embeddings for similarity calculation
        topic_texts = list(current_topics)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(topic_texts)
            similarities = cosine_similarity(tfidf_matrix)
            
            # Update topic graph
            for i, topic1 in enumerate(topic_texts):
                for j, topic2 in enumerate(topic_texts[i+1:], i+1):
                    similarity = similarities[i][j]
                    if similarity >= self.config['topic_similarity_threshold']:
                        # Update bidirectional relationship
                        self.topic_graph[topic1][topic2] = similarity
                        self.topic_graph[topic2][topic1] = similarity
                        
                        # Prune weak connections
                        self._prune_topic_relationships(topic1)
                        self._prune_topic_relationships(topic2)
                        
        except Exception as e:
            logger.warning(f"Error updating topic relationships: {e}")

    def _prune_topic_relationships(self, topic: str):
        """
        Keep only the strongest topic relationships
        """
        if topic in self.topic_graph:
            relationships = self.topic_graph[topic]
            if len(relationships) > self.config['max_topic_associations']:
                # Keep only the strongest associations
                sorted_relationships = sorted(
                    relationships.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                self.topic_graph[topic] = dict(
                    sorted_relationships[:self.config['max_topic_associations']]
                )

    def _update_working_memory(self, analysis: Dict, timestamp: datetime):
        """
        Update current working memory with active context
        """
        self.working_memory = {
            'timestamp': timestamp.isoformat(),
            'current_topics': analysis['topics'],
            'current_intent': analysis['intent'],
            'current_entities': analysis['entities'],
            'recent_topics': self._get_recent_topics(),
            'related_topics': self._get_related_topics(analysis['topics']),
            'active_context': self._get_active_context(analysis)
        }

    def _get_recent_topics(self) -> Set[str]:
        """
        Get topics from recent interactions
        """
        recent_topics = set()
        for memory in self.short_term_memory:
            recent_topics.update(memory.topics)
        return recent_topics

    def _get_related_topics(self, current_topics: Set[str]) -> Dict[str, float]:
        """
        Get topics related to current topics
        """
        related_topics = defaultdict(float)
        for topic in current_topics:
            if topic in self.topic_graph:
                for related, strength in self.topic_graph[topic].items():
                    related_topics[related] = max(related_topics[related], strength)
        return dict(related_topics)

    def _get_active_context(self, analysis: Dict) -> Dict:
        """
        Determine currently active context
        """
        return {
            'topics': analysis['topics'],
            'intent': analysis['intent'],
            'sentiment': analysis['sentiment'],
            'user_preferences': analysis['user_preferences'],
            'recent_memories': [
                memory.content for memory in self.short_term_memory
            ]
        }

    def _maintain_memory_system(self):
        """
        Perform regular maintenance on the memory system
        """
        current_time = datetime.now()
        
        # Clean up old memories
        self._cleanup_old_memories(current_time)
        
        # Apply memory decay
        self._apply_memory_decay(current_time)
        
        # Consolidate similar memories
        self._consolidate_memories()

    def _cleanup_old_memories(self, current_time: datetime):
        """
        Remove old, unimportant memories
        """
        cutoff_date = current_time - timedelta(
            days=self.config['long_term_memory_cleanup_days']
        )
        
        # Remove old memories unless they're important
        self.long_term_memory = {
            key: memory for key, memory in self.long_term_memory.items()
            if (memory.timestamp > cutoff_date or 
                memory.importance > self.config['memory_importance_threshold'])
        }

    def get_relevant_memories(self, topics: Set[str], limit: int = 5) -> List[Memory]:
        """
        Retrieve memories relevant to given topics
        """
        relevant_memories = []
        for memory in self.long_term_memory.values():
            if topics & memory.topics:  # If there's any overlap in topics
                relevant_memories.append(memory)
        
        # Sort by relevance (importance and recency)
        relevant_memories.sort(
            key=lambda m: (len(topics & m.topics) * m.importance, m.timestamp),
            reverse=True
        )
        
        return relevant_memories[:limit]

    def get_current_context(self) -> Dict:
        """
        Get current context for response generation
        """
        return {
            'working_memory': self.working_memory,
            'recent_interactions': list(self.short_term_memory),
            'active_topics': self._get_recent_topics(),
            'topic_relationships': dict(self.topic_graph)
        }

    def _extract_entities(self, doc) -> List[Dict]:
        """
        Extract named entities from the document
        """
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities

    def _analyze_sentiment(self, doc) -> Dict:
        """
        Analyze sentiment of the document
        """
        # Simple rule-based sentiment analysis
        positive_words = {'good', 'great', 'excellent', 'happy', 'love', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'sad', 'hate', 'poor'}
        
        text = doc.text.lower()
        words = set(token.text.lower() for token in doc)
        
        positive_score = len(words & positive_words)
        negative_score = len(words & negative_words)
        
        if positive_score > negative_score:
            sentiment = 'positive'
            score = min(1.0, positive_score / len(words))
        elif negative_score > positive_score:
            sentiment = 'negative'
            score = -min(1.0, negative_score / len(words))
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score
        }

    def _extract_key_phrases(self, doc) -> List[str]:
        """
        Extract important key phrases from the document
        """
        key_phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop:
                key_phrases.append(chunk.text)
        
        # Extract verb phrases
        for token in doc:
            if token.pos_ == 'VERB' and not token.is_stop:
                phrase = token.text
                # Include objects of verbs
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj'):
                        phrase += ' ' + child.text
                key_phrases.append(phrase)
        
        return list(set(key_phrases))

    def _extract_preferences(self, doc) -> Dict[str, List[str]]:
        """
        Extract user preferences from the document
        """
        preferences = {
            'likes': [],
            'dislikes': [],
            'interests': []
        }
        
        like_verbs = {'like', 'love', 'enjoy', 'prefer'}
        dislike_verbs = {'dislike', 'hate', 'despise'}
        interest_indicators = {'interested', 'curious', 'fascinated'}
        
        for token in doc:
            if token.lemma_ in like_verbs:
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj'):
                        preferences['likes'].append(child.text)
            elif token.lemma_ in dislike_verbs:
                for child in token.children:
                    if child.dep_ in ('dobj', 'pobj'):
                        preferences['dislikes'].append(child.text)
            elif token.lemma_ in interest_indicators:
                for child in token.children:
                    if child.dep_ in ('prep', 'pobj'):
                        preferences['interests'].append(child.text)
        
        return preferences

    def _calculate_importance(self, doc) -> float:
        """
        Calculate the importance of the input based on various factors
        """
        importance_score = 0.5  # Base importance
        
        # Increase importance for questions
        if any(token.text.endswith('?') for token in doc):
            importance_score += 0.1
        
        # Increase for emotional content
        emotion_words = {'love', 'hate', 'angry', 'happy', 'sad', 'excited'}
        if any(token.lemma_ in emotion_words for token in doc):
            importance_score += 0.1
        
        # Increase for named entities
        if doc.ents:
            importance_score += 0.1
        
        # Increase for imperative sentences
        if doc[0].pos_ == 'VERB':
            importance_score += 0.1
        
        return min(1.0, importance_score)

    def _apply_memory_decay(self, current_time: datetime):
        """
        Apply decay to memories based on time elapsed and access patterns
        """
        for memory_key, memory in self.long_term_memory.items():
            # Calculate time since last access
            time_elapsed = current_time - memory.last_accessed
            days_elapsed = time_elapsed.days
            
            # Calculate decay factor
            base_decay = self.config['memory_decay_rate'] * days_elapsed
            
            # Adjust decay based on access frequency
            access_factor = 1.0 / (1.0 + memory.access_count)
            total_decay = base_decay * access_factor
            
            # Apply decay to importance
            memory.importance = max(0.1, memory.importance - total_decay)

    def _consolidate_memories(self):
        """
        Consolidate similar memories to prevent redundancy
        """
        # Get all memory texts for similarity comparison
        memory_texts = [memory.content for memory in self.long_term_memory.values()]
        
        if not memory_texts:
            return
            
        try:
            # Calculate similarity matrix
            tfidf_matrix = self.vectorizer.fit_transform(memory_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Track memories to merge
            to_merge = []
            processed = set()
            
            # Find similar memories
            for i in range(len(memory_texts)):
                if i in processed:
                    continue
                    
                similar_indices = []
                for j in range(i + 1, len(memory_texts)):
                    if j not in processed and similarity_matrix[i][j] > 0.8:  # Similarity threshold
                        similar_indices.append(j)
                
                if similar_indices:
                    to_merge.append([i] + similar_indices)
                    processed.update(similar_indices)
            
            # Merge similar memories
            memory_items = list(self.long_term_memory.items())
            for merge_group in to_merge:
                self._merge_memories([memory_items[i][1] for i in merge_group])
                
        except Exception as e:
            logger.warning(f"Error in memory consolidation: {e}")

    def _merge_memories(self, memories: List[Memory]):
        """
        Merge a group of similar memories into a single memory
        """
        if not memories:
            return
            
        # Use the most recent memory as the base
        base_memory = max(memories, key=lambda m: m.timestamp)
        
        # Combine topics and contexts
        all_topics = set()
        combined_context = {}
        
        for memory in memories:
            all_topics.update(memory.topics)
            if memory.context:
                for key, value in memory.context.items():
                    if key in combined_context:
                        if isinstance(combined_context[key], list):
                            combined_context[key].append(value)
                        else:
                            combined_context[key] = [combined_context[key], value]
                    else:
                        combined_context[key] = value
        
        # Update base memory
        base_memory.topics = all_topics
        base_memory.context.update(combined_context)
        base_memory.importance = min(1.0, base_memory.importance * 1.2)  # Increase importance slightly
        
        # Remove other memories
        self.long_term_memory = {
            key: mem for key, mem in self.long_term_memory.items()
            if mem is base_memory or mem not in memories
        }

    def _calculate_memory_relevance(self, memory: Memory, current_topics: Set[str], 
                                current_time: datetime) -> float:
        """
        Calculate how relevant a memory is to the current context
        """
        # Topic overlap
        topic_relevance = len(memory.topics & current_topics) / max(len(memory.topics), 1)
        
        # Time decay
        time_factor = 1.0 / (1.0 + (current_time - memory.timestamp).days)
        
        # Access frequency
        access_factor = min(1.0, memory.access_count / 10.0)
        
        # Combine factors
        relevance = (0.4 * topic_relevance + 
                    0.3 * time_factor + 
                    0.3 * access_factor) * memory.importance
                    
        return relevance

    def get_relevant_memories(self, current_topics: Set[str], 
                            limit: int = 5, threshold: float = 0.2) -> List[Memory]:
        """
        Get memories relevant to the current context
        """
        current_time = datetime.now()
        relevant_memories = []
        
        for memory in self.long_term_memory.values():
            relevance = self._calculate_memory_relevance(memory, current_topics, current_time)
            if relevance >= threshold:
                relevant_memories.append((memory, relevance))
        
        # Sort by relevance and return top memories
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in relevant_memories[:limit]]

class VoiceSystem:
    # def __init__(self):
    #     self.recognizer = sr.Recognizer()
    #     self.audio = pyaudio.PyAudio()
    #     pygame.mixer.init()
    def __init__(self, name="Samantha",knowledge_base=None, nlp_pipeline=None):
        self.name = name
        self.user_name = None
        self.conversation_history = []
    
        # Initialize advanced components
        self.nlp_pipeline = AdvancedNLPPipeline()
        self.knowledge_base = KnowledgeBase()
        print(self.knowledge_base)
        self.continuous_learning = ContinuousLearningSystem(self.knowledge_base)

        # self.knowledge_base = knowledge_base
        # self.nlp_pipeline = nlp_pipeline or AdvancedNLPPipeline()
    
        # if knowledge_base:
        #     self.continuous_learning = ContinuousLearningSystem(self.knowledge_base)
    
        # Enhanced emotional state and context management
        self.emotional_state = EmotionalSystem()
        self.context_manager = ContextManager()
    
        # Load pre-trained models
        try:
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained("best_emotion_model")
            self.intent_classifier = pipeline("text-classification", model="best_intent_model")
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
        
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

        # Initialize continuous learning system
        self.continuous_learning = ContinuousLearningSystem(self.knowledge_base)
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased")
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained("emotion_model")
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            raise
        
        logger.info("AI initialization complete")
    
    # def process_input(self, user_input):
        # # Analyze sentiment
        # sentiment = self.sentiment_analyzer(user_input)[0]
        
        # # Update emotional state
        # self.emotional_state.update_emotional_state(
        #     user_input,
        #     sentiment['score'],
        #     self.context_manager.current_context
        # )
        
        # # Generate response
        # response = self.knowledge_base.get_response(
        #     user_input,
        #     self.context_manager.current_context
        # )
        
        # # Update context
        # self.context_manager.update_context(user_input, response)
        
        # # Add to conversation history
        # self.conversation_history.append({
        #     "timestamp": datetime.now().isoformat(),
        #     "user_input": user_input,
        #     "ai_response": response,
        #     "emotional_state": self.emotional_state.emotions.copy()
        # })
        
        # return response
    def process_input(self, user_input):
        # Comprehensive NLP analysis
        nlp_analysis = self.voice_system.nlp_pipeline.comprehensive_analysis(user_input)
    
        # Emotion and sentiment detection
        sentiment = nlp_analysis['sentiment']
        intent = nlp_analysis['intent']
        entities = nlp_analysis['entities']
    
        # Update emotional state with more context
        self.emotional_state.update_emotional_state(
            user_input,
            sentiment['score'],
            {
                'intent': intent['label'],
                'entities': entities
            }
        )
    
        # Generate contextually rich response
        response = self._generate_contextual_response(
            user_input, 
            nlp_analysis
        )
    
        # Update context and conversation history
        self.context_manager.update_context(user_input, response)
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'user_input': user_input,
            'ai_response': response,
            'emotional_state': self.emotional_state.emotions
        })
    
        # Continuous learning
        self.continuous_learning.record_interaction(user_input, response)
    
        return response

    def _generate_contextual_response(self, user_input, nlp_analysis):
        # Advanced response generation logic
        intent = nlp_analysis['intent']['label']
        entities = nlp_analysis['entities']
        
        # Use knowledge base with enhanced context
        response_strategies = {
            'question': self._handle_question,
            'statement': self._handle_statement,
            'emotion': self._handle_emotional_input
        }
        
        response_generator = response_strategies.get(
            intent.lower(), 
            self.knowledge_base.get_response
        )
        
        return response_generator(user_input, self.context_manager.get_current_context())
    
    def _handle_question(self, user_input, context):
        # Advanced question handling
        entities = self.nlp_pipeline.named_entity_recognizer(user_input)
        if entities:
            # Find most relevant entity
            primary_entity = max(entities, key=lambda x: x['score'])
            return self.knowledge_base.get_response(primary_entity['word'], context)
        
        return "Could you provide more details about your question?"

    def _handle_statement(self, user_input, context):
        # Empathetic response generation
        sentiment = self.nlp_pipeline.sentiment_analyzer(user_input)[0]
        
        if sentiment['label'] == 'POSITIVE':
            return "That sounds wonderful! Tell me more."
        else:
            return "I'm sorry to hear that. How can I support you?"

    def _handle_emotional_input(self, user_input, context):
        # Emotion-specific response
        dominant_emotion = self._detect_dominant_emotion(user_input)
        
        empathy_responses = {
            'joy': "Your happiness is contagious!",
            'sadness': "I'm here to listen and support you.",
            'anger': "I understand you're feeling frustrated.",
            # Add more emotion-specific responses
        }
        
        return empathy_responses.get(
            dominant_emotion, 
            "I sense you're experiencing strong emotions."
        )

    def _detect_dominant_emotion(self, text):
        # Use emotion model to detect dominant emotion
        emotion_probabilities = self.emotion_model(text)[0]
        return self.emotion_labels[np.argmax(emotion_probabilities)]

    # def save_state(self):
    #     state = {
    #         "conversation_history": self.conversation_history,
    #         "emotional_state": self.emotional_state.emotions,
    #         "mood_history": self.emotional_state.mood_history,
    #         "long_term_memory": self.context_manager.long_term_memory
    #     }
        
    #     with open("ai_state.json", "w") as f:
    #         json.dump(state, f)
    
    # def load_state(self):
    #     try:
    #         with open("ai_state.json", "r") as f:
    #             state = json.load(f)
    #             self.conversation_history = state["conversation_history"]
    #             self.emotional_state.emotions = state["emotional_state"]
    #             self.emotional_state.mood_history = state["mood_history"]
    #             self.context_manager.long_term_memory = state["long_term_memory"]
    #     except FileNotFoundError:
    #         logger.info("No previous state file found")

    def save_state(self):
        state = {
            "conversation_history": self.conversation_history,
            "emotional_state": self.emotional_state.emotions,
            "context": self.context_manager.current_context,
            "learned_knowledge": self.continuous_learning.knowledge_base.facts
        }
        
        with open("advanced_ai_state.json", "w") as f:
            json.dump(state, f, default=str)

    def load_state(self):
        try:
            with open("advanced_ai_state.json", "r") as f:
                state = json.load(f)
                self.conversation_history = state["conversation_history"]
                self.emotional_state.emotions = state["emotional_state"]
                self.context_manager.current_context = state["context"]
                # Merge learned knowledge
                self.knowledge_base.facts.update(state["learned_knowledge"])
        except FileNotFoundError:
            logger.info("No previous state found")

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
