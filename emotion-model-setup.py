# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from sklearn.model_selection import train_test_split
# import json
# import numpy as np
# import os

# class EmotionDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]

#         encoding = self.tokenizer(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=False,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# def prepare_emotion_model():
#     # Define paths properly using os.path
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     model_save_path = os.path.join(base_dir, "emotion_model")
#     emotion_triggers_path = os.path.join(base_dir, "emotion_triggers.json")

#     # Using a pre-trained model as starting point
#     model_name = "distilbert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=7  # joy, sadness, anger, fear, surprise, trust, anticipation
#     )

#     # Load emotion triggers for training data generation
#     with open(emotion_triggers_path, "r") as f:
#         emotion_triggers = json.load(f)

#     # Prepare training data
#     texts = []
#     labels = []
#     emotion_to_label = {emotion: i for i, emotion in enumerate(emotion_triggers.keys())}

#     # Generate training examples from emotion triggers
#     for emotion, triggers in emotion_triggers.items():
#         label = emotion_to_label[emotion]
#         # Create sentences using triggers
#         for trigger in triggers:
#             # Generate multiple examples per trigger
#             texts.extend([
#                 f"I feel {trigger}",
#                 f"Today was a {trigger} day",
#                 f"The situation is {trigger}",
#                 f"Everything seems {trigger}"
#             ])
#             labels.extend([label] * 4)

#     # Split into train and validation sets
#     train_texts, val_texts, train_labels, val_labels = train_test_split(
#         texts, labels, test_size=0.2, random_state=42
#     )

#     # Create datasets
#     train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
#     val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

#     # Training parameters
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=16)

#     # Training configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     num_epochs = 3

#     # Training loop
#     print("Starting training...")
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()
            
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )

#             loss = outputs.loss
#             total_loss += loss.item()
            
#             loss.backward()
#             optimizer.step()

#         avg_train_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch + 1}/{num_epochs}, Average training loss: {avg_train_loss:.4f}")

#     # Create model save directory if it doesn't exist
#     os.makedirs(model_save_path, exist_ok=True)
    
#     # Save the model and tokenizer
#     print(f"Saving model to {model_save_path}")
#     model.save_pretrained(model_save_path)
#     tokenizer.save_pretrained(model_save_path)
    
#     return model_save_path

# if __name__ == "__main__":
#     model_path = prepare_emotion_model()
#     print(f"Emotion model saved to: {model_path}")

# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import classification_report
# import numpy as np
# import json
# import os

# class AdvancedEmotionModelTrainer:
#     def __init__(self, model_name="distilbert-base-uncased"):
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.base_model = AutoModelForSequenceClassification.from_pretrained(
#             model_name, num_labels=7
#         )
#         self.emotion_labels = [
#             "joy", "sadness", "anger", 
#             "fear", "surprise", "trust", 
#             "anticipation"
#         ]
    
#     def _load_enhanced_dataset(self):
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         triggers_path = os.path.join(base_dir, "emotion_triggers.json")
        
#         with open(triggers_path, "r") as f:
#             emotion_triggers = json.load(f)
        
#         texts = []
#         labels = []
#         emotion_to_label = {emotion: i for i, emotion in enumerate(self.emotion_labels)}
        
#         # Enhanced data generation with context and complexity
#         for emotion, triggers in emotion_triggers.items():
#             label = emotion_to_label[emotion]
#             for trigger in triggers:
#                 contexts = [
#                     f"I feel deeply {trigger} right now",
#                     f"The situation makes me {trigger}",
#                     f"Something happened that feels {trigger}",
#                     f"My experience is {trigger}"
#                 ]
#                 texts.extend(contexts)
#                 labels.extend([label] * len(contexts))
        
#         return texts, labels
    
#     def train_model(self, epochs=5, learning_rate=2e-5):
#         texts, labels = self._load_enhanced_dataset()
        
#         # Advanced train/validation split
#         train_texts, val_texts, train_labels, val_labels = train_test_split(
#             texts, labels, test_size=0.2, stratify=labels, random_state=42
#         )
        
#         # Training configuration
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=learning_rate)
        
#         # Training loop with validation
#         best_val_score = 0
#         for epoch in range(epochs):
#             self.base_model.train()
#             train_loss = self._train_epoch(train_texts, train_labels, optimizer, device)
            
#             val_score = self._validate_model(val_texts, val_labels)
            
#             if val_score > best_val_score:
#                 best_val_score = val_score
#                 self._save_best_model()
            
#             print(f"Epoch {epoch+1}: Train Loss = {train_loss}, Validation Score = {val_score}")
    
#     def _train_epoch(self, texts, labels, optimizer, device):
#         # Implement training logic
#         pass
    
#     def _validate_model(self, texts, labels):
#         # Cross-validation scoring
#         return np.mean(cross_val_score(
#             self.base_model, 
#             texts, 
#             labels, 
#             scoring='f1_macro'
#         ))
    
#     def _save_best_model(self):
#         save_path = os.path.join(
#             os.path.dirname(__file__), 
#             "best_emotion_model"
#         )
#         self.base_model.save_pretrained(save_path)
#         self.tokenizer.save_pretrained(save_path)

# def main():
#     trainer = AdvancedEmotionModelTrainer()
#     trainer.train_model()

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import json
import numpy as np
import os

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_emotion_model():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    emotion_triggers_path = os.path.join(base_dir, "emotion_triggers.json")
    model_save_path = os.path.join(base_dir, "emotion_model")

    # Load emotion triggers
    with open(emotion_triggers_path, "r") as f:
        emotion_triggers = json.load(f)

    # Prepare training data
    texts = []
    labels = []
    emotion_to_label = {emotion: i for i, emotion in enumerate(emotion_triggers.keys())}

    # Generate training examples
    for emotion, triggers in emotion_triggers.items():
        label = emotion_to_label[emotion]
        for trigger in triggers:
            # Create multiple context-rich examples
            contexts = [
                f"I am feeling {trigger}",
                f"The situation makes me feel {trigger}",
                f"Everything seems {trigger}",
                f"My emotions are {trigger}"
            ]
            texts.extend(contexts)
            labels.extend([label] * len(contexts))

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(emotion_triggers)
    )

    # Create datasets
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(model_save_path, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train model
    trainer.train()

    # Save model and tokenizer
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    return model_save_path

if __name__ == "__main__":
    model_path = prepare_emotion_model()
    print(f"Emotion model saved to: {model_path}")