import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from transformers import BertModel, BertTokenizer
import json
import warnings
#from IPython.display import display

class VQAModel(nn.Module):
    def __init__(self, cnn_feature_size=2048, bert_feature_size=768, num_classes=1001):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(cnn_feature_size + bert_feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, cnn_features, bert_features):
        combined_features = torch.cat((cnn_features, bert_features), dim=1)
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x

def load_model_safely(model_path, device):
    """
    Safely load a PyTorch model with proper error handling.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            model_data = torch.load(model_path, map_location=device)

        model = VQAModel(cnn_feature_size=2048, bert_feature_size=768, num_classes=1001)

        if isinstance(model_data, dict):
            if 'state_dict' in model_data:
                model.load_state_dict(model_data['state_dict'])
            else:
                model.load_state_dict(model_data)
        elif isinstance(model_data, nn.Module):
            model = model_data
        else:
            raise ValueError(f"Unexpected model data type: {type(model_data)}")

        model = model.to(device)
        return model

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

class VQADemo:
    def __init__(self, model_path, answers_vocab_path):
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        try:
            with open('data/answer_vocab.json', 'r') as f:
                vocab_data = json.load(f)
                # Create a reverse mapping from index to answer
                self.answers_vocab = {str(v): k for k, v in vocab_data.items()}
        except Exception as e:
            raise Exception(f"Failed to load answers vocabulary: {str(e)}")

        # Initialize CNN (ResNet50)
        try:
            self.cnn_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.cnn_model.fc = nn.Identity()
            self.cnn_model = self.cnn_model.to(self.device)
            self.cnn_model.eval()
        except Exception as e:
            raise Exception(f"Failed to initialize ResNet50: {str(e)}")

        # Initialize BERT
        try:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased', local_files_only=False)
            self.bert_model = self.bert_model.to(self.device)
            self.bert_model.eval()

            # Initialize tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=False)
        except Exception as e:
            raise Exception(f"Failed to initialize BERT: {str(e)}")

        # Load and move the VQA model to device
        try:
            self.model = load_model_safely(model_path, self.device)
            self.model.eval()
        except Exception as e:
            raise Exception(f"Failed to load VQA model: {str(e)}")

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path):
        """Process image for model input"""
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            # Store the original image for display
            self.original_image = image
            image_tensor = self.transform(image)
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            return image_tensor
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def process_question(self, question):
        """Process question text for model input"""
        try:
            # Tokenize question
            encoded = self.tokenizer.encode_plus(
                question,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return encoded['input_ids'], encoded['attention_mask']
        except Exception as e:
            raise Exception(f"Error processing question: {str(e)}")

    def extract_cnn_features(self, image_tensor):
        """Extract features using ResNet50"""
        try:
            with torch.no_grad():
                # Ensure image tensor is on correct device
                image_tensor = image_tensor.to(self.device)
                cnn_features = self.cnn_model(image_tensor)
            return cnn_features.to(self.device)
        except Exception as e:
            raise Exception(f"Error extracting CNN features: {str(e)}")

    def extract_bert_features(self, input_ids, attention_mask):
        """Extract features using BERT"""
        try:
            with torch.no_grad():
                # Ensure inputs are on correct device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                bert_features = outputs.last_hidden_state[:, 0, :]
            return bert_features.to(self.device)
        except Exception as e:
            raise Exception(f"Error extracting BERT features: {str(e)}")

    def get_answer(self, image_path, question):
        """Get model prediction and display the image"""
        try:
            with torch.no_grad():
                # Process inputs
                image_tensor = self.process_image(image_path)

                # Display the original image
                print("Input Image:")
                #display(self.original_image)

                # Extract features
                cnn_features = self.extract_cnn_features(image_tensor)
                input_ids, attention_mask = self.process_question(question)
                bert_features = self.extract_bert_features(input_ids, attention_mask)

                # Forward pass
                outputs = self.model(cnn_features, bert_features)
                pred_idx = outputs.argmax(dim=1).item()

                # Get answer using the reversed vocabulary mapping
                answer = self.answers_vocab.get(str(pred_idx), f"Unknown answer index: {pred_idx}")

                print(f"\nQuestion: {question}")
                print(f"Answer: {answer}")

            return answer
        except Exception as e:
            return f"Error getting answer: {str(e)}"