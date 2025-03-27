import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from src.model import FaceRecognitionModel

class FaceRecognizer:
    def __init__(self, model_path, num_classes=None, class_to_idx=None):
        """
        Initialize face recognition inference
        
        Args:
            model_path (str): Path to saved model
            num_classes (int, optional): Number of classes to use
            class_to_idx (dict, optional): Mapping of classes to indices
        """
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Use provided number of classes or from checkpoint
        if num_classes is None:
            num_classes = checkpoint.get('num_classes', len(checkpoint['class_to_idx']))
        
        # Recreate model
        self.model = FaceRecognitionModel(num_classes)
        
        # Dynamically modify the last layer if needed
        if num_classes != checkpoint.get('num_classes', num_classes):
            print(f"Warning: Adjusting model output layer from {checkpoint.get('num_classes', num_classes)} to {num_classes} classes")
            
            # Recreate the last layer
            num_features = self.model.backbone.fc[0].in_features
            self.model.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
        
        # Load state dict
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.' prefix if model was trained with DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Use provided class_to_idx or from checkpoint
        self.class_to_idx = class_to_idx or checkpoint.get('class_to_idx', {})
        
        # Invert class to idx mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=3, confidence_threshold=0.7):
        """
        Predict person in the image
        
        Args:
            image_path (str): Path to input image
            top_k (int): Number of top predictions to return
            confidence_threshold (float): Minimum confidence for prediction
        
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = probabilities.topk(top_k)
        
        # Convert to numpy for easier processing
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Process results
        results = []
        for prob, idx in zip(top_probs, top_indices):
            # Use index mapping if available, otherwise use index as fallback
            person_name = self.idx_to_class.get(idx, f"Person_{idx}")
            results.append({
                'person': person_name,
                'confidence': float(prob)
            })
        
        # Filter by confidence threshold
        results = [r for r in results if r['confidence'] >= confidence_threshold]
        
        return {
            'predictions': results,
            'top_prediction': results[0] if results else None
        }

def main():
    # Example usage with more flexible initialization
    recognizer = FaceRecognizer(
        model_path='face_recognition_model.pth',
        num_classes=4,  # Specify number of classes if needed
        class_to_idx={'person1': 0, 'person2': 1, 'person3': 2, 'person4': 3}
    )
    
    result = recognizer.predict('path/to/test/image.jpg')
    print(result)

if __name__ == '__main__':
    main()