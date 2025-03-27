import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from src.model import FaceRecognitionModel

class FaceRecognizer:
    def __init__(self, model_path, class_to_idx):
        """
        Initialize face recognition inference
        
        Args:
            model_path (str): Path to saved model
            class_to_idx (dict): Mapping of classes to indices
        """
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model
        self.num_classes = len(class_to_idx)
        self.model = FaceRecognitionModel(self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Invert class to idx mapping
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        
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
            results.append({
                'person': self.idx_to_class[idx],
                'confidence': float(prob)
            })
        
        # Filter by confidence threshold
        results = [r for r in results if r['confidence'] >= confidence_threshold]
        
        return {
            'predictions': results,
            'top_prediction': results[0] if results else None
        }

def main():
    # Example usage
    recognizer = FaceRecognizer(
        model_path='face_recognition_model.pth',
        class_to_idx={'person1': 0, 'person2': 1}
    )
    
    result = recognizer.predict('path/to/test/image.jpg')
    print(result)

if __name__ == '__main__':
    main()