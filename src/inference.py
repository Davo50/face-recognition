import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from src.model import FaceRecognitionModel

class FaceRecognizer:
    def __init__(self, model_path, num_classes=None, class_to_idx=None):
        """
        Инициализация инференса распознавания лиц
        
        Args:
            model_path (str): Путь к сохранённой модели
            num_classes (int, optional): Количество классов
            class_to_idx (dict, optional): Словарь сопоставления классов с индексами
        """
        # Определение устройства
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Загрузка контрольной точки (checkpoint)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Используем переданное количество классов или берём из контрольной точки
        if num_classes is None:
            num_classes = checkpoint.get('num_classes', len(checkpoint['class_to_idx']))
        
        # Создаём модель
        self.model = FaceRecognitionModel(num_classes)
        
        # При необходимости модифицируем последний слой модели
        if num_classes != checkpoint.get('num_classes', num_classes):
            print(f"Warning: Adjusting model output layer from {checkpoint.get('num_classes', num_classes)} to {num_classes} classes")
            
            num_features = self.model.backbone.fc[0].in_features
            self.model.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, num_classes)
            )
        
        # Загрузка state dict
        state_dict = checkpoint['model_state_dict']
        
        # Убираем префикс 'module.', если модель обучалась с DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Используем переданный class_to_idx или берем из контрольной точки
        self.class_to_idx = class_to_idx or checkpoint.get('class_to_idx', {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Преобразование изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path, top_k=3, confidence_threshold=0.5):
        """
        Распознавание лица на изображении
        
        Args:
            image_path (str): Путь к изображению
            top_k (int): Количество выдаваемых вариантов
            confidence_threshold (float): Порог уверенности
        
        Returns:
            Словарь с результатами распознавания
        """
        # Разрешение абсолютного пути для изображения
        resolved_path = os.path.abspath(image_path)
        print(f"Resolved path: {resolved_path}")
        
        # Проверка наличия файла
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Изображение не найдено: {resolved_path}")
        
        # Загрузка и предобработка изображения
        image = Image.open(resolved_path).convert('RGB')
        print("Изображение загружено:", image.size)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Выполнение предсказания
        with torch.no_grad():
            outputs = self.model(image_tensor)
            print("Выход модели:", outputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print("Вероятности:", probabilities)
            top_probs, top_indices = probabilities.topk(top_k)

        # Перевод в numpy для удобства обработки
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            person_name = self.idx_to_class.get(idx, f"Person_{idx}")
            results.append({
                'person': person_name,
                'confidence': float(prob)
            })
        
        results = [r for r in results if r['confidence'] >= confidence_threshold]
        
        return {
            'predictions': results,
            'top_prediction': results[0] if results else None
        }

def main():
    print("Старт инференса")
    recognizer = FaceRecognizer(
        model_path='face_recognition_model.pth',
        num_classes=4,
        class_to_idx={'Andron': 0, 'David': 1, 'Kirill': 2, 'Nikita': 3}
    )
    
    while True:
        print("\nМеню:")
        print("1. Распознать лицо")
        print("2. Выход")
        choice = input("Выберите опцию (1/2): ")
        
        if choice == "1":
            image_path = input("Введите путь к изображению: ")
            try:
                result = recognizer.predict(image_path)
                print("Результат инференса:", result)
            except Exception as e:
                print(f"Ошибка: {e}")
        elif choice == "2":
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте ещё раз.")

if __name__ == '__main__':
    main()
