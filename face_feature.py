import cv2
import dlib
import numpy as np
import os
from PIL import Image

class FacialFeatureDetector:
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

        self.FEATURE_COLORS = {
            "Face": (0, 255, 0),        # Green
            "Left Eyebrow": (255, 150, 0),  # Orange
            "Right Eyebrow": (255, 200, 0), # Light Orange
            "Left Eye": (255, 0, 0),     # Blue
            "Right Eye": (0, 0, 255),    # Red
            "Nose": (255, 0, 255),       # Purple
            "Mouth": (0, 255, 255),      # Yellow
            "Jawline": (255, 255, 0)     # Cyan
        }

    def identify_facial_features(self, image_path):
        """Identify and label facial features with colored bounding boxes"""
        # Load image with original size
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        #scale factor 
        height, width = image.shape[:2]
        max_dimension = 1000  
        scale = 1.0
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            height, width = image.shape[:2]
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        marked_image = image.copy()
        
        faces = self.detector(image_rgb)
        
        if len(faces) == 0:
            raise ValueError("No faces detected in the image")
        
        for face in faces:
            # box around face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), self.FEATURE_COLORS["Face"], 2)
            
            # Predict facial landmarks
            shape = self.predictor(image_rgb, face)
            points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            features = {
                "Left Eyebrow": list(range(17, 22)),  # Points 17-21
                "Right Eyebrow": list(range(22, 27)), # Points 22-26
                "Nose": list(range(27, 36)),         # Points 27-35
                "Left Eye": list(range(36, 42)),      # Points 36-41
                "Right Eye": list(range(42, 48)),     # Points 42-47
                "Mouth": list(range(48, 68)),         # Points 48-67
                "Jawline": list(range(0, 17))         # Points 0-16
            }
            
            font_scale = max(0.5, min(2.0, width / 1000))
            thickness = max(1, int(width / 500))
            
            # Process each feature separately
            for feature_name, indices in features.items():
                feature_points = points[indices]
                
                x, y, w, h = cv2.boundingRect(feature_points.reshape(-1, 1, 2))
                
                padding = max(5, int(width / 200))
                x, y = max(0, x - padding), max(0, y - padding)
                w, h = min(width - x, w + 2*padding), min(height - y, h + 2*padding)
                
                # Draw rectangle
                color = self.FEATURE_COLORS[feature_name]
                cv2.rectangle(marked_image, (x, y), (x + w, y + h), color, thickness)
                cv2.putText(marked_image, feature_name, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                
                # Draw the individual landmarks
                for (x_point, y_point) in feature_points:
                    cv2.circle(marked_image, (x_point, y_point), max(2, int(width / 300)), color, -1)
        
        legend_height = 200
        legend = np.zeros((legend_height, width, 3), dtype=np.uint8)
        
        cv2.putText(legend, "Facial Feature Legend", (width//2 - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        features_column1 = ["Face", "Left Eyebrow", "Right Eyebrow", "Left Eye"]
        features_column2 = ["Right Eye", "Nose", "Mouth", "Jawline"]
        
        y_pos = 60
        for i, feature in enumerate(features_column1):
            color = self.FEATURE_COLORS[feature]
            cv2.circle(legend, (width//4 - 50, y_pos + i*40), 10, color, -1)
            cv2.putText(legend, feature, (width//4 - 30, y_pos + i*40 + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 255), 1)
        
        for i, feature in enumerate(features_column2):
            color = self.FEATURE_COLORS[feature]
            cv2.circle(legend, (3*width//4 - 50, y_pos + i*40), 10, color, -1)
            cv2.putText(legend, feature, (3*width//4 - 30, y_pos + i*40 + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 255), 1)
        
        final_image = np.vstack([marked_image, legend])
        
        return final_image