"""
Image Tampering Detection Module
Uses ELA (Error Level Analysis) and Metadata Validation
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from exif import Image as ExifImage
from urllib.request import urlopen
import json
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings('ignore')


class TamperingDetector:
    # Weather code mapping
    WEATHER_DICT = {
        0: 'Sunny', 1: 'Clear or partly cloudy', 2: 'Clear or partly cloudy',
        3: 'Slight Rain', 45: 'Fog', 48: 'Fog',
        51: 'Slight Raining', 53: 'Slight Raining', 55: 'Slight Raining',
        56: 'Slight Raining', 57: 'Slight Raining',
        61: 'Raining', 63: 'Raining', 65: 'Raining', 66: 'Raining', 67: 'Raining',
        71: 'Snow', 73: 'Snow', 75: 'Snow', 77: 'Snow',
        80: 'Raining', 81: 'Raining', 82: 'Raining',
        85: 'Snow', 86: 'Snow',
        95: 'Lightning', 96: 'Lightning', 99: 'Lightning'
    }
    
    ELA_CLASSES = ['Real', 'Tampered']
    WEATHER_CLASSES = ['Lightning', 'Rainy', 'Snow', 'Sunny']
    
    def __init__(self, ela_model_path, weather_model_path):
        """Initialize tampering detection models"""
        self.ela_model = None
        self.weather_model = None
        self._load_models(ela_model_path, weather_model_path)
    
    def _load_models(self, ela_model_path, weather_model_path):
        """Load ELA and Weather models"""
        import tensorflow as tf
        
        print("Loading tampering detection models...")
        
        # Load ELA model
        try:
            print(f"  Loading ELA model from {ela_model_path}...")
            try:
                import tf_keras
                self.ela_model = tf_keras.models.load_model(ela_model_path, compile=False)
            except ImportError:
                self.ela_model = tf.keras.models.load_model(ela_model_path, compile=False, safe_mode=False)
            print("  ELA model loaded successfully!")
        except Exception as e:
            print(f"  Error loading ELA model: {e}")
            self.ela_model = None
        
        # Load Weather model
        try:
            print(f"  Loading Weather model from {weather_model_path}...")
            try:
                import tf_keras
                self.weather_model = tf_keras.models.load_model(weather_model_path, compile=False)
            except ImportError:
                self.weather_model = tf.keras.models.load_model(weather_model_path, compile=False)
            print("  Weather model loaded successfully!")
        except Exception as e:
            print(f"  Error loading Weather model: {e}")
            self.weather_model = None
        
        print("Tampering detection models initialized!\n")
    
    def _convert_to_ela_image(self, path, quality=90):
        """Perform Error Level Analysis on an image"""
        temp_filename = 'temp_file_name_ela.jpg'
        
        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality=quality)
        temp_image = Image.open(temp_filename)
        
        ela_image = ImageChops.difference(image, temp_image)
        
        extrema = ela_image.getextrema()
        max_diff = sum([ex[1] for ex in extrema]) / 3
        if max_diff == 0:
            max_diff = 1
        
        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return ela_image
    
    def _prepare_image_for_ela(self, image_path):
        """Prepare image for ELA model"""
        ela_img = self._convert_to_ela_image(image_path, 90)
        img = np.array(ela_img.resize((128, 128))).flatten() / 255.0
        img = img.reshape(128, 128, 3)
        return np.expand_dims(img, axis=0), ela_img
    
    def _prepare_image_for_weather(self, image_path):
        """Prepare image for weather model"""
        img = np.array(Image.open(image_path).convert('RGB').resize((128, 128))) / 255.0
        img = img.reshape(128, 128, 3)
        return np.expand_dims(img, axis=0)
    
    def _decimal_coords(self, coords, ref):
        """Convert GPS coordinates to decimal format"""
        decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
        if ref == "S" or ref == 'W':
            decimal_degrees = -decimal_degrees
        return decimal_degrees
    
    def _extract_metadata(self, image_path):
        """Extract EXIF metadata from image"""
        try:
            with open(image_path, 'rb') as src:
                img = ExifImage(src)
            
            if not img.has_exif:
                return None, None, None, False
            
            try:
                latitude = self._decimal_coords(img.gps_latitude, img.gps_latitude_ref)
                longitude = self._decimal_coords(img.gps_longitude, img.gps_longitude_ref)
            except AttributeError:
                return None, None, None, False
            
            try:
                date_time = img.datetime_original
            except AttributeError:
                try:
                    date_time = img.gps_datestamp + " 12:00:00"
                except:
                    return None, None, None, False
            
            return date_time, latitude, longitude, True
        except Exception as e:
            return None, None, None, False
    
    def _get_historical_weather(self, date_time, lat, lon):
        """Fetch historical weather data"""
        try:
            date = date_time[:10].replace(':', '-')
            time = date_time[11:]
            hour = int(time[:2])
            
            geoLoc = Nominatim(user_agent="FraudDetector")
            locname = geoLoc.reverse(f"{lat},{lon}")
            
            base_url = 'https://archive-api.open-meteo.com/v1/era5?'
            url = f"{base_url}latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&hourly=weathercode&timezone=Asia%2FBangkok"
            
            response = urlopen(url)
            data_json = json.loads(response.read())
            
            weather_code = data_json['hourly']['weathercode']
            
            if weather_code[hour - 1] is None:
                return str(locname), date, "NA"
            
            weather_condition = self.WEATHER_DICT.get(weather_code[hour - 1], "Unknown")
            return str(locname), date, weather_condition
        except Exception as e:
            return "Unknown", "Unknown", "NA"
    
    def detect(self, image_path):
        """
        Detect if an image has been tampered with
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: {
                'is_tampered': bool,
                'tampering_score': float,
                'confidence': str,
                'ela_prediction': str,
                'weather_match': bool,
                'details': list
            }
        """
        result = {
            'is_tampered': False,
            'tampering_score': 0.0,
            'confidence': 'Low',
            'ela_prediction': None,
            'weather_match': None,
            'metadata_available': False,
            'details': []
        }
        
        if not os.path.exists(image_path):
            result['details'].append(f"Error: Image not found at {image_path}")
            return result
        
        # ELA ANALYSIS
        if self.ela_model is not None:
            try:
                np_img_input, ela_image = self._prepare_image_for_ela(image_path)
                predictions = self.ela_model.predict(np_img_input, verbose=0)
                
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                result['ela_prediction'] = self.ELA_CLASSES[predicted_class]
                
                if predicted_class == 1:  # Tampered
                    result['tampering_score'] += confidence * 0.6
                    result['is_tampered'] = True
                else:  # Real
                    result['tampering_score'] += (100 - confidence) * 0.6
                
                result['details'].append(
                    f"ELA Analysis: {self.ELA_CLASSES[predicted_class]} ({confidence:.2f}% confident)"
                )
            except Exception as e:
                result['details'].append(f"ELA Analysis failed: {e}")
        else:
            result['details'].append("ELA model not available")
        
        # METADATA EXTRACTION
        date_time, lat, lon, has_metadata = self._extract_metadata(image_path)
        
        if has_metadata:
            result['metadata_available'] = True
            result['details'].append(f"Metadata found: {date_time} at ({lat:.4f}, {lon:.4f})")
        else:
            result['details'].append("No EXIF metadata available")
        
        # WEATHER VALIDATION
        if has_metadata and self.weather_model is not None:
            try:
                weather_input = self._prepare_image_for_weather(image_path)
                weather_predictions = self.weather_model.predict(weather_input, verbose=0)
                
                predicted_weather_idx = np.argmax(weather_predictions[0])
                predicted_weather = self.WEATHER_CLASSES[predicted_weather_idx]
                
                location, date, historical_weather = self._get_historical_weather(date_time, lat, lon)
                
                if historical_weather != "NA":
                    predicted_lower = predicted_weather.lower()
                    historical_lower = historical_weather.lower()
                    
                    weather_match = False
                    if predicted_lower in historical_lower or historical_lower in predicted_lower:
                        weather_match = True
                    elif ('rain' in predicted_lower and 'rain' in historical_lower):
                        weather_match = True
                    elif ('sunny' in predicted_lower or 'clear' in predicted_lower) and \
                         ('sunny' in historical_lower or 'clear' in historical_lower):
                        weather_match = True
                    
                    result['weather_match'] = weather_match
                    
                    if not weather_match:
                        result['tampering_score'] += 40 * 0.4
                        result['is_tampered'] = True
                        result['details'].append(
                            f"Weather Mismatch: Detected '{predicted_weather}' vs Historical '{historical_weather}'"
                        )
                    else:
                        result['details'].append(
                            f"Weather Match: '{predicted_weather}' matches '{historical_weather}'"
                        )
            except Exception as e:
                result['details'].append(f"Weather validation failed: {e}")
        
        # Ensure score is between 0 and 100
        result['tampering_score'] = min(100, max(0, result['tampering_score']))
        
        # Determine confidence
        if result['tampering_score'] < 30:
            result['confidence'] = "High (Authentic)"
        elif result['tampering_score'] < 60:
            result['confidence'] = "Medium (Uncertain)"
        else:
            result['confidence'] = "High (Tampered)"
        
        return result


if __name__ == "__main__":
    # Test
    detector = TamperingDetector(
        "Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/ELA_Training/model_ela.h5",
        "Image-Tampering-Detection-using-ELA-and-Metadata-Analysis/WeatherCNNTraining/Weather_Model.h5"
    )
    result = detector.detect("test_image.jpg")
    print(result)
