# File: main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import os
from pathlib import Path
import json
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import hashlib
import numpy as np

# Create necessary directories first (before logging setup)
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ["logs", "uploads", "results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# Create directories before configuring logging
create_directories()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Configuration
class Config:
    MODEL_PATH = "/Volumes/KODAK/folder 02/Brest_cancer_prediction/model/fine_tuning_model/breast_cancer_vit_fine_tuned_20250701_100324/best_model.pth"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    LOG_DIR = "logs"
    UPLOAD_DIR = "uploads"
    RESULTS_DIR = "results"

# Pydantic models
class PredictionRequest(BaseModel):
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Additional patient information")

class PredictionResponse(BaseModel):
    prediction_id: str
    timestamp: datetime
    prediction: str
    confidence: float
    risk_level: str
    recommendations: Dict[str, Any]
    processing_time: float
    patient_id: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime
    version: str

# Model wrapper class
class BreastCancerModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_transform()
        # Default class mapping - might need to be reversed based on training
        self.class_names = ["Normal", "Breast Cancer"]
        self.class_mapping_verified = False
        
    def _get_transform(self):
        """Define image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            logger.info("Successfully loaded checkpoint")
            
            # Check what type of object we have
            checkpoint_type = type(checkpoint).__name__
            logger.info(f"Checkpoint type: {checkpoint_type}")
            
            # Check if there's class mapping information in the checkpoint
            if isinstance(checkpoint, dict):
                if 'class_to_idx' in checkpoint:
                    self._update_class_mapping(checkpoint['class_to_idx'])
                elif 'class_names' in checkpoint:
                    self.class_names = checkpoint['class_names']
                    self.class_mapping_verified = True
                    logger.info(f"Found class names in checkpoint: {self.class_names}")
            
            # Handle different checkpoint formats
            if self._is_direct_model_object(checkpoint):
                logger.info("Detected direct model object")
                self.model = checkpoint
            elif self._is_transformers_checkpoint(checkpoint):
                logger.info("Detected transformers checkpoint")
                self.model = self._load_transformers_model(checkpoint)
            elif self._is_torchvision_checkpoint(checkpoint):
                logger.info("Detected torchvision checkpoint")
                self.model = self._load_torchvision_model(checkpoint)
            else:
                # Try to handle as a generic state dict
                logger.info("Attempting to load as generic state dict")
                self.model = self._load_generic_model(checkpoint)
            
            if self.model is None:
                raise RuntimeError("Failed to initialize model")
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully and moved to {self.device}")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Class mapping: {self.class_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _update_class_mapping(self, class_to_idx: Dict[str, int]):
        """Update class mapping based on training data"""
        logger.info(f"Found class_to_idx mapping: {class_to_idx}")
        
        # Create idx_to_class mapping
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Update class_names based on the mapping
        self.class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
        self.class_mapping_verified = True
        
        logger.info(f"Updated class mapping: {self.class_names}")
    
    def _is_direct_model_object(self, checkpoint):
        """Check if the checkpoint is a direct model object"""
        try:
            # Check if it's a transformers model
            if hasattr(checkpoint, 'config') and hasattr(checkpoint, 'forward'):
                return True
            # Check if it's a torch model
            if isinstance(checkpoint, torch.nn.Module):
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking direct model object: {str(e)}")
            return False
    
    def _is_transformers_checkpoint(self, checkpoint):
        """Check if the checkpoint is from a transformers model"""
        try:
            if not isinstance(checkpoint, dict):
                return False
                
            # Get the state dict
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                return False
            
            # Look for transformers-specific keys
            transformers_patterns = [
                'vit.embeddings',
                'vit.encoder',
                'classifier.weight',
                'classifier.bias',
                'embeddings.patch_embeddings',
                'embeddings.position_embeddings',
                'encoder.layer'
            ]
            
            state_dict_keys = list(state_dict.keys())
            for pattern in transformers_patterns:
                if any(pattern in key for key in state_dict_keys):
                    return True
                    
            # Also check for config
            if 'config' in checkpoint:
                config = checkpoint['config']
                if isinstance(config, dict) and config.get('model_type') == 'vit':
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking transformers checkpoint: {str(e)}")
            return False
    
    def _is_torchvision_checkpoint(self, checkpoint):
        """Check if the checkpoint is from a torchvision model"""
        try:
            if not isinstance(checkpoint, dict):
                return False
                
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                return False
            
            # Look for torchvision ViT patterns
            torchvision_patterns = [
                'conv_proj.weight',
                'conv_proj.bias',
                'encoder.layers',
                'heads.head.weight',
                'heads.head.bias'
            ]
            
            state_dict_keys = list(state_dict.keys())
            for pattern in torchvision_patterns:
                if any(pattern in key for key in state_dict_keys):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking torchvision checkpoint: {str(e)}")
            return False
    
    def _extract_state_dict(self, checkpoint):
        """Extract state dict from checkpoint"""
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            else:
                # Assume the checkpoint itself is the state dict
                return checkpoint
        return None
    
    def _load_transformers_model(self, checkpoint):
        """Load transformers ViT model"""
        try:
            # Try to import transformers
            try:
                from transformers import ViTForImageClassification, ViTConfig
            except ImportError:
                logger.error("Transformers library not installed. Install with: pip install transformers")
                raise ImportError("Transformers library required but not installed")
            
            # Get state dict
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                raise ValueError("Could not extract state dict from checkpoint")
            
            # Try to load config if available
            config = None
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                try:
                    config = ViTConfig.from_dict(checkpoint['config'])
                except Exception as e:
                    logger.warning(f"Failed to load config from checkpoint: {str(e)}")
            
            # Create default config if not available
            if config is None:
                config = ViTConfig(
                    image_size=224,
                    patch_size=16,
                    num_channels=3,
                    num_labels=2,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                )
            
            # Ensure num_labels is correct
            config.num_labels = 2
            
            # Create model
            model = ViTForImageClassification(config)
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.info("Successfully loaded transformers model")
            return model
            
        except Exception as e:
            logger.error(f"Error loading transformers model: {str(e)}")
            raise
    
    def _load_torchvision_model(self, checkpoint):
        """Load torchvision ViT model"""
        try:
            from torchvision.models import vit_b_16
            
            # Create model
            model = vit_b_16(weights=None)  # Use weights=None instead of pretrained=False
            
            # Modify the classifier head for 2 classes
            num_classes = 2
            if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            elif hasattr(model, 'head'):
                model.head = nn.Linear(model.head.in_features, num_classes)
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            
            # Get state dict
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                raise ValueError("Could not extract state dict from checkpoint")
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.info("Successfully loaded torchvision model")
            return model
            
        except Exception as e:
            logger.error(f"Error loading torchvision model: {str(e)}")
            raise
    
    def _load_generic_model(self, checkpoint):
        """Attempt to load as a generic model"""
        try:
            # Try transformers first
            try:
                logger.info("Attempting to load as transformers model")
                return self._load_transformers_model(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load as transformers model: {str(e)}")
            
            # Try torchvision
            try:
                logger.info("Attempting to load as torchvision model")
                return self._load_torchvision_model(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load as torchvision model: {str(e)}")
            
            raise RuntimeError("Could not load model with any known format")
            
        except Exception as e:
            logger.error(f"Error in generic model loading: {str(e)}")
            raise
    
    def predict(self, image: Image.Image, debug_mode: bool = False) -> Dict[str, Any]:
        """Make prediction on input image"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Preprocess image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                # Handle different model output formats
                if hasattr(outputs, 'logits'):
                    # Transformers model
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    # Direct tensor output
                    logits = outputs
                else:
                    # Try to get the first element if it's a tuple/list
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                # Ensure logits is 2D
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                probabilities = torch.nn.functional.softmax(logits[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
                
                # Convert to numpy for easier debugging
                prob_array = probabilities.cpu().numpy()
                pred_class_idx = predicted_class.item()
                confidence_score = confidence.item()
            
            # Debug information
            debug_info = {
                "raw_logits": logits[0].cpu().numpy().tolist(),
                "probabilities": prob_array.tolist(),
                "predicted_class_idx": pred_class_idx,
                "class_names": self.class_names,
                "class_mapping_verified": self.class_mapping_verified
            }
            
            logger.info(f"Debug info: {debug_info}")
            
            # Handle potential class mapping issues
            if not self.class_mapping_verified:
                # If we're not sure about the mapping, we might need to flip it
                # This is a heuristic based on the assumption that cancer should be less frequent
                if pred_class_idx == 0 and confidence_score > 0.8:
                    # If model is very confident about class 0, but we expect most images to be normal
                    # Check if we should flip the mapping
                    pass  # We'll handle this in the logic below
            
            # Get prediction - handle both possible mappings
            prediction = self.class_names[pred_class_idx]
            
            # Additional check: if mapping seems wrong, warn user
            if debug_mode and not self.class_mapping_verified:
                logger.warning("Class mapping not verified from checkpoint. "
                             "If predictions seem inverted, the model may have been trained with different class ordering.")
            
            # Determine risk level
            risk_level = self._determine_risk_level(prediction, confidence_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(prediction, confidence_score)
            
            result = {
                "prediction": prediction,
                "confidence": confidence_score,
                "risk_level": risk_level,
                "recommendations": recommendations,
                "probabilities": {
                    self.class_names[0]: prob_array[0],
                    self.class_names[1]: prob_array[1]
                }
            }
            
            if debug_mode:
                result["debug_info"] = debug_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def verify_class_mapping(self, known_cancer_image_path: str, known_normal_image_path: str):
        """
        Verify class mapping using known cancer and normal images
        This should be called during setup if you have reference images
        """
        try:
            # Load known cancer image
            cancer_image = Image.open(known_cancer_image_path).convert('RGB')
            cancer_result = self.predict(cancer_image, debug_mode=True)
            
            # Load known normal image  
            normal_image = Image.open(known_normal_image_path).convert('RGB')
            normal_result = self.predict(normal_image, debug_mode=True)
            
            logger.info(f"Known cancer image predicted as: {cancer_result['prediction']} "
                       f"(confidence: {cancer_result['confidence']:.2%})")
            logger.info(f"Known normal image predicted as: {normal_result['prediction']} "
                       f"(confidence: {normal_result['confidence']:.2%})")
            
            # Check if predictions match expectations
            if (cancer_result['prediction'] == 'Normal' and cancer_result['confidence'] > 0.7) or \
               (normal_result['prediction'] == 'Breast Cancer' and normal_result['confidence'] > 0.7):
                logger.warning("Class mapping appears to be inverted!")
                # Flip the class names
                self.class_names = [self.class_names[1], self.class_names[0]]
                logger.info(f"Flipped class mapping to: {self.class_names}")
                
        except Exception as e:
            logger.error(f"Error verifying class mapping: {str(e)}")
    
    def _determine_risk_level(self, prediction: str, confidence: float) -> str:
        """Determine risk level based on prediction and confidence"""
        if prediction == "Normal":
            return "Low"
        else:  # Breast Cancer
            if confidence >= 0.9:
                return "High"
            elif confidence >= 0.7:
                return "Medium-High"
            elif confidence >= 0.5:
                return "Medium"
            else:
                return "Low-Medium"
    
    def _generate_recommendations(self, prediction: str, confidence: float) -> Dict[str, Any]:
        """Generate medical recommendations based on prediction"""
        if prediction == "Normal":
            return {
                "immediate_action": "No immediate action required",
                "follow_up": "Continue regular screening as per guidelines",
                "lifestyle": [
                    "Maintain healthy lifestyle",
                    "Regular exercise",
                    "Balanced diet",
                    "Limit alcohol consumption"
                ],
                "next_screening": "Follow standard screening schedule",
                "disclaimer": "This is an AI prediction. Always consult with healthcare professionals."
            }
        else:  # Breast Cancer detected
            urgency = "HIGH" if confidence >= 0.8 else "MEDIUM"
            return {
                "immediate_action": f"URGENT: Contact oncologist immediately - {urgency} priority",
                "follow_up": "Schedule comprehensive diagnostic workup within 48-72 hours",
                "required_tests": [
                    "Detailed mammography",
                    "Ultrasound examination",
                    "Possible biopsy",
                    "MRI if recommended by oncologist"
                ],
                "specialist_referral": "Oncologist consultation required immediately",
                "support_resources": [
                    "Cancer support groups",
                    "Patient navigation services",
                    "Psychological counseling"
                ],
                "confidence_note": f"Model confidence: {confidence:.2%}",
                "disclaimer": "CRITICAL: This is an AI screening tool. Immediate professional medical evaluation is essential."
            }

# Global model instance
model_instance = None

# Startup/shutdown handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_instance
    logger.info("Starting up FastAPI server...")
    
    # Create directories
    for dir_path in [Config.LOG_DIR, Config.UPLOAD_DIR, Config.RESULTS_DIR]:
        Path(dir_path).mkdir(exist_ok=True)
    
    # Load model
    model_instance = BreastCancerModel(Config.MODEL_PATH)
    success = model_instance.load_model()
    
    if not success:
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")
    
    # Optional: Verify class mapping if you have reference images
    # Uncomment and provide paths to known cancer and normal images
    # model_instance.verify_class_mapping(
    #     "/path/to/known_cancer_image.jpg",
    #     "/path/to/known_normal_image.jpg"
    # )
    
    logger.info("FastAPI server started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI server...")

# Create FastAPI app
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Professional AI-powered breast cancer prediction from MRI images",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for API key validation (optional)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your token validation logic here
    # For now, we'll skip validation
    return credentials.credentials

# Utility functions
def validate_image(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        return False
    
    # Check file size (if size is available)
    if hasattr(file, 'size') and file.size and file.size > Config.MAX_FILE_SIZE:
        return False
    
    return True

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Breast Cancer Prediction API",
        "status": "active",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_instance is not None and model_instance.model is not None,
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_breast_cancer(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    debug_mode: bool = False,
    # token: str = Depends(verify_token)  # Uncomment for auth
):
    """
    Predict breast cancer from MRI image
    
    Args:
        file: MRI image file (JPG, PNG, etc.)
        patient_id: Optional patient identifier
        debug_mode: Include debug information in response
        
    Returns:
        Prediction results with recommendations
    """
    start_time = datetime.now()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if not validate_image(file):
            raise HTTPException(
                status_code=400,
                detail="Invalid file. Must be image file under 10MB"
            )
        
        # Read file content
        content = await file.read()
        file_hash = calculate_file_hash(content)
        
        # Log upload
        logger.info(f"Processing prediction {prediction_id} for file {file.filename}")
        
        # Save uploaded file
        upload_path = Path(Config.UPLOAD_DIR) / f"{prediction_id}_{file.filename}"
        with open(upload_path, 'wb') as f:
            f.write(content)
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make prediction
        prediction_result = model_instance.predict(image, debug_mode=debug_mode)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = PredictionResponse(
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            risk_level=prediction_result["risk_level"],
            recommendations=prediction_result["recommendations"],
            processing_time=processing_time,
            patient_id=patient_id,
            debug_info=prediction_result.get("debug_info") if debug_mode else None
        )
        
        # Save results
        result_path = Path(Config.RESULTS_DIR) / f"{prediction_id}_result.json"
        with open(result_path, 'w') as f:
            f.write(response.model_dump_json(indent=2))
        
        # Log result
        logger.info(f"Prediction {prediction_id} completed: {prediction_result['prediction']} "
                   f"(confidence: {prediction_result['confidence']:.2%})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Cleanup uploaded file after processing
        try:
            if 'upload_path' in locals():
                upload_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup upload file: {str(e)}")

@app.get("/prediction/{prediction_id}")
async def get_prediction_result(prediction_id: str):
    """Retrieve prediction result by ID"""
    try:
        result_path = Path(Config.RESULTS_DIR) / f"{prediction_id}_result.json"
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Prediction result not found")
        
        with open(result_path, 'r') as f:
            result_data = f.read()
        
        return json.loads(result_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve prediction")

@app.delete("/prediction/{prediction_id}")
async def delete_prediction_result(prediction_id: str):
    """Delete prediction result by ID"""
    try:
        result_path = Path(Config.RESULTS_DIR) / f"{prediction_id}_result.json"
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Prediction result not found")
        
        result_path.unlink()
        
        return {"message": f"Prediction {prediction_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete prediction")

@app.post("/verify-mapping")
async def verify_class_mapping(
    cancer_image: UploadFile = File(...),
    normal_image: UploadFile = File(...),
):
    """
    Verify and potentially fix class mapping using known cancer and normal images
    """
    try:
        # Read images
        cancer_content = await cancer_image.read()
        normal_content = await normal_image.read()
        
        # Create temporary files
        cancer_path = Path(Config.UPLOAD_DIR) / f"temp_cancer_{uuid.uuid4().hex}.jpg"
        normal_path = Path(Config.UPLOAD_DIR) / f"temp_normal_{uuid.uuid4().hex}.jpg"
        
        with open(cancer_path, 'wb') as f:
            f.write(cancer_content)
        with open(normal_path, 'wb') as f:
            f.write(normal_content)
        
        # Verify mapping
        old_mapping = model_instance.class_names.copy()
        model_instance.verify_class_mapping(str(cancer_path), str(normal_path))
        new_mapping = model_instance.class_names.copy()
        
        # Cleanup temp files
        cancer_path.unlink(missing_ok=True)
        normal_path.unlink(missing_ok=True)
        
        return {
            "message": "Class mapping verification completed",
            "old_mapping": old_mapping,
            "new_mapping": new_mapping,
            "mapping_changed": old_mapping != new_mapping
        }
        
    except Exception as e:
        logger.error(f"Error verifying class mapping: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Class mapping verification failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        workers=1,  # Adjust based on your needs
        log_level="info"
    )
   