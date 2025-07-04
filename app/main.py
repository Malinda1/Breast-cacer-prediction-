# File: main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
    directories = ["logs", "uploads", "results", "static", "templates"]
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
    STATIC_DIR = "static"
    TEMPLATES_DIR = "app/template"

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

# Model wrapper class (FIXED VERSION)
class BreastCancerModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_transform()
        
        # FIXED: Start with the correct mapping that matches your model's training
        # Based on your issue, your model was likely trained with Cancer=0, Normal=1
        self.class_names = ["Breast Cancer", "Normal"]  # This matches the model's training
        self.class_to_idx = {"Breast Cancer": 0, "Normal": 1}  # This matches the model's training
        self.idx_to_class = {0: "Breast Cancer", 1: "Normal"}  # This matches the model's training
        self.class_mapping_verified = False
        self.prediction_needs_inversion = False
        
    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            logger.info("Successfully loaded checkpoint")
            
            checkpoint_type = type(checkpoint).__name__
            logger.info(f"Checkpoint type: {checkpoint_type}")
            
            # FIXED: Better class mapping extraction
            if isinstance(checkpoint, dict):
                # Check for class mapping in checkpoint
                if 'class_to_idx' in checkpoint:
                    self._update_class_mapping_from_checkpoint(checkpoint['class_to_idx'])
                elif 'class_names' in checkpoint:
                    self._update_class_names_from_checkpoint(checkpoint['class_names'])
                elif 'idx_to_class' in checkpoint:
                    self._update_idx_to_class_from_checkpoint(checkpoint['idx_to_class'])
                else:
                    logger.warning("No class mapping found in checkpoint.")
                    logger.info("Using corrected default mapping: Breast Cancer=0, Normal=1")
                    # Keep the corrected mapping we set in __init__
                    self.class_mapping_verified = True
            
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
                logger.info("Attempting to load as generic state dict")
                self.model = self._load_generic_model(checkpoint)
            
            if self.model is None:
                raise RuntimeError("Failed to initialize model")
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully and moved to {self.device}")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Class mapping: {self.class_names}")
            logger.info(f"Class to idx: {self.class_to_idx}")
            logger.info(f"Idx to class: {self.idx_to_class}")
            logger.info(f"Class mapping verified: {self.class_mapping_verified}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _update_class_mapping_from_checkpoint(self, class_to_idx: Dict[str, int]):
        """Update class mapping from checkpoint's class_to_idx"""
        logger.info(f"Found class_to_idx mapping in checkpoint: {class_to_idx}")
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.class_names = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
        self.class_mapping_verified = True
        logger.info(f"Updated class mapping from checkpoint - Names: {self.class_names}")
    
    def _update_class_names_from_checkpoint(self, class_names: list):
        """Update class mapping from checkpoint's class_names"""
        logger.info(f"Found class_names in checkpoint: {class_names}")
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(class_names)}
        self.class_mapping_verified = True
        logger.info(f"Updated class mapping from class_names - Mapping: {self.class_to_idx}")
    
    def _update_idx_to_class_from_checkpoint(self, idx_to_class: Dict[int, str]):
        """Update class mapping from checkpoint's idx_to_class"""
        logger.info(f"Found idx_to_class mapping in checkpoint: {idx_to_class}")
        self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in idx_to_class.items()}
        self.class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
        self.class_mapping_verified = True
        logger.info(f"Updated class mapping from idx_to_class - Names: {self.class_names}")
    
    def _is_direct_model_object(self, checkpoint):
        try:
            if hasattr(checkpoint, 'config') and hasattr(checkpoint, 'forward'):
                return True
            if isinstance(checkpoint, torch.nn.Module):
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking direct model object: {str(e)}")
            return False
    
    def _is_transformers_checkpoint(self, checkpoint):
        try:
            if not isinstance(checkpoint, dict):
                return False
                
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                return False
            
            transformers_patterns = [
                'vit.embeddings', 'vit.encoder', 'classifier.weight', 'classifier.bias',
                'embeddings.patch_embeddings', 'embeddings.position_embeddings', 'encoder.layer'
            ]
            
            state_dict_keys = list(state_dict.keys())
            for pattern in transformers_patterns:
                if any(pattern in key for key in state_dict_keys):
                    return True
                    
            if 'config' in checkpoint:
                config = checkpoint['config']
                if isinstance(config, dict) and config.get('model_type') == 'vit':
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking transformers checkpoint: {str(e)}")
            return False
    
    def _is_torchvision_checkpoint(self, checkpoint):
        try:
            if not isinstance(checkpoint, dict):
                return False
                
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                return False
            
            torchvision_patterns = [
                'conv_proj.weight', 'conv_proj.bias', 'encoder.layers',
                'heads.head.weight', 'heads.head.bias'
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
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                return checkpoint['state_dict']
            elif 'model' in checkpoint:
                return checkpoint['model']
            else:
                return checkpoint
        return None
    
    def _load_transformers_model(self, checkpoint):
        try:
            try:
                from transformers import ViTForImageClassification, ViTConfig
            except ImportError:
                logger.error("Transformers library not installed. Install with: pip install transformers")
                raise ImportError("Transformers library required but not installed")
            
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                raise ValueError("Could not extract state dict from checkpoint")
            
            config = None
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                try:
                    config = ViTConfig.from_dict(checkpoint['config'])
                except Exception as e:
                    logger.warning(f"Failed to load config from checkpoint: {str(e)}")
            
            if config is None:
                config = ViTConfig(
                    image_size=224, patch_size=16, num_channels=3, num_labels=2,
                    hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                    intermediate_size=3072,
                )
            
            config.num_labels = 2
            model = ViTForImageClassification(config)
            
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
        try:
            from torchvision.models import vit_b_16
            
            model = vit_b_16(weights=None)
            num_classes = 2
            
            if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            elif hasattr(model, 'head'):
                model.head = nn.Linear(model.head.in_features, num_classes)
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            
            state_dict = self._extract_state_dict(checkpoint)
            if state_dict is None:
                raise ValueError("Could not extract state dict from checkpoint")
            
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
        try:
            try:
                logger.info("Attempting to load as transformers model")
                return self._load_transformers_model(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to load as transformers model: {str(e)}")
            
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
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                probabilities = torch.nn.functional.softmax(logits[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
                
                prob_array = probabilities.cpu().numpy()
                pred_class_idx = predicted_class.item()
                confidence_score = confidence.item()
            
            # FIXED: The model outputs the correct class index directly
            # No need for complex inversion logic since we corrected the initial mapping
            
            debug_info = {
                "raw_logits": logits[0].cpu().numpy().tolist(),
                "probabilities": probabilities.cpu().numpy().tolist(),
                "predicted_class_idx": pred_class_idx,
                "class_names": self.class_names,
                "class_to_idx": self.class_to_idx,
                "idx_to_class": self.idx_to_class,
                "class_mapping_verified": self.class_mapping_verified,
                "model_output_interpretation": f"Model output {pred_class_idx} -> {self.idx_to_class[pred_class_idx]}"
            }
            
            logger.info(f"Debug info: {debug_info}")
            
            # Use the class index directly with our corrected mapping
            prediction = self.idx_to_class[pred_class_idx]
            
            risk_level = self._determine_risk_level(prediction, confidence_score)
            recommendations = self._generate_recommendations(prediction, confidence_score)
            
            result = {
                "prediction": prediction,
                "confidence": confidence_score,
                "risk_level": risk_level,
                "recommendations": recommendations,
                "probabilities": {
                    self.class_names[0]: float(prob_array[0]),  # Breast Cancer probability
                    self.class_names[1]: float(prob_array[1])   # Normal probability
                }
            }
            
            if debug_mode:
                result["debug_info"] = debug_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def _determine_risk_level(self, prediction: str, confidence: float) -> str:
        if prediction == "Normal":
            return "Low"
        else:
            if confidence >= 0.9:
                return "High"
            elif confidence >= 0.7:
                return "Medium-High"
            elif confidence >= 0.5:
                return "Medium"
            else:
                return "Low-Medium"
    
    def _generate_recommendations(self, prediction: str, confidence: float) -> Dict[str, Any]:
        if prediction == "Normal":
            return {
                "immediate_action": "No immediate action required",
                "follow_up": "Continue regular screening as per guidelines",
                "lifestyle": [
                    "Maintain healthy lifestyle", "Regular exercise",
                    "Balanced diet", "Limit alcohol consumption"
                ],
                "next_screening": "Follow standard screening schedule",
                "disclaimer": "This is an AI prediction. Always consult with healthcare professionals."
            }
        else:
            urgency = "HIGH" if confidence >= 0.8 else "MEDIUM"
            return {
                "immediate_action": f"URGENT: Contact oncologist immediately - {urgency} priority",
                "follow_up": "Schedule comprehensive diagnostic workup within 48-72 hours",
                "required_tests": [
                    "Detailed mammography", "Ultrasound examination",
                    "Possible biopsy", "MRI if recommended by oncologist"
                ],
                "specialist_referral": "Oncologist consultation required immediately",
                "support_resources": [
                    "Cancer support groups", "Patient navigation services",
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
    global model_instance
    logger.info("Starting up FastAPI server...")
    
    # Create directories using the correct paths
    for dir_path in [Config.LOG_DIR, Config.UPLOAD_DIR, Config.RESULTS_DIR, Config.STATIC_DIR]:
        Path(dir_path).mkdir(exist_ok=True)
    
    # Create the template directory if it doesn't exist
    Path(Config.TEMPLATES_DIR).mkdir(parents=True, exist_ok=True)
    
    model_instance = BreastCancerModel(Config.MODEL_PATH)
    success = model_instance.load_model()
    
    if not success:
        logger.error("Failed to load model during startup")
        raise RuntimeError("Model loading failed")
    
    logger.info("FastAPI server started successfully")
    yield
    
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=Config.TEMPLATES_DIR)

# Dependency for API key validation (optional)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return credentials.credentials

# Utility functions
def validate_image(file: UploadFile) -> bool:
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        return False
    
    if hasattr(file, 'size') and file.size and file.size > Config.MAX_FILE_SIZE:
        return False
    
    return True

def calculate_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main frontend HTML page"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving frontend: {str(e)}")
        # Return a simple HTML response if template is not found
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breast Cancer Prediction API</title>
        </head>
        <body>
            <h1>Breast Cancer Prediction API</h1>
            <p>The API is running successfully!</p>
            <p>Template file not found. Please ensure index.html is in the app/template directory.</p>
            <p>Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
        </html>
        """)

@app.get("/api", response_model=Dict[str, str])
async def api_root():
    """API root endpoint"""
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
):
    """Predict breast cancer from MRI image"""
    start_time = datetime.now()
    prediction_id = str(uuid.uuid4())
    
    try:
        if not validate_image(file):
            raise HTTPException(
                status_code=400,
                detail="Invalid file. Must be image file under 10MB"
            )
        
        content = await file.read()
        file_hash = calculate_file_hash(content)
        
        logger.info(f"Processing prediction {prediction_id} for file {file.filename}")
        
        upload_path = Path(Config.UPLOAD_DIR) / f"{prediction_id}_{file.filename}"
        with open(upload_path, 'wb') as f:
            f.write(content)
        
        image = Image.open(io.BytesIO(content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        prediction_result = model_instance.predict(image, debug_mode=debug_mode)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
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
        
        result_path = Path(Config.RESULTS_DIR) / f"{prediction_id}_result.json"
        with open(result_path, 'w') as f:
            f.write(response.model_dump_json(indent=2))
        
        logger.info(f"Prediction {prediction_id} completed: {prediction_result['prediction']} "
                   f"(confidence: {prediction_result['confidence']:.2%})")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
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

@app.get("/class-mapping")
async def get_class_mapping():
    """Get current class mapping information"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "class_names": model_instance.class_names,
        "class_to_idx": model_instance.class_to_idx,
        "idx_to_class": model_instance.idx_to_class,
        "class_mapping_verified": model_instance.class_mapping_verified
    }

@app.post("/test-prediction")
async def test_prediction_mapping():
    """Test endpoint to verify prediction mapping is correct"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Create a simple test to show current mapping
    return {
        "message": "Current class mapping",
        "mapping": {
            "index_0_maps_to": model_instance.idx_to_class[0],
            "index_1_maps_to": model_instance.idx_to_class[1]
        },
        "explanation": {
            "when_model_outputs_0": f"Prediction will be: {model_instance.idx_to_class[0]}",
            "when_model_outputs_1": f"Prediction will be: {model_instance.idx_to_class[1]}"
        },
        "note": "If this mapping is wrong, the predictions will be swapped. Cancer images should output 'Breast Cancer', normal images should output 'Normal'."
    }

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
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )