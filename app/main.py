"""
Customer Churn Prediction API
FastAPI application for serving ML predictions with SHAP explanations
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import asyncio
from contextlib import asynccontextmanager
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import time

# Import custom modules
from app.utils.shap_explainer import ChurnExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REGISTRY = CollectorRegistry()
prediction_counter = Counter('predictions_total', 'Total predictions made', ['model_version', 'prediction'], registry=REGISTRY)
prediction_histogram = Histogram('prediction_duration_seconds', 'Prediction request duration', registry=REGISTRY)
feedback_counter = Counter('feedback_total', 'Total feedback received', ['feedback_type'], registry=REGISTRY)
api_request_counter = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status_code'], registry=REGISTRY)

# Global variables for model and explainer
model = None
explainer = None
model_version = "1.0.0"
model_metadata = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Customer Churn Prediction API...")
    await load_models()
    logger.info("Models loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Churn Prediction API...")

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready API for customer churn prediction with SHAP explanations",
    version=model_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Grafana
        "http://localhost:8080",  # Frontend
        "http://localhost:8000",  # API docs
        "https://your-domain.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response models
class CustomerData(BaseModel):
    """Customer data model for prediction requests"""
    monthly_charges: float = Field(..., ge=0, description="Monthly subscription charges")
    total_charges: float = Field(..., ge=0, description="Total charges to date")
    contract_length: int = Field(..., ge=1, le=60, description="Contract length in months")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    days_since_last_login: int = Field(..., ge=0, description="Days since last login")
    never_logged_in: int = Field(..., ge=0, le=1, description="1 if never logged in, 0 otherwise")
    age: int = Field(..., ge=18, le=120, description="Customer age")
    total_transactions: int = Field(..., ge=0, description="Total number of transactions")
    total_amount: float = Field(..., ge=0, description="Total transaction amount")
    avg_amount: float = Field(..., ge=0, description="Average transaction amount")
    std_amount: float = Field(..., ge=0, description="Standard deviation of transaction amounts")
    failure_rate: float = Field(..., ge=0, le=1, description="Transaction failure rate")
    days_since_last_transaction: int = Field(..., ge=0, description="Days since last transaction")
    failed_transactions: int = Field(..., ge=0, description="Number of failed transactions")
    total_tickets: int = Field(..., ge=0, description="Total support tickets")
    avg_satisfaction: float = Field(..., ge=0, le=5, description="Average satisfaction score")
    min_satisfaction: float = Field(..., ge=0, le=5, description="Minimum satisfaction score")
    avg_resolution_time: float = Field(..., ge=0, description="Average resolution time in hours")
    billing_tickets: int = Field(..., ge=0, description="Number of billing tickets")
    technical_tickets: int = Field(..., ge=0, description="Number of technical tickets")
    cancellation_tickets: int = Field(..., ge=0, description="Number of cancellation tickets")
    customer_value_score: float = Field(..., ge=0, description="Customer value score")
    risk_score: float = Field(..., ge=0, description="Customer risk score")
    subscription_type: str = Field(..., description="Subscription type (Basic/Premium/Enterprise)")
    province: str = Field(..., description="Canadian province")
    payment_method: str = Field(..., description="Payment method")
    paperless_billing: int = Field(..., ge=0, le=1, description="1 if paperless billing, 0 otherwise")
    auto_pay: int = Field(..., ge=0, le=1, description="1 if auto pay enabled, 0 otherwise")
    
    @validator('subscription_type')
    def validate_subscription_type(cls, v):
        allowed_types = ['Basic', 'Premium', 'Enterprise']
        if v not in allowed_types:
            raise ValueError(f'subscription_type must be one of {allowed_types}')
        return v
    
    @validator('province')
    def validate_province(cls, v):
        allowed_provinces = ['ON', 'QC', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB', 'NL', 'PE', 'NT', 'YT', 'NU']
        if v not in allowed_provinces:
            raise ValueError(f'province must be one of {allowed_provinces}')
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction requests"""
    customer_id: str
    prediction: Dict[str, Any]
    explanation: Dict[str, Any]
    model_version: str
    timestamp: datetime

class FeedbackData(BaseModel):
    """Feedback data model"""
    customer_id: str = Field(..., description="Customer identifier")
    predicted_churn: bool = Field(..., description="Original prediction")
    actual_churn: bool = Field(..., description="Actual churn outcome")
    feedback_type: str = Field(..., description="Type of feedback")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in feedback")
    notes: Optional[str] = Field(None, description="Additional notes")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    model_loaded: bool

# Dependency for request metrics
async def track_request_metrics(request: Request):
    """Track API request metrics"""
    start_time = time.time()
    
    yield
    
    duration = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    # Note: status_code would be available in middleware
    api_request_counter.labels(endpoint=endpoint, method=method, status_code="200").inc()

async def load_models():
    """Load ML model and SHAP explainer"""
    global model, explainer, model_metadata
    
    try:
        # Load model
        model_path = 'models/churn_prediction_model.pkl'
        feature_names_path = 'models/feature_names.pkl'
        preprocessing_info_path = 'models/preprocessing_info.pkl'
        
        model = joblib.load(model_path)
        
        # Load explainer
        explainer = ChurnExplainer(model_path, feature_names_path, preprocessing_info_path)
        
        # Load metadata
        model_metadata = {
            'model_type': type(model).__name__,
            'model_version': model_version,
            'loaded_at': datetime.now().isoformat(),
            'feature_count': len(explainer.feature_names)
        }
        
        logger.info(f"Model loaded: {model_metadata}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": model_version,
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None and explainer is not None else "unhealthy",
        timestamp=datetime.now(),
        version=model_version,
        model_loaded=model is not None and explainer is not None
    )

@app.post("/predict/{customer_id}", response_model=PredictionResponse)
async def predict_churn(
    customer_id: str,
    customer_data: CustomerData,
    background_tasks: BackgroundTasks,
    request_tracker: Any = Depends(track_request_metrics)
):
    """
    Predict customer churn probability with SHAP explanations
    
    This endpoint:
    - Validates input data
    - Makes churn prediction
    - Provides SHAP-based explanations
    - Logs prediction for monitoring
    - Returns comprehensive response
    """
    
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with prediction_histogram.time():
            # Convert Pydantic model to dict
            customer_dict = customer_data.dict()
            customer_dict['customer_id'] = customer_id
            
            # Get prediction with explanation
            explanation = explainer.explain_prediction(customer_dict)
            
            # Create response
            response = PredictionResponse(
                customer_id=customer_id,
                prediction=explanation['prediction'],
                explanation={
                    'top_factors': explanation['top_factors'],
                    'summary': explanation['summary']
                },
                model_version=model_version,
                timestamp=datetime.now()
            )
            
            # Update metrics
            prediction_counter.labels(
                model_version=model_version,
                prediction=str(explanation['prediction']['will_churn'])
            ).inc()
            
            # Log prediction (background task)
            background_tasks.add_task(
                log_prediction,
                customer_id,
                explanation['prediction'],
                explanation['summary']
            )
            
            return response
            
    except Exception as e:
        logger.error(f"Error making prediction for {customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackData,
    background_tasks: BackgroundTasks
):
    """
    Submit feedback for model improvement
    
    This endpoint:
    - Accepts feedback about prediction accuracy
    - Stores feedback for model retraining
    - Updates monitoring metrics
    - Returns confirmation
    """
    
    try:
        # Validate feedback
        if feedback.predicted_churn == feedback.actual_churn:
            feedback_type = "correct_prediction"
        else:
            feedback_type = "incorrect_prediction"
        
        # Update metrics
        feedback_counter.labels(feedback_type=feedback_type).inc()
        
        # Store feedback (background task)
        background_tasks.add_task(store_feedback, feedback)
        
        return {
            "message": "Feedback received successfully",
            "customer_id": feedback.customer_id,
            "feedback_type": feedback_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    metrics_data = generate_latest(REGISTRY)
    if not metrics_data.endswith(b'# EOF\n'):
        metrics_data += b'# EOF\n'
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/model/info")
async def get_model_info():
    """Get model information and metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_metadata": model_metadata,
        "feature_count": len(explainer.feature_names),
        "feature_names": explainer.feature_names[:10],  # First 10 features
        "model_type": type(model).__name__
    }

@app.post("/batch_predict")
async def batch_predict(
    customers: List[CustomerData],
    background_tasks: BackgroundTasks
):
    """
    Batch prediction endpoint for multiple customers
    
    Useful for:
    - Bulk processing
    - Scheduled batch jobs
    - Analysis of customer segments
    """
    
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(customers) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        results = []
        
        for i, customer_data in enumerate(customers):
            customer_dict = customer_data.dict()
            # Generate customer ID for batch processing
            customer_dict['customer_id'] = f"BATCH_CUSTOMER_{i+1:03d}"
            explanation = explainer.explain_prediction(customer_dict)
            
            result = {
                "customer_id": customer_dict['customer_id'],
                "prediction": explanation['prediction'],
                "summary": explanation['summary']
            }
            results.append(result)
            
            # Update metrics
            prediction_counter.labels(
                model_version=model_version,
                prediction=str(explanation['prediction']['will_churn'])
            ).inc()
        
        return {
            "results": results,
            "batch_size": len(customers),
            "model_version": model_version,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Background tasks

async def log_prediction(customer_id: str, prediction: Dict[str, Any], summary: str):
    """Log prediction for monitoring and analysis"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "churn_probability": prediction['churn_probability'],
            "will_churn": prediction['will_churn'],
            "confidence": prediction['confidence'],
            "summary": summary,
            "model_version": model_version
        }
        
        logger.info(f"Prediction logged: {log_entry}")
        
        # In production, you would store this in a database or data lake
        # For now, we'll just log it
        
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

async def store_feedback(feedback: FeedbackData):
    """Store feedback for model improvement"""
    try:
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": feedback.customer_id,
            "predicted_churn": feedback.predicted_churn,
            "actual_churn": feedback.actual_churn,
            "feedback_type": feedback.feedback_type,
            "confidence": feedback.confidence,
            "notes": feedback.notes
        }
        
        logger.info(f"Feedback stored: {feedback_entry}")
        
        # In production, you would store this in a database
        # This data would be used for model retraining
        
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Customer Churn Prediction API",
        version=model_version,
        description="""
        ## Customer Churn Prediction API
        
        Production-ready API for predicting customer churn with explainable AI.
        
        ### Features
        - **Real-time predictions** with SHAP explanations
        - **Batch processing** for multiple customers
        - **Feedback system** for model improvement
        - **Comprehensive monitoring** with Prometheus metrics
        - **Production-ready** with proper error handling and logging
        
        ### Use Cases
        - Customer retention programs
        - Proactive customer engagement
        - Risk assessment and prioritization
        - Business intelligence and analytics
        
        ### Model Information
        - **Model Type**: Logistic Regression
        - **Features**: 37 engineered features
        - **Accuracy**: 77.1%
        - **AUC Score**: 0.745
        """,
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )