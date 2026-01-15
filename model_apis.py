"""
API interfaces for different models
Supports different models such as GPT and Qwen
"""

import os
import base64
import json
import time
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
import requests
from openai import OpenAI
from openai import RateLimitError, APIError

# Optional imports for Qwen
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

try:
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None

# ==================== Base Model Interface ====================
class BaseModelAPI(ABC):
    """Base model API class"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Args:
            api_key: API key
            base_url: API base URL (optional)
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url
    
    @abstractmethod
    def predict(self, 
               images: List[str], 
               prompt: str,
               **kwargs) -> str:
        """
        Model prediction interface
        
        Args:
            images: List of image paths
            prompt: Prompt text
            **kwargs: Other parameters
            
        Returns:
            Model output answer text
        """
        pass
    
    def _load_image_base64(self, image_path: str) -> str:
        """Convert image to base64 encoding"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _prepare_images(self, images: List[str]) -> List[Dict]:
        """Prepare image data (can be overridden by subclasses)"""
        return [{"path": img} for img in images]

# ==================== GPT API Interface ====================
class GPTModelAPI(BaseModelAPI):
    """OpenAI GPT model API interface"""
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gpt-4o",
                 base_url: str = None,
                 max_tokens: int = None,
                 temperature: float = 0.0):
        """
        Args:
            api_key: OpenAI API key
            model: Model name, e.g., "gpt-4o", "gpt-4-vision-preview"
            base_url: Custom API base URL (optional, for proxy)
            max_tokens: Maximum number of tokens
            temperature: Temperature parameter
        """
        super().__init__(api_key, base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def predict(self, 
               images: List[str], 
               prompt: str,
               system_message: str = None,
               **kwargs) -> str:
        """
        Use GPT model for prediction
        
        Args:
            images: List of image paths
            prompt: Prompt text
            system_message: System message (optional)
            **kwargs: Other parameters (e.g., max_tokens, temperature, etc.)
        """
        # Prepare message list
        messages = []
        
        # Add system message (if provided)
        if system_message:
            messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Prepare user message content
        content = [{"type": "text", "text": prompt}]
        
        # Add images
        for img_path in images:
            if not Path(img_path).exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Read image and convert to base64
            image_base64 = self._load_image_base64(img_path)
            
            # Determine mime_type based on image format
            img_ext = Path(img_path).suffix.lower()
            mime_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(img_ext, 'image/png')
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Call API with retry logic
        max_retries = 3
        retry_delay = 1  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                api_params = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': kwargs.get('temperature', self.temperature)
                }
                # Only add max_tokens if explicitly provided
                max_tokens_value = kwargs.get('max_tokens', self.max_tokens)
                if max_tokens_value is not None:
                    api_params['max_tokens'] = max_tokens_value
                
                response = self.client.chat.completions.create(**api_params)
                
                return response.choices[0].message.content.strip()
            
            except RateLimitError as e:
                # Rate limit error - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[GPT API] Rate limit exceeded (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[GPT API ERROR] Rate limit exceeded after {max_retries} attempts")
                    return f"Error: Rate limit exceeded - {str(e)}"
            
            except (APIError, Exception) as e:
                # Other API errors or general exceptions - retry with exponential backoff
                error_type = type(e).__name__
                is_retryable = (
                    error_type in ['APIConnectionError', 'APITimeoutError', 'InternalServerError'] or
                    (hasattr(e, 'status_code') and e.status_code in [429, 500, 502, 503, 504])
                )
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[GPT API] {error_type} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[GPT API ERROR] {error_type}: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"[GPT API ERROR] Failed after {max_retries} attempts")
                    return f"Error: {str(e)}"
        
        # Should not reach here, but just in case
        return "Error: Max retries exceeded"

# ==================== Ksyun API Interface ====================
class KsyunModelAPI(BaseModelAPI):
    """Ksyun model API interface (compatible with OpenAI format)"""
    
    def __init__(self,
                 api_key: str = None,
                 model: str = "qwen3-vl-235b-a22b-thinking",
                 base_url: str = "https://kspmas.ksyun.com/v1/",
                 max_tokens: int = None,
                 temperature: float = 0.0):
        """
        Args:
            api_key: Ksyun API key
            model: Model name for API calls
            base_url: API base URL, default Ksyun address
            max_tokens: Maximum number of tokens
            temperature: Temperature parameter
        """
        super().__init__(api_key, base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Check if API key is provided
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "Ksyun API key is required. Please set it in model_config.json or "
                "provide it via --test_api_key argument or KSYUN_API_KEY environment variable."
            )
        
        # Initialize OpenAI client (using Ksyun base_url)
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Ksyun API client: {e}. "
                f"Please check your API key and base_url configuration."
            ) from e
    
    def predict(self,
               images: List[str],
               prompt: str,
               system_message: str = None,
               **kwargs) -> str:
        """
        Use Ksyun model for prediction
        
        Args:
            images: List of image paths
            prompt: Prompt text (may be complete system prompt or user message)
            system_message: System message (optional, if explicitly provided)
            **kwargs: Other parameters
        """
        # Prepare message list
        messages = []
        
        # Determine if prompt is a complete system prompt
        # Features: contains keywords like "Vision-Language Model", "Strict rules", "Options:"
        is_system_prompt = (
            "Vision-Language Model" in prompt or
            "Strict rules" in prompt or
            ("Task:" in prompt and "Options:" in prompt) or
            prompt.strip().startswith("You are")
        )
        
        # Prepare user message content
        user_content = []
        
        # Add images first (following the example format)
        for img_path in images:
            if not Path(img_path).exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Read image and convert to base64
            image_base64 = self._load_image_base64(img_path)
            
            # Determine mime_type based on image format
            img_ext = Path(img_path).suffix.lower()
            mime_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            mime_type = mime_type_map.get(img_ext, 'image/png')
            
            # Add image using base64 data URL format
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            })
        
        # Determine if prompt is a complete system prompt
        is_system_prompt = (
            "Vision-Language Model" in prompt or
            "Strict rules" in prompt or
            ("Task:" in prompt and "Options:" in prompt) or
            prompt.strip().startswith("You are")
        )
        
        if is_system_prompt:
            # prompt itself is a complete system prompt
            messages.append({
                "role": "system",
                "content": prompt
            })
            # Add text after images in user message
            user_content.append({"type": "text", "text": "Please analyze the images and provide your answer."})
        else:
            # prompt is a regular user message
            # If system_message exists, use it
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            # Add text after images
            user_content.append({"type": "text", "text": prompt})
        
        # Add user message with images and text
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Call API with retry logic
        max_retries = 3
        retry_delay = 1  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Prepare API call parameters (no max_tokens limit)
                api_params = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': kwargs.get('temperature', self.temperature)
                }
                # Only add max_tokens if explicitly provided in kwargs
                if 'max_tokens' in kwargs and kwargs['max_tokens'] is not None:
                    api_params['max_tokens'] = kwargs['max_tokens']
                
                response = self.client.chat.completions.create(**api_params)
                
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    message = choice.message
                    content = message.content
                    
                    if content:
                        return content.strip()
                    else:
                        # Try to get from dict form
                        if isinstance(message, dict):
                            content = message.get('content', '')
                        elif hasattr(message, '__dict__'):
                            content = message.__dict__.get('content', '')
                        return content.strip() if content else ""
                else:
                    print(f"[Ksyun API ERROR] No choices in response!")
                    return ""
            
            except RateLimitError as e:
                # Rate limit error - retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"[Ksyun API] Rate limit exceeded (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[Ksyun API ERROR] Rate limit exceeded after {max_retries} attempts")
                    return f"Error: Rate limit exceeded - {str(e)}"
            
            except (APIError, Exception) as e:
                # Other API errors or general exceptions - retry with exponential backoff
                error_type = type(e).__name__
                is_retryable = (
                    error_type in ['APIConnectionError', 'APITimeoutError', 'InternalServerError'] or
                    (hasattr(e, 'status_code') and e.status_code in [429, 500, 502, 503, 504])
                )
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[Ksyun API] {error_type} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    print(f"[Ksyun API ERROR] {error_type}: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"[Ksyun API ERROR] Failed after {max_retries} attempts")
                    return f"Error: {str(e)}"
        
        # Should not reach here, but just in case
        return "Error: Max retries exceeded"

# ==================== Qwen API Interface ====================
class QwenModelAPI(BaseModelAPI):
    """Alibaba Cloud Qwen model API interface"""
    
    def __init__(self,
                 api_key: str = None,
                 model: str = "qwen-vl-max",
                 max_tokens: int = None,
                 temperature: float = 0.0):
        """
        Args:
            api_key: DashScope API key
            model: Model name, e.g., "qwen-vl-max", "qwen-vl-plus"
            max_tokens: Maximum number of tokens
            temperature: Temperature parameter
        """
        super().__init__(api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set API key
        if self.api_key:
            os.environ['DASHSCOPE_API_KEY'] = self.api_key
    
    def predict(self,
               images: List[str],
               prompt: str,
               **kwargs) -> str:
        """
        Use Qwen model for prediction
        
        Args:
            images: List of image paths
            prompt: Prompt text
            **kwargs: Other parameters
        """
        # Prepare message content
        messages = []
        
        # Prepare content list
        content_list = []
        
        # Add images
        for img_path in images:
            if not Path(img_path).exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Qwen needs to add image path to content
            content_list.append({
                "image": img_path
            })
        
        # Add text prompt (after all images)
        if prompt:
            content_list.append({
                "text": prompt
            })
        
        # If only text without images
        if not content_list and prompt:
            content_list.append({
                "text": prompt
            })
        
        # Build message
        if content_list:
            messages.append({
                "role": "user",
                "content": content_list
            })
        
        # Call API with retry logic
        max_retries = 3
        retry_delay = 1  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                api_params = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': kwargs.get('temperature', self.temperature)
                }
                # Only add max_tokens if explicitly provided
                max_tokens_value = kwargs.get('max_tokens', self.max_tokens)
                if max_tokens_value is not None:
                    api_params['max_tokens'] = max_tokens_value
                
                response = MultiModalConversation.call(**api_params)
                
                if response.status_code == 200:
                    # Extract text content
                    if hasattr(response, 'output') and hasattr(response.output, 'choices'):
                        content = response.output.choices[0].message.content
                        # content may be list or string
                        if isinstance(content, list):
                            # Find text type content
                            for item in content:
                                if isinstance(item, dict) and item.get('type') == 'text':
                                    return item.get('text', '').strip()
                            # If text not found, try to get first element directly
                            if content:
                                return str(content[0]).strip()
                        elif isinstance(content, str):
                            return content.strip()
                    return "Error: Unable to parse response"
                else:
                    # Check if error is retryable
                    error_msg = getattr(response, 'message', 'Unknown error')
                    status_code = getattr(response, 'status_code', None)
                    is_retryable = status_code in [429, 500, 502, 503, 504]
                    
                    if is_retryable and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"[Qwen API] Error {status_code} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[Qwen API ERROR] {error_msg}")
                        return f"Error: {error_msg}"
            
            except Exception as e:
                error_type = type(e).__name__
                is_retryable = (
                    error_type in ['ConnectionError', 'TimeoutError', 'HTTPError'] or
                    (hasattr(e, 'status_code') and e.status_code in [429, 500, 502, 503, 504])
                )
                
                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[Qwen API] {error_type} (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[Qwen API ERROR] {error_type}: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"[Qwen API ERROR] Failed after {max_retries} attempts")
                    return f"Error: {str(e)}"
        
        # Should not reach here, but just in case
        return "Error: Max retries exceeded"

# ==================== Model Factory ====================
class ModelFactory:
    """Model factory class for creating different types of model instances"""
    
    @staticmethod
    def create_model(model_type: str = None, 
                    model_alias: str = None,
                    config_manager = None,
                    **kwargs) -> BaseModelAPI:
        """
        Create model instance
        
        Args:
            model_type: Model type, e.g., "gpt", "qwen", "ksyun", "mog" (ignored if model_alias is provided)
            model_alias: Model alias, e.g., "qwen3-vl-235b", will read config from config file
            config_manager: Config manager instance (optional, will create default instance if not provided)
            **kwargs: Model initialization parameters (will override config file parameters)
            
        Returns:
            Model API instance
        """
        # If model_alias is provided, read from config
        actual_model_type = None
        if model_alias:
            try:
                try:
                    from .config_manager import ModelConfigManager
                except ImportError:
                    from config_manager import ModelConfigManager
                
                if config_manager is None:
                    config_manager = ModelConfigManager()
                
                # Get model type and parameters
                actual_model_type = config_manager.get_model_type(model_alias)
                model_params = config_manager.get_model_params(model_alias)
                
                # Merge kwargs (kwargs has higher priority)
                model_params.update(kwargs)
                
                # Create using actual model type
                return ModelFactory.create_model(
                    model_type=actual_model_type,
                    **model_params
                )
            except Exception as e:
                print(f"Warning: Failed to load model config for '{model_alias}': {e}")
                if actual_model_type:
                    # If we got the model type before failure, use it for fallback
                    print(f"Falling back to model_type '{actual_model_type}' with provided kwargs")
                    model_type = actual_model_type
                else:
                    print("Falling back to direct model_type specification")
        
        # If model_alias not provided or loading failed, use model_type
        if model_type is None:
            raise ValueError("Either model_type or model_alias must be provided")
        
        model_type = model_type.lower()
        
        if model_type in ["gpt", "gpt-4", "gpt-4o", "gpt-4-vision"]:
            return GPTModelAPI(**kwargs)
        elif model_type in ["qwen", "qwen-vl", "qwen-vl-max", "qwen-vl-plus"]:
            return QwenModelAPI(**kwargs)
        elif model_type in ["ksyun", "mog", "mog-1", "qwen3-vl-235b"]:
            # Set default values
            if "base_url" not in kwargs:
                kwargs["base_url"] = "https://kspmas.ksyun.com/v1/"
            if "model" not in kwargs:
                kwargs["model"] = "qwen3-vl-235b-a22b-thinking"
            return KsyunModelAPI(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def create_from_config(config: Dict) -> BaseModelAPI:
        """
        Create model from config dictionary
        
        Args:
            config: Config dictionary containing model_type or model_alias and other parameters
            
        Returns:
            Model API instance
        """
        # Support model_alias
        if 'model_alias' in config:
            model_alias = config.pop('model_alias')
            return ModelFactory.create_model(model_alias=model_alias, **config)
        
        model_type = config.pop('model_type', 'gpt')
        return ModelFactory.create_model(model_type=model_type, **config)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例1: 使用GPT模型
    print("Testing GPT Model...")
    gpt_model = GPTModelAPI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        max_tokens=500
    )
    
    # 示例2: 使用Qwen模型
    print("Testing Qwen Model...")
    qwen_model = QwenModelAPI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-vl-max",
        max_tokens=500
    )
    
    # 示例3: 使用工厂创建
    print("Testing Model Factory...")
    model = ModelFactory.create_model(
        model_type="gpt",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o"
    )

