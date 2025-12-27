"""
API interfaces for different models
Supports different models such as GPT and Qwen
"""

import os
import base64
import json
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
import requests
from openai import OpenAI

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
                 max_tokens: int = 500,
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
        
        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            return f"Error: {str(e)}"

# ==================== Ksyun API Interface ====================
class KsyunModelAPI(BaseModelAPI):
    """Ksyun model API interface (compatible with OpenAI format)"""
    
    def __init__(self,
                 api_key: str = None,
                 model: str = "qwen3-vl-235b-a22b-thinking",
                 base_url: str = "https://kspmas.ksyun.com/v1/",
                 max_tokens: int = 500,
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
        
        # Initialize OpenAI client (using Ksyun base_url)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
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
        
        # Call API
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error calling Ksyun API: {e}")
            print(f"  Model: {self.model}")
            print(f"  Base URL: {self.base_url}")
            print(f"  Messages structure: {len(messages)} message(s)")
            if messages:
                print(f"  First message role: {messages[0].get('role', 'N/A')}")
                if 'content' in messages[0]:
                    content = messages[0]['content']
                    if isinstance(content, list):
                        print(f"  First message content: {len(content)} items")
                        for i, item in enumerate(content[:3]):
                            if isinstance(item, dict):
                                if 'type' in item:
                                    print(f"    Item {i}: type={item['type']}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

# ==================== Qwen API Interface ====================
class QwenModelAPI(BaseModelAPI):
    """Alibaba Cloud Qwen model API interface"""
    
    def __init__(self,
                 api_key: str = None,
                 model: str = "qwen-vl-max",
                 max_tokens: int = 500,
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
        
        # Call API
        try:
            response = MultiModalConversation.call(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            
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
                error_msg = getattr(response, 'message', 'Unknown error')
                print(f"Qwen API Error: {error_msg}")
                return f"Error: {error_msg}"
        
        except Exception as e:
            print(f"Error calling Qwen API: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

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

