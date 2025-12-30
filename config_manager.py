"""
Model Configuration Manager
Supports reading model configurations from config files, using aliases to map to actual models
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

class ModelConfigManager:
    """Model configuration manager"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to config file, if None then use default path
        """
        if config_path is None:
            # Default config file path
            default_config = Path(__file__).parent / "model_config.json"
            config_path = str(default_config)
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration file"""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            # If config file doesn't exist, create default config
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            return self._get_default_config()
    
    def _save_config(self, config: Dict):
        """Save configuration file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "models": {
                "qwen3-vl-235b": {
                    "type": "ksyun",
                    "model": "qwen3-vl-235b-a22b-thinking",
                    "api_key": "a6088cfd-b3c4-4793-ae12-4396240cbbed",
                    "base_url": "https://kspmas.ksyun.com/v1/",
                    "temperature": 0.0,
                    "description": "Kingsoft Cloud Qwen3 VL 235B Model"
                }
            },
            "default_model": "qwen3-vl-235b",
            "default_judge_model": "qwen3-vl-235b"
        }
    
    def get_model_config(self, model_alias: str) -> Optional[Dict]:
        """
        Get configuration by model alias
        
        Args:
            model_alias: Model alias, e.g., "gpt-5"
            
        Returns:
            Model configuration dictionary, returns None if not found
        """
        models = self.config.get("models", {})
        return models.get(model_alias)
    
    def get_default_model(self) -> str:
        """Get default model alias"""
        return self.config.get("default_model", "qwen3-vl-235b")
    
    def get_default_judge_model(self) -> str:
        """Get default judge model alias"""
        return self.config.get("default_judge_model", "qwen3-vl-235b")
    
    def list_models(self) -> Dict[str, Dict]:
        """List all available model configurations"""
        return self.config.get("models", {})
    
    def add_model(self, alias: str, config: Dict):
        """
        Add new model configuration
        
        Args:
            alias: Model alias
            config: Model configuration dictionary
        """
        if "models" not in self.config:
            self.config["models"] = {}
        
        self.config["models"][alias] = config
        self._save_config(self.config)
    
    def update_model(self, alias: str, **kwargs):
        """
        Update model configuration
        
        Args:
            alias: Model alias
            **kwargs: Configuration items to update
        """
        if alias not in self.config.get("models", {}):
            raise ValueError(f"Model alias '{alias}' not found")
        
        self.config["models"][alias].update(kwargs)
        self._save_config(self.config)
    
    def get_model_params(self, model_alias: str, override_api_key: str = None) -> Dict:
        """
        Get model parameters for creating model instance
        
        Args:
            model_alias: Model alias
            override_api_key: Optional API key override
            
        Returns:
            Model parameters dictionary
        """
        model_config = self.get_model_config(model_alias)
        if model_config is None:
            raise ValueError(f"Model alias '{model_alias}' not found in config")
        
        # Copy configuration
        params = model_config.copy()
        
        # If override_api_key is provided, use it
        if override_api_key:
            params["api_key"] = override_api_key
        elif not params.get("api_key"):
            # If no api_key in config, try to get from environment variables
            env_key_map = {
                "gpt": "OPENAI_API_KEY",
                "qwen": "DASHSCOPE_API_KEY",
                "ksyun": "KSYUN_API_KEY"
            }
            model_type = params.get("type", "").lower()
            env_key = env_key_map.get(model_type)
            if env_key:
                params["api_key"] = os.getenv(env_key, "")
        
        # Remove unnecessary fields
        params.pop("description", None)
        params.pop("type", None)  # type needs to be handled separately
        
        return params
    
    def get_model_type(self, model_alias: str) -> str:
        """
        Get model type
        
        Args:
            model_alias: Model alias
            
        Returns:
            Model type, e.g., "gpt", "qwen", "ksyun"
        """
        model_config = self.get_model_config(model_alias)
        if model_config is None:
            raise ValueError(f"Model alias '{model_alias}' not found in config")
        
        return model_config.get("type", "gpt")

