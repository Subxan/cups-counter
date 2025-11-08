"""Configuration management with Pydantic models."""

import os
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class CameraConfig(BaseModel):
    """Camera configuration."""

    source: Literal["picamera2", "gstreamer", "opencv"] = "picamera2"
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    rotate: int = Field(default=0, ge=0, le=270)

    @field_validator("rotate")
    @classmethod
    def validate_rotate(cls, v):
        """Ensure rotation is multiple of 90."""
        if v % 90 != 0:
            raise ValueError("rotate must be multiple of 90")
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    hef_path: str = "./models/yolov8n_coco.hef"
    class_filter: List[str] = ["cup"]
    conf_thresh: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_thresh: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=50, ge=1)
    mock_mode: bool = False


class TrackingConfig(BaseModel):
    """Tracking configuration."""

    type: Literal["bytetrack", "ocsort"] = "bytetrack"
    track_thresh: float = Field(default=0.50, ge=0.0, le=1.0)
    match_thresh: float = Field(default=0.80, ge=0.0, le=1.0)
    min_box_area: int = Field(default=150, ge=1)
    lost_ttl: int = Field(default=30, ge=1)


class LineConfig(BaseModel):
    """Line configuration for counting."""

    start: List[int] = Field(default=[100, 360], min_length=2, max_length=2)
    end: List[int] = Field(default=[1180, 360], min_length=2, max_length=2)


class CountingConfig(BaseModel):
    """Counting configuration."""

    line: LineConfig = Field(default_factory=LineConfig)
    direction: Literal["bar_to_counter", "counter_to_bar"] = "bar_to_counter"
    hysteresis_px: int = Field(default=6, ge=0)
    min_visible_frames: int = Field(default=3, ge=1)


class StorageConfig(BaseModel):
    """Storage configuration."""

    sqlite_path: str = "./data/cups.db"
    csv_dir: str = "./data/exports"
    export_daily_time: str = "23:59"  # HH:MM
    wal_mode: bool = True

    @field_validator("export_daily_time")
    @classmethod
    def validate_time(cls, v):
        """Validate time format HH:MM."""
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError("export_daily_time must be HH:MM")
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError("Invalid time")
        return v


class UIConfig(BaseModel):
    """UI configuration."""

    http_port: int = Field(default=8080, ge=1, le=65535)
    preview: bool = True
    draw_overlays: bool = True
    timezone: str = "Asia/Baku"


class OpsConfig(BaseModel):
    """Operations configuration."""

    save_debug_video: bool = False
    debug_dir: str = "./debug"
    metrics_port: int = Field(default=9109, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(BaseModel):
    """Main application configuration."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    counting: CountingConfig = Field(default_factory=CountingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    ops: OpsConfig = Field(default_factory=OpsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_env_override(cls, base_config: "AppConfig") -> "AppConfig":
        """Create config with environment variable overrides."""
        # Simple env override - can be extended
        config_dict = base_config.model_dump()
        # Example: CUPS_MOCK_MODE=true
        if os.getenv("CUPS_MOCK_MODE", "").lower() == "true":
            config_dict["model"]["mock_mode"] = True
        return cls(**config_dict)

