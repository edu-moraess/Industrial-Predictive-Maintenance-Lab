"""Vision module configuration defaults."""

SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DEFAULT_CONFIDENCE = 0.35
DEFAULT_FRAME_STRIDE = 5
MAX_IMAGE_SIDE = 1280
MAX_VIDEO_FRAMES_ANALYZED = 60
YOLO_MODEL_NAME = "yolov8n.pt"  # lightweight COCO weights; downloads on first use
