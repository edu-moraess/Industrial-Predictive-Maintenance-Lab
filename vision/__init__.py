"""Computer Vision inspection lab (independent from sensor ML pipeline).

VisionInspectionService (vision/inspection.py) is a COCO-detector +
heuristic reference-comparison pipeline. It is NOT wired into the
Streamlit app or the API — the live inspection page (app/vision_page.py)
talks to vision.industrial_model.IndustrialAnomalyModel (trained PCA
residual detector) directly instead. Unlike the industrial model,
VisionInspectionService's own visual_status *does* depend on whether the
optional COCO detector is online — keep that in mind before wiring it
into anything that reports industrial anomaly/visual-health results.
"""

from vision.inspection import VisionInspectionService

__all__ = ["VisionInspectionService"]
