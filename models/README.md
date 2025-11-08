# Models Directory

Place your Hailo-compiled YOLO model (`.hef` file) in this directory.

## Model Requirements

- Format: `.hef` (Hailo Executable Format)
- Input: RGB image, typically 640x640 (model-dependent)
- Output: YOLO-style detections

## Default Model

The default config expects: `yolov8n_coco.hef`

To use a different model, update `model.hef_path` in your config file.

## Label Mapping

The `labelmap.json` file maps class names to class IDs. The default mapping assumes COCO format where "cup" is class ID 41.

To customize:
1. Edit `labelmap.json`
2. Update `model.class_filter` in your config to match desired classes

## Getting a Model

1. Train or download a YOLO model
2. Compile to `.hef` using Hailo's tools
3. Place the `.hef` file in this directory
4. Update config with correct path

## Mock Mode

If you don't have a `.hef` file yet, you can run in mock mode:
- Set `model.mock_mode: true` in config, or
- Run with `--mock` flag

Mock mode generates synthetic detections for testing the pipeline.

