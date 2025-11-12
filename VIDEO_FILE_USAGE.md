# Using Video Files as Input Source

The cups-counter application now supports using video files as input instead of live cameras.

## Quick Start

### Method 1: Edit Config File

Edit `configs/site-default.yaml`:

```yaml
camera:
  source: "opencv"
  index: "./path/to/your/video.mp4"  # Change from 0 (camera) to file path
  width: 1280
  height: 720
  fps: 30
  rotate: 0
```

Then run:
```bash
python apps/edge_service.py
```

### Method 2: Use Example Config

1. Copy the example config:
   ```bash
   cp configs/video-file.yaml configs/my-video.yaml
   ```

2. Edit `configs/my-video.yaml` and set the video file path:
   ```yaml
   camera:
     index: "./data/my_video.mp4"  # Your video file path
   ```

3. Run with the config:
   ```bash
   python apps/edge_service.py --config configs/my-video.yaml
   ```

## Supported Video Formats

OpenCV supports common video formats:
- `.mp4` (H.264, H.265)
- `.avi`
- `.mov`
- `.mkv`
- Most formats supported by OpenCV/FFmpeg

## How It Works

1. **Automatic Properties**: The system automatically reads:
   - Video resolution (width × height)
   - Frame rate (FPS)
   - Total frame count

2. **Video Looping**: When the video reaches the end, it automatically loops back to the beginning

3. **Frame Timing**: Frames are played back at the video's original FPS (not the configured FPS)

## Example Use Cases

### Testing Without Hardware
```bash
# Record a test video first, then:
python apps/edge_service.py --config configs/video-file.yaml
```

### Processing Recorded Footage
```bash
# Process a recorded video for counting
python apps/edge_service.py --config configs/video-file.yaml
# Access web UI to see counts in real-time
```

### Regression Testing
```bash
# Use the same test video to verify counting accuracy
python apps/edge_service.py --config configs/video-file.yaml
```

## Configuration Tips

**For Video Files:**
- Set `autocal.enabled: false` (no need for auto-calibration on video)
- Set `drift.enabled: false` (drift monitoring not needed for video)
- Set `tuner.enabled: false` (tuning runs on live feeds)

**File Paths:**
- Use relative paths: `"./data/video.mp4"`
- Use absolute paths: `"/home/user/videos/test.mp4"`
- Paths are resolved relative to the working directory

## Troubleshooting

**Video won't open:**
- Check file path is correct
- Verify file exists: `ls -lh /path/to/video.mp4`
- Check file format is supported by OpenCV
- Try absolute path instead of relative

**Video plays too fast/slow:**
- The system uses the video's native FPS
- Check video FPS: `ffprobe video.mp4` (if ffprobe installed)
- The configured `fps` in config is only a fallback

**Video loops unexpectedly:**
- This is expected behavior - video loops when it ends
- To process once and exit, you'd need to modify the code or use `backfill_replay.py` instead

## Comparison: Video File vs backfill_replay.py

| Feature | Video File Source | backfill_replay.py |
|---------|------------------|-------------------|
| Live counting | ✅ Yes | ❌ No |
| Web UI | ✅ Yes | ❌ No |
| Loops video | ✅ Yes | ❌ No (processes once) |
| CSV export | ✅ Daily | ✅ Immediate |
| Real-time metrics | ✅ Yes | ❌ No |

Use **video file source** for:
- Testing the full system with recorded footage
- Demonstrations
- Continuous processing of a test video

Use **backfill_replay.py** for:
- One-time processing of historical footage
- Batch processing multiple videos
- Generating CSV exports from recordings

