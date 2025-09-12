# YOLO Person Counter

A simple person counting application using YOLO object detection and tracking.
This tool counts people crossing a defined line in a video stream with high
accuracy.

## Features

- Real-time person detection and tracking using YOLO11
- Configurable counting line with directional counting
- Support for environment variables and command-line arguments
- Optional window display for debugging
- High-precision tracking with line intersection detection

## Requirements

- [mise](https://mise.jdx.dev/)

```bash
mise install && \
    pip install -r requirements.txt
```

## Usage

### Command line

```bash
python main.py --video-path video.mp4 --line-start 100,200 --line-end 500,200 --enable-window true
```

### Environment variables

```bash
VIDEO_PATH="video.mp4" \
    LINE_START="100,200" \
    LINE_END="500,200" \
    ENABLE_WINDOW="true" \
    WINDOW_SCALE="0.5" \
    REVERSE_DIRECTION="false" \
    IMGSZ="640" \
    DEVICE="cpu" \
    python main.py
```

or

```bash
python main.py \
    --video-path "video.mp4" \
    --line-start "100,200" \
    --line-end "500,200" \
    --enable-window true \
    --window-scale 0.5 \
    --reverse-direction false \
    --imgsz 640
    --device cuda
```

## Configuration

| Parameter           | Description                                                   | Default | Required |
| ------------------- | ------------------------------------------------------------- | ------- | -------- |
| `VIDEO_PATH`        | Path to input video file                                      | -       | Yes      |
| `LINE_START`        | Start coordinates of counting line (x,y)                      | -       | Yes      |
| `LINE_END`          | End coordinates of counting line (x,y)                        | -       | Yes      |
| `WINDOW_SCALE`      | Display scale factor for window                               | 0.5     | No       |
| `REVERSE_DIRECTION` | Reverse counting direction                                    | false   | No       |
| `ENABLE_WINDOW`     | Enable window display                                         | false   | No       |
| `IMGSZ`             | Image size for YOLO inference (0 for original)                | 0       | No       |
| `DEVICE`            | Device to use for inference (cpu, cuda, cuda:0, cuda:1, etc.) | cpu     | No       |

## How it works

1. The application loads a YOLO11 model for person detection
2. Processes video frames and tracks detected persons
3. Monitors when tracked persons cross the defined counting line
4. Counts persons moving in the specified direction only
5. Outputs the total count at the end

The counting line is defined by two points and includes a directional arrow
indicator. Only movement in the positive direction (configurable) is counted to
avoid double-counting.

## License

[MIT](./LICENSE)
