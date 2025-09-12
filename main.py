import cv2
import numpy as np
import os
import argparse
from ultralytics import YOLO
from collections import defaultdict

def parse_coordinates(coord_str):
    """Parse coordinate string 'x,y' to tuple (x, y)"""
    try:
        x, y = map(int, coord_str.split(','))
        return (x, y)
    except:
        raise ValueError(f"Invalid coordinate format: {coord_str}. Expected format: 'x,y'")

def parse_arguments():
    """Parse command line arguments and environment variables"""
    parser = argparse.ArgumentParser(description='Person counting with YOLO')

    # Add arguments with defaults from environment variables
    parser.add_argument('--video-path', 
                       type=str,
                       default=os.getenv('VIDEO_PATH'),
                       help='Path to input video file (required)')

    parser.add_argument('--window-scale', 
                       type=float,
                       default=float(os.getenv('WINDOW_SCALE', '0.5')),
                       help='Display scale factor for window (default: 0.5)')

    parser.add_argument('--line-start', 
                       type=str,
                       default=os.getenv('LINE_START'),
                       help='Start coordinates of counting line (x,y) (required)')

    parser.add_argument('--line-end',
                       type=str,
                       default=os.getenv('LINE_END'), 
                       help='End coordinates of counting line (x,y) (required)')

    parser.add_argument('--reverse-direction',
                       type=bool,
                       default=os.getenv('REVERSE_DIRECTION', 'False').lower() in ('true', '1', 'yes'),
                       help='Reverse counting direction (default: False)')

    parser.add_argument('--enable-window',
                       type=bool,
                       default=os.getenv('ENABLE_WINDOW', 'False').lower() in ('true', '1', 'yes'),
                       help='Enable window display (default: False)')

    parser.add_argument('--imgsz',
                       type=int,
                       default=int(os.getenv('IMGSZ', '0')),
                       help='Image size for YOLO inference (0 for original video resolution) (default: 0)')

    parser.add_argument('--device',
                       type=str,
                       default=os.getenv('DEVICE', 'cpu'),
                       help='Device to use for inference (cpu, cuda, cuda:0, cuda:1, etc.) (default: cpu)')

    args = parser.parse_args()

    # Check required arguments
    missing_args = []
    if not args.video_path:
        missing_args.append('video-path (--video-path or args.video_path env var)')
    if not args.line_start:
        missing_args.append('line-start (--line-start or args.line_start env var)')
    if not args.line_end:
        missing_args.append('line-end (--line-end or args.line_end env var)')
    
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(missing_args)}")

    args.line_start = parse_coordinates(args.line_start)
    args.line_end = parse_coordinates(args.line_end)

    return args

# Parse arguments and environment variables
args = parse_arguments()

# Load YOLO model
model = YOLO('yolo11l.pt')

# Open video
cap = cv2.VideoCapture(args.video_path)

# Get original video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original video resolution: {width}x{height}")

# Determine inference size
if args.imgsz == 0:
    # Use original video resolution for inference
    inference_size = max(width, height)
    print(f"Using original resolution for inference: {inference_size}")
else:
    inference_size = args.imgsz
    print(f"Using custom inference size: {inference_size}")

# Tracking variables
track_history = defaultdict(lambda: [])
bbox_history = defaultdict(lambda: [])
crossed_ids = set()
person_count = 0

def line_intersect_bbox(bbox1, bbox2, line_start, line_end):
    """Check if bounding box movement crosses the line"""
    def is_counter_clockwise(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def line_intersects_segment(p1, p2, line_start, line_end):
        return is_counter_clockwise(line_start, p1, p2) != is_counter_clockwise(line_end, p1, p2) and \
               is_counter_clockwise(line_start, line_end, p1) != is_counter_clockwise(line_start, line_end, p2)
    
    # Extract bbox coordinates
    x1_prev, y1_prev, w_prev, h_prev = bbox1
    x1_curr, y1_curr, w_curr, h_curr = bbox2
    
    # Convert center coordinates to corner coordinates
    left_prev = int(x1_prev - w_prev/2)
    right_prev = int(x1_prev + w_prev/2)
    top_prev = int(y1_prev - h_prev/2)
    bottom_prev = int(y1_prev + h_prev/2)
    
    left_curr = int(x1_curr - w_curr/2)
    right_curr = int(x1_curr + w_curr/2)
    top_curr = int(y1_curr - h_curr/2)
    bottom_curr = int(y1_curr + h_curr/2)
    
    # Check if any edge of the bounding box crosses the line
    bbox_edges = [
        # Top edge
        ((left_prev, top_prev), (left_curr, top_curr)),
        ((right_prev, top_prev), (right_curr, top_curr)),
        # Bottom edge
        ((left_prev, bottom_prev), (left_curr, bottom_curr)),
        ((right_prev, bottom_prev), (right_curr, bottom_curr)),
        # Left edge
        ((left_prev, top_prev), (left_curr, top_curr)),
        ((left_prev, bottom_prev), (left_curr, bottom_curr)),
        # Right edge
        ((right_prev, top_prev), (right_curr, top_curr)),
        ((right_prev, bottom_prev), (right_curr, bottom_curr)),
        # Center point (for fallback)
        ((int(x1_prev), int(y1_prev)), (int(x1_curr), int(y1_curr)))
    ]
    
    for (p1_prev, p1_curr) in bbox_edges:
        if line_intersects_segment(p1_prev, p1_curr, line_start, line_end):
            return True
    
    return False

def get_direction(p1, p2, line_start, line_end):
    """Determine passage direction (1 or -1)"""
    # If line is vertical
    if line_start[0] == line_end[0]:
        return 1 if p2[0] > p1[0] else -1
    # If line is horizontal
    else:
        return 1 if p2[1] > p1[1] else -1

def draw_arrow(frame, line_start, line_end):
    """Draw an arrow perpendicular to the line to indicate counting direction"""
    # Calculate line midpoint
    mid_x = (line_start[0] + line_end[0]) // 2
    mid_y = (line_start[1] + line_end[1]) // 2
    
    # Calculate line vector
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    
    # Calculate perpendicular vector (90° rotation)
    # For positive direction, take the right normal
    perp_dx = -dy
    perp_dy = dx
    
    # Reverse direction if requested
    if args.reverse_direction:
        perp_dx = -perp_dx
        perp_dy = -perp_dy
    
    # Normalize perpendicular vector
    length = np.sqrt(perp_dx**2 + perp_dy**2)
    if length > 0:
        perp_dx = perp_dx / length * 50  # Arrow length
        perp_dy = perp_dy / length * 50
    
    # Arrow points
    arrow_start = (mid_x, mid_y)
    arrow_end = (int(mid_x + perp_dx), int(mid_y + perp_dy))
    
    if args.enable_window:
        # Draw arrow
        cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Detection and tracking
    results = model.track(frame, persist=True, imgsz=inference_size, device=args.device)
    
    if args.enable_window:
        # Draw counting line
        cv2.line(frame, args.line_start, args.line_end, (0, 255, 0), 3)
        
        # Draw arrow to indicate counting direction
        draw_arrow(frame, args.line_start, args.line_end)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            # Filter only persons (class 0 in COCO)
            if cls == 0:
                x, y, w, h = box
                center = (int(x), int(y))
                
                # Add position to history
                track_history[track_id].append(center)
                bbox_history[track_id].append((x, y, w, h))
                
                # Keep only last 10 positions
                if len(track_history[track_id]) > 10:
                    track_history[track_id].pop(0)
                if len(bbox_history[track_id]) > 10:
                    bbox_history[track_id].pop(0)
                
                # Check line crossing using bounding boxes
                if len(bbox_history[track_id]) >= 2 and track_id not in crossed_ids:
                    bbox1 = bbox_history[track_id][-2]
                    bbox2 = bbox_history[track_id][-1]
                    
                    if line_intersect_bbox(bbox1, bbox2, args.line_start, args.line_end):
                        p1 = track_history[track_id][-2]
                        p2 = track_history[track_id][-1]
                        direction = get_direction(p1, p2, args.line_start, args.line_end)
                        # Reverse direction if requested
                        if args.reverse_direction:
                            direction = -direction
                        # Count only in one direction (direction == 1)
                        if direction == 1:
                            person_count += 1
                            crossed_ids.add(track_id)
                            print(f"Person counted! Total: {person_count}")
                
                if args.enable_window:
                    # Draw bounding box
                    left = int(x - w/2)
                    right = int(x + w/2)
                    top = int(y - h/2)
                    bottom = int(y + h/2)
                    
                    # Choose color based on whether person has been counted
                    color = (0, 255, 255) if track_id in crossed_ids else (255, 0, 0)  # Yellow if counted, Blue if not
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw tracking center point
                    cv2.circle(frame, center, 5, color, -1)
                    
                    # Draw track ID and status
                    label = f"ID: {track_id}"
                    if track_id in crossed_ids:
                        label += " (COUNTED)"
                    
                    cv2.putText(frame, label, 
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if args.enable_window:
        # Display counter
        cv2.putText(frame, f"Persons: {person_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize frame for display only (processing is done at full resolution)
        if args.window_scale != 1.0:
            display_frame = cv2.resize(frame, None, fx=args.window_scale, fy=args.window_scale)
        else:
            display_frame = frame
        
        cv2.imshow("Person Counting", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Total number of persons who crossed the line: {person_count}")