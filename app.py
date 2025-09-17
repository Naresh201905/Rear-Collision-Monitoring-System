# Rear Collision Avoidance System Using YOLOv8 and Camera
# Based on the referenced paper methodology

import cv2
import numpy as np
import serial
import time
import os
import threading
import pygame
from ultralytics import YOLO
from datetime import datetime

# Initialize pygame for audio alerts
pygame.mixer.init()

# Load audio alerts
try:
    vehicle_alert = pygame.mixer.Sound("vehicle_alert.mp3")
    person_alert = pygame.mixer.Sound("person_alert.mp3")
    animal_alert = pygame.mixer.Sound("animal_alert.mp3")
    barrier_alert = pygame.mixer.Sound("barrier_alert.mp3")
    general_alert = pygame.mixer.Sound("alert.mp3")  # Fallback alert sound
except:
    print("Warning: Some alert sounds couldn't be loaded.")
    # Create a simple tone as fallback
    pygame.mixer.Sound.play(pygame.mixer.Sound.tone(440, 1000))

# Global variables
last_alert_time = time.time()
ALERT_COOLDOWN = 2.0  # Seconds between alerts to prevent spamming

# Initialize YOLOv8 model
try:
    # Try to load a local custom model first (if you've trained one as per the paper)
    if os.path.exists("best.pt"):
        model = YOLO("best.pt")
        print("Loaded custom YOLOv8 model")
    else:
        # Fall back to standard YOLOv8 model
        model = YOLO("yolov8n.pt")
        print("Loaded standard YOLOv8 model")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

# Define the classes we're interested in (from the paper)
target_classes = {
    2: "car",         # car
    3: "motorcycle",  # motorcycle
    5: "bus",         # bus
    7: "truck",       # truck
    0: "person",      # person
    15: "cat",        # cat (animal)
    16: "dog",        # dog (animal)
    17: "horse",      # horse (animal)
    18: "sheep",      # sheep (animal)
    19: "cow",        # cow (animal)
    11: "stop sign",  # barrier
    13: "bench"       # barrier
}

# Object tracking variables
detected_objects = {}

def connect_to_arduino(port=None):
    """Attempt to connect to Arduino via serial"""
    if port is None:
        # Common serial ports for Arduino
        possible_ports = [
            '/dev/ttyUSB0',
            '/dev/ttyACM0',
            'COM3',
            'COM4',
            'COM6'
        ]
        
        for p in possible_ports:
            try:
                ser = serial.Serial(p, 9600, timeout=1)
                print(f"Connected to Arduino on {p}")
                time.sleep(2)  # Wait for Arduino to reset
                return ser
            except:
                continue
        
        print("Could not connect to Arduino. Running without sensor data.")
        return None
    else:
        try:
            ser = serial.Serial(port, 9600, timeout=1)
            print(f"Connected to Arduino on {port}")
            time.sleep(2)  # Wait for Arduino to reset
            return ser
        except:
            print(f"Could not connect to Arduino on {port}. Running without sensor data.")
            return None

def read_arduino_data(ser):
    """Read and process data from Arduino"""
    while True:
        if ser is None:
            time.sleep(1)
            continue
            
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()
                
                if line.startswith("Distance:"):
                    distance = int(line.split(":")[1].strip())
                    # Process distance data if needed
                
                elif line.startswith("ALERT:"):
                    alert_type = line.split(":")[1]
                    if alert_type == "COLLISION_RISK":
                        # Ultrasonic sensor detected something close
                        print("Ultrasonic Alert: Object too close!")
                        pygame.mixer.Sound.play(general_alert)
        except Exception as e:
            print(f"Error reading from Arduino: {e}")
            time.sleep(0.5)

def estimate_risk(detection, frame_width, frame_height):
    """
    Estimate collision risk based on detection properties
    Returns a risk score between 0-100
    """
    # Extract bounding box coordinates
    x1, y1, x2, y2 = detection[0:4]
    
    # Calculate box dimensions
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Calculate area ratio (how much of the frame is occupied by object)
    frame_area = frame_width * frame_height
    area_ratio = area / frame_area
    
    # Calculate vertical position (lower in frame means closer to vehicle)
    vertical_position = y2 / frame_height
    
    # Calculate risk based on size and position
    # Objects that are larger and lower in the frame pose higher risk
    risk_score = (area_ratio * 50) + (vertical_position * 50)
    
    # Cap at 100
    return min(risk_score * 100, 100)

def alert_driver(risk_score, object_class, ser):
    """Alert driver based on risk level and object class"""
    global last_alert_time
    
    # Respect cooldown to prevent alert spam
    current_time = time.time()
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return
    
    print(f"ALERT: {object_class} detected with risk score: {risk_score:.1f}")
    
    # Play appropriate sound based on object class
    if "car" in object_class or "truck" in object_class or "bus" in object_class or "motorcycle" in object_class:
        pygame.mixer.Sound.play(vehicle_alert)
    elif "person" in object_class:
        pygame.mixer.Sound.play(person_alert)
    elif "cat" in object_class or "dog" in object_class or "horse" in object_class or "sheep" in object_class or "cow" in object_class:
        pygame.mixer.Sound.play(animal_alert)
    elif "stop sign" in object_class or "bench" in object_class:
        pygame.mixer.Sound.play(barrier_alert)
    else:
        pygame.mixer.Sound.play(general_alert)
    
    # Send alert to Arduino to trigger buzzer
    if ser:
        ser.write(b"BUZZ\n")
    
    last_alert_time = current_time

def process_frame(frame, ser):
    """Process a single frame with YOLOv8"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Run YOLOv8 detection
    results = model(frame)
    
    # Process detections
    high_risk_detected = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract box information
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Only consider target classes with confidence > 0.5
            if cls in target_classes and conf > 0.5:
                # Calculate risk score
                risk_score = estimate_risk((x1, y1, x2, y2), width, height)
                
                # Only alert on high risk (> 60)
                if risk_score > 60:
                    object_class = target_classes[cls]
                    alert_driver(risk_score, object_class, ser)
                    high_risk_detected = True
                
                # Draw bounding box
                color = (0, 0, 255) if risk_score > 60 else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Add label
                object_name = target_classes.get(cls, "unknown")
                label = f"{object_name}: {conf:.2f}, Risk: {risk_score:.1f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame, high_risk_detected

def main():
    """Main function to run the system"""
    # Ask for Arduino port
    print("Enter Arduino port (leave blank for auto-detection):")
    port = input().strip()
    port = None if port == "" else port
    
    # Connect to Arduino
    ser = connect_to_arduino(port)
    
    # Start thread to read Arduino data
    if ser is not None:
        arduino_thread = threading.Thread(target=read_arduino_data, args=(ser,), daemon=True)
        arduino_thread.start()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use laptop camera (0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    print("Rear Collision Avoidance System Running")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            # Process frame
            processed_frame, high_risk = process_frame(frame, ser)
            
            # Add system info to frame
            cv2.putText(processed_frame, "Rear Collision Avoidance System", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, timestamp, (10, processed_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow("Rear Collision Avoidance System", processed_frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        if ser:
            ser.close()
        print("System shutdown")

if __name__ == "__main__":
    main()
