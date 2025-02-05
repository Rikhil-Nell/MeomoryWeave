import streamlit as st
import cv2
import asyncio
from datetime import datetime
from ultralytics import YOLO
from Filter import filter_detections
from agent import Bard

# Set page title and layout
st.set_page_config(page_title="Live YOLO Inference", layout="wide")

# Load or upload YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize session state
if "run" not in st.session_state:
    st.session_state.update({
        "run": False,
        "results": [],
        "frame_count": 0
    })

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_option = st.radio("Choose Model", ["Default (YOLOv8n)", "Custom Model"])
    
    if model_option == "Custom Model":
        model_file = st.file_uploader("Upload YOLO Model", type=["pt"])
        if model_file:
            model = load_model(model_file.name)
        else:
            model = None
    else:
        model = load_model("yolo11n.pt")
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Main app interface
st.title("Real-Time Object Detection with YOLO")
st.caption("Powered by Streamlit, OpenCV, and Ultralytics YOLO")

# Start/stop camera buttons
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Start Camera")
with col2:
    stop_button = st.button("Stop Camera")

if start_button:
    st.session_state["run"] = True
    st.session_state["results"] = []  # Reset previous results
    st.session_state["frame_count"] = 0

if stop_button:
    st.session_state["run"] = False
    if st.session_state["results"]:
        # Convert results to JSON string
        json_data = st.session_state["results"]

        filter_json = filter_detections(json_data)
        
        response = asyncio.run(Bard.run(user_prompt=filter_json))

        st.write(response.data)


# Video display placeholder
video_placeholder = st.empty()

# Camera capture and processing
if st.session_state["run"] and model is not None:
    cap = cv2.VideoCapture(0)
    
    while st.session_state["run"] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Perform YOLO inference
        results = model.predict(
            source=frame,
            conf=confidence_threshold,
            verbose=False
        )
        
        # Process and store results
        frame_data = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": st.session_state["frame_count"],
            "detections": []
        }
        
        # Extract detection information
        for result in results:
            for box in result.boxes:
                detection = {
                    "class_id": int(box.cls),
                    "class_name": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                }
                frame_data["detections"].append(detection)
        
        st.session_state["results"].append(frame_data)
        st.session_state["frame_count"] += 1
        
        # Render results on frame
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
    
    cap.release()
    cv2.destroyAllWindows()
elif model is None and st.session_state["run"]:
    st.error("Please load a model first!")

# Instructions
st.markdown("### Instructions")
st.markdown("""
1. Select model type in the sidebar
2. Adjust confidence threshold as needed
3. Click 'Start Camera' to begin inference
4. Click 'Stop Camera' to end session and download results
""")