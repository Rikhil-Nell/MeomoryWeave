import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from datetime import datetime
from ultralytics import YOLO
from Filter import filter_detections
from agent import agent
import asyncio

# Set page title and layout
st.set_page_config(page_title="Live YOLO Inference", layout="wide")

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
st.caption("Powered by Streamlit, WebRTC, and Ultralytics YOLO")

# Video processing callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format
    
    if model is not None:
        # Perform YOLO inference
        results = model.predict(
            source=img,
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
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
    
    return frame

# WebRTC streamer
ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Stop button to process results
if st.button("Stop and Analyze"):
    if st.session_state["results"]:
        # Convert results to JSON string
        json_data = st.session_state["results"]
        
        # Filter detections
        filter_json = filter_detections(json_data)
        
        # Get agent response
        response = asyncio.run(agent.run(user_prompt=filter_json))
        
        # Display response
        st.write(response.data)
    else:
        st.warning("No results to analyze. Start the camera first.")

# Instructions
st.markdown("### Instructions")
st.markdown("""
1. Select model type in the sidebar
2. Adjust confidence threshold as needed
3. Click 'Start' to begin inference
4. Click 'Stop and Analyze' to process results
""")