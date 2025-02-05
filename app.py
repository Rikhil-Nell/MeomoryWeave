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

# Initialize session state
if "results" not in st.session_state:
    st.session_state.update({
        "results": [],
        "frame_count": 0,
        "model": None,
        "webrtc_ctx": None
    })

# Load or upload YOLO model
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model_option = st.radio("Choose Model", ["Default (YOLOv8n)", "Custom Model"])
    
    if model_option == "Custom Model":
        model_file = st.file_uploader("Upload YOLO Model", type=["pt"])
        if model_file:
            st.session_state.model = load_model(model_file.name)
    else:
        if not st.session_state.model:
            st.session_state.model = load_model("yolo11n.pt")
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

# Main app interface
st.title("Real-Time Object Detection with YOLO")
st.caption("Powered by Streamlit, WebRTC, and Ultralytics YOLO")

# Video processing callback
def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        
        if st.session_state.model:
            # Perform YOLO inference
            results = st.session_state.model.predict(
                source=img,
                conf=confidence_threshold,
                verbose=False,
                device="cpu"  # Force CPU for cloud compatibility
            )
            
            # Process and store results
            frame_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_number": st.session_state.frame_count,
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
            
            st.session_state.results.append(frame_data)
            st.session_state.frame_count += 1
            
            # Render results on frame
            annotated_frame = results[0].plot()
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        
        return frame
    except Exception as e:
        st.error(f"Frame processing error: {str(e)}")
        return frame

# WebRTC component
class WebRTCComponent:
    def __init__(self):
        self.ctx = None

    def __call__(self):
        self.ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_state=st.session_state.get("webrtc_running", False)
        )
        return self.ctx

# Create WebRTC component instance
webrtc_component = WebRTCComponent()
ctx = webrtc_component()

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera") and not st.session_state.get("webrtc_running"):
        st.session_state.webrtc_running = True


with col2:
    if st.button("Stop Camera") and st.session_state.get("webrtc_running"):
        st.session_state.webrtc_running = False
        if ctx:
            ctx.stop()


# Analysis section
if st.button("Analyze Results") and st.session_state.results:
    try:
        # Filter detections
        filter_json = filter_detections(st.session_state.results)
        
        # Get agent response
        response = asyncio.run(agent.run(user_prompt=filter_json))
        
        # Display response
        st.subheader("Analysis Results")
        st.write(response.data)
        
        # Clear results after analysis
        st.session_state.results = []
        st.session_state.frame_count = 0
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# Instructions
st.markdown("### Instructions")
st.markdown("""
1. Select model type in the sidebar
2. Adjust confidence threshold as needed
3. Click 'Start Camera' to begin inference
4. Click 'Stop Camera' to end the stream
5. Click 'Analyze Results' to process collected data
""")