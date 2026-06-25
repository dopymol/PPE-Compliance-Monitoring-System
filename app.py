import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import pandas as pd
import tempfile

from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from ppe_matcher import assign_ppe_to_workers


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="PPE Compliance AI Dashboard",
    page_icon="🦺",
    layout="wide"
)


# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
}
.subtitle {
    font-size: 18px;
    color: #4b5563;
}
.section-card {
    background-color: #f8fafc;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
}
.footer {
    color: #6b7280;
    font-size: 14px;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="PPE Compliance AI Dashboard",
        options=[
            "Dashboard",
            "Image Detection",
            "Video Detection",
            "Live Camera",
            "About Project"
        ],
        icons=[
            "speedometer2",
            "image",
            "camera-video",
            "camera",
            "info-circle"
        ],
        menu_icon="shield-check",
        default_index=0
    )

    st.markdown("---")
    st.subheader("⚙️ Detection Settings")

    model_path = st.text_input("Model Path", "models/best.pt")
    confidence = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    iou = st.slider("IoU Threshold", 0.10, 0.90, 0.40, 0.05)

    st.markdown("---")
    st.info("""
    Recommended Settings  
    Confidence: 0.25 – 0.40  
    IoU: 0.40 – 0.50
    """)


# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)


model = load_model(model_path)


# -------------------------------
# Header
# -------------------------------
st.markdown(
    '<div class="main-title">🦺 PPE Compliance Monitoring System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered workplace safety monitoring using YOLO object detection.</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# -------------------------------
# Dashboard Page
# -------------------------------
if selected == "Dashboard":

    st.subheader("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "YOLO")
    col2.metric("Modes", "Image / Video / Live")
    col3.metric("Report", "CSV")
    col4.metric("UI", "Streamlit")

    style_metric_cards(
        background_color="#FFFFFF",
        border_left_color="#2563EB",
        border_color="#E5E7EB",
        box_shadow=True
    )

    st.markdown("""
    <div class="section-card">
    This PPE Compliance Monitoring System detects workers and safety equipment such as
    helmets, safety vests, gloves, goggles, and boots. It performs worker-wise compliance
    checking and generates downloadable safety audit reports.
    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# Image Detection Page
# -------------------------------
elif selected == "Image Detection":

    st.subheader("🖼️ Image PPE Detection")
    st.write("Upload one or more workplace images and run PPE compliance analysis.")

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi_image_upload"
    )

    if not uploaded_files:
        st.info("Upload one or more images to start PPE compliance detection.")

    else:
        run_detection = st.button(
            "🚀 Run PPE Detection on All Images",
            use_container_width=True
        )

        if run_detection:

            all_reports = []
            uploaded_preview_images = []
            detection_preview_images = []

            with st.spinner("Running YOLO detection on uploaded images..."):

                for uploaded_file in uploaded_files:

                    image = Image.open(uploaded_file).convert("RGB")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                        image.save(temp_file.name)
                        temp_image_path = temp_file.name

                    results = model(temp_image_path, conf=confidence, iou=iou)

                    persons = []
                    ppe_objects = []

                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        class_name = results[0].names[class_id].lower()
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        item = {
                            "class_name": class_name,
                            "confidence": round(conf_score, 2),
                            "box": [x1, y1, x2, y2]
                        }

                        if class_name == "person":
                            persons.append(item)
                        else:
                            ppe_objects.append(item)

                    annotated_image = results[0].plot()

                    uploaded_preview_images.append((uploaded_file.name, image))
                    detection_preview_images.append((uploaded_file.name, annotated_image))

                    if len(persons) == 0:
                        all_reports.append({
                            "Image": uploaded_file.name,
                            "Worker ID": "No worker detected",
                            "Helmet": "-",
                            "Vest": "-",
                            "Gloves": "-",
                            "Goggles": "-",
                            "Boots/Shoes": "-",
                            "Status": "No Worker",
                            "Violations": "No person detected"
                        })
                        continue

                    worker_ppe = assign_ppe_to_workers(persons, ppe_objects)

                    for idx, person in enumerate(persons, start=1):
                        ppe = worker_ppe[idx - 1]

                        has_helmet = ppe["helmet"]
                        has_vest = ppe["vest"]
                        has_gloves = ppe["gloves"]
                        has_goggles = ppe["goggles"]
                        has_boots = ppe["boots"]

                        has_no_helmet = ppe["no_helmet"]
                        has_no_gloves = ppe["no_gloves"]
                        has_no_goggle = ppe["no_goggle"]

                        violations = []

                        if has_no_helmet:
                            violations.append("Helmet missing")

                        if not has_vest:
                            violations.append("Vest missing")

                        if has_no_gloves:
                            violations.append("Gloves missing")

                        if has_no_goggle:
                            violations.append("Goggles missing")

                        status = "Violation" if violations else "Compliant"

                        all_reports.append({
                            "Image": uploaded_file.name,
                            "Worker ID": idx,
                            "Helmet": "Yes" if has_helmet else "Not confidently detected",
                            "Vest": "Yes" if has_vest else "No",
                            "Gloves": "Yes" if has_gloves else "Not confidently detected",
                            "Goggles": "Yes" if has_goggles else "Not confidently detected",
                            "Boots/Shoes": "Yes" if has_boots else "Not confidently detected",
                            "Status": status,
                            "Violations": "; ".join(violations) if violations else "None"
                        })

            st.markdown("---")

            for (name, img), (_, detected_img) in zip(
                uploaded_preview_images,
                detection_preview_images
            ):
                st.markdown(f"### {name}")

                c1, c2 = st.columns(2)

                with c1:
                    st.image(
                        img,
                        caption="Original Image",
                        use_container_width=True
                    )

                with c2:
                    st.image(
                        detected_img,
                        caption="Detection Result",
                        use_container_width=True
                    )

                st.markdown("---")

            if all_reports:

                df = pd.DataFrame(all_reports)

                st.subheader("📊 Dashboard Summary")

                total_workers = len(df[df["Status"] != "No Worker"])
                compliant_workers = len(df[df["Status"] == "Compliant"])
                violation_workers = len(df[df["Status"] == "Violation"])
                compliance_rate = round((compliant_workers / total_workers) * 100, 2) if total_workers > 0 else 0

                m1, m2, m3, m4 = st.columns(4)

                m1.metric("👷 Total Workers", total_workers)
                m2.metric("✅ Compliant", compliant_workers)
                m3.metric("⚠️ Violations", violation_workers)
                m4.metric("📈 Compliance Rate", f"{compliance_rate}%")

                style_metric_cards(
                    background_color="#FFFFFF",
                    border_left_color="#22C55E",
                    border_color="#E5E7EB",
                    box_shadow=True
                )

                st.subheader("📋 Combined Worker-wise Compliance Report")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Combined CSV Report",
                    data=csv,
                    file_name="combined_worker_ppe_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# -------------------------------
# Video Detection Page
# -------------------------------
elif selected == "Video Detection":

    st.subheader("🎥 Video Detection")
    st.write("Upload a workplace video and process it with YOLO.")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video is None:
        st.info("Upload a video file to start PPE video detection.")

    else:
        st.video(uploaded_video)

        process_video = st.button(
            "🚀 Run PPE Detection on Video",
            use_container_width=True
        )

        if process_video:

            with st.spinner("Processing video... This may take some time."):

                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_input.write(uploaded_video.read())
                temp_input.close()

                input_video_path = temp_input.name
                output_video_path = os.path.join(
                    "output",
                    "videos",
                    f"detected_{os.path.splitext(uploaded_video.name)[0]}.avi"
                )

                os.makedirs("output/videos", exist_ok=True)

                cap = cv2.VideoCapture(input_video_path)


                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps == 0:
                    fps = 25

                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                out = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    fps,
                    (640,480)
                )

                frame_count = 0
                progress_bar = st.progress(0)

                frame_skip = 3

                while True:
                    success, frame = cap.read()

                    if not success:
                        break

                    frame_count += 1

                    # Process only every 3rd frame
                    if frame_count % frame_skip != 0:
                        resized_frame = cv2.resize(frame, (640, 480))
                        out.write(resized_frame)
                        continue

                    frame = cv2.resize(frame, (640, 480))

                    results = model(frame, conf=confidence, iou=iou)
                    annotated_frame = results[0].plot()

                    out.write(annotated_frame)

                    if total_frames > 0:
                        progress_bar.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                cap.release()
                out.release()

            st.success(f"Video processing completed. Total frames processed: {frame_count}")

            st.subheader("✅ Processed Video with PPE Detection")

            st.warning("Preview may not play in browser due to codec limits, but the processed video is saved and downloadable.")
            with open(output_video_path, "rb") as f:
                video_bytes = f.read()

            st.video(video_bytes)

            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="ppe_detected_video.avi",
                    mime="video/x-msvideo",
                    use_container_width=True
                )


# -------------------------------
# Live Camera Page
# -------------------------------
elif selected == "Live Camera":

    st.subheader("📷 Live PPE Detection")

    st.markdown("""
        ### Real-Time Workplace Monitoring

        This module performs real-time PPE compliance detection using a webcam feed.

        **Features**
        - Person Detection
        - Helmet Detection
        - Vest Detection
        - Gloves Detection
        - Goggles Detection
        - Boots Detection
        - Live Compliance Monitoring

        **Instructions**
        1. Connect your webcam.
        2. Open terminal.
        3. Run the command below.
        4. Press **Q** to stop detection.
        """)

    st.code(
        "python real_time_detection.py",
        language="bash"
    )

    st.success("Ready for real-time PPE monitoring.")


# -------------------------------
# About Project Page
# -------------------------------
elif selected == "About Project":

    st.subheader("ℹ️ About Project")

    st.markdown("""
    ### PPE Compliance Monitoring System

    This project is an end-to-end Computer Vision application for workplace safety monitoring.

    **Core Functionalities**
    - Worker/person detection
    - PPE object detection
    - Worker-wise PPE assignment
    - Compliance rule checking
    - CSV report generation
    - Image detection
    - Video detection
    - Real-time webcam detection

    **Tech Stack**
    - Python
    - YOLO / Ultralytics
    - OpenCV
    - Streamlit
    - Pandas
    - PIL
    """)


# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div class="footer">
Built using YOLO, Streamlit, Python, OpenCV and Pandas | PPE Compliance AI System
</div>
""", unsafe_allow_html=True)