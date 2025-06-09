#!/bin/bash

# Set environment variables untuk optimasi
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
export OPENCV_VIDEOIO_PRIORITY_V4L2=1
export PYTHONUNBUFFERED=1

# Tingkatkan prioritas proses
streamlit run drowsiness_dashboard.py --server.port 8501 --server.address 0.0.0.0
