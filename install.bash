#!/bin/bash

pip install ultralytics==8.3.136
pip uninstall -y opencv-python
pip install opencv-python-headless==4.10.0.84
pip install roboflow
pip install torch