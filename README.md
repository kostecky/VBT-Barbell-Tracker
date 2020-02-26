# VBT-Barbell-Tracker
Optically track a barbell through its range of motion using OpenCV to give the lifter feedback on concentric avg velocity and displacement for Velocity Based Training.

## Motivation

...

## Requirements
- Python 3.7
- A webcam or IP camera that supports RTSP
- Green paper/paint/etc.

## Installation

1. Create a new python 3.7 virtualenv
```mkvirtualenv VBT-barbell-tracker```

2. Install python requirements
```pip install -r requirements.txt```

3. Generate the intrinsic camera distortion values
...

## Usage

### Running it on a video
```
python vbt_barbell_tracker.py --video press.mp4
```

### Running it with your webcam
```
python VBT-barbell-tracker.py
```

### Running it on a RTSP stream
```
python VBT-barbell-tracker.py --video 'rtsp://USER:PASSWORD@192.168.1.127?tcp'
```



