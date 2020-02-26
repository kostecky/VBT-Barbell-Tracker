# VBT-Barbell-Tracker
Optically track a barbell through its range of motion using OpenCV to give the lifter realtime feedback on concentric avg velocity, cutoff velocity, and displacement for a Velocity Based Training program.

## Motivation

...

## Journey

...

## VBT
https://www.scienceforsport.com/velocity-based-training

## Requirements
- Python 3.7
- A webcam or IP camera that supports RTSP
- Lime Green paper/paint/etc.

## Installation

1. Create a new python 3.7 virtualenv
```
mkvirtualenv VBT-barbell-tracker
```

2. Install python requirements
```
pip install -r requirements.txt
```

3. Generate intrinsic camera values to undistort fisheye/barrel effect
Generate intrinsic camera distortion values to remove any barrel/fisheye distortion that your camera may have. You can easily spot this by looking at the outer perimeter of your camera to see if straight lines appear curved.

    a. Using your webcam, take 10 or more snapshots of the chessboard printout. Save the images as `.png` files. Adhere the printout to a board, so it's very flat. The images should be taken to ensure you cover all areas of your image plane, paying attention to the outer perimeter where most of the distortion will take place. You can find the opencv chessboard image here: https://github.com/opencv/opencv/blob/master/doc/pattern.png

    b. Place these images in the `./images/` directory

    c. Take another image of the area you want to undistort as a test. Save it as `test.png` and place it in the `./images/` directory.

    d. Run the `python undistort_fisheye.py` script to discover the intrinsict values. They will be dumped out in a json file called `fisheye_calibration_data.json`

4. Place lime green paper/paint/etc. on the end of the barbell that faces the camera, ensuring that it covers the entire face of it. Then, take a measurement of the diameter of the face of the barbell end you covered. Mine comes out to 50mm, so the radius would be 25mm. This is **essential** to calibrating the distance scale within the app. You can use any colour you want, but you will have to adjust parameters accordingly. The idea here is to use a colour that is highly differentiated from anything being captured in your surroundings or your clothing.

![Barbell with green marking on end](/images/test.png?raw=true "Barbell with green marking on end")

## Usage

```
workon VBT-barbell-tracker
```

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

## Hotkeys

`r` Clear set and reset all counters and bar tracker. This also happens automatically after 2 minutes of no movement.

## Roadmap
- [ ] Refactor POC
- [ ] Use QT or simpleGUI to reformulate into a proper app with input/output to adjust parameters on the fly
