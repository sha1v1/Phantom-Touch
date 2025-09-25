# Phantom-Touch

### requirements
Python version: 3.12 


### Set up instructions
1. Make sure you have the correct python version (3.12) installed. This is essential for Mediapipe to run, otherwise the filter won't function as intended.
2. Create a virtual environment
3. install dependencies
    ``` 
    pip install -r requirements.txt
    ```

### How to run 

To run this live, **make sure your bluetooth is turned off**:
```
python3 phantom-touch-live.py
```

### How it works?
Mediapipe tracks the fingertips and using the Lucas-Kanade method, the "tap" gesture is identified. A "tap" is defined as a moving index finger followed by a sudden stop. Once a tap is recognized, a distortion filter, based on a sine-wave and gaussian envelope, is applied on that spot. The idea is to mimic an actual ripple on water i.e. an expanding ring which fades out as is grows and is very local to the region of the tap. Edge detection is being used as an additional check to see if the index finger is actually stretched out (like this ☝🏽) so the filter isn't triggerd with a closed fist (because mediapipe, by default, always outputs fingertip locations even if they are not visible). 
