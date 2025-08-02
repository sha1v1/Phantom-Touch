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
If you have a video input file:
```
python3 phantom-touch.py <input_file.mp4>
```

If you'd like to run this live, **make sure your bluetooth is turned off**:
```
python3 phantom-touch-live.py
```