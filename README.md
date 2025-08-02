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
This will result in an output file named **output.mp4**. In case the filter does not work as expected, try out the live version below.

To run this live, **make sure your bluetooth is turned off**:
```
python3 phantom-touch-live.py
```

### Guide to the files
1. **demo.mp4**: Demo and explanation of the filter
2. **phantom-touch.py**: The file to be used when testing using an input file
3. **phantom-touch_live.py**: The file to be used when testing using live video
4. **input.mp4 and output.mpr**: Record of my runs. 
