'## Title: Tracking Tomato Plant Growth and Disease

### Overview
```
Our project uses edge detection methods and neural networks in order to create a web application
that can let a user know how their plant is growing and diagnose it with any diseases
```
### Installation and setup
1. You will need to have openCV, matplotlib, numpy, pickle, flask and pytorch libraries installed
2. After pulling the code, create three organized folders for the downloaded images, the edge detection output, the output for the linear regression, and the output for the corresponding pickle files
3. To have the flask server run successfully, templates must be in the same directory as server.py

### Running
For the edge detection file
```
python3 edge_detection.py -i <full filepath for the training image folder> -o <full filepath for the edge output folder> --minwidth 30 --minheight 30 -p <full filepath for the pickle output folder>
```
For the growth detection file
```
python3 growth_detection.py -f <full filepath for the pickle output folder> -o <full filepath for the regression output folder>
```

### Sample output
- An example can be found at sampleOutput/AcGraph.png

Credit to Github user @meijieru for the dehaze
Credit to Github user @FienSoP for the canny_edge_detector
