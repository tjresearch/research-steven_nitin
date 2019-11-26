## Title: Tracking Tomato Plant Growth and Disease

### Overview
```
Our project uses edge detection methods and neural networks in order to create a web application
that can let a user know how their plant is growing and diagnose it with any diseases
```
### Installation and setup
1. You will need to have openCV, matplotlib, numpy, pickle, and pytorch libraries installed
2. After pulling the code, create three organized folders for the downloaded images, the edge detection output, and the output for the corresponding pickle files

### Running
For the edge detection file
```
python3 edge_detection.py -i <full filepath for the training image folder> -o <full filepath for the edge output folder> --minwidth <minimum width of each edge> --minheight <minimum height of each edge> -p <full filepath for the pickle output folder>
```
For the growth detection file
```
python3 growth_detection.py -f <full filepath for the pickle output folder>
```

### Sample output
- An example can be found at sampleOutput/AcGraph.png
