# Online-Realtime-Action-Recognition-based-on-OpenPose
A skeleton-based real-time online action recognition project, classifying and recognizing base on framewise joints.


------
## Introduction
*The **pipline** of this work is:*   
 - Realtime pose estimation by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose);   
 - Online human tracking for multi-people scenario by [DeepSort algorithm](https://github.com/nwojke/deep_sortv);   
 - Action recognition with DNN for each person based on single framewise joints detected from Openpose.


------
## Dependencies
 - python >= 3.5
 - Opencv >= 3.4.1   
 - sklearn
 - tensorflow & keras
 - numpy & scipy 
 - pathlib
 
 install_dependencies.sh is a script that will install core dependencies and will download the VGG_origin model.
 
------
## Usage
 - Download the openpose VGG tf-model with command line `./download.sh`(/Pose/graph_models/VGG_origin) or fork [here](https://pan.baidu.com/s/1XT8pHtNP1FQs3BPHgD5f-A#list/path=%2Fsharelink1864347102-902260820936546%2Fopenpose%2Fopenpose%20graph%20model%20coco&parentPath=%2Fsharelink1864347102-902260820936546), and place it under the corresponding folder; 
 - `python main.py`, it will **start the webcam**. 
 (you can choose to test video with command `python main.py --video=test.mp4`, however I just tested the webcam mode)   
 - By the way, you can choose different openpose pretrained model in script.    
 **VGG_origin**: training with the VGG net, as same as the CMU providing caffemodel, more accurate but slower, **mobilenet_thin**:  training with the Mobilenet, much smaller than the origin VGG, faster but less accurate.   
 **However, Please attention that the Action Dataset in this repo is collected along with the** ***VGG model*** **running**.


------
## Training with own dataset
 - prepare data(actions) by running `main.py`, remember to ***uncomment the code of data collecting***, the origin data will be saved as a `.txt`.
 - transforming the `.txt` to `.csv`, you can use EXCEL to do this.
 - do the training with the `traing.py` in `Action/training/`, remember to ***change the action_enum and output-layer of model***.
 
-----
## Labelling Data
- `action_labeller.py` is a GUI application that is meant to label images with the intention of training action recognition. It can be launched with `python3 action_labeller.py`. To run on the CPU, run `python3 action_labeller.py --use_cpu 1`. Three test images are provided in `test_data`. It can also process videos in the data directory by first extracting individual frames.

------
## Acknowledge
Thanks to the following awesome works:    
 - [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)   
 - [deep_sort_yolov3](https://github.com/Qidian213/deep_sort_yolov3)    
 - [Real-Time-Action-Recognition](https://github.com/TianzhongSong/Real-Time-Action-Recognition)
