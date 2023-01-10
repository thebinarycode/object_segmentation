# object_segmentation
Instance segmentation model training using UNET or FCN_8 architecture

Goal: To build/train custom Instance segmentation model

Build dataset: You can use LabelImg for tagging the objects in the image Refer the link below for installation and dataset preparation https://github.com/heartexlabs/labelImg

To train in a model directly, TrainModel.py file can be used

To train inside a Docker container, follow as below

Steps to execute:

    Set the path to current directory in command prompt
    Command to run the complete process: docker compose up
    Command to build the image from DockerFile: docker build -t [preferred container name] .
    Command to run the container image in interactive type: docker run -it [assigned container name] bash
    
refernce/credits:
https://github.com/seth814/Semantic-Shapes

