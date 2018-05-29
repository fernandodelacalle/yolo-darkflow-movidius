# yolo-darkflow-movidius

Need to have darkflow (https://github.com/thtrieu/darkflow) installed in the path.

You can follow the next steps: 
```
git clone https://github.com/fernandodelacalle/yolo-darkflow-movidius.git

cd yolo-darkflow-movidius/
pip install --upgrade cython

git clone https://github.com/thtrieu/darkflow darkflow_all
cd darkflow_all
python3 setup.py build_ext --inplace
pip install -e .

cd ..
ln -s darkflow_all/darkflow darkflow
```

Next download the tiny yolo v2 weights for voc and save the built graph to a protobuf file (.pb):

```
cd darkflow_all
wget -O tiny-yolo-voc.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny-voc.cfg
wget -O tiny-yolo-voc.weights https://pjreddie.com/media/files/yolov2-tiny-voc.weights
## Saving graph and weights to protobuf file
python3 flow --model tiny-yolo-voc.cfg --load tiny-yolo-voc.weights --savepb
cd ..
mv  darkflow_all/built_graph/ . 
```

Finally compile the pb to a movidus graph as follows:
```
mvNCCompile built_graph/tiny-yolo-voc.pb -s 12 -in input -on output -o built_graph/tiny-yolo-voc.graph
```