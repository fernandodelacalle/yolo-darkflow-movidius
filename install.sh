pip install --upgrade cython

git clone https://github.com/thtrieu/darkflow darkflow_all
cd darkflow_all
python3 setup.py build_ext --inplace
pip install -e .

cd ..
ln -s darkflow_all/darkflow darkflow

cd darkflow_all
wget -O tiny-yolo-voc.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny-voc.cfg
wget -O tiny-yolo-voc.weights https://pjreddie.com/media/files/yolov2-tiny-voc.weights
# Saving graph and weights to protobuf file
python3 flow --model tiny-yolo-voc.cfg --load tiny-yolo-voc.weights --savepb
cd ..
mv  darkflow_all/built_graph/ . 
# Compile the model
mvNCCompile built_graph/tiny-yolo-voc.pb -s 12 -in input -on output -o built_graph/tiny-yolo-voc.graph
