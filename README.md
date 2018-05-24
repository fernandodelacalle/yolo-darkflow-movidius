# yolo-darkflow-movidius

Need to have darkflow  installed in the path: https://github.com/thtrieu/darkflow

Save the built graph to a protobuf file (.pb)

```
## Saving the lastest checkpoint to protobuf file
flow --model cfg/yolo-new.cfg --load -1 --savepb

## Saving graph and weights to protobuf file
flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb
```

For the Movidius compile the pb to movidus graph as follows:
```
mvNCCompile yolov2-tiny-voc.pb -s 12 -in input -on output -o yolov2-tiny-voc.graph
```