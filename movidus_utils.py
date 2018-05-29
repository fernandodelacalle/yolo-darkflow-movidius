import mvnc.mvncapi as mvncapi

def get_mvnc_device():
    mvncapi.global_set_option(mvncapi.GlobalOption.RW_LOG_LEVEL, 0)
    # get a list of names for all the devices plugged into the system
    devices = mvncapi.enumerate_devices()
    if (len(devices) < 1):
        print("Error - no NCS devices detected, verify an NCS device is connected.")
        quit()
    # get the first NCS device by its name.  For this program we will always open the first NCS device.
    dev = mvncapi.Device(devices[0])
    # try to open the device.  this will throw an exception if someone else has it open already
    try:
        dev.open()
    except:
        print("Error - Could not open NCS device.")
        quit()
    return dev


def load_graph(dev, graph_file):
    with open(graph_file, mode='rb') as f:
        graph_file_buffer = f.read()
    graph = mvncapi.Graph('graph1')
    # input_fifo, output_fifo = graph.allocate_with_fifos(dev, graph_file_buffer,
    #     input_fifo_type=mvncapi.FifoType.HOST_WO, input_fifo_data_type=mvncapi.FifoDataType.FP32, input_fifo_num_elem=2,
    #     output_fifo_type=mvncapi.FifoType.HOST_RO, output_fifo_data_type=mvncapi.FifoDataType.FP32, output_fifo_num_elem=2)
    input_fifo, output_fifo = graph.allocate_with_fifos(dev, graph_file_buffer)
    return graph, input_fifo, output_fifo
