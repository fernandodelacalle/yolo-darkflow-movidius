import mvnc.mvncapi as mvncapi

def get_mvnc_device():
    mvncapi.global_set_option(mvncapi.GlobalOption.RW_LOG_LEVEL, 0)
    devices = mvncapi.enumerate_devices()
    if (len(devices) < 1):
        print("Error - no NCS devices detected")
        quit()
    dev = mvncapi.Device(devices[0])
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
    input_fifo, output_fifo = graph.allocate_with_fifos(dev, graph_file_buffer)
    return graph, input_fifo, output_fifo