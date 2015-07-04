# Caffe-Layer-Extractor

Dumps activations after each layer into text files.

`./extractor --network alexnet.protoxt --weights alexnet.caffemodel --image test.jpg --tolayer 5`

Todo:
- Name output files as layer names
- Bounds checking if tolayer > sizeof(network)
