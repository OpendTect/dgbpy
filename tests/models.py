# Create simple img2img onnx, keras and pytorch models of arbitrary shape, 
# number of inputs, number of outputs for testing
#
# Shape - use None to define a dynamic axis size
# Batch size is always dynamic
# Model ouput varies depending on numin and numout:
#   For numin=numout=1 model outputs input*2
#   For numin=1, numout>1 model outputs input/numout to each output channel 
#   For numin>1, numout=1 model outputs sum(inputs) 
#   For numin>1, numout>1 model outputs sum(inputs)/numout to each output channel
#
#

import numpy as np
import onnx
import tensorflow as tf
import torch
from onnx2torch import convert

def make_models(shape, numin, numout, data_format):
    make_onnx_model(shape, numin, numout, data_format)
    make_keras_model(shape, numin, numout, data_format)
    if data_format=='channels_first':
        make_pytorch_model(shape, numin, numout)

class ReduceSumModule(torch.nn.Module):
    def __init__(self, chaxis=1):
        super().__init__()
        self.chaxis = chaxis

    def forward(self, input):
        sum = torch.sum(input, dim=self.chaxis, keepdim=True)
        return sum

class DoubleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        sum = torch.add(input, input)
        return sum

class DivideModule(torch.nn.Module):
    def __init__(self, num=1):
        super().__init__()
        self.num = num

    def forward(self, input):
        res = torch.div(input, self.num)
        return res

class ConCatModule(torch.nn.Module):
    def __init__(self, num=1, chaxis=1):
        super().__init__()
        self.chaxis = chaxis
        self.num = num

    def forward(self, input):
        res = torch.cat([input for i in range(self.num)], dim=self.chaxis)
        return res

def make_pytorch_model(shape, numin, numout):
    model_inshape = [numin, *shape]
    model_outshape = [numout, *shape]
    model_name = "test_" + '-'.join(map(str, model_inshape)) + "_" + '-'.join(map(str, model_outshape))
    model_def = None
    if numin==1:
        if numout==1:
            model_def = torch.nn.Sequential(
                DoubleModule()
            )
        else:
            model_def = torch.nn.Sequential(
                DivideModule(num=numout),
                ConCatModule(num=numout, chaxis=1)
            )

    else:
        if numout==1:
            model_def = torch.nn.Sequential(
                ReduceSumModule(chaxis=1)
            )
        else:
            model_def = torch.nn.Sequential(
                ReduceSumModule(chaxis=1),
                DivideModule(num=numout),
                ConCatModule(num=numout, chaxis=1)
            )

    model = torch.jit.script(model_def)
    model.save(model_name+".pth")
#    torch.save(model_def, model_name+".pt")

def make_keras_model(shape, numin, numout, data_format):
    model_inshape = [numin, *shape] if data_format=="channels_first" else [*shape, numin]
    model_outshape = [numout, *shape] if data_format=="channels_first" else [*shape, numout]
    model_name = "test_" + '-'.join(map(str, model_inshape)) + "_" + '-'.join(map(str, model_outshape))

    inputs = tf.keras.Input(shape=model_inshape, name='X', dtype='float32')
    outputs = None
    chaxis = 1 if  data_format=="channels_first" else -1
    if numin==1:
        if numout==1:
            outputs = tf.keras.layers.Add(name='Y')([inputs, inputs])
        else:
            div = tf.keras.layers.Lambda(lambda x: x/numout)(inputs)
            tlist = [div for i in range(numout)]
            outputs = tf.keras.layers.Concatenate(axis=chaxis, name='Y')(tlist)
        
    else:
        inlist = tf.unstack(inputs, axis=chaxis)
        sum = tf.keras.layers.Add()(inlist)
        if numout>1:
            div = tf.expand_dims(tf.keras.layers.Lambda(lambda x: x/numout)(sum), axis=chaxis)
            tlist = [div for i in range(numout)]
            outputs = tf.keras.layers.Concatenate(axis=chaxis, name='Y')(tlist)
        else:
            outputs = tf.keras.layers.Lambda(lambda x: x, name='Y') (tf.expand_dims(sum, axis=chaxis))

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    tf.keras.models.save_model(model, model_name+".h5",overwrite=True)

def get_model_filename(shape):
    return "test_" + '-'.join(map(str, shape)) + "_" + '-'.join(map(str, shape))

def make_onnx_model(shape, numin, numout, data_format):
    model_inshape = [numin, *shape] if data_format=="channels_first" else [*shape, numin]
    model_outshape = [numout, *shape] if data_format=="channels_first" else [*shape, numout]
    model_name = get_model_filename(shape)

    inshape = [None, *model_inshape]
    outshape = [None, *model_outshape]
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, inshape)
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, outshape)
    nout = onnx.helper.make_tensor("nout", onnx.TensorProto.FLOAT, (1,), vals=[numout])
    shape = onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, (len(outshape),))

    graph_def = None
    nodes = []
    chaxis = 1 if  data_format=="channels_first" else len(inshape)-1
    if numin==1:
        if numout==1:
            nodes.append(onnx.helper.make_node(
                            name="sum",
                            op_type="Sum",
                            inputs=["X", "X"],
                            outputs=["Y"])
            )
        else:
            nodes.append(onnx.helper.make_node(
                            name="divide",
                            op_type="Div",
                            inputs=["X","nout"],
                            outputs=["node1"])
            )
            nodes.append(onnx.helper.make_node(
                            name="concat",
                            op_type="Concat",
                            inputs=["node1" for i in range(numout)],
                            outputs=["Y"],
                            axis=chaxis)
            )
        
        graph_def = onnx.helper.make_graph(
            nodes=nodes,
            name="testmodel",
            inputs=[X],
            outputs=[Y],
            initializer=[nout] if numout>1 else [])
    else:
        axes = onnx.helper.make_tensor(
            name="axes",
            data_type=onnx.TensorProto.INT64,
            dims=(1,),
            vals=[chaxis])
        nodes.append(onnx.helper.make_node(
            name="sum",
            op_type="ReduceSum",
            inputs=["X", "axes"],
            outputs=["node1" if numout>1 else "Y"],
            keepdims=1)
        )
        if numout>1:
            nodes.append(onnx.helper.make_node(
                            name="divide",
                            op_type="Div",
                            inputs=["node1","nout"],
                            outputs=["node2"])
            )
            nodes.append(onnx.helper.make_node(
                            name="concat",
                            op_type="Concat",
                            inputs=["node2" for i in range(numout)],
                            outputs=["Y"],
                            axis=chaxis)
            )

        graph_def = onnx.helper.make_graph(
            nodes=nodes,
            name="testmodel",
            inputs=[X],
            outputs=[Y],
            initializer=[axes, nout] if numout>1 else [axes])

    model_def = onnx.helper.make_model(graph_def, producer_name="test")
    model_def.opset_import[0].version = 16
    model_def.ir_version = 8
    model_def = onnx.shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    onnx.save(model_def, model_name+".onnx")



# make_models((128,256), 1, 1, "channels_first")
# make_models((128,256), 2, 1, "channels_first")
# make_models((128,256), 1, 2, "channels_first")
# make_models((128,256), 2, 2, "channels_first")
# make_models((None,None), 1, 1, "channels_first")
# make_models((None,None), 2, 1, "channels_first")
# make_models((None,None), 1, 2, "channels_first")
# make_models((None,None), 2, 2, "channels_first")

# make_models((128,128,128), 1, 1, "channels_first")
# make_models((128,128,128), 2, 1, "channels_first")
# make_models((128,128,128), 1, 2, "channels_first")
# make_models((128,128,128), 2, 2, "channels_first")
# make_models((None,None,None), 1, 1, "channels_first")
# make_models((None,None,None), 2, 1, "channels_first")
# make_models((None,None,None), 1, 2, "channels_first")
# make_models((None,None,None), 2, 2, "channels_first")

# make_models((128,256), 1, 1, "channels_last")
# make_models((128,256), 2, 1, "channels_last")
# make_models((128,256), 1, 2, "channels_last")
# make_models((128,256), 2, 2, "channels_last")
# make_models((None,None), 1, 1, "channels_last")
# make_models((None,None), 2, 1, "channels_last")
# make_models((None,None), 1, 2, "channels_last")
# make_models((None,None), 2, 2, "channels_last")

# make_models((128,128,128), 1, 1, "channels_last")
# make_models((128,128,128), 2, 1, "channels_last")
# make_models((128,128,128), 1, 2, "channels_last")
# make_models((128,128,128), 2, 2, "channels_last")
# make_models((None,None,None), 1, 1, "channels_last")
# make_models((None,None,None), 2, 1, "channels_last")
# make_models((None,None,None), 1, 2, "channels_last")
# make_models((None,None,None), 2, 2, "channels_last")