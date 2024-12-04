import torch
import onnx
import os
import cv2
from tqdm import tqdm
from copy import deepcopy
import re

from my_utils import *


def torch_to_onnx(model_name, fuse_conv_bn, input_shape, onnx_path):    
    # onnx_path = get_onnx_path(model_name)
    
    torch_model = get_yolo_model(model_name)
    if torch_model is None:
        raise ValueError(
            f"Cannot load torch model. YOLO model {model_name} not found/supported"
        )

    if fuse_conv_bn:
        torch_model = torch_model.fuse()
    
    torch_model.eval()
    
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        torch_model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )
    
    edited_onnx_model = edit_onnx_model(model_name, onnx_path)
    onnx.save(onnx.shape_inference.infer_shapes(edited_onnx_model), onnx_path)
            
    print(f"Model {model_name} converted to ONNX format")
    return True


def optimize_onnx(onnx_path, simplify=False, optimize=False):
    onnx_model = onnx.load(onnx_path)
    
    if simplify:
        from onnxsim import simplify, model_info
        opt_onnx_model, check = simplify(onnx_model)
        assert check, "ONNX model could not be simplified"
        model_info.print_simplifying_info(onnx_model, opt_onnx_model)
        onnx_model = opt_onnx_model
    
    if optimize:
        import onnxoptimizer
        onnx_model = onnxoptimizer.optimize(onnx_model)
    
    onnx.save(onnx_model, onnx_path)


def quantize_onnx(
    onnx_path,
    quant_onnx_name,
    calib_method="MIN_MAX_ASYM",
    num_calib_data=500,
    input_type="uint8", output_type="float32", 
    save_step=None,
    calib_p=99.999,
    calib_data_dir="/home/furiosa/work_space/val2017",
):
    """
        - onnx_path: path of the ONNX model to quantize
        - quant_onnx_name: name of the quantized ONNX model
        - input_type: type of the input tensor (uint8 or float32)
        - output_type: type of the output tensor (int8, uint8, or float32)
        - save_step: path to save the quantized model at each step of calibration
            - ex) save_step = 100 -> save the model every 50 steps of calibration 
            - useful for running experiments
    """
    quant_onnx_path = get_quant_onnx_path(quant_onnx_name)
    
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(quant_onnx_path)):
        os.makedirs(os.path.dirname(quant_onnx_path))
    
    # If quantized model exists, stop
    if os.path.exists(quant_onnx_path):
        print(f"Quantized model {quant_onnx_name} already exists")
        return
    
    # If ONNX model doesn't exist, stop
    if not os.path.exists(onnx_path):
        print(f"ONNX model {onnx_path} not found")
        return
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    from my_utils.preprocess import YOLOPreProcessor
    from furiosa.optimizer import optimize_model
    from furiosa.quantizer import (
        CalibrationMethod, Calibrator, quantize
    )
    
    input_shape = [1, 3, 640, 640]
    new_shape = input_shape[2:]
    
    # optimmize model with furiosa_optimizer
    onnx_model = optimize_model(
        onnx_model,
        opset_version=13,
        input_shapes = {"input": input_shape} 
    )
    
    # Initialize calibrator and preprocessor
    calibrator = Calibrator(
        onnx_model,
        CalibrationMethod._member_map_[calib_method],
        percentage=calib_p,
    )
    preprocessor = YOLOPreProcessor()
    
    # Get calibration dataset
    calib_dataset = get_calib_dataset(num_calib_data, calib_data_dir)    
        
    # Collect data for calibration
    for i, calib_data in tqdm(
        enumerate(calib_dataset), desc=f'calibrating with {calib_method}...', total=num_calib_data,
    ):
        input_img = cv2.imread(calib_data)
        input_, _ = preprocessor(input_img, new_shape, tensor_type="float32")
        calibrator.collect_data([[input_]])
        
        if save_step is not None and (i+1) % save_step == 0:
            # get copy of model (to preserve original model)
            onnx_model_copy = deepcopy(onnx_model)
            
            # change IO types
            onnx_model = onnx_change_io_type(onnx_model, input_type, output_type)
            
            # calculate calibration range
            calib_range = calibrator.compute_range()
            
            # get quantized model
            quantized_model = quantize(onnx_model, calib_range)
            
            # save quantized model
            # file name will have "..._nc100_"
            # replace the number after "nc" to the current calibration step
            new_quant_onnx_name = re.sub(r"nc\d+", f"nc{i+1}", quant_onnx_name)
            new_quant_onnx_path = get_quant_onnx_path(new_quant_onnx_name)
            with open(new_quant_onnx_path, "wb") as f:
                f.write(bytes(quantized_model))
            
            print(f"Quantized model saved at {new_quant_onnx_path}")
            
            # restore original model
            onnx_model = onnx_model_copy
    
    # Final quantization if this doesn't exist
    if not os.path.exists(quant_onnx_path):
        onnx_model = onnx_change_io_type(onnx_model, input_type, output_type)
        calib_range = calibrator.compute_range()
        quantized_model = quantize(onnx_model, calib_range)
        with open(quant_onnx_path, "wb") as f:
            f.write(bytes(quantized_model))
        
    return True


def get_onnx_path(model_name):
    task = get_yolo_task(model_name)
    return f"{ONNX_DIR}/{task}/{model_name}.onnx"


def get_quant_onnx_path(quant_onnx_name):
    longest_model_name = ""
    saved_task_type = None

    for task in TASKS:
        for model_name in TEST_MODEL_LIST[task]:
            if model_name in quant_onnx_name:
                if len(model_name) > len(longest_model_name):
                    longest_model_name = model_name
                    saved_task_type = task
    
    # Save or return saved_task_type as needed
    return f"{QUANT_ONNX_DIR}/{saved_task_type}/{quant_onnx_name}.onnx"


def get_yolo_task(model_name):
    for task in TASKS:
        for name in TEST_MODEL_LIST[task]:
            if name == model_name:
                return task
    raise ValueError(f"Task unfound: YOLO model {model_name} not found/supported")


def get_yolo_model(model_name):
    from ultralytics import YOLO
    
    yolo_version = int(model_name.split('v')[-1][0])
    task_type = get_yolo_task(model_name)
    
    if model_name not in TEST_MODEL_LIST[task_type]:
        print(model_name)
        raise ValueError(f"YOLO model {model_name} not in TEST_MODEL_LIST")
    
    print(f"YOLO_VERSION: {yolo_version}")
    
    weight_file_name = f"{model_name}.pt"
    weight_file = os.path.join(WEIGHTS_DIR, task_type, weight_file_name)
    
    print(weight_file)
    
    torch_model = None            
    if yolo_version in [8, 9]:
        if task_type == "instance_segmentation" and yolo_version == 9:
            torch_model = torch.hub.load("WongKinYiu/yolov9", "custom", weight_file, force_reload=True).to(
                torch.device("cpu")
            )
        else:
            torch_model = YOLO(weight_file).model
    elif yolo_version == 7:
        torch_model = torch.hub.load("WongKinYiu/yolov7", "custom", weight_file).to(
            torch.device("cpu")
        )
    elif yolo_version == 5:
        if "u" not in model_name:
            torch_model = torch.hub.load(
                "ultralytics/yolov5", "custom", weight_file
            ).to(torch.device("cpu"))
        else:
            torch_model = YOLO(weight_file).model
    else:
        try:
            torch_model = YOLO(weight_file).model
        except:
            raise ValueError(f"YOLO model {weight_file} not found/supported")
    
    return torch_model


def edit_onnx_model(model_name, onnx_path):
    from onnx.utils import Extractor
    from my_utils import yolo_onnx_utils
    
    # onnx_path = get_onnx_path(model_name)
    
    onnx_model = onnx.load(onnx_path)
    task = get_yolo_task(model_name)
    
    input_to_shape, output_to_shape = yolo_onnx_utils.get_onnx_graph_info(
        task, model_name, onnx_path, None
    )
    
    edited_graph = Extractor(onnx_model).extract_model(
        input_names=list(input_to_shape), output_names=list(output_to_shape)
    )
    
    for value_info in edited_graph.graph.input:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(
            input_to_shape[value_info.name]
        )
    for value_info in edited_graph.graph.output:
        del value_info.type.tensor_type.shape.dim[:]
        value_info.type.tensor_type.shape.dim.extend(
            output_to_shape[value_info.name]
        )
    return edited_graph


def get_calib_dataset(
    num_calib_data=500,
    calib_data_dir="/home/furiosa/work_space/val2017",
):
    import imghdr, glob, random
    
    calib_data = []
    
    data = glob.glob(calib_data_dir + "/**", recursive=True)
    data = random.choices(data, k=min(num_calib_data, len(data)))
    
    for img in data:
        if os.path.isdir(img) or imghdr.what(img) is None:
            continue
        calib_data.append(img)

    return calib_data


def onnx_change_io_type(onnx_model, input_type, output_type):
    from furiosa.quantizer import (
        ModelEditor, ModelEditor, TensorType,
        get_pure_input_names, get_output_names,
    )
    
    editor = ModelEditor(onnx_model)
    
    # Change input_type
    if input_type in ['uint8']:
        input_names = get_pure_input_names(onnx_model)[0]
        editor.convert_input_type(input_names, TensorType.UINT8)
    
    # Change output_type
    if output_type in ['int8', 'uint8']:
        output_names = get_output_names(onnx_model)[0]
        if output_type == 'int8':
            editor.convert_output_type(output_names, TensorType.INT8)
        else:
            editor.convert_output_type(output_names, TensorType.UINT8)
            
    return onnx_model