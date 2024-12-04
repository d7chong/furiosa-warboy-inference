import onnx
from furiosa.quantizer.calibrator import Calibrator, CalibrationMethod
import cv2
from tqdm import tqdm

from my_utils.quantize import get_calib_dataset
from my_utils.preprocess import YOLOPreProcessor

def print_calibration_ranges(
    onnx_path, 
    calib_method="MIN_MAX_ASYM", 
    calib_p=99.999, 
    num_calib_data=1, 
    calib_data_dir="/home/furiosa/work_space/val2017"
):
    """
    Print calibration ranges for an ONNX model.
    
    Args:
        onnx_path (str): Path to the ONNX model.
        calib_method (str): Calibration method (e.g., MIN_MAX_ASYM, MSE_SYM).
        calib_p (float): Percentile for calibration.
        num_calib_data (int): Number of calibration samples to use.
        calib_data_dir (str): Directory containing calibration data.
    """
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Set up calibration method and calibrator
    calibrator = Calibrator(
        onnx_model,
        CalibrationMethod._member_map_[calib_method],
        percentage=calib_p,
    )
    
    # Load calibration data
    calib_dataset = get_calib_dataset(num_calib_data, calib_data_dir)
    
    preprocessor = YOLOPreProcessor()

    # Collect calibration data
    print("Collecting calibration data...")
    for calib_data_path in tqdm(calib_dataset, desc="Calibration progress"):
        input_img = cv2.imread(calib_data_path)
        input_, _ = preprocessor(input_img, [640, 640], tensor_type="float32")
        calibrator.collect_data([[input_]])

    # Compute calibration ranges
    print("Computing calibration ranges...")
    calib_ranges = calibrator.compute_range()

    # Print and log calibration ranges
    print("\nCalibration Ranges:")
    for tensor_name, (tensor_min, tensor_max) in calib_ranges.items():
        print(f"{tensor_name}: min={tensor_min}, max={tensor_max}")

def plot_onnx_weight_distributions(onnx_path, dequantize=False):
    import matplotlib.pyplot as plt
    import numpy as np

    w_dtype = np.int8 if "i8" in onnx_path else np.float32
    
    onnx_model = onnx.load(onnx_path)
    weights = {init.name: np.frombuffer(init.raw_data, dtype=w_dtype) for init in onnx_model.graph.initializer}
    num_layers = len(weights)
    cols = 8
    rows = (num_layers + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axs = axs.flatten()

    for ax, (layer_name, data) in tqdm(zip(axs, weights.items()), desc='Plotting layers', total=num_layers):
        if w_dtype == np.int8 and dequantize:
            scale = np.abs(np.max(data) - np.min(data)) / 255
            data = np.float32(data * scale)
        
        # print(f"{layer_name}: min={np.min(data)}, max={np.max(data)}")
        # print(f"layer dtype: {data.dtype}")
        
        w_min, w_max = (-2.0, 2.0) if data.dtype == np.float32 else (-128, 127)
        w_bins = 2**10 if w_dtype == np.float32 else 256
        
        ax.hist(data, bins=w_bins, range=(w_min, w_max))
        ax.set_title(layer_name, fontsize=8)
        
        # print(f"{layer_name}: min={np.min(data)}, max={np.max(data)}")

    for ax in axs[num_layers:]:
        ax.remove()
    
    # TODO - PLOT QUANTIZED WEIGHTS, DEQUANTIZED WEIGHTS, AND OVERLAP WITH ORIGINAL WEIGHTS! 

    plt.tight_layout()
    plt.savefig("onnx_weight_distributions.png")
    plt.show()

if __name__ == "__main__":
    # onnx_path = "/home/furiosa/dev/hyo/furiosa-warboy-inference/models/onnx/object_detection/yolov5nu_simp1_opt0_fuse0.onnx"
    # print_calibration_ranges(onnx_path)
    
    quant_onnx_path = "/home/furiosa/dev/hyo/furiosa-warboy-inference/models/quant_onnx/object_detection/yolov5nu_simp1_opt0_fuse0_MIN_MAX_SYM_nc1000_bs16_pefuse0_iuint8_ofloat32_i8.onnx"
    plot_onnx_weight_distributions(quant_onnx_path, dequantize=True)