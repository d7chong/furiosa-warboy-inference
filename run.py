import torch
import numpy as np
from tqdm import tqdm
import asyncio
import argparse
from pathlib import Path
from pycocotools.cocoeval import COCOeval
import random
import time

from my_utils import (
    ANCHORS,
    CONF_THRES,
    IOU_THRES,
    ONNX_DIR,
    QUANT_ONNX_DIR,
    TASK_TO_ANNOTATION_PATH,
    TARGET_DET_ACCURACY, 
    TARGET_POSE_ACCURACY,
    TARGET_BBOX_ACCURACY,
)

from my_utils.data_utils import MSCOCODataset, get_dataloader
from my_utils.preprocess import YOLOPreProcessor
from my_utils.postprocess import getPostProcesser
from my_utils.scheduler import round_robin, queue_runner
from my_utils.monitor_npu import track_npu_stats, draw_npu_stats_fig
from my_utils.quantize import (
    torch_to_onnx,
    optimize_onnx,
    quantize_onnx,
    get_yolo_task,
)

from furiosa.runtime.profiler import profile

random.seed(42)
np.random.seed(42)

# Change the main function to be asynchronous
async def main():
    parser = argparse.ArgumentParser()
    # params for model
    parser.add_argument("--model", type=str, required=True, help="model name")
    # params for throughput
    parser.add_argument("--num_workers", type=int, default=1, help="number of workers")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--save_step", type=int, default=None, help="save step for quantized onnx")
    # params for calibration
    parser.add_argument("--num_calib_imgs", type=int, default=100, help="number of calibration images")
    parser.add_argument("--calib_method", type=str, default="MIN_MAX_ASYM", help="calibration method")
    parser.add_argument("--calib_p", type=float, default=99.999, help="calibration percentile")
    # params for devices
    parser.add_argument("--device", type=str, default="npu:0:0-1", help="device config for npus")
    # params for compilation
    parser.add_argument("--input_type", type=str, default="uint8", help="input type")
    parser.add_argument("--output_type", type=str, default="float32", help="output type")
    # params for torch model
    parser.add_argument("--fuse_conv_bn", action="store_true", help="fuse conv and bn")
    # params for onnx model
    parser.add_argument("--simplify_onnx", action="store_true", help="simplify onnx model")
    parser.add_argument("--optimize_onnx", action="store_true", help="optimize onnx model")
    # params for profiling
    parser.add_argument("--do_trace", action="store_true", help="profile model")
    parser.add_argument("--do_profile", action="store_true", help="profile model")
    # params for scheduling
    parser.add_argument(
        "--scheduling", type=str, choices=["round_robin", "queue"], default="queue",
        help="Choose the scheduling method: round_robin or queue"
    )
    # params for queue
    parser.add_argument("--input_queue_size", type=int, default=1000, help="input queue size")
    parser.add_argument("--output_queue_size", type=int, default=1000, help="output queue size")
    # params for npu_stats
    args = parser.parse_args()
    
    print("-"*50)
    # print args
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    cprofiler = None
    if args.do_profile:
        import cProfile
        cprofiler = cProfile.Profile()
        cprofiler.enable()
    
    # get task
    task = get_yolo_task(args.model)
    print(f"task: {task}")
    
    # get preprocesser
    preprocessor = YOLOPreProcessor()
    
    # get postprocessor
    for model_keyword in ANCHORS.keys():
        if model_keyword in args.model:
            print(f"model_keyword: {model_keyword}")
            anchors = ANCHORS[model_keyword]
            print(f"anchors: {anchors}")
            break
    if "u" in args.model:
        anchors = [None]
    model_cfg = {"conf_thres": CONF_THRES, "iou_thres": IOU_THRES, "anchors": anchors}
    postprocessor = getPostProcesser(task, args.model, model_cfg, None, False).postprocess_func
    
    # get data and annotation paths
    data_path = Path("/home/furiosa/work_space/val2017")
    annotation_path = TASK_TO_ANNOTATION_PATH[task]
    
    # get dataset
    dataset = MSCOCODataset(
        data_path,
        annotation_path,
        preprocessor,
        input_shape=[1, 3, 640, 640],
    )
    
    # get data loader
    data_loader = get_dataloader(
        data_path,
        annotation_path,
        preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_shape=[1, 3, 640, 640],
    )
    
    # save whether pe fusion
    pe_fusion = 1 if "-" in args.device else 0
        
    # get onnx and onnx_path paths
    # ADD: CHANGE ONNX FILE PATH, BASED ON ONNX OPTIMIZATIONS!
    ONNX_PATH = f"{ONNX_DIR}/{task}/{args.model}_simp{int(args.simplify_onnx)}_opt{int(args.optimize_onnx)}_fuse{int(args.fuse_conv_bn)}.onnx"
    QUANT_ONNX_NAME = f"{args.model}_simp{int(args.simplify_onnx)}_opt{int(args.optimize_onnx)}_fuse{int(args.fuse_conv_bn)}_{args.calib_method}_nc{args.num_calib_imgs}_bs{args.batch_size}_pefuse{pe_fusion}_i{args.input_type}_o{args.output_type}_i8"
    QUANT_ONNX_PATH = f"{QUANT_ONNX_DIR}/{task}/{QUANT_ONNX_NAME}.onnx"
    
    # torch --> onnx    
    if not Path(ONNX_PATH).exists():
        torch_to_onnx(args.model, args.fuse_conv_bn, input_shape=[1, 3, 640, 640], onnx_path=ONNX_PATH)
    else:
        print(f"ONNX model for {args.model} already exists")
    
    # optimize onnx
    if args.simplify_onnx or args.optimize_onnx:
        optimize_onnx(ONNX_PATH, simplify=args.simplify_onnx, optimize=args.optimize_onnx)
    
    # quantize onnx
    quantize_onnx(
        ONNX_PATH,
        QUANT_ONNX_NAME,
        calib_method=args.calib_method,
        num_calib_data=args.num_calib_imgs,
        input_type=args.input_type,
        output_type=args.output_type,
        save_step=args.save_step,
        calib_p=args.calib_p,
    )
    
    # create inference task, based on scheduling
    if args.scheduling == "queue":
        print(f"Using device: {args.device} for queue_runner")
        inference_task = asyncio.create_task(
            queue_runner(
                QUANT_ONNX_PATH,
                args.num_workers,
                data_loader,
                postprocessor,
                task,
                args.input_queue_size,
                args.output_queue_size,
                device=args.device
            )
        )
    else:
        print(f"Using device: {args.device} for round_robin")
        inference_task = asyncio.create_task(
            round_robin(
                QUANT_ONNX_PATH,
                args.num_workers,
                data_loader,
                postprocessor,
                task,
                device=args.device
            )
        )
    
    tik = time.time()

    # inference, either with or without tracing/profiling
    if args.do_trace:
        with open(f"{args.model}_{args.scheduling}_i{args.input_queue_size}_o{args.output_queue_size}_bs{args.batch_size}_nw{args.num_workers}.json", "w") as output:
            with profile(file=output) as profiler:
                with profiler.record("inference") as record:
                    results = await inference_task
                    inference_time = time.time() - tik
    else:
        results = await inference_task
        inference_time = time.time() - tik
    
    print(f"INFERENCE TIME: {inference_time:.2f} seconds")

    coco_result = dataset.coco.loadRes(results)
    coco_evals = {
        "object_detection": [COCOeval(dataset.coco, coco_result, "bbox")],
        "pose_estimation": [COCOeval(dataset.coco, coco_result, "keypoints")],
        "instance_segmentation": [
            COCOeval(dataset.coco, coco_result, "segm"),
            COCOeval(dataset.coco, coco_result, "bbox")
        ]
    }
    
    # print results
    for coco_eval in coco_evals[task]:
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    
    tok = time.time()
    
    print(f"END-TO-END TIME: {tok - tik:.2f} seconds")
    
    if args.do_profile:
        cprofiler.disable()
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        cprofiler.dump_stats(f"run_py_profiler_{args.model}_{args.scheduling}_bs{args.batch_size}_nw{args.num_workers}_{time_stamp}.prof")
    
    # get target accuracy all models
    target_accs = {
        "object_detection":         [TARGET_DET_ACCURACY],
        "pose_estimation":          [TARGET_POSE_ACCURACY],
        "instance_segmentation":    [TARGET_BBOX_ACCURACY, TARGET_DET_ACCURACY],
    }
    
    # check target accuracy and calculate quantization accuracy drop
    for target_acc in target_accs[task]:
        for model_name, acc in target_acc.items():            
            if model_name in args.model:
                print(f"Target Accuracy: {acc}")
                
                # Calculate and log quantization accuracy drop
                orig_acc = acc  # Assuming target accuracy is the original accuracy
                quant_acc = coco_eval.stats[0]  # Assuming quantized accuracy is AP
                accuracy_drop = 100 - 100 * (orig_acc - quant_acc) / orig_acc
                print(f"Quantization Accuracy Drop: {accuracy_drop}")
                
                break


# Use asyncio.run to execute the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())