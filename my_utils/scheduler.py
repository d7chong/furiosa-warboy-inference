import numpy as np
from tqdm import tqdm
import asyncio
import pycocotools.mask as mask_util

from furiosa.runtime import create_runner, create_queue

from my_utils import YOLO_CATEGORY_TO_COCO_CATEGORY
from my_utils.yolo_onnx_utils import xyxy2xywh


COMPILER_CONFIG = {
    'progress_mode': 'ProgressBar',
    # 'use_loop_estimator': True,
    # 'estimator_mode': 'accurate',
    # 'keep_unsignedness': True,
    'lower_tabulated_dequantize': True,
    # 'remove_lower': False,
    # 'remove_unlower': True,
    # 'max_operation_memory_ratio': 0.9,
    # 'split_policy': 'MultiOperator',
    # 'graph_lowering_scheme': 'aggressive',
    # 'split_batch_lower_unlower': True,
    # 'max_num_partitioning_axes': 4,
    # 'allow_to_partition_last_axis': True,
    # 'use_hybrid_cast': True,
    # 'bail_on_split_failure': False,
    # 'optimize_external_operators': True,
    # 'optimize_lower_unlower': True,
    # 'dont_care_bridge_cost': False,
    # 'allow_precision_error': True,
    # 'use_heuristic_lowering': True,
    # 'use_peak_memory_scheduler': True,
    # 'io_scheduling_mode': 'Greedy',
    # 'no_weight_pinning': False,
    # 'no_buffer_sharing': False,
    # 'use_atrr_lifetime': True,
    # 'target_pes': [0, 1],
    # 'mimic_sync_io': False,
    # 'use_legacy_task_model': False,
    # 'use_subroutine': True,
    # 'disable_buffer_cache': False,
    'use_program_loading': True,
    # 'first_conv_to_matmul': True,
    # 'skip_split': False,
    # 'use_repartition_conv': True,
    # 'external_operators_to_mlir': True,
    # 'use_projection_shape': True,
    # 'skip_lower': False,
    # 'panic_on_pdb_miss': False,
    # 'use_repartition': True,
    # 'permute_input': [[0, 3, 1, 2]],
    # 'dump_in_nvp': False,
    'use_tensor_dma': True,
    # 'split_width_first': True,
    # 'permute_strides': True,
    # 'allow_rf_segment': True,
    # 'abs_static_dram_addr': True
}


async def round_robin(model, num_workers, data_loader, postprocessor, task_type, device):
    try:
        async def task(
            runner, data_loader, worker_id, worker_num
        ):
            results = []
            
            for idx, (images, annotations, contexts) in enumerate(tqdm(data_loader, desc=f"worker id: {worker_id}")):
                if idx % worker_num != worker_id:
                    continue
                
                # stack images into a single numpy array: (B, 3, 640, 640)
                batch_images = np.stack(list(images)).squeeze(1)
                
                
                # convert tuples into lists
                annotations, contexts = list(annotations), list(contexts)
                
                # pad batch to batch_size
                pad_size = data_loader.batch_size - batch_images.shape[0]
                if pad_size > 0:
                    # pad images, annotations, and contexts
                     # Ensure padding is of type uint8, as stated here: https://furiosa-ai.github.io/docs/latest/en/software/performance.html#optimizing-quantize-operator
                    batch_images = np.concatenate([batch_images, np.zeros((pad_size, 3, 640, 640))], axis=0).astype(np.uint8)
                    annotations = annotations + [None] * pad_size
                    contexts = contexts + [None] * pad_size
                
                try:
                    preds = await runner.run(batch_images)
                except ValueError as e:
                    print(f"Error in worker {worker_id}: {e}")
                    continue
                
                # Split predictions into comprehensible preds for each image
                # [(N, a, b), (N, c, d)] -> [[(a, b), (c, d)], [(a, b), (c, d)], ...]
                preds = [list(split) for split in zip(*(np.split(pred, preds[0].shape[0]) for pred in preds))]
                
                # slice if padding was added
                if pad_size > 0:
                    preds = preds[:-pad_size]
                    annotations = annotations[:-pad_size]
                    contexts = contexts[:-pad_size]
                    
                # BASELINE SOLUTION
                for pred, annotation, context in zip(preds, annotations, contexts):
                    # Get outputs
                    img_shape = tuple(images[0].shape[2:])
                    if task_type in ["object_detection", "pose_estimation"]:
                        outputs = postprocessor(pred, context, img_shape)[0]
                    else: # task_type == "instance_segmentation":
                        outputs, pred_masks = postprocessor(pred, context, img_shape)[0]
                    
                    # Extract ouptuts
                    if task_type != "pose_estimation":
                        bboxes = xyxy2xywh(outputs[:, :4])
                        bboxes[:, :2] -= bboxes[:, 2:] / 2
                    if task_type == "pose_estimation":
                        keypoints = outputs[:, 5:]
                    if task_type == "instance_segmentation":
                        rles = [
                            mask_util.encode(
                                np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
                            )[0]
                            for mask in pred_masks
                        ]
                        
                    # Evaluate each output    
                    for i, output in enumerate(outputs):
                        result = {
                            "image_id": annotation["id"],
                            "score": round(output[4], 5),
                        }

                        if task_type == "object_detection":
                            result["category_id"] = YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])]
                            result["bbox"] = [round(x, 3) for x in bboxes[i]]
                        elif task_type == "instance_segmentation":
                            result["category_id"] = YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])]
                            result["bbox"] = [round(x, 3) for x in bboxes[i]]
                            result["segmentation"] = rles[i]
                        else:  # task_type == "pose_estimation":
                            result["category_id"] = 1
                            result["keypoints"] = keypoints[i].tolist()

                        results.append(result)
            return results
        
        async with create_runner(model, worker_num=num_workers, batch_size=data_loader.batch_size, device=device, compiler_config=None) as runner:
            results = await asyncio.gather(
                *(
                    task(runner, data_loader, idx, num_workers)
                    for idx in range(num_workers)
                )
            )
            
            results = sum(results, [])
            
            return results
    except Exception as e:
        print(f"An error occurred during inference: {e}")


async def queue_runner(model, num_workers, data_loader, postprocessor, task_type, input_queue_size, output_queue_size, device):
    results = []
    
    # Log the device being used
    print(f"Creating queue with device: {device}")
    
    submitter, receiver = await create_queue(
        model,
        worker_num=num_workers,
        batch_size=data_loader.batch_size,
        device=device,
        input_queue_size=input_queue_size,
        output_queue_size=output_queue_size,
        compiler_config=COMPILER_CONFIG,
    )
    
    async def submit_tasks():
        async with submitter:
            for images, annotations, contexts in tqdm(data_loader, desc="Submitting"):
                # Get the images, annotations, and contexts
                batch_images = np.stack(list(images)).squeeze(1)
                annotations, contexts = list(annotations), list(contexts)
                
                # Pad the batch if it's smaller than batch_size
                pad_size = data_loader.batch_size - batch_images.shape[0]
                if pad_size > 0:
                    # Ensure padding is of type uint8, as stated here: https://furiosa-ai.github.io/docs/latest/en/software/performance.html#optimizing-quantize-operator
                    batch_images = np.pad(batch_images, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0).astype(np.uint8)
                    annotations.extend([None] * pad_size)
                    contexts.extend([None] * pad_size)
                
                await submitter.submit(batch_images, context=(annotations, contexts, pad_size))  # Wrapped in a list

    async def receive_tasks():
        async with receiver:
            async for context, preds in receiver:
                # unpack context
                annotations, contexts, pad_size = context
                
                # Split predictions into comprehensible preds for each image
                # preds = [list(split) for split in zip(*(np.split(pred, preds[0].shape[0]) for pred in preds))]
                preds = [[pred.astype(np.float32) for pred in image_preds] for image_preds in zip(*(np.split(pred, preds[0].shape[0]) for pred in preds))]
                
                # slice if padding was added
                if pad_size > 0:
                    preds = preds[:-pad_size]
                    annotations = annotations[:-pad_size]
                    contexts = contexts[:-pad_size]
                
                # assert len(preds) == len(annotations) == len(contexts)
                # print(f"type(preds[0][0].dtype): {type(preds[0][0].dtype)}")
                
                # Process each prediction
                img_shape = (640, 640)  # Assuming fixed image shape
                for pred, annotation, context in zip(preds, annotations, contexts):
                    if task_type in ["object_detection", "pose_estimation"]:
                        outputs = postprocessor(pred, context, img_shape)[0]
                        
                    else:  # task_type == "instance_segmentation":
                        outputs, pred_masks = postprocessor(pred, context, img_shape)[0]
                    
                    # extract relevant info from outputs
                    if task_type != "pose_estimation":
                        bboxes = xyxy2xywh(outputs[:, :4])
                        bboxes[:, :2] -= bboxes[:, 2:] / 2
                    elif task_type == "pose_estimation":
                        keypoints = outputs[:, 5:]
                    else: #task_type == "instance_segmentation":
                        rles = [
                            mask_util.encode(
                                np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
                            )[0]
                            for mask in pred_masks
                        ]
                            
                    # eval each output
                    for i, output in enumerate(outputs):
                        result = {
                            "image_id": annotation["id"],
                            "score": round(output[4], 5),
                        }

                        if task_type == "object_detection":
                            result["category_id"] = YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])]
                            result["bbox"] = np.round(bboxes[i], 3).tolist()
                        elif task_type == "instance_segmentation":
                            result["category_id"] = YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])]
                            result["bbox"] = np.round(bboxes[i], 3).tolist()
                            result["segmentation"] = rles[i]
                        else:  # task_type == "pose_estimation":
                            result["category_id"] = 1
                            result["keypoints"] = keypoints[i].tolist()

                        results.append(result)
    
    await asyncio.gather(submit_tasks(), receive_tasks())
    return results