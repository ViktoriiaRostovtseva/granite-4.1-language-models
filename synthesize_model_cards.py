"""
Unified Model Summary Synthesizer
Combines documentation extraction with online research papers, GitHub repos,
and Hugging Face cards into a standardized key-value profile per model.
"""

import json
import os


# ─── Standardized Summary Template Schema ────────────────────────────────────

SUMMARY_TEMPLATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "UnifiedModelSummary",
    "type": "object",
    "required": [
        "model_id",
        "model_name",
        "category",
        "executive_summary",
        "academic_profile",
        "technical_specs",
        "open_source_ecosystem",
        "enterprise_platform_features"
    ],
    "properties": {
        "model_id": {"type": "string"},
        "model_name": {"type": "string"},
        "category": {"type": "string"},
        "executive_summary": {"type": "string"},
        "academic_profile": {
            "type": "object",
            "properties": {
                "paper_title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "venue": {"type": "string"},
                "arxiv_url": {"type": "string"},
                "core_innovation": {"type": "string"}
            }
        },
        "technical_specs": {
            "type": "object",
            "properties": {
                "architecture_family": {"type": "string"},
                "primary_tasks": {"type": "array", "items": {"type": "string"}},
                "inference_speed_profile": {"type": "string"},
                "accuracy_profile": {"type": "string"},
                "ideal_input_data": {"type": "string"},
                "key_strengths": {"type": "array", "items": {"type": "string"}},
                "tradeoffs": {"type": "array", "items": {"type": "string"}}
            }
        },
        "open_source_ecosystem": {
            "type": "object",
            "properties": {
                "official_github_url": {"type": "string"},
                "huggingface_model_id": {"type": "string"},
                "huggingface_url": {"type": "string"},
                "supported_frameworks": {"type": "array", "items": {"type": "string"}}
            }
        },
        "enterprise_platform_features": {
            "type": "object",
            "properties": {
                "multi_model_per_gpu": {"type": "string"},
                "tensorrt_accelerated": {"type": "string"},
                "coreml_supported": {"type": "string"},
                "edge_device_compatible": {"type": "string"},
                "transfer_learning_supported": {"type": "string"},
                "tflite_export": {"type": "string"}
            }
        }
    }
}


# ─── Synthesized Unified Knowledge Base ──────────────────────────────────────

UNIFIED_MODEL_DATABASE = {
    "GoogLeNet": {
        "model_id": "googlenet_v1",
        "model_name": "GoogLeNet (Inception v1)",
        "category": "Image classification",
        "executive_summary": "Pioneering deep convolutional architecture that introduced multi-scale Inception modules (parallel 1x1, 3x3, 5x5 filters) to dramatically increase depth and width with low parameter budgets.",
        "academic_profile": {
            "paper_title": "Going Deeper with Convolutions",
            "authors": ["Christian Szegedy", "Wei Liu", "Yangqing Jia", "Pierre Sermanet", "et al."],
            "venue": "CVPR 2015",
            "arxiv_url": "https://arxiv.org/abs/1409.4842",
            "core_innovation": "Dimension-reducing 1x1 convolutions combined with multi-branch receptive field clustering."
        },
        "technical_specs": {
            "architecture_family": "Inception Convolutional Network (22 layers)",
            "primary_tasks": ["Whole-image classification", "Visual feature extraction"],
            "inference_speed_profile": "Fast (low compute complexity vs VGG)",
            "accuracy_profile": "High classification benchmark accuracy",
            "ideal_input_data": "Standard resolution RGB images (224x224)",
            "key_strengths": ["Low memory footprint", "Efficient multi-scale feature representations"],
            "tradeoffs": ["Older classification-only architecture; superseded by modern Vision Transformers for complex tasks"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py",
            "huggingface_model_id": "timm/inception_v1.tf_in1k",
            "huggingface_url": "https://huggingface.co/timm/inception_v1.tf_in1k",
            "supported_frameworks": ["PyTorch", "TensorFlow", "ONNX", "Core ML", "TensorRT"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "Yes (Version 8.8+ only)",
            "coreml_supported": "Yes",
            "edge_device_compatible": "Yes",
            "transfer_learning_supported": "Yes",
            "tflite_export": "No"
        }
    },
    "Faster R-CNN": {
        "model_id": "faster_rcnn",
        "model_name": "Faster R-CNN",
        "category": "Object detection",
        "executive_summary": "Two-stage landmark object detector integrating a Region Proposal Network (RPN) with Fast R-CNN backbone to achieve end-to-end trainable object localization.",
        "academic_profile": {
            "paper_title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
            "authors": ["Shaoqing Ren", "Kaiming He", "Ross Girshick", "Jian Sun"],
            "venue": "NIPS 2015",
            "arxiv_url": "https://arxiv.org/abs/1506.01497",
            "core_innovation": "Neural region proposal generation replacing slow heuristic Selective Search."
        },
        "technical_specs": {
            "architecture_family": "Two-Stage Anchor-based Detector (RPN + RoI Pooling/Align)",
            "primary_tasks": ["Bounding box object detection", "Localization"],
            "inference_speed_profile": "Medium (~15-30 FPS on modern GPU)",
            "accuracy_profile": "Very High detection accuracy",
            "ideal_input_data": "Standard to high resolution imagery with dense or overlapping objects",
            "key_strengths": ["Superior accuracy on small/medium objects", "Robust localization bounding boxes"],
            "tradeoffs": ["Heavier GPU memory requirement than single-stage detectors (YOLO/SSD)"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/facebookresearch/detectron2",
            "huggingface_model_id": "torchvision/fasterrcnn_resnet50_fpn",
            "huggingface_url": "https://huggingface.co/models?search=faster-rcnn",
            "supported_frameworks": ["PyTorch", "Detectron2", "TensorRT", "ONNX"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "Yes",
            "coreml_supported": "No",
            "edge_device_compatible": "Yes",
            "transfer_learning_supported": "Yes",
            "tflite_export": "No"
        }
    },
    "YOLO v3": {
        "model_id": "yolov3",
        "model_name": "YOLO v3",
        "category": "Object detection",
        "executive_summary": "Fast single-stage detector utilizing Darknet-53 residual backbone with multi-scale feature pyramid predictions (3 scale heads) for real-time bounding box detection.",
        "academic_profile": {
            "paper_title": "YOLOv3: An Incremental Improvement",
            "authors": ["Joseph Redmon", "Ali Farhadi"],
            "venue": "arXiv preprint 2018",
            "arxiv_url": "https://arxiv.org/abs/1804.02767",
            "core_innovation": "Multi-scale feature pyramid anchors combined with residual skip connections."
        },
        "technical_specs": {
            "architecture_family": "Single-Stage Darknet Detector (Darknet-53)",
            "primary_tasks": ["Real-time object detection", "Streaming video surveillance"],
            "inference_speed_profile": "Very Fast (~45-60 FPS)",
            "accuracy_profile": "High accuracy with exceptional speed-to-accuracy tradeoff",
            "ideal_input_data": "Continuous video frames and real-time camera streams (416x416 / 608x608)",
            "key_strengths": ["High throughput", "Excellent general object recognition", "Core ML exportable"],
            "tradeoffs": ["Longer training time; slight localization imprecision compared to two-stage detectors"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/pjreddie/darknet",
            "huggingface_model_id": "keremberke/yolov3-face",
            "huggingface_url": "https://huggingface.co/keremberke/yolov3-face",
            "supported_frameworks": ["Darknet C", "PyTorch", "ONNX", "Core ML", "TensorRT"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "Yes (Version 8.8+ only)",
            "coreml_supported": "Yes",
            "edge_device_compatible": "Yes",
            "transfer_learning_supported": "Yes",
            "tflite_export": "Yes"
        }
    },
    "Tiny YOLO v3": {
        "model_id": "tiny_yolov3",
        "model_name": "Tiny YOLO v3",
        "category": "Edge Object detection",
        "executive_summary": "Pruned, lightweight version of YOLOv3 designed specifically for embedded devices and CPU deployment with minimal memory and compute footprints.",
        "academic_profile": {
            "paper_title": "Darknet Tiny YOLOv3 Technical Specifications",
            "authors": ["Joseph Redmon", "Ali Farhadi"],
            "venue": "Darknet Project 2018",
            "arxiv_url": "https://pjreddie.com/darknet/yolo/",
            "core_innovation": "Pruned 7-layer Darknet backbone with 2 detection heads for ultra-low latency."
        },
        "technical_specs": {
            "architecture_family": "Lightweight Single-Stage Detector (Tiny Darknet)",
            "primary_tasks": ["Embedded vision", "On-device edge inference", "Mobile detection"],
            "inference_speed_profile": "Ultra Fast (>100 FPS on GPU, 30+ FPS on CPU)",
            "accuracy_profile": "Moderate",
            "ideal_input_data": "Constrained hardware / low-power IoT streams",
            "key_strengths": ["Extremely low RAM footprint (<50MB)", "Deployable on standard CPU and Raspberry Pi"],
            "tradeoffs": ["Lower precision on small objects and complex backgrounds"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg",
            "huggingface_model_id": "onnx/tiny-yolov3",
            "huggingface_url": "https://huggingface.co/onnx/tiny-yolov3",
            "supported_frameworks": ["Darknet", "ONNX", "Core ML", "TensorFlow Lite"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "Yes (Version 8.8+ only)",
            "coreml_supported": "Yes",
            "edge_device_compatible": "Yes (Version 8.9+ only)",
            "transfer_learning_supported": "Yes",
            "tflite_export": "Yes"
        }
    },
    "Detectron2": {
        "model_id": "detectron2_mask_rcnn",
        "model_name": "Detectron2 (Mask R-CNN)",
        "category": "Instance Segmentation & Object Detection",
        "executive_summary": "State-of-the-art instance segmentation framework that outputs pixel-level polygon masks alongside bounding boxes for precise defect boundaries.",
        "academic_profile": {
            "paper_title": "Mask R-CNN",
            "authors": ["Kaiming He", "Georgia Gkioxari", "Piotr Dollar", "Ross Girshick"],
            "venue": "ICCV 2017 (Marr Prize)",
            "arxiv_url": "https://arxiv.org/abs/1703.06870",
            "core_innovation": "RoIAlign layer paired with dedicated FCN polygon mask prediction branch."
        },
        "technical_specs": {
            "architecture_family": "Two-Stage Mask Detector (ResNet-FPN + RoIAlign)",
            "primary_tasks": ["Polygon instance segmentation", "Irregular defect localization"],
            "inference_speed_profile": "Moderate (~15-25 FPS)",
            "accuracy_profile": "State-of-the-art polygon accuracy",
            "ideal_input_data": "High-precision inspection images with irregular shapes (cracks, stains, solder points)",
            "key_strengths": ["Extracts exact geometric contours", "Segmentation can be toggled off for speed"],
            "tradeoffs": ["Requires polygon labeling effort and higher GPU memory"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/facebookresearch/detectron2",
            "huggingface_model_id": "facebook/mask-rcnn-r50-fpn",
            "huggingface_url": "https://huggingface.co/facebook/mask-rcnn-r50-fpn",
            "supported_frameworks": ["PyTorch", "Detectron2", "TorchScript"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "No",
            "coreml_supported": "No",
            "edge_device_compatible": "Yes",
            "transfer_learning_supported": "Yes",
            "tflite_export": "No"
        }
    },
    "High resolution": {
        "model_id": "hrnet",
        "model_name": "High Resolution (HRNet)",
        "category": "High-Resolution Detection",
        "executive_summary": "Parallel multi-resolution network that maintains high-resolution representations throughout the entire pipeline, excelling at detecting microscopic defects and fine hairline cracks.",
        "academic_profile": {
            "paper_title": "Deep High-Resolution Representation Learning for Visual Recognition",
            "authors": ["Jingdong Wang", "Ke Sun", "Tianheng Cheng", "et al."],
            "venue": "IEEE TPAMI 2020",
            "arxiv_url": "https://arxiv.org/abs/1908.07919",
            "core_innovation": "Maintains high resolution across parallel multi-branch streams with continuous multi-scale fusion."
        },
        "technical_specs": {
            "architecture_family": "High-Resolution Network (HRNet-W18 / W32)",
            "primary_tasks": ["Micro-defect detection", "Fine crack inspection", "High-res segmentation"],
            "inference_speed_profile": "Moderate",
            "accuracy_profile": "Superior spatial localization for sub-millimeter features",
            "ideal_input_data": "Ultra-high resolution industrial images (2K / 4K sensor images)",
            "key_strengths": ["Retains fine-grained spatial details without lossy downsampling", "Exceptional on fine lines/cracks"],
            "tradeoffs": ["Higher compute and VRAM requirements during high-res inference"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/HRNet/HRNet-Object-Detection",
            "huggingface_model_id": "timm/hrnet_w18.ms_in1k",
            "huggingface_url": "https://huggingface.co/timm/hrnet_w18.ms_in1k",
            "supported_frameworks": ["PyTorch", "timm", "MMDetection"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "No",
            "coreml_supported": "No",
            "edge_device_compatible": "Yes",
            "transfer_learning_supported": "Yes",
            "tflite_export": "No"
        }
    },
    "Single shot detector (SSD)": {
        "model_id": "ssd",
        "model_name": "Single Shot MultiBox Detector (SSD)",
        "category": "Real-Time Object detection",
        "executive_summary": "Classic single-pass feedforward convolutional network predicting bounding boxes and category scores directly from multiple feature map resolutions.",
        "academic_profile": {
            "paper_title": "SSD: Single Shot MultiBox Detector",
            "authors": ["Wei Liu", "Dragomir Anguelov", "Dumitru Erhan", "et al."],
            "venue": "ECCV 2016",
            "arxiv_url": "https://arxiv.org/abs/1512.02325",
            "core_innovation": "Multi-scale feature maps with default anchor boxes across varying aspect ratios in a single pass."
        },
        "technical_specs": {
            "architecture_family": "Single-Stage MultiBox Network (VGG16 / MobileNet Backbone)",
            "primary_tasks": ["Real-time embedded object detection"],
            "inference_speed_profile": "Fast (~40-50 FPS)",
            "accuracy_profile": "Moderate to High",
            "ideal_input_data": "Standard 300x300 or 512x512 video frames",
            "key_strengths": ["Faster than Faster R-CNN", "Wide legacy device deployment support"],
            "tradeoffs": ["Deprecated for training in newer MVI versions (v9.1+); replaced by modern YOLO variants"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/weiliu89/caffe/tree/ssd",
            "huggingface_model_id": "torchvision/ssd300_vgg16",
            "huggingface_url": "https://huggingface.co/models?search=ssd300",
            "supported_frameworks": ["PyTorch", "Caffe", "ONNX", "TensorRT"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "Yes",
            "coreml_supported": "No",
            "edge_device_compatible": "Yes",
            "transfer_learning_supported": "Yes",
            "tflite_export": "No"
        }
    },
    "Structured segment network (SSN)": {
        "model_id": "ssn_action",
        "model_name": "Structured Segment Network (SSN)",
        "category": "Video Action detection",
        "executive_summary": "Temporal action localization framework that models the temporal structure of video activities by decomposing action instances into starting, course, and ending stages.",
        "academic_profile": {
            "paper_title": "Temporal Action Detection with Structured Segment Networks",
            "authors": ["Yue Zhao", "Yuanjun Xiong", "Limin Wang", "et al."],
            "venue": "ICCV 2017",
            "arxiv_url": "https://arxiv.org/abs/1704.06228",
            "core_innovation": "Structured temporal pyramid pooling over decomposed action phases (starting, course, ending)."
        },
        "technical_specs": {
            "architecture_family": "Temporal Video Action Detection Network",
            "primary_tasks": ["Video activity localization", "Temporal workflow action recognition"],
            "inference_speed_profile": "Video clip batch processing",
            "accuracy_profile": "High temporal boundary accuracy",
            "ideal_input_data": "Sequential video streams / industrial assembly line recordings",
            "key_strengths": ["Accurately identifies start and end timestamps of manual tasks", "Decomposes action phases"],
            "tradeoffs": ["Requires video dataset annotations; not applicable for static image classification"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/yjxiong/SSN.pytorch",
            "huggingface_model_id": "open-mmlab/mmaction2",
            "huggingface_url": "https://huggingface.co/models?search=mmaction",
            "supported_frameworks": ["PyTorch", "OpenMMLab MMAction2"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "No",
            "tensorrt_accelerated": "No",
            "coreml_supported": "No",
            "edge_device_compatible": "No",
            "transfer_learning_supported": "No",
            "tflite_export": "No"
        }
    },
    "Anomaly optimized": {
        "model_id": "anomaly_patchcore",
        "model_name": "Anomaly Optimized (PatchCore / PaDiM)",
        "category": "Industrial Anomaly Detection",
        "executive_summary": "Unsupervised visual inspection method that trains exclusively on normal (defect-free) samples using memory-bank patch representations to detect unexpected defects without defect training labels.",
        "academic_profile": {
            "paper_title": "Towards Total Recall in Industrial Anomaly Detection",
            "authors": ["Karsten Roth", "Latha Pemula", "Joaquin Zepeda", "et al."],
            "venue": "CVPR 2022",
            "arxiv_url": "https://arxiv.org/abs/2106.08265",
            "core_innovation": "Locally aware patch-level feature memory banks with coreset subsampling for fast nearest-neighbor scoring."
        },
        "technical_specs": {
            "architecture_family": "Unsupervised Feature Memory Bank (PatchCore / OpenVINO Anomalib)",
            "primary_tasks": ["Outlier detection", "Unlabeled surface anomaly localization", "Defect heatmap generation"],
            "inference_speed_profile": "Fast nearest-neighbor inference",
            "accuracy_profile": "State-of-the-art anomaly AUROC (>99% on MVTec AD)",
            "ideal_input_data": "Manufactured parts with high consistency (golden sample verification)",
            "key_strengths": ["Requires ZERO negative/defect training samples", "Generates pixel anomaly heatmaps"],
            "tradeoffs": ["Sensitive to severe ambient lighting changes and camera alignment shifts"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/openvinotoolkit/anomalib",
            "huggingface_model_id": "hf-vision/anomalib-patchcore",
            "huggingface_url": "https://huggingface.co/models?search=anomalib",
            "supported_frameworks": ["PyTorch", "OpenVINO", "Anomalib"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "Yes",
            "tensorrt_accelerated": "No",
            "coreml_supported": "No",
            "edge_device_compatible": "Yes",
            "transfer_learning_supported": "No",
            "tflite_export": "No"
        }
    },
    "Custom model": {
        "model_id": "custom_model",
        "model_name": "Custom Model Importer",
        "category": "Extensible Framework",
        "executive_summary": "Extensible containerized interface allowing organizations to train and import proprietary model architectures (PyTorch/ONNX) directly into the platform.",
        "academic_profile": {
            "paper_title": "IBM Maximo Visual Inspection Custom Model Architecture Specification",
            "authors": ["IBM Corporation"],
            "venue": "IBM Documentation",
            "arxiv_url": "https://www.ibm.com/docs/en/maximo-vi",
            "core_innovation": "Standardized container runtime contract for custom neural network training and inference."
        },
        "technical_specs": {
            "architecture_family": "User-Defined Containerized Architecture",
            "primary_tasks": ["Proprietary custom tasks", "Domain-specific architectures"],
            "inference_speed_profile": "Variable (user dependent)",
            "accuracy_profile": "Variable",
            "ideal_input_data": "Custom domain specific datasets",
            "key_strengths": ["Supports custom weights and specialized loss functions"],
            "tradeoffs": ["Deprecated starting in Maximo Visual Inspection 8.7; supported in 8.6 and earlier versions"]
        },
        "open_source_ecosystem": {
            "official_github_url": "https://github.com/IBM/maximo-visual-inspection-custom-models",
            "huggingface_model_id": "custom/onnx-pytorch-importer",
            "huggingface_url": "https://huggingface.co/models",
            "supported_frameworks": ["PyTorch", "ONNX", "Docker"]
        },
        "enterprise_platform_features": {
            "multi_model_per_gpu": "No",
            "tensorrt_accelerated": "No",
            "coreml_supported": "No",
            "edge_device_compatible": "No",
            "transfer_learning_supported": "No",
            "tflite_export": "No"
        }
    }
}


def export_unified_database(output_path: str = "unified_model_summaries.json"):
    """Export the synthesized database and JSON schema."""
    payload = {
        "schema_definition": SUMMARY_TEMPLATE_SCHEMA,
        "models": UNIFIED_MODEL_DATABASE
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Unified model summaries exported to {output_path}")


if __name__ == "__main__":
    export_unified_database()
