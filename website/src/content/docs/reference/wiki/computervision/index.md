---
title: "Computer Vision"
description: "All 266 public types in the AiDotNet.computervision namespace, organized by kind."
section: "API Reference"
---

**266** public types in this namespace, organized by kind.

## Models & Types (154)

| Type | Summary |
|:-----|:--------|
| [`AnatomicalStructure`](/docs/reference/wiki/computervision/anatomicalstructure/) | Metadata for a single segmented anatomical structure. |
| [`AnchorGenerator<T>`](/docs/reference/wiki/computervision/anchorgenerator/) | Generates anchor boxes for object detection models. |
| [`AnchorMatchResult<T>`](/docs/reference/wiki/computervision/anchormatchresult/) | Result of anchor-to-ground-truth matching. |
| [`AnchorMatcher<T>`](/docs/reference/wiki/computervision/anchormatcher/) | Matches anchor boxes to ground truth boxes for training object detectors. |
| [`BatchDetectionResult<T>`](/docs/reference/wiki/computervision/batchdetectionresult/) | Represents a batch of detection results for multiple images. |
| [`BiFPN<T>`](/docs/reference/wiki/computervision/bifpn/) | Bidirectional Feature Pyramid Network (BiFPN) with weighted feature fusion. |
| [`BiomedParse<T>`](/docs/reference/wiki/computervision/biomedparse/) | BiomedParse: Biomedical image parsing with text prompts. |
| [`BitmapFont<T>`](/docs/reference/wiki/computervision/bitmapfont/) | A simple 5x7 bitmap font for rendering text on images. |
| [`ByteTrack<T>`](/docs/reference/wiki/computervision/bytetrack/) | ByteTrack: Multi-Object Tracking by Associating Every Detection Box. |
| [`CATSeg<T>`](/docs/reference/wiki/computervision/catseg/) | CAT-Seg: Cost Aggregation for open-vocabulary semantic segmentation. |
| [`CIoULoss<T>`](/docs/reference/wiki/computervision/ciouloss/) | Complete Intersection over Union (CIoU) loss for bounding box regression. |
| [`CRAFT<T>`](/docs/reference/wiki/computervision/craft/) | CRAFT (Character Region Awareness for Text) detector. |
| [`CRNN<T>`](/docs/reference/wiki/computervision/crnn/) | CRNN (Convolutional Recurrent Neural Network) for text recognition. |
| [`CSPDarknet<T>`](/docs/reference/wiki/computervision/cspdarknet/) | CSP-Darknet backbone network used in YOLO family models (v5, v7, v8). |
| [`CUPS<T>`](/docs/reference/wiki/computervision/cups/) | CUPS: Unified Panoptic Segmentation with Comprehensive Use of Pixels and Semantics. |
| [`CascadeRCNN<T>`](/docs/reference/wiki/computervision/cascadercnn/) | Cascade R-CNN - Multi-stage object detection with progressive refinement. |
| [`CombinedMaskLoss<T>`](/docs/reference/wiki/computervision/combinedmaskloss/) | Combined mask loss using BCE + Dice. |
| [`Concerto<T>`](/docs/reference/wiki/computervision/concerto/) | Concerto: Hybrid Mamba-Transformer backbone for 3D point clouds. |
| [`DBNet<T>`](/docs/reference/wiki/computervision/dbnet/) | DBNet (Differentiable Binarization Network) text detector. |
| [`DETRSetLoss<T>`](/docs/reference/wiki/computervision/detrsetloss/) | DETR Set Prediction Loss with Hungarian Matching for end-to-end object detection. |
| [`DETR<T>`](/docs/reference/wiki/computervision/detr/) | DETR (DEtection TRansformer) - End-to-end object detection with transformers. |
| [`DEVA<T>`](/docs/reference/wiki/computervision/deva/) | DEVA: Tracking Anything with Decoupled Video Segmentation. |
| [`DINO<T>`](/docs/reference/wiki/computervision/dino/) | DINO (DETR with Improved deNoising anchOr boxes) - State-of-the-art DETR variant. |
| [`DIoULoss<T>`](/docs/reference/wiki/computervision/diouloss/) | Distance Intersection over Union (DIoU) loss for bounding box regression. |
| [`DIoUNMS<T>`](/docs/reference/wiki/computervision/diounms/) | Implements Distance-IoU based Non-Maximum Suppression for improved localization. |
| [`DeepSORT<T>`](/docs/reference/wiki/computervision/deepsort/) | DeepSORT (Deep SORT) tracking with appearance features. |
| [`DetectionResult<T>`](/docs/reference/wiki/computervision/detectionresult/) | Contains the results of object detection on an image. |
| [`DetectionStatistics<T>`](/docs/reference/wiki/computervision/detectionstatistics/) | Statistics about detection results. |
| [`DetectionVisualizer<T>`](/docs/reference/wiki/computervision/detectionvisualizer/) | Visualizes object detection results on images. |
| [`Detection<T>`](/docs/reference/wiki/computervision/detection/) | Represents a single detected object. |
| [`DiffCutSegmentation<T>`](/docs/reference/wiki/computervision/diffcutsegmentation/) | DiffCut: Diffusion-based zero-shot segmentation via graph cuts. |
| [`DiffCut<T>`](/docs/reference/wiki/computervision/diffcut/) | DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut. |
| [`DiffSeg<T>`](/docs/reference/wiki/computervision/diffseg/) | DiffSeg: Unsupervised Semantic Segmentation from Diffusion Model Self-Attention Maps. |
| [`DocumentBlock<T>`](/docs/reference/wiki/computervision/documentblock/) | A block of text in a document (paragraph, header, etc.). |
| [`DocumentBlock<T>`](/docs/reference/wiki/computervision/documentblock-2/) | A block in a document (paragraph, table, figure, etc.). |
| [`DocumentLayoutResult<T>`](/docs/reference/wiki/computervision/documentlayoutresult/) | Result of document layout analysis. |
| [`DocumentLine<T>`](/docs/reference/wiki/computervision/documentline/) | A line of text in a document. |
| [`DocumentOCRResult<T>`](/docs/reference/wiki/computervision/documentocrresult/) | Result of document OCR. |
| [`DocumentReader<T>`](/docs/reference/wiki/computervision/documentreader/) | Document reader for OCR with layout analysis. |
| [`EAST<T>`](/docs/reference/wiki/computervision/east/) | EAST (Efficient and Accurate Scene Text) detector. |
| [`EdgeSAM<T>`](/docs/reference/wiki/computervision/edgesam/) | EdgeSAM: Prompt-in-the-Loop Distillation for SAM on edge devices. |
| [`EfficientNet<T>`](/docs/reference/wiki/computervision/efficientnet/) | EfficientNet backbone for efficient feature extraction. |
| [`EfficientSAM<T>`](/docs/reference/wiki/computervision/efficientsam/) | EfficientSAM: Leveraged Masked Image Pretraining for efficient SAM. |
| [`EfficientTAM<T>`](/docs/reference/wiki/computervision/efficienttam/) | EfficientTAM: Efficient Track Anything Model for edge video segmentation. |
| [`EoMT<T>`](/docs/reference/wiki/computervision/eomt/) | EoMT: Encoder-only Mask Transformer for universal image segmentation. |
| [`FPN<T>`](/docs/reference/wiki/computervision/fpn/) | Feature Pyramid Network (FPN) for multi-scale feature fusion. |
| [`FastSAM<T>`](/docs/reference/wiki/computervision/fastsam/) | FastSAM: Fast Segment Anything Model based on YOLOv8-Seg. |
| [`FasterRCNN<T>`](/docs/reference/wiki/computervision/fasterrcnn/) | Faster R-CNN - Two-stage object detection with region proposal network. |
| [`GIoULoss<T>`](/docs/reference/wiki/computervision/giouloss/) | Generalized Intersection over Union (GIoU) loss for bounding box regression. |
| [`GLaMM<T>`](/docs/reference/wiki/computervision/glamm/) | GLaMM: Grounding Large Multimodal Model for pixel-level understanding. |
| [`GroundedSAM2<T>`](/docs/reference/wiki/computervision/groundedsam2/) | Grounded SAM 2: Text-grounded tracking and segmentation. |
| [`GroundingToken`](/docs/reference/wiki/computervision/groundingtoken/) | A grounding token linking a text span to an image region. |
| [`InstanceMask<T>`](/docs/reference/wiki/computervision/instancemask/) | Represents a single instance with bounding box and segmentation mask. |
| [`InstanceSegmentationResult<T>`](/docs/reference/wiki/computervision/instancesegmentationresult/) | Result of instance segmentation. |
| [`InternImage<T>`](/docs/reference/wiki/computervision/internimage/) | InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions. |
| [`KMaXDeepLab<T>`](/docs/reference/wiki/computervision/kmaxdeeplab/) | kMaX-DeepLab: k-means Mask Transformer for panoptic segmentation. |
| [`LISA<T>`](/docs/reference/wiki/computervision/lisa/) | LISA: Reasoning Segmentation via Large Language Model. |
| [`Mask2Former<T>`](/docs/reference/wiki/computervision/mask2former/) | Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation. |
| [`MaskAdapter<T>`](/docs/reference/wiki/computervision/maskadapter/) | Mask-Adapter: Adding SAM to open-vocabulary segmentation via mask prediction. |
| [`MaskBCELoss<T>`](/docs/reference/wiki/computervision/maskbceloss/) | Binary Cross-Entropy loss for mask prediction. |
| [`MaskDINO<T>`](/docs/reference/wiki/computervision/maskdino/) | Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation. |
| [`MaskDiceLoss<T>`](/docs/reference/wiki/computervision/maskdiceloss/) | Dice loss for mask prediction. |
| [`MaskFocalLoss<T>`](/docs/reference/wiki/computervision/maskfocalloss/) | Focal loss for mask prediction (addresses class imbalance). |
| [`MaskHead<T>`](/docs/reference/wiki/computervision/maskhead/) | Mask prediction head for instance segmentation. |
| [`MaskRCNN<T>`](/docs/reference/wiki/computervision/maskrcnn/) | Mask R-CNN for instance segmentation. |
| [`MaskVisualizer<T>`](/docs/reference/wiki/computervision/maskvisualizer/) | Visualizes instance segmentation results on images. |
| [`MedNeXt<T>`](/docs/reference/wiki/computervision/mednext/) | MedNeXt: Transformer-driven scaling of ConvNets for medical segmentation. |
| [`MedSAM2<T>`](/docs/reference/wiki/computervision/medsam2/) | MedSAM-2: SAM 2 adapted for medical image and video segmentation. |
| [`MedSAM<T>`](/docs/reference/wiki/computervision/medsam/) | MedSAM: Segment Anything in Medical Images. |
| [`MedSegDiffV2Segmentation<T>`](/docs/reference/wiki/computervision/medsegdiffv2segmentation/) | MedSegDiff-V2 Segmentation: Diffusion-based medical image segmentation pipeline. |
| [`MedSegDiffV2<T>`](/docs/reference/wiki/computervision/medsegdiffv2/) | MedSegDiff-V2: Spectrum-space diffusion for medical segmentation. |
| [`MedicalSegmentationOutput<T>`](/docs/reference/wiki/computervision/medicalsegmentationoutput/) | Output for medical image segmentation with volumetric support and clinical metadata. |
| [`MobileSAM<T>`](/docs/reference/wiki/computervision/mobilesam/) | MobileSAM: Faster Segment Anything with TinyViT encoder. |
| [`NMS<T>`](/docs/reference/wiki/computervision/nms/) | Implements Non-Maximum Suppression (NMS) algorithms for removing duplicate detections. |
| [`NnUNet<T>`](/docs/reference/wiki/computervision/nnunet/) | nnU-Net: Self-configuring framework for medical image segmentation. |
| [`OCRResult<T>`](/docs/reference/wiki/computervision/ocrresult/) | Result of OCR processing. |
| [`OCRVisualizer<T>`](/docs/reference/wiki/computervision/ocrvisualizer/) | Visualizes OCR (Optical Character Recognition) results on images. |
| [`ODISESegmentation<T>`](/docs/reference/wiki/computervision/odisesegmentation/) | ODISE Segmentation: Panoptic segmentation via diffusion model features. |
| [`ODISE<T>`](/docs/reference/wiki/computervision/odise/) | ODISE: Open-vocabulary DIffusion-based panoptic SEgmentation. |
| [`OMGLLaVA<T>`](/docs/reference/wiki/computervision/omgllava/) | OMG-LLaVA: Bridging Image-Level, Object-Level, and Pixel-Level understanding. |
| [`OMGSeg<T>`](/docs/reference/wiki/computervision/omgseg/) | OMG-Seg: Is One Model Good Enough For All Segmentation? |
| [`ObjectTrackSummary`](/docs/reference/wiki/computervision/objecttracksummary/) | Summary of an object's track across a video sequence. |
| [`OneFormer<T>`](/docs/reference/wiki/computervision/oneformer/) | OneFormer: One Transformer to Rule Universal Image Segmentation. |
| [`OpenVocabSAM<T>`](/docs/reference/wiki/computervision/openvocabsam/) | Open-Vocabulary SAM: SAM with text-based open-vocabulary recognition. |
| [`OpenVocabSegmentationOutput<T>`](/docs/reference/wiki/computervision/openvocabsegmentationoutput/) | Output for open-vocabulary and referring segmentation models. |
| [`PANet<T>`](/docs/reference/wiki/computervision/panet/) | Path Aggregation Network (PANet) for enhanced multi-scale feature fusion. |
| [`PIDNet<T>`](/docs/reference/wiki/computervision/pidnet/) | PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers. |
| [`PixelLM<T>`](/docs/reference/wiki/computervision/pixellm/) | PixelLM: Pixel Reasoning with Large Multimodal Model. |
| [`PointCloudSegmentationOutput<T>`](/docs/reference/wiki/computervision/pointcloudsegmentationoutput/) | Output for 3D point cloud segmentation. |
| [`PointTransformerV3<T>`](/docs/reference/wiki/computervision/pointtransformerv3/) | Point Transformer V3: Simpler, Faster, Stronger 3D point cloud segmentation. |
| [`PrototypeMaskHead<T>`](/docs/reference/wiki/computervision/prototypemaskhead/) | Prototype-based mask head for YOLO and SOLOv2. |
| [`QueryMeldNet<T>`](/docs/reference/wiki/computervision/querymeldnet/) | QueryMeldNet (MQ-Former): Dynamic Query Melding for Multi-Dataset Segmentation. |
| [`RPN<T>`](/docs/reference/wiki/computervision/rpn/) | Region Proposal Network (RPN) - Generates object proposals for two-stage detectors. |
| [`RTDETR<T>`](/docs/reference/wiki/computervision/rtdetr/) | RT-DETR (Real-Time DEtection TRansformer) - First real-time end-to-end object detector. |
| [`RecognizedText<T>`](/docs/reference/wiki/computervision/recognizedtext/) | Represents recognized text in an image region. |
| [`RepViTSAM<T>`](/docs/reference/wiki/computervision/repvitsam/) | RepViT-SAM: Real-time SAM with RepViT backbone. |
| [`ResNet<T>`](/docs/reference/wiki/computervision/resnet/) | ResNet backbone network for feature extraction. |
| [`SAM21<T>`](/docs/reference/wiki/computervision/sam21/) | SAM 2.1: Segment Anything Model 2.1 with refined checkpoints for images and videos. |
| [`SAMHQ<T>`](/docs/reference/wiki/computervision/samhq/) | SAM-HQ: Segment Anything in High Quality. |
| [`SAM<T>`](/docs/reference/wiki/computervision/sam/) | Segment Anything Model (SAM): the first promptable foundation model for image segmentation. |
| [`SAN<T>`](/docs/reference/wiki/computervision/san/) | SAN: Side Adapter Network for open-vocabulary semantic segmentation. |
| [`SED<T>`](/docs/reference/wiki/computervision/sed/) | SED: A Simple Encoder-Decoder for open-vocabulary semantic segmentation. |
| [`SEEM<T>`](/docs/reference/wiki/computervision/seem/) | SEEM: Segment Everything Everywhere All at Once. |
| [`SOLOv2<T>`](/docs/reference/wiki/computervision/solov2/) | SOLOv2 (Segmenting Objects by Locations v2) for instance segmentation. |
| [`SORT<T>`](/docs/reference/wiki/computervision/sort/) | SORT (Simple Online and Realtime Tracking) implementation. |
| [`SceneTextReader<T>`](/docs/reference/wiki/computervision/scenetextreader/) | End-to-end scene text reader that combines detection and recognition. |
| [`SegFormer<T>`](/docs/reference/wiki/computervision/segformer/) | SegFormer: Simple and Efficient Semantic Segmentation with Transformers. |
| [`SegGPT<T>`](/docs/reference/wiki/computervision/seggpt/) | SegGPT: Segmenting Everything In Context via in-context learning. |
| [`SegMamba<T>`](/docs/reference/wiki/computervision/segmamba/) | SegMamba: long-range sequential modeling Mamba for 3D medical image segmentation (Xing et al., 2024, arXiv:2401.13560). |
| [`SegNeXt<T>`](/docs/reference/wiki/computervision/segnext/) | SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation. |
| [`SegmentInfo<T>`](/docs/reference/wiki/computervision/segmentinfo/) | Metadata for a single segment in panoptic/instance output. |
| [`SegmentationEvaluation`](/docs/reference/wiki/computervision/segmentationevaluation/) | Comprehensive evaluation metrics for segmentation models covering all standard benchmarks. |
| [`SegmentationMask<T>`](/docs/reference/wiki/computervision/segmentationmask/) | Represents a single segmentation mask with associated metadata. |
| [`SegmentationOutput<T>`](/docs/reference/wiki/computervision/segmentationoutput/) | Unified output type for all segmentation tasks, combining semantic, instance, and panoptic results. |
| [`SegmentationPrompt<T>`](/docs/reference/wiki/computervision/segmentationprompt/) | Represents a user prompt for interactive/promptable segmentation models. |
| [`SegmentationSample<T>`](/docs/reference/wiki/computervision/segmentationsample/) | Represents a segmentation dataset sample with image, masks, and metadata. |
| [`SlimSAM<T>`](/docs/reference/wiki/computervision/slimsam/) | SlimSAM: Pruned and distilled SAM for efficient segmentation. |
| [`SoftNMS<T>`](/docs/reference/wiki/computervision/softnms/) | Implements Soft-NMS algorithm which reduces confidence of overlapping boxes instead of removing them. |
| [`Sonata<T>`](/docs/reference/wiki/computervision/sonata/) | Sonata: A Mamba-based 3D point cloud backbone for efficient segmentation. |
| [`SwinTransformer<T>`](/docs/reference/wiki/computervision/swintransformer/) | Swin Transformer backbone for hierarchical vision transformer feature extraction. |
| [`SwinUNETR<T>`](/docs/reference/wiki/computervision/swinunetr/) | Swin UNETR: Swin Transformer encoder for 3D medical segmentation. |
| [`TensorData`](/docs/reference/wiki/computervision/tensordata/) | Represents tensor data extracted from a weight file. |
| [`TextDetectionResult<T>`](/docs/reference/wiki/computervision/textdetectionresult/) | Result of text detection containing detected text regions. |
| [`TextRegion<T>`](/docs/reference/wiki/computervision/textregion/) | Represents a detected text region in an image. |
| [`TrOCR<T>`](/docs/reference/wiki/computervision/trocr/) | TrOCR (Transformer-based OCR) for text recognition. |
| [`Track<T>`](/docs/reference/wiki/computervision/track/) | Represents a tracked object across frames. |
| [`TrackingResult<T>`](/docs/reference/wiki/computervision/trackingresult/) | Result of tracking on a frame. |
| [`TransUNet<T>`](/docs/reference/wiki/computervision/transunet/) | TransUNet: Transformers make strong encoders for medical segmentation. |
| [`U2Seg<T>`](/docs/reference/wiki/computervision/u2seg/) | U2Seg: Unified Unsupervised Segmentation framework. |
| [`UMamba<T>`](/docs/reference/wiki/computervision/umamba/) | U-Mamba: Hybrid CNN-Mamba architecture for medical segmentation. |
| [`UNINEXT<T>`](/docs/reference/wiki/computervision/uninext/) | UNINEXT: Universal Instance Perception as Object Discovery and Retrieval. |
| [`UniVS<T>`](/docs/reference/wiki/computervision/univs/) | UniVS: Unified and Universal Video Segmentation. |
| [`UniverSeg<T>`](/docs/reference/wiki/computervision/universeg/) | UniverSeg: Universal Medical Image Segmentation via cross-attention. |
| [`VMamba<T>`](/docs/reference/wiki/computervision/vmamba/) | VMamba: Visual State Space Model with Cross-Scan for 2D understanding. |
| [`ViMUNet<T>`](/docs/reference/wiki/computervision/vimunet/) | ViM-UNet: Vision Mamba for medical image segmentation in U-Net. |
| [`ViTAdapter<T>`](/docs/reference/wiki/computervision/vitadapter/) | ViT-Adapter: Vision Transformer Adapter for Dense Predictions. |
| [`ViTCoMer<T>`](/docs/reference/wiki/computervision/vitcomer/) | ViT-CoMer: Vision Transformer with Convolutional Multi-scale Feature Interaction. |
| [`VideoFrameSegmentation<T>`](/docs/reference/wiki/computervision/videoframesegmentation/) | Segmentation result for a single video frame. |
| [`VideoLISA<T>`](/docs/reference/wiki/computervision/videolisa/) | Video-LISA: Language-instructed video segmentation with reasoning. |
| [`VideoSegmentationOutput<T>`](/docs/reference/wiki/computervision/videosegmentationoutput/) | Output for video segmentation containing per-frame masks with temporal tracking. |
| [`VisionMamba<T>`](/docs/reference/wiki/computervision/visionmamba/) | Vision Mamba (Vim): Bidirectional State Space Model for vision. |
| [`WeightDownloader`](/docs/reference/wiki/computervision/weightdownloader/) | Downloads and caches pre-trained model weights from remote URLs. |
| [`WeightLoader`](/docs/reference/wiki/computervision/weightloader/) | Loads pre-trained model weights from various file formats. |
| [`XDecoder<T>`](/docs/reference/wiki/computervision/xdecoder/) | X-Decoder: Generalized Decoding for Pixel, Image, and Language. |
| [`YOLO11Seg<T>`](/docs/reference/wiki/computervision/yolo11seg/) | YOLO11-Seg: Ultralytics next-generation real-time instance segmentation. |
| [`YOLO26Seg<T>`](/docs/reference/wiki/computervision/yolo26seg/) | YOLO26-Seg: Latest YOLO family instance segmentation with advanced CSP design. |
| [`YOLOSeg<T>`](/docs/reference/wiki/computervision/yoloseg/) | YOLOv8-Seg for instance segmentation. |
| [`YOLOv10<T>`](/docs/reference/wiki/computervision/yolov10/) | YOLOv10 object detector with NMS-free training and consistent dual assignments. |
| [`YOLOv11<T>`](/docs/reference/wiki/computervision/yolov11/) | YOLOv11 object detector with enhanced feature extraction and attention mechanisms. |
| [`YOLOv12Seg<T>`](/docs/reference/wiki/computervision/yolov12seg/) | YOLOv12-Seg: Attention-centric real-time instance segmentation. |
| [`YOLOv8Seg<T>`](/docs/reference/wiki/computervision/yolov8seg/) | YOLOv8-Seg: Ultralytics real-time instance segmentation. |
| [`YOLOv8<T>`](/docs/reference/wiki/computervision/yolov8/) | YOLOv8 object detector - anchor-free, decoupled head architecture. |
| [`YOLOv9Seg<T>`](/docs/reference/wiki/computervision/yolov9seg/) | YOLOv9-Seg: Instance segmentation with Programmable Gradient Information. |
| [`YOLOv9<T>`](/docs/reference/wiki/computervision/yolov9/) | YOLOv9 object detector with Programmable Gradient Information (PGI). |

## Base Classes (15)

| Type | Summary |
|:-----|:--------|
| [`InstanceSegmentationBase<T>`](/docs/reference/wiki/computervision/instancesegmentationbase/) | Abstract base class for instance segmentation models that detect and mask individual object instances. |
| [`InstanceSegmenterBase<T>`](/docs/reference/wiki/computervision/instancesegmenterbase/) | Base class for instance segmentation models. |
| [`MedicalSegmentationBase<T>`](/docs/reference/wiki/computervision/medicalsegmentationbase/) | Abstract base class for medical image segmentation models handling 2D slices and 3D volumes. |
| [`NeckBase<T>`](/docs/reference/wiki/computervision/neckbase/) | Base class for neck modules that perform multi-scale feature fusion. |
| [`OCRBase<T>`](/docs/reference/wiki/computervision/ocrbase/) | Base class for OCR models. |
| [`ObjectDetectorBase<T>`](/docs/reference/wiki/computervision/objectdetectorbase/) | Base class for all object detection models. |
| [`ObjectTrackerBase<T>`](/docs/reference/wiki/computervision/objecttrackerbase/) | Base class for object trackers. |
| [`OpenVocabSegmentationBase<T>`](/docs/reference/wiki/computervision/openvocabsegmentationbase/) | Abstract base class for open-vocabulary segmentation models that segment objects described by arbitrary text without being limited to a fixed class set. |
| [`PanopticSegmentationBase<T>`](/docs/reference/wiki/computervision/panopticsegmentationbase/) | Abstract base class for panoptic segmentation models that unify semantic and instance segmentation. |
| [`PromptableSegmentationBase<T>`](/docs/reference/wiki/computervision/promptablesegmentationbase/) | Abstract base class for promptable segmentation models like SAM that accept user prompts (points, boxes, masks) to segment specific objects. |
| [`ReferringSegmentationBase<T>`](/docs/reference/wiki/computervision/referringsegmentationbase/) | Abstract base class for referring segmentation models that segment objects from natural language descriptions with complex reasoning about spatial relationships and attributes. |
| [`SegmentationModelBase<T>`](/docs/reference/wiki/computervision/segmentationmodelbase/) | Abstract base class for all segmentation models, providing common dual-mode (native + ONNX) infrastructure, batch handling, forward/backward passes, and serialization. |
| [`SemanticSegmentationBase<T>`](/docs/reference/wiki/computervision/semanticsegmentationbase/) | Abstract base class for semantic segmentation models that assign a class label to every pixel. |
| [`TextDetectorBase<T>`](/docs/reference/wiki/computervision/textdetectorbase/) | Base class for text detection models. |
| [`VideoSegmentationBase<T>`](/docs/reference/wiki/computervision/videosegmentationbase/) | Abstract base class for video segmentation models that track and segment objects across frames. |

## Enums (15)

| Type | Summary |
|:-----|:--------|
| [`DecayMethod<T>`](/docs/reference/wiki/computervision/decaymethod/) | Soft-NMS decay method. |
| [`DocumentBlockType`](/docs/reference/wiki/computervision/documentblocktype/) | Type of document block. |
| [`DocumentBlockType`](/docs/reference/wiki/computervision/documentblocktype-2/) | Types of document blocks. |
| [`EfficientNetVariant`](/docs/reference/wiki/computervision/efficientnetvariant/) | EfficientNet variant enumeration. |
| [`InstanceSegmentationArchitecture`](/docs/reference/wiki/computervision/instancesegmentationarchitecture/) | Instance segmentation architectures. |
| [`MatchLabel`](/docs/reference/wiki/computervision/matchlabel/) | Label indicating how an anchor should be treated during training. |
| [`OCRMode`](/docs/reference/wiki/computervision/ocrmode/) | OCR processing modes. |
| [`PromptType`](/docs/reference/wiki/computervision/prompttype/) | Type of segmentation prompt. |
| [`SegmentationTaskType`](/docs/reference/wiki/computervision/segmentationtasktype/) | Type of segmentation task. |
| [`SwinVariant`](/docs/reference/wiki/computervision/swinvariant/) | Swin Transformer variant enumeration. |
| [`TextDetectionArchitecture`](/docs/reference/wiki/computervision/textdetectionarchitecture/) | Text detection architecture types. |
| [`TextDetectionModel`](/docs/reference/wiki/computervision/textdetectionmodel/) | Text detection model types. |
| [`TextRecognitionModel`](/docs/reference/wiki/computervision/textrecognitionmodel/) | Text recognition model types. |
| [`TextRegionType`](/docs/reference/wiki/computervision/textregiontype/) | Type of text region. |
| [`TrackState`](/docs/reference/wiki/computervision/trackstate/) | Track lifecycle states. |

## Options & Configuration (78)

| Type | Summary |
|:-----|:--------|
| [`BiomedParseOptions`](/docs/reference/wiki/computervision/biomedparseoptions/) | Configuration options for BiomedParse biomedical foundation segmentation. |
| [`CATSegOptions`](/docs/reference/wiki/computervision/catsegoptions/) | Configuration options for CAT-Seg cost aggregation open-vocabulary segmentation. |
| [`CUPSOptions`](/docs/reference/wiki/computervision/cupsoptions/) | Configuration options for CUPS unsupervised panoptic segmentation. |
| [`ConcertoOptions`](/docs/reference/wiki/computervision/concertooptions/) | Configuration options for Concerto joint 2D-3D segmentation. |
| [`DEVAOptions`](/docs/reference/wiki/computervision/devaoptions/) | Configuration options for DEVA decoupled video segmentation. |
| [`DiffCutOptions`](/docs/reference/wiki/computervision/diffcutoptions/) | Configuration options for the DiffCut semantic segmentation model. |
| [`DiffCutSegmentationOptions`](/docs/reference/wiki/computervision/diffcutsegmentationoptions/) | Configuration options for DiffCut diffusion-based zero-shot segmentation. |
| [`DiffSegOptions`](/docs/reference/wiki/computervision/diffsegoptions/) | Configuration options for the DiffSeg semantic segmentation model. |
| [`EdgeSAMOptions`](/docs/reference/wiki/computervision/edgesamoptions/) | Configuration options for EdgeSAM edge-optimized SAM. |
| [`EfficientSAMOptions`](/docs/reference/wiki/computervision/efficientsamoptions/) | Configuration options for EfficientSAM (SAMI-pretrained fast SAM). |
| [`EfficientTAMOptions`](/docs/reference/wiki/computervision/efficienttamoptions/) | Configuration options for EfficientTAM lightweight video segmentation. |
| [`EoMTOptions`](/docs/reference/wiki/computervision/eomtoptions/) | Configuration options for the EoMT (Encoder-only Mask Transformer) model. |
| [`FastSAMOptions`](/docs/reference/wiki/computervision/fastsamoptions/) | Configuration options for FastSAM (CNN-based fast Segment Anything). |
| [`GLaMMOptions`](/docs/reference/wiki/computervision/glammoptions/) | Configuration options for GLaMM grounding language model segmentation. |
| [`GroundedSAM2Options`](/docs/reference/wiki/computervision/groundedsam2options/) | Configuration options for Grounded SAM 2 text-prompted detection and tracking. |
| [`InstanceSegmentationOptions<T>`](/docs/reference/wiki/computervision/instancesegmentationoptions/) | Options for instance segmentation models. |
| [`InternImageOptions`](/docs/reference/wiki/computervision/internimageoptions/) | Configuration options for the InternImage semantic segmentation model. |
| [`KMaXDeepLabOptions`](/docs/reference/wiki/computervision/kmaxdeeplaboptions/) | Configuration options for kMaX-DeepLab panoptic segmentation. |
| [`LISAOptions`](/docs/reference/wiki/computervision/lisaoptions/) | Configuration options for LISA reasoning segmentation. |
| [`Mask2FormerOptions`](/docs/reference/wiki/computervision/mask2formeroptions/) | Configuration options for the Mask2Former universal segmentation model. |
| [`MaskAdapterOptions`](/docs/reference/wiki/computervision/maskadapteroptions/) | Configuration options for Mask-Adapter open-vocabulary segmentation. |
| [`MaskDINOOptions`](/docs/reference/wiki/computervision/maskdinooptions/) | Configuration options for the Mask DINO model. |
| [`MedNeXtOptions`](/docs/reference/wiki/computervision/mednextoptions/) | Configuration options for MedNeXt medical segmentation. |
| [`MedSAM2Options`](/docs/reference/wiki/computervision/medsam2options/) | Configuration options for MedSAM 2 3D medical segmentation. |
| [`MedSAMOptions`](/docs/reference/wiki/computervision/medsamoptions/) | Configuration options for MedSAM universal medical segmentation. |
| [`MedSegDiffV2Options`](/docs/reference/wiki/computervision/medsegdiffv2options/) | Configuration options for MedSegDiff-V2 transformer diffusion medical segmentation. |
| [`MedSegDiffV2SegmentationOptions`](/docs/reference/wiki/computervision/medsegdiffv2segmentationoptions/) | Configuration options for MedSegDiff-V2 transformer diffusion medical segmentation. |
| [`MobileSAMOptions`](/docs/reference/wiki/computervision/mobilesamoptions/) | Configuration options for MobileSAM (distilled lightweight SAM). |
| [`NeckConfig`](/docs/reference/wiki/computervision/neckconfig/) | Configuration for neck modules. |
| [`NnUNetOptions`](/docs/reference/wiki/computervision/nnunetoptions/) | Configuration options for nnU-Net v2 medical segmentation. |
| [`OCROptions<T>`](/docs/reference/wiki/computervision/ocroptions/) | Options for OCR models. |
| [`ODISEOptions`](/docs/reference/wiki/computervision/odiseoptions/) | Configuration options for ODISE open-vocabulary panoptic segmentation. |
| [`ODISESegmentationOptions`](/docs/reference/wiki/computervision/odisesegmentationoptions/) | Configuration options for ODISE diffusion-based panoptic segmentation. |
| [`OMGLLaVAOptions`](/docs/reference/wiki/computervision/omgllavaoptions/) | Configuration options for OMG-LLaVA unified pixel-level reasoning. |
| [`OMGSegOptions`](/docs/reference/wiki/computervision/omgsegoptions/) | Configuration options for the OMG-Seg model. |
| [`OneFormerOptions`](/docs/reference/wiki/computervision/oneformeroptions/) | Configuration options for the OneFormer universal segmentation model. |
| [`OpenVocabSAMOptions`](/docs/reference/wiki/computervision/openvocabsamoptions/) | Configuration options for Open-Vocabulary SAM interactive recognition. |
| [`PIDNetOptions`](/docs/reference/wiki/computervision/pidnetoptions/) | Configuration options for PIDNet real-time segmentation. |
| [`PixelLMOptions`](/docs/reference/wiki/computervision/pixellmoptions/) | Configuration options for PixelLM pixel-level reasoning segmentation. |
| [`PointTransformerV3Options`](/docs/reference/wiki/computervision/pointtransformerv3options/) | Configuration options for Point Transformer V3 3D segmentation. |
| [`QueryMeldNetOptions`](/docs/reference/wiki/computervision/querymeldnetoptions/) | Configuration options for the QueryMeldNet (MQ-Former) model. |
| [`RepViTSAMOptions`](/docs/reference/wiki/computervision/repvitsamoptions/) | Configuration options for RepViT-SAM real-time mobile SAM. |
| [`SAM21Options`](/docs/reference/wiki/computervision/sam21options/) | Configuration options for SAM 2.1 (Segment Anything Model 2.1). |
| [`SAMHQOptions`](/docs/reference/wiki/computervision/samhqoptions/) | Configuration options for the SAM-HQ (High-Quality Segment Anything) model. |
| [`SAMOptions`](/docs/reference/wiki/computervision/samoptions/) | Configuration options for the Segment Anything Model (SAM). |
| [`SANOptions`](/docs/reference/wiki/computervision/sanoptions/) | Configuration options for SAN (Side Adapter Network) open-vocabulary segmentation. |
| [`SEDOptions`](/docs/reference/wiki/computervision/sedoptions/) | Configuration options for SED simple encoder-decoder open-vocabulary segmentation. |
| [`SEEMOptions`](/docs/reference/wiki/computervision/seemoptions/) | Configuration options for SEEM interactive segmentation. |
| [`SegFormerOptions`](/docs/reference/wiki/computervision/segformeroptions/) | Configuration options for the SegFormer semantic segmentation model. |
| [`SegGPTOptions`](/docs/reference/wiki/computervision/seggptoptions/) | Configuration options for SegGPT in-context segmentation. |
| [`SegMambaOptions`](/docs/reference/wiki/computervision/segmambaoptions/) | Configuration options for SegMamba 3D volumetric Mamba segmentation. |
| [`SegNeXtOptions`](/docs/reference/wiki/computervision/segnextoptions/) | Configuration options for the SegNeXt semantic segmentation model. |
| [`SegmentationDatasetConfig`](/docs/reference/wiki/computervision/segmentationdatasetconfig/) | Configuration for a segmentation dataset. |
| [`SegmentationVisualizationConfig`](/docs/reference/wiki/computervision/segmentationvisualizationconfig/) | Visualization settings and utilities for segmentation outputs. |
| [`SlimSAMOptions`](/docs/reference/wiki/computervision/slimsamoptions/) | Configuration options for SlimSAM (pruned SAM with 1.4% params). |
| [`SonataOptions`](/docs/reference/wiki/computervision/sonataoptions/) | Configuration options for Sonata 3D segmentation with self-distillation. |
| [`SwinUNETROptions`](/docs/reference/wiki/computervision/swinunetroptions/) | Configuration options for Swin-UNETR brain tumor segmentation. |
| [`TextDetectionOptions<T>`](/docs/reference/wiki/computervision/textdetectionoptions/) | Options for text detection models. |
| [`TrackingOptions<T>`](/docs/reference/wiki/computervision/trackingoptions/) | Options for object tracking. |
| [`TransUNetOptions`](/docs/reference/wiki/computervision/transunetoptions/) | Configuration options for TransUNet medical segmentation. |
| [`U2SegOptions`](/docs/reference/wiki/computervision/u2segoptions/) | Configuration options for the U2Seg model. |
| [`UMambaOptions`](/docs/reference/wiki/computervision/umambaoptions/) | Configuration options for U-Mamba CNN+Mamba medical segmentation. |
| [`UNINEXTOptions`](/docs/reference/wiki/computervision/uninextoptions/) | Configuration options for the UNINEXT model. |
| [`UniVSOptions`](/docs/reference/wiki/computervision/univsoptions/) | Configuration options for UniVS universal video segmentation. |
| [`UniverSegOptions`](/docs/reference/wiki/computervision/universegoptions/) | Configuration options for UniverSeg few-shot medical segmentation. |
| [`VMambaOptions`](/docs/reference/wiki/computervision/vmambaoptions/) | Configuration options for VMamba visual state space model segmentation. |
| [`ViMUNetOptions`](/docs/reference/wiki/computervision/vimunetoptions/) | Configuration options for ViM-UNet Vision Mamba + U-Net biomedical segmentation. |
| [`ViTAdapterOptions`](/docs/reference/wiki/computervision/vitadapteroptions/) | Configuration options for the ViT-Adapter semantic segmentation model. |
| [`ViTCoMerOptions`](/docs/reference/wiki/computervision/vitcomeroptions/) | Configuration options for the ViT-CoMer semantic segmentation model. |
| [`VideoLISAOptions`](/docs/reference/wiki/computervision/videolisaoptions/) | Configuration options for VideoLISA video reasoning segmentation. |
| [`VisionMambaOptions`](/docs/reference/wiki/computervision/visionmambaoptions/) | Configuration options for Vision Mamba (Vim) bidirectional SSM segmentation. |
| [`VisualizationOptions`](/docs/reference/wiki/computervision/visualizationoptions/) | Options for visualization. |
| [`XDecoderOptions`](/docs/reference/wiki/computervision/xdecoderoptions/) | Configuration options for the X-Decoder model. |
| [`YOLO11SegOptions`](/docs/reference/wiki/computervision/yolo11segoptions/) | Configuration options for YOLO11-Seg instance segmentation. |
| [`YOLO26SegOptions`](/docs/reference/wiki/computervision/yolo26segoptions/) | Configuration options for YOLO26-Seg instance segmentation. |
| [`YOLOv12SegOptions`](/docs/reference/wiki/computervision/yolov12segoptions/) | Configuration options for YOLOv12-Seg instance segmentation. |
| [`YOLOv8SegOptions`](/docs/reference/wiki/computervision/yolov8segoptions/) | Configuration options for the YOLOv8-Seg instance segmentation model. |
| [`YOLOv9SegOptions`](/docs/reference/wiki/computervision/yolov9segoptions/) | Configuration options for YOLOv9-Seg instance segmentation. |

## Helpers & Utilities (4)

| Type | Summary |
|:-----|:--------|
| [`AnchorPresets`](/docs/reference/wiki/computervision/anchorpresets/) | Pre-defined anchor configurations for common models. |
| [`PretrainedRegistry`](/docs/reference/wiki/computervision/pretrainedregistry/) | Registry of pre-trained model weights with their download URLs. |
| [`SegmentationDatasets`](/docs/reference/wiki/computervision/segmentationdatasets/) | Configuration for standard segmentation datasets. |
| [`SegmentationTensorOps`](/docs/reference/wiki/computervision/segmentationtensorops/) | Static helper methods for common segmentation tensor operations (argmax, softmax, etc.) used by models implementing segmentation interfaces. |

