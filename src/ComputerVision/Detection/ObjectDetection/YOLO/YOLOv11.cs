using AiDotNet.Tensors.Engines;
using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Extensions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// YOLOv11 object detector with enhanced feature extraction and attention mechanisms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> YOLOv11 is the latest YOLO version with improved
/// feature extraction using attention mechanisms and more efficient architecture.
/// It builds upon YOLOv8-v10 innovations while adding new enhancements.</para>
///
/// <para>Key features:
/// - C3k2 blocks with attention for enhanced feature extraction
/// - Spatial Pyramid Pooling Fast (SPPF) with larger kernel
/// - Multi-head self-attention in neck
/// - Improved small object detection
/// </para>
///
/// <para>Reference: Ultralytics, "YOLOv11" 2024</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("YOLO11",
    "https://github.com/ultralytics/ultralytics",
    Year = 2024,
    Authors = "Glenn Jocher, Jing Qiu")]
public partial class YOLOv11<T> : ObjectDetectorBase<T>, IDetectionTrainingModel<T>
{
    private readonly AiDotNet.ComputerVision.Detection.Losses.TaskAlignedDetectionLoss<T> _detectionLoss;
    private readonly YOLOv8Head<T> _head;
    private readonly int[] _strides;

    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"YOLOv11-{Options.Size}";

    /// <summary>
    /// Creates a new YOLOv11 detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public YOLOv11(ObjectDetectionOptions<T> options) : base(options)
    {
        // YOLO11's own architecture (yolo11.yaml): C3k2 stages, SPPF and C2PSA attention in the backbone,
        // C3k2 in the PAN neck. This was the YOLOv4/v5 CSPDarknet with an SPPF and a generic attention
        // block bolted onto every neck output, neither of which is YOLO11's design.
        Backbone = new YOLOv11Backbone<T>(new YoloBackboneOptions { Size = options.Size });
        Neck = new YOLOv11Neck<T>(options.Size);
        _head = new YOLOv8Head<T>(Neck.LevelChannels.ToArray(), options.NumClasses);

        _strides = Backbone.Strides.ToArray();
        _detectionLoss = new AiDotNet.ComputerVision.Detection.Losses.TaskAlignedDetectionLoss<T>(options.NumClasses,
            _head.RegMax, options.TaskAlignedLoss ?? new AiDotNet.ComputerVision.Detection.Losses.TaskAlignedLossOptions());
        _nms = new NMS<T>();
    }

    /// <summary>Trains the head with task-aligned assignment, BCE classification, CIoU and distribution focal loss.</summary>
    /// <remarks>
    /// Uses the YOLOv8-family objective (alpha 0.5, beta 6, top-10; box/class/DFL gains 7.5/0.5/1.5); override it
    /// with <see cref="ObjectDetectionOptions{T}.TaskAlignedLoss"/>. Inputs are model-ready NCHW tensors, as for
    /// Predict, and targets are normalized against that input size.
    /// </remarks>
    public void TrainDetections(Tensor<T> input, DetectionTrainingBatch<T> targets)
    {
        YoloDetectionTraining.Validate(input, targets, Options.NumClasses, "YOLOv11");
        int height = input.Shape[2];
        int width = input.Shape[3];
        TrainWithTargets(input, targets, (heads, batch) => YoloDetectionTraining.HeadLoss(
            _detectionLoss, heads, 0, _strides.Length, _strides, height, width, batch, _detectionLoss.TopK));
    }

    /// <inheritdoc/>
    public override DetectionResult<T> Detect(Tensor<T> image, double confidenceThreshold, double nmsThreshold)
    {
        var startTime = DateTime.UtcNow;

        int originalHeight = image.Shape[2];
        int originalWidth = image.Shape[3];

        var input = Preprocess(image);
        var outputs = Forward(input);
        var detections = PostProcess(outputs, originalWidth, originalHeight, confidenceThreshold, nmsThreshold);

        return new DetectionResult<T>
        {
            Detections = detections,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = originalWidth,
            ImageHeight = originalHeight
        };
    }

    /// <inheritdoc/>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        // Backbone feature extraction
        var backboneFeatures = EnsureBackbone.ExtractFeatures(input);

        // Neck feature fusion, then the decoupled head.
        var (clsOutputs, regOutputs) = _head.Forward(EnsureNeck.Forward(backboneFeatures));

        var outputs = new List<Tensor<T>>();
        outputs.AddRange(clsOutputs);
        outputs.AddRange(regOutputs);

        return outputs;
    }

    /// <inheritdoc/>
    protected override List<Detection<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold,
        double nmsThreshold)
    {
        int numLevels = outputs.Count / 2;
        var clsOutputs = outputs.Take(numLevels).ToList();
        var regOutputs = outputs.Skip(numLevels).ToList();

        var decoded = _head.DecodeOutputs(clsOutputs, regOutputs, _strides, imageHeight, imageWidth);

        float[] boxes = decoded[0].boxes;
        float[] scores = decoded[0].scores;
        int[] classIds = decoded[0].classIds;

        // Build detection list with confidence filtering
        var candidateDetections = new List<Detection<T>>();

        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] >= confidenceThreshold)
            {
                var box = new BoundingBox<T>(
                    NumOps.FromDouble(boxes[i * 4]),
                    NumOps.FromDouble(boxes[i * 4 + 1]),
                    NumOps.FromDouble(boxes[i * 4 + 2]),
                    NumOps.FromDouble(boxes[i * 4 + 3]));

                int classId = classIds[i];
                candidateDetections.Add(new Detection<T>(
                    box,
                    classId,
                    NumOps.FromDouble(scores[i]),
                    classId < ClassNames.Length ? ClassNames[classId] : null));
            }
        }

        // Apply NMS
        var nmsResults = _nms.Apply(candidateDetections, nmsThreshold);

        // Limit to max detections
        if (nmsResults.Count > Options.MaxDetections)
        {
            return nmsResults.Take(Options.MaxDetections).ToList();
        }

        return nmsResults;
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        return _head.GetParameterCount();

    }

    /// <inheritdoc/>
    public override async Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        byte[] data;
        if (pathOrUrl.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            pathOrUrl.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            using var client = new System.Net.Http.HttpClient();
            data = await client.GetByteArrayWithCancellationAsync(pathOrUrl, cancellationToken);
        }
        else
        {
            // Use Task.Run for net471 compatibility (ReadAllBytesAsync not available)
            data = await Task.Run(() => File.ReadAllBytes(pathOrUrl), cancellationToken);
        }

        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and verify magic number and version
        int magic = reader.ReadInt32();
        if (magic != 0x594F4C4F) // "YOLO" in ASCII
        {
            throw new InvalidDataException("Invalid weight file format: incorrect magic number.");
        }

        int version = reader.ReadInt32();
        // Version 2: YOLO11's own backbone and neck; version 1 carried the old bolt-on SPPF and attention.
        if (version != 2)

        {
            throw new InvalidDataException($"Unsupported weight file version: {version}.");
        }

        // Read model configuration
        string modelName = reader.ReadString();
        if (!modelName.StartsWith("YOLOv11"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not YOLOv11.");
        }

        // Read backbone parameters
        if (Backbone is null)
        {
            throw new InvalidOperationException("YOLOv11 backbone must be initialized before loading weights.");
        }
        Backbone.ReadParameters(reader);


        // Read neck parameters
        if (Neck is null)
        {
            throw new InvalidOperationException("YOLOv11 neck must be initialized before loading weights.");
        }
        Neck.ReadParameters(reader);


        // Read head parameters
        _head.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write magic number and version for identification
        writer.Write(0x594F4C4F); // "YOLO" in ASCII
        writer.Write(2); // Version 2


        // Write model configuration
        writer.Write(Name);

        // Write backbone parameters
        EnsureBackbone.WriteParameters(writer);


        // Write neck parameters
        EnsureNeck.WriteParameters(writer);


        // Write head parameters
        _head.WriteParameters(writer);
    }
}
