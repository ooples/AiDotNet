using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// YOLOv9 object detector: the GELAN architecture with Programmable Gradient Information (PGI).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> YOLOv9 builds its network from GELAN blocks (RepNCSPELAN4) and adds, only while
/// training, an auxiliary branch with its own detection head. That branch feeds the main network reliable
/// gradients (the paper's "programmable gradient information") and is dropped at inference, so it costs
/// nothing at test time.</para>
/// <para>
/// Each <see cref="ModelSize"/> is its own paper configuration: Nano = t, Small = s, Medium = m, Large = c and
/// XLarge = e. See <see cref="YOLOv9Backbone{T}"/> for the per-size PGI branches. This model previously ran a
/// YOLOv4/v5 CSPDarknet with a residual 3x3 convolution per level, and had no auxiliary branch.
/// </para>
/// <para>Reference: Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information", 2024</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information",
    "https://arxiv.org/abs/2402.13616",
    Year = 2024,
    Authors = "Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao")]
public partial class YOLOv9<T> : ObjectDetectorBase<T>, IDetectionTrainingModel<T>
{
    /// <summary>Weight of the auxiliary (PGI) head's loss relative to the main head (utils/loss_tal_dual.py).</summary>
    internal const double AuxiliaryLossWeight = 0.25;

    private readonly AiDotNet.ComputerVision.Detection.Losses.TaskAlignedDetectionLoss<T> _detectionLoss;
    private readonly YOLOv8Head<T> _head;
    private readonly YOLOv8Head<T> _auxHead; // PGI's auxiliary head: trained jointly, dropped at inference.
    private readonly int[] _strides;
    private readonly NMS<T> _nms;
    private bool _auxHeadShapesResolved;

    /// <inheritdoc/>
    public override string Name => $"YOLOv9-{Options.Size}";
    /// <summary>
    /// Creates a new YOLOv9 detector with default options derived from the architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="size">Model size variant (default: Small for faster construction).</param>
    /// <param name="numClasses">Number of detection classes (default: 80 for COCO).</param>
    public YOLOv9(
        NeuralNetworkArchitecture<T> architecture,
        ModelSize size = ModelSize.Small,
        int numClasses = 80)
        : this(new ObjectDetectionOptions<T>
        {
            Size = size,
            NumClasses = numClasses,
            InputSize = new[] { architecture.InputHeight > 0 ? architecture.InputHeight : 640, architecture.InputWidth > 0 ? architecture.InputWidth : 640 },
        })
    {
    }

    /// <summary>
    /// Creates a new YOLOv9 detector with the specified options.
    /// </summary>
    /// <param name="options">Detection options; <see cref="ObjectDetectionOptions{T}.Size"/> selects t/s/m/c/e.</param>
    public YOLOv9(ObjectDetectionOptions<T> options) : base(options)
    {
        var backbone = new YOLOv9Backbone<T>(new YoloBackboneOptions { Size = options.Size });
        Backbone = backbone;
        Neck = new YOLOv9Neck<T>(options.Size);
        _head = new YOLOv8Head<T>(Neck.LevelChannels.ToArray(), options.NumClasses);
        _auxHead = new YOLOv8Head<T>(backbone.AuxiliaryChannels.ToArray(), options.NumClasses);

        _strides = Backbone.Strides.ToArray();
        _detectionLoss = new AiDotNet.ComputerVision.Detection.Losses.TaskAlignedDetectionLoss<T>(options.NumClasses,
            _head.RegMax, options.TaskAlignedLoss ?? new AiDotNet.ComputerVision.Detection.Losses.TaskAlignedLossOptions());
        _nms = new NMS<T>();
    }

    private YOLOv9Backbone<T> EnsureYoloV9Backbone => EnsureBackbone as YOLOv9Backbone<T>
        ?? throw new InvalidOperationException("YOLOv9 requires its own YOLOv9Backbone.");

    /// <summary>Trains both heads with task-aligned assignment, BCE classification, CIoU and distribution focal loss.</summary>
    /// <remarks>
    /// The loss is the main head's plus 0.25 x the auxiliary head's, as in the reference dual loss
    /// (utils/loss_tal_dual.py); box/class/DFL gains default to 7.5/0.5/1.5 and can be overridden with
    /// <see cref="ObjectDetectionOptions{T}.TaskAlignedLoss"/>. Inputs are model-ready NCHW tensors, as for
    /// Predict, and targets are normalized against that input size.
    /// </remarks>
    public void TrainDetections(Tensor<T> input, DetectionTrainingBatch<T> targets)
    {
        YoloDetectionTraining.Validate(input, targets, Options.NumClasses, "YOLOv9");
        int height = input.Shape[2];
        int width = input.Shape[3];
        int levels = _strides.Length;
        TrainWithTargets(input, targets, ForwardTrainingHeads, (heads, batch) => Engine.TensorAdd(
            YoloDetectionTraining.HeadLoss(_detectionLoss, heads, 0, levels, _strides, height, width, batch, _detectionLoss.TopK),
            Engine.TensorMultiplyScalar(
                YoloDetectionTraining.HeadLoss(_detectionLoss, heads, 2 * levels, levels, _strides, height, width, batch, _detectionLoss.TopK),
                NumOps.FromDouble(AuxiliaryLossWeight))));
    }

    /// <inheritdoc/>
    /// <remarks>The generic tensor objective supervises both heads, with the same 0.25 weight on the auxiliary one.</remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expectedOutput is null) throw new ArgumentNullException(nameof(expectedOutput));
        TrainWithTargets(input, expectedOutput, ForwardTrainingHeads, (heads, target) =>
        {
            int half = heads.Count / 2;
            return Engine.TensorAdd(
                TensorModelTrainer<T>.MeanSquaredError(CvTensorOps<T>.ConcatenateOutputs(heads.GetRange(0, half)), target),
                Engine.TensorMultiplyScalar(
                    TensorModelTrainer<T>.MeanSquaredError(CvTensorOps<T>.ConcatenateOutputs(heads.GetRange(half, half)), target),
                    NumOps.FromDouble(AuxiliaryLossWeight)));
        });
    }

    /// <summary>Main head outputs (class then distribution per level), followed by the auxiliary head's.</summary>
    internal List<Tensor<T>> ForwardTrainingHeads(Tensor<T> input)
    {
        var (main, auxiliary) = EnsureYoloV9Backbone.ExtractWithAuxiliary(input);
        var (mainClasses, mainDistributions) = _head.Forward(EnsureNeck.Forward(main));
        var (auxClasses, auxDistributions) = _auxHead.Forward(auxiliary);
        _auxHeadShapesResolved = true;
        var outputs = new List<Tensor<T>>();
        outputs.AddRange(mainClasses);
        outputs.AddRange(mainDistributions);
        outputs.AddRange(auxClasses);
        outputs.AddRange(auxDistributions);
        return outputs;
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
        if (!_auxHeadShapesResolved)
        {
            // The auxiliary head trains only, but its lazily sized convolutions must exist whenever the
            // parameters are enumerated, saved or cloned; size them on the first forward.
            ForwardTrainingHeads(input);
        }

        var (clsOutputs, regOutputs) = _head.Forward(EnsureNeck.Forward(EnsureBackbone.ExtractFeatures(input)));
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
    protected override long GetHeadParameterCount() => _head.GetParameterCount() + _auxHead.GetParameterCount();

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using var stream = File.OpenRead(pathOrUrl);
        using var reader = new BinaryReader(stream);

        int magic = reader.ReadInt32();
        if (magic != 0x594F4C4F) // "YOLO" in ASCII
        {
            throw new InvalidDataException("Invalid weight file format: incorrect magic number.");
        }

        // Version 2: the GELAN/PGI architecture; version 1 stored the old CSPDarknet with per-level GELAN convs.
        int version = reader.ReadInt32();
        if (version != 2)
        {
            throw new InvalidDataException($"Unsupported weight file version: {version}.");
        }

        string modelName = reader.ReadString();
        if (!modelName.StartsWith("YOLOv9"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not YOLOv9.");
        }

        EnsureBackbone.ReadParameters(reader);
        EnsureNeck.ReadParameters(reader);
        _head.ReadParameters(reader);
        _auxHead.ReadParameters(reader);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(0x594F4C4F); // "YOLO" in ASCII
        writer.Write(2); // Version 2
        writer.Write(Name);
        EnsureBackbone.WriteParameters(writer);
        EnsureNeck.WriteParameters(writer);
        _head.WriteParameters(writer);
        _auxHead.WriteParameters(writer);
    }
}
