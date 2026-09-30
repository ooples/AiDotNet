using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// RT-DETR (Real-Time DEtection TRansformer) - the first real-time end-to-end object detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> RT-DETR predicts a fixed set of boxes end to end, like DETR, but at YOLO
/// speed. It gets there two ways. A hybrid encoder runs full attention only on the coarsest feature map and
/// fuses the scales with convolutions. The decoder starts from the encoder's highest-scoring positions
/// instead of learned queries. No NMS is needed.</para>
/// <para>
/// The architecture follows the reference implementation (lyuwenyu/RT-DETR, rtdetr_pytorch):
/// <list type="bullet">
/// <item>ResNet C3-C5.</item>
/// <item>The hybrid encoder: a ConvNorm projection, AIFI on S5 with a 2-D sin-cos encoding, and a CCFM
/// FPN + PAN of CSPRep blocks.</item>
/// <item>IoU-aware query selection of the top 300 encoder tokens, whose detached features become the
/// content queries.</item>
/// <item>Decoder layers with <see cref="MultiScaleDeformableAttention{T}"/> (3 levels x 4 points) and
/// per-layer heads with iterative refinement.</item>
/// <item>Contrastive denoising.</item>
/// </list>
/// The training objective is the reference's: varifocal + L1 + GIoU with Hungarian matching on every
/// decoder layer and on the encoder's selected queries, plus the fixed-assignment denoising loss of every
/// layer.
/// </para>
/// <para>Reference: Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DETRs Beat YOLOs on Real-time Object Detection",
    "https://arxiv.org/abs/2304.08069",
    Year = 2024,
    Authors = "Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, Jie Chen")]
public partial class RTDETR<T> : ObjectDetectorBase<T>, IDetectionTrainingModel<T>
{
    private readonly RtdetrHybridEncoder<T> _encoder;
    private readonly RtdetrDecoderTransformer<T> _decoder;
    private readonly RTDETROptions<T> _rtdetr;
    private readonly NMS<T> _nms;
    private readonly AiDotNet.ComputerVision.Detection.Losses.DETRSetLoss<T> _detectionLoss;
    private readonly int _trainingClassCount;
    private readonly Random _denoisingRandom;

    private static readonly int[] BackboneTaps = { 1, 2, 3 };

    /// <inheritdoc/>
    public override string Name => $"RT-DETR-{Options.Size}";

    /// <summary>
    /// Creates RT-DETR. Pass an <see cref="RTDETROptions{T}"/> to change the decoder or denoising; plain
    /// <see cref="ObjectDetectionOptions{T}"/> use the paper's values.
    /// </summary>
    public RTDETR(ObjectDetectionOptions<T> options) : base(options)
    {
        _rtdetr = options as RTDETROptions<T> ?? new RTDETROptions<T>();
        if (_rtdetr.NumQueries <= 0) throw new ArgumentException("NumQueries must be positive.", nameof(options));

        var size = SizeConfig(options.Size);
        Backbone = new ResNet<T>(options: new ResNetBackboneOptions { Variant = size.Backbone });
        var channels = BackboneTaps.Select(tap => Backbone.OutputChannels[tap]).ToArray();
        _encoder = new RtdetrHybridEncoder<T>(channels, size.EncoderHidden, _rtdetr.NumHeads, size.EncoderFeedForward,
            size.Expansion, _rtdetr.PositionalTemperature);
        _decoder = new RtdetrDecoderTransformer<T>(size.EncoderHidden, _rtdetr.DecoderHiddenDimension, _rtdetr.NumHeads,
            size.DecoderLayers, _rtdetr.DecoderFeedForwardDimension, _rtdetr.NumQueries, _rtdetr.NumSamplingPoints, options.NumClasses);

        var lossOptions = options.SetPredictionLoss ?? AiDotNet.ComputerVision.Detection.Losses.DetrSetLossOptions.ForRtDetr();
        if (lossOptions.ClassificationLoss == SetPredictionClassificationLoss.SoftmaxCrossEntropy)
            throw new ArgumentException("RT-DETR's class head has independent sigmoid classes and no no-object class; use a varifocal or sigmoid focal set loss.", nameof(options));
        _trainingClassCount = options.NumClasses;
        _detectionLoss = new AiDotNet.ComputerVision.Detection.Losses.DETRSetLoss<T>(options.NumClasses, lossOptions);
        _denoisingRandom = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        _nms = new NMS<T>();

        // BatchNorm layers start in training mode. A detector starts in inference mode, as YOLO's blocks do,
        // so Predict and Detect normalize with the running statistics rather than with the batch's.
        _encoder.SetTrainingMode(false);
        _decoder.SetTrainingMode(false);
    }

    /// <summary>
    /// The paper's per-backbone configuration: R18 and R34 use CSP expansion 0.5, and R101's hybrid encoder
    /// is 384 wide with a 2048 feed-forward.
    /// </summary>
    private static (ResNetVariant Backbone, int EncoderHidden, int EncoderFeedForward, double Expansion, int DecoderLayers) SizeConfig(ModelSize size) => size switch
    {
        ModelSize.Nano => (ResNetVariant.ResNet18, 256, 1024, 0.5, 3),
        ModelSize.Small => (ResNetVariant.ResNet34, 256, 1024, 0.5, 4),
        ModelSize.Medium => (ResNetVariant.ResNet50, 256, 1024, 1.0, 6),
        ModelSize.Large or ModelSize.XLarge => (ResNetVariant.ResNet101, 384, 2048, 1.0, 6),
        _ => throw new ArgumentOutOfRangeException(nameof(size), size, "RT-DETR has no configuration for this size.")
    };

    /// <summary>The decoder, for tests that configure controlled weights.</summary>
    /// <remarks>A method, not a property: the parameter registry names components after the members that hold them.</remarks>
    internal RtdetrDecoderTransformer<T> GetDecoder() => _decoder;

    /// <summary>The full inference-shaped forward with every head output, for tests that differentiate it.</summary>
    internal DetrPass<T> ForwardPass(Tensor<T> input) => _decoder.Forward(_encoder.Forward(BackboneLevels(input)), null);

    /// <summary>The pre-update outputs, denoising plan and recorded loss of the last <see cref="TrainDetections"/> step.</summary>
    internal DetrTrainingRecord<T>? LastTrainingRecord { get; private set; }

    /// <inheritdoc/>
    /// <remarks>
    /// One step of the reference objective: the Hungarian varifocal + L1 + GIoU loss of every decoder layer and
    /// of the encoder's selected queries, plus the fixed-assignment denoising loss of every layer. Inputs are
    /// model-ready NCHW tensors, as for Predict, and targets are normalized against that input.
    /// </remarks>
    public void TrainDetections(Tensor<T> input, DetectionTrainingBatch<T> targets)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (input.Rank != 4 || input.Shape[0] <= 0 || input.Shape[1] != 3 || input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new ArgumentException("RT-DETR training requires a nonempty NCHW three-channel image batch.", nameof(input));
        targets.ValidateForModel(input.Shape[0], _trainingClassCount, _rtdetr.NumQueries);

        var plan = ContrastiveDenoising<T>.Plan(targets, _trainingClassCount, _rtdetr.DenoisingQueries,
            _rtdetr.LabelNoiseRatio, _rtdetr.BoxNoiseScale, _denoisingRandom);
        DetrPass<T>? pass = null;
        TrainWithTargets(input, targets,
            x =>
            {
                pass = _decoder.Forward(_encoder.Forward(BackboneLevels(x)), plan);
                return pass.All();
            },
            (heads, batch) =>
            {
                var current = pass ?? throw new InvalidOperationException("RT-DETR's training forward did not run.");
                var loss = Objective(current, batch);
                LastTrainingRecord = DetrTrainingRecord<T>.Capture(current, NumOps.ToDouble(loss.ToArray()[0]));
                return loss;
            });
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The generic tensor objective covers the same outputs as <see cref="TrainDetections"/>, each weighted
    /// equally as in the reference: the encoder's selected queries and every decoder layer, each in Predict's
    /// [class logits, box logits] format, scored by mean squared error against the target.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expectedOutput is null) throw new ArgumentNullException(nameof(expectedOutput));
        DetrPass<T>? pass = null;
        TrainWithTargets(input, expectedOutput,
            x =>
            {
                pass = _decoder.Forward(_encoder.Forward(BackboneLevels(x)), null);
                return pass.All();
            },
            (heads, target) =>
            {
                var current = pass ?? throw new InvalidOperationException("RT-DETR's training forward did not run.");
                var total = TensorModelTrainer<T>.MeanSquaredError(
                    CvTensorOps<T>.ConcatenateOutputs(new List<Tensor<T>> { current.EncoderClasses, current.EncoderBoxLogits }), target);
                for (int layer = 0; layer < current.Classes.Count; layer++)
                    total = Engine.TensorAdd(total, TensorModelTrainer<T>.MeanSquaredError(
                        CvTensorOps<T>.ConcatenateOutputs(new List<Tensor<T>> { current.Classes[layer], current.BoxLogits[layer] }), target));
                return total;
            });
    }

    private Tensor<T> Objective(DetrPass<T> pass, DetectionTrainingBatch<T> targets)
    {
        var total = _detectionLoss.ComputeTapeLoss(pass.EncoderClasses, pass.EncoderBoxes, targets);
        for (int layer = 0; layer < pass.Classes.Count; layer++)
            total = Engine.TensorAdd(total, _detectionLoss.ComputeTapeLoss(pass.Classes[layer], pass.Boxes[layer], targets));
        if (pass.Denoising is { } plan)
        {
            for (int layer = 0; layer < pass.DenoisingClasses.Count; layer++)
                total = Engine.TensorAdd(total, _detectionLoss.ComputeTapeLoss(
                    pass.DenoisingClasses[layer], pass.DenoisingBoxes[layer], plan.Targets, plan.Assignments));
        }
        return total;
    }

    private List<Tensor<T>> BackboneLevels(Tensor<T> input)
    {
        var features = EnsureBackbone.ExtractFeatures(input);
        return BackboneTaps.Select(tap => features[tap]).ToList();
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool training)
    {
        base.SetTrainingMode(training);
        _encoder.SetTrainingMode(training);
        _decoder.SetTrainingMode(training);
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
    /// <remarks>The last decoder layer's class logits <c>[N, K, C]</c> and box logits <c>[N, K, 4]</c> (pre-sigmoid).</remarks>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        var pass = _decoder.Forward(_encoder.Forward(BackboneLevels(input)), null);
        return new List<Tensor<T>> { pass.Classes[pass.Classes.Count - 1], pass.FinalBoxLogits };
    }

    /// <inheritdoc/>
    protected override List<Detection<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold,
        double nmsThreshold)
    {
        // RTDETRPostProcessor: sigmoid scores over every (query, class) pair, top-K, no NMS.
        var detections = DetrHeads<T>.TopKSigmoid(outputs[0], outputs[1], imageWidth, imageHeight, confidenceThreshold, ClassNames);
        var kept = _nms.Apply(detections, EffectiveNmsThreshold(nmsThreshold));
        return kept.Count > Options.MaxDetections ? kept.Take(Options.MaxDetections).ToList() : kept;
    }

    /// <inheritdoc/>
    /// <remarks>RT-DETR is NMS-free: its set loss trains one query per object, and the reference post-processing applies no suppression.</remarks>
    public override double EffectiveNmsThreshold(double requested) => 1.0;

    /// <inheritdoc/>
    protected override long GetHeadParameterCount() => _encoder.ParameterCount + _decoder.ParameterCount;

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var stream = File.OpenRead(pathOrUrl);
        using var reader = new BinaryReader(stream);
        if (reader.ReadInt32() != 0x52544452) // "RTDR"
            throw new InvalidDataException("Invalid weight file format: incorrect magic number.");
        int version = reader.ReadInt32();
        if (version != 2)
            throw new InvalidDataException($"Unsupported RT-DETR weight file version {version}; this build reads version 2 (hybrid encoder + deformable decoder).");
        string modelName = reader.ReadString();
        if (!modelName.StartsWith("RT-DETR", StringComparison.Ordinal))
            throw new InvalidDataException($"Weight file is for {modelName}, not RT-DETR.");
        EnsureBackbone.ReadParameters(reader);
        _encoder.Read(reader);
        _decoder.Read(reader);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(0x52544452); // "RTDR"
        writer.Write(2);
        writer.Write(Name);
        EnsureBackbone.WriteParameters(writer);
        _encoder.Write(writer);
        _decoder.Write(writer);
    }
}
