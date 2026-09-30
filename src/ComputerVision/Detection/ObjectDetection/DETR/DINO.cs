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
/// DINO (DETR with Improved deNoising anchOr boxes).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DINO is an end-to-end detector: it predicts a fixed set of boxes and learns
/// which of them are objects, so it needs no anchors or non-maximum suppression. It trains faster and more
/// accurately than the original DETR through three ideas. Contrastive denoising teaches the decoder to
/// repair noised copies of the real boxes and to reject harder negatives. Mixed query selection starts the
/// queries from the encoder's best positions. "Look forward twice" lets each layer's box correction
/// improve the layer before it.</para>
/// <para>
/// The architecture follows the reference implementation (IDEA-Research/DINO) at DINO-4scale:
/// <list type="bullet">
/// <item>The backbone's C3-C5 (C2-C5 at five scales) plus an extra stride-2 level are projected to 256
/// channels with GroupNorm.</item>
/// <item>Six deformable encoder layers (<see cref="MultiScaleDeformableAttention{T}"/>).</item>
/// <item>Two-stage proposals: the encoder's top-K positions are the anchors, and the content queries
/// are learnable.</item>
/// <item>Six decoder layers (self-attention, deformable cross-attention, FFN) with iterative box
/// refinement.</item>
/// <item>Shared sigmoid class and box heads.</item>
/// </list>
/// The training objective is the reference's: focal + L1 + GIoU with Hungarian matching on the final
/// layer, on each intermediate layer and on the encoder proposals. On top of that come the contrastive
/// denoising losses of every layer.
/// </para>
/// <para>Reference: Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection", ICLR 2023</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection",
    "https://arxiv.org/abs/2203.03605",
    Year = 2023,
    Authors = "Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, Heung-Yeung Shum")]
public partial class DINO<T> : ObjectDetectorBase<T>, IDetectionTrainingModel<T>
{
    private readonly DinoDetectionTransformer<T> _transformer;
    private readonly DINOOptions<T> _dino;
    private readonly int[] _backboneTaps;
    private readonly NMS<T> _nms;
    private readonly AiDotNet.ComputerVision.Detection.Losses.DETRSetLoss<T> _detectionLoss;
    private readonly int _trainingClassCount;
    private readonly Random _denoisingRandom;

    /// <inheritdoc/>
    public override string Name => $"DINO-{Options.Size}";

    /// <summary>
    /// Creates DINO. Pass a <see cref="DINOOptions{T}"/> to change the transformer; plain
    /// <see cref="ObjectDetectionOptions{T}"/> use the paper's DINO-4scale values.
    /// </summary>
    public DINO(ObjectDetectionOptions<T> options) : base(options)
    {
        _dino = options as DINOOptions<T> ?? new DINOOptions<T>();
        if (_dino.NumQueries <= 0) throw new ArgumentException("NumQueries must be positive.", nameof(options));
        if (_dino.NumDecoderLayers <= 0) throw new ArgumentException("DINO needs at least one decoder layer.", nameof(options));

        // ModelSize selects the paper's backbone: ResNet-50 4-scale, Swin-L 4-scale, or Swin-L 5-scale.
        bool fiveScale = options.Size == ModelSize.XLarge;
        if (options.Size is ModelSize.Large or ModelSize.XLarge)
            Backbone = new SwinTransformer<T>(new SwinTransformerOptions { Variant = SwinVariant.SwinLarge });
        else
            Backbone = new ResNet<T>(options: new ResNetBackboneOptions { Variant = ResNetVariant.ResNet50 });
        _backboneTaps = fiveScale ? new[] { 0, 1, 2, 3 } : new[] { 1, 2, 3 };
        var channels = _backboneTaps.Select(tap => Backbone.OutputChannels[tap]).ToArray();

        _transformer = new DinoDetectionTransformer<T>(
            new DINOOptionsView(_dino.HiddenDimension, _dino.NumHeads, _dino.NumEncoderLayers, _dino.NumDecoderLayers,
                _dino.FeedForwardDimension, _dino.NumQueries, _dino.NumSamplingPoints, _dino.PositionalTemperature),
            channels, options.NumClasses);

        var lossOptions = options.SetPredictionLoss ?? AiDotNet.ComputerVision.Detection.Losses.DetrSetLossOptions.ForDino();
        if (lossOptions.ClassificationLoss == SetPredictionClassificationLoss.SoftmaxCrossEntropy)
            throw new ArgumentException("DINO's class head has independent sigmoid classes and no no-object class; use a sigmoid focal or varifocal set loss.", nameof(options));
        _trainingClassCount = options.NumClasses;
        _detectionLoss = new AiDotNet.ComputerVision.Detection.Losses.DETRSetLoss<T>(options.NumClasses, lossOptions);
        _denoisingRandom = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        _nms = new NMS<T>();
    }

    /// <summary>
    /// The pre-update outputs, denoising plan and recorded loss of the last <see cref="TrainDetections"/> step,
    /// for tests that re-derive the objective independently.
    /// </summary>
    internal DetrTrainingRecord<T>? LastTrainingRecord { get; private set; }

    /// <summary>The transformer and heads, for tests that configure controlled weights.</summary>
    /// <remarks>A method, not a property: the parameter registry names components after the members that hold them.</remarks>
    internal DinoDetectionTransformer<T> GetTransformer() => _transformer;

    /// <inheritdoc/>
    /// <remarks>
    /// One step of the reference objective: the Hungarian set loss of the final layer, of every intermediate
    /// layer and of the encoder proposals, plus the fixed-assignment denoising loss of every layer. Inputs are
    /// model-ready NCHW tensors, as for Predict, and targets are normalized against that input.
    /// </remarks>
    public void TrainDetections(Tensor<T> input, DetectionTrainingBatch<T> targets)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (input.Rank != 4 || input.Shape[0] <= 0 || input.Shape[1] != 3 || input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new ArgumentException("DINO training requires a nonempty NCHW three-channel image batch.", nameof(input));
        targets.ValidateForModel(input.Shape[0], _trainingClassCount, _dino.NumQueries);

        var plan = ContrastiveDenoising<T>.Plan(targets, _trainingClassCount, _dino.DenoisingQueries,
            _dino.LabelNoiseRatio, _dino.BoxNoiseScale, _denoisingRandom);
        DetrPass<T>? pass = null;
        TrainWithTargets(input, targets,
            x =>
            {
                pass = _transformer.Forward(BackboneLevels(x), plan);
                return pass.All();
            },
            (heads, batch) =>
            {
                var current = pass ?? throw new InvalidOperationException("DINO's training forward did not run.");
                var loss = Objective(current, batch);
                LastTrainingRecord = DetrTrainingRecord<T>.Capture(current, NumOps.ToDouble(loss.ToArray()[0]));
                return loss;
            });
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The generic tensor objective covers the same outputs as <see cref="TrainDetections"/>, each layer
    /// weighted equally as in the reference: the encoder proposals and every decoder layer, each in Predict's
    /// [class logits, box logits] format, scored by mean squared error against the target. There are no
    /// targets to noise here, so no denoising queries run.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expectedOutput is null) throw new ArgumentNullException(nameof(expectedOutput));
        DetrPass<T>? pass = null;
        TrainWithTargets(input, expectedOutput,
            x =>
            {
                pass = _transformer.Forward(BackboneLevels(x), null);
                return pass.All();
            },
            (heads, target) =>
            {
                var current = pass ?? throw new InvalidOperationException("DINO's training forward did not run.");
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
        return _backboneTaps.Select(tap => features[tap]).ToList();
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
    /// <remarks>The final decoder layer's class logits <c>[N, K, C]</c> and box logits <c>[N, K, 4]</c> (pre-sigmoid).</remarks>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        var pass = _transformer.Forward(BackboneLevels(input), null);
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
        // The reference PostProcess: every (query, class) pair scored by its own sigmoid, no NMS (see
        // EffectiveNmsThreshold).
        var detections = DetrHeads<T>.TopKSigmoid(outputs[0], outputs[1], imageWidth, imageHeight, confidenceThreshold, ClassNames);
        var kept = _nms.Apply(detections, EffectiveNmsThreshold(nmsThreshold)); return kept.Count > Options.MaxDetections ? kept.Take(Options.MaxDetections).ToList() : kept;
    }

    /// <inheritdoc/>
    /// <remarks>DINO is NMS-free: one query per object is what its set loss trains, and the reference post-processing applies no suppression.</remarks>
    public override double EffectiveNmsThreshold(double requested) => 1.0;

    /// <inheritdoc/>
    protected override long GetHeadParameterCount() => _transformer.ParameterCount;

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var stream = File.OpenRead(pathOrUrl);
        using var reader = new BinaryReader(stream);
        if (reader.ReadInt32() != 0x44494E4F) // "DINO" in ASCII
            throw new InvalidDataException("Invalid weight file format: incorrect magic number.");
        int version = reader.ReadInt32();
        if (version != 2)
            throw new InvalidDataException($"Unsupported DINO weight file version {version}; this build reads version 2 (deformable DINO).");
        string modelName = reader.ReadString();
        if (!modelName.StartsWith("DINO", StringComparison.Ordinal))
            throw new InvalidDataException($"Weight file is for {modelName}, not DINO.");
        EnsureBackbone.ReadParameters(reader);
        _transformer.Read(reader);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(0x44494E4F); // "DINO" in ASCII
        writer.Write(2);
        writer.Write(Name);
        EnsureBackbone.WriteParameters(writer);
        _transformer.Write(writer);
    }

}

/// <summary>Host copies of one DINO training step's head outputs and its recorded loss.</summary>
internal sealed class DetrTrainingRecord<T>
{
    private DetrTrainingRecord(double loss, int queries, int classes, double[][] classes2, double[][] boxes,
        double[][] denoisingClasses, double[][] denoisingBoxes, double[] encoderClasses, double[] encoderBoxes, ContrastiveDenoisingPlan<T>? plan)
    {
        Loss = loss;
        Queries = queries;
        Classes = classes;
        LayerClasses = classes2;
        LayerBoxes = boxes;
        DenoisingClasses = denoisingClasses;
        DenoisingBoxes = denoisingBoxes;
        EncoderClasses = encoderClasses;
        EncoderBoxes = encoderBoxes;
        Denoising = plan;
    }

    public double Loss { get; }
    public int Queries { get; }
    public int Classes { get; }
    public double[][] LayerClasses { get; }
    public double[][] LayerBoxes { get; }
    public double[][] DenoisingClasses { get; }
    public double[][] DenoisingBoxes { get; }
    public double[] EncoderClasses { get; }
    public double[] EncoderBoxes { get; }
    public ContrastiveDenoisingPlan<T>? Denoising { get; }

    internal static DetrTrainingRecord<T> Capture(DetrPass<T> pass, double loss)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double[] Host(Tensor<T> t) => t.ToArray().Select(v => ops.ToDouble(v)).ToArray();
        return new DetrTrainingRecord<T>(loss, pass.Classes[0].Shape[1], pass.Classes[0].Shape[2],
            pass.Classes.Select(Host).ToArray(), pass.Boxes.Select(Host).ToArray(),
            pass.DenoisingClasses.Select(Host).ToArray(), pass.DenoisingBoxes.Select(Host).ToArray(),
            Host(pass.EncoderClasses), Host(pass.EncoderBoxes), pass.Denoising);
    }
}
