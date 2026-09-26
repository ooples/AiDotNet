using AiDotNet.Tensors.Engines;
using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;

/// <summary>
/// Cascade R-CNN - Multi-stage object detection with progressive refinement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Cascade R-CNN improves upon Faster R-CNN by using multiple
/// detection stages, each trained with progressively higher IoU thresholds. This allows
/// the model to produce higher quality detections through progressive refinement.</para>
///
/// <para>Key features:
/// - Multiple cascade stages (typically 3) for progressive refinement
/// - Each stage uses higher IoU threshold for training
/// - Bounding boxes are refined at each stage
/// - Achieves higher accuracy than Faster R-CNN at slight speed cost
/// </para>
///
/// <para>Reference: Cai and Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection",
/// CVPR 2018</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Cascade R-CNN: Delving into High Quality Object Detection",
    "https://arxiv.org/abs/1712.00726",
    Year = 2018,
    Authors = "Zhaowei Cai, Nuno Vasconcelos")]
public partial class CascadeRCNN<T> : ObjectDetectorBase<T>, IDetectionTrainingModel<T>
{
    private readonly AiDotNet.ComputerVision.Detection.Losses.TwoStageDetectionLoss<T> _detectionLoss;

    [AiDotNet.Attributes.Scratch]
    private Random? _trainingRandom;
    private readonly RPN<T> _rpn;
    private readonly RoIAlign<T> _roiAlign;
    private readonly List<CascadeStage<T>> _stages;
    private readonly int _roiOutputSize;
    private readonly int _numStages;
    private readonly double[] _iouThresholds;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"Cascade-RCNN-{Options.Size}";

    /// <summary>
    /// Creates a new Cascade R-CNN detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    /// <param name="numStages">Number of cascade stages (default 3).</param>
    public CascadeRCNN(ObjectDetectionOptions<T> options, int numStages = 3) : base(options)
    {
        _numStages = numStages;
        _iouThresholds = new[] { 0.5, 0.6, 0.7 }; // Progressive IoU thresholds

        var (hiddenDim, roiOutputSize) = GetSizeConfig(options.Size);
        _roiOutputSize = roiOutputSize;

        // Backbone: ResNet-50 with FPN
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);
        Neck = new FPN<T>(Backbone.OutputChannels.ToArray(), outputChannels: 256);

        // Region Proposal Network
        _rpn = new RPN<T>(256, hiddenDim);

        // RoI feature extraction
        _roiAlign = new RoIAlign<T>(roiOutputSize, samplingRatio: 2);

        // Cascade stages
        int roiFeatureSize = 256 * roiOutputSize * roiOutputSize;
        _stages = new List<CascadeStage<T>>();
        for (int i = 0; i < numStages; i++)
        {
            _stages.Add(new CascadeStage<T>(roiFeatureSize, hiddenDim, options.NumClasses + 1));
        }

        _nms = new NMS<T>();
        _detectionLoss = new AiDotNet.ComputerVision.Detection.Losses.TwoStageDetectionLoss<T>(options.NumClasses, numStages,
            options.TwoStageLoss ?? new AiDotNet.ComputerVision.Detection.Losses.TwoStageDetectionLossOptions());
    }

    private static (int hiddenDim, int roiOutputSize) GetSizeConfig(ModelSize size) => size switch
    {
        ModelSize.Nano => (128, 5),
        ModelSize.Small => (192, 7),
        ModelSize.Medium => (256, 7),
        ModelSize.Large => (384, 7),
        ModelSize.XLarge => (512, 7),
        _ => (256, 7)
    };

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
        var stages = ForwardStages(input, null, out _, out var objectness, out var bboxDeltas);
        if (stages is null)
        {
            return new List<Tensor<T>>
            {
                new Tensor<T>(new[] { 0, Options.NumClasses + 1 }),
                new Tensor<T>(new[] { 0, (Options.NumClasses + 1) * 4 }),
                new Tensor<T>(new[] { 0, 4 }),
                objectness,
                bboxDeltas
            };
        }

        // PostProcess reads the first three entries: the last stage's logits, deltas and the boxes that stage
        // received. The earlier stages' outputs and the RPN's follow, so every head reaches a training objective.
        var last = stages[stages.Count - 1];
        var outputs = new List<Tensor<T>> { last.ClassLogits, last.BoxDeltas, last.Boxes };
        for (int stage = 0; stage < stages.Count - 1; stage++)
        {
            outputs.Add(stages[stage].ClassLogits);
            outputs.Add(stages[stage].BoxDeltas);
        }
        outputs.Add(objectness);
        outputs.Add(bboxDeltas);
        return outputs;
    }

    /// <summary>Trains the proposal network and every cascade stage with their published objectives.</summary>
    /// <remarks>
    /// <para>
    /// One update sums the region proposal loss (Ren et al. 2015) and, for each stage t, the region-of-interest loss
    /// L_cls + [y_t &gt;= 1] L_loc on the boxes that stage actually received, labeled at that stage's IoU threshold
    /// (Cai and Vasconcelos 2018, Eq. 8; thresholds 0.5, 0.6, 0.7). As in the reference implementation, the object
    /// boxes join the first stage's proposals, and each later stage resamples the previous stage's regressed boxes.
    /// Override the sampling and weights with <see cref="ObjectDetectionOptions{T}.TwoStageLoss"/>.
    /// </para>
    /// <para>
    /// Inputs are model-ready NCHW tensors, as for Predict, and targets are normalized against that input size. The
    /// stages classify the proposals of a single image per forward pass, so each step takes one image.
    /// </para>
    /// </remarks>
    public void TrainDetections(Tensor<T> input, DetectionTrainingBatch<T> targets)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (input.Rank != 4 || input.Shape[1] != 3 || input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new ArgumentException("Cascade R-CNN training requires a three-channel NCHW image batch.", nameof(input));
        if (input.Shape[0] != 1)
            throw new ArgumentException("Cascade R-CNN stages classify one image's proposals per forward pass; train one image per step.", nameof(input));
        targets.ValidateForModel(1, Options.NumClasses, int.MaxValue);
        int height = input.Shape[2];
        int width = input.Shape[3];
        var gold = targets[0].Select(target => TwoStageTargets.PixelCorners(target, width, height)).ToList();
        var goldClasses = targets[0].Select(target => target.ClassId).ToList();
        var random = _trainingRandom ??= Options.RandomSeed is int seed
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        List<BoundingBox<T>> anchors = new();
        TrainWithTargets(input, targets,
            image => ForwardForTraining(image, gold, out anchors),
            (heads, batch) =>
            {
                var loss = _detectionLoss.ComputeProposalLoss(TwoStageTargets.FirstImage(heads[heads.Count - 2]),
                    TwoStageTargets.FirstImage(heads[heads.Count - 1]), anchors, gold, _rpn.AnchorsPerLocation, random);
                int stages = (heads.Count - 2) / 3;
                for (int stage = 0; stage < stages; stage++)
                    loss = Engine.TensorAdd(loss, _detectionLoss.ComputeStageLoss(
                        heads[3 * stage], heads[3 * stage + 1], heads[3 * stage + 2], gold, goldClasses, stage, random));
                return loss;
            });
    }

    /// <summary>Each stage's logits, deltas and received boxes in order, then the RPN's objectness and deltas.</summary>
    private List<Tensor<T>> ForwardForTraining(Tensor<T> input, IReadOnlyList<double[]> gold, out List<BoundingBox<T>> anchors)
    {
        var stages = ForwardStages(input, gold, out anchors, out var objectness, out var bboxDeltas);
        var outputs = new List<Tensor<T>>();
        if (stages is not null)
        {
            foreach (var stage in stages)
            {
                outputs.Add(stage.ClassLogits);
                outputs.Add(stage.BoxDeltas);
                outputs.Add(stage.Boxes);
            }
        }
        outputs.Add(objectness);
        outputs.Add(bboxDeltas);
        return outputs;
    }

    /// <summary>
    /// Runs the backbone, proposal network and every cascade stage, optionally adding boxes to the first stage's
    /// proposals. Returns null when no stage receives a box.
    /// </summary>
    private List<CascadeStageOutput>? ForwardStages(Tensor<T> input, IReadOnlyList<double[]>? extraProposals,
        out List<BoundingBox<T>> anchors, out Tensor<T> objectness, out Tensor<T> bboxDeltas)
    {
        int imageHeight = input.Shape[2];
        int imageWidth = input.Shape[3];

        // Extract backbone features
        var backboneFeatures = EnsureBackbone.ExtractFeatures(input);

        // Apply FPN neck to get multi-scale features
        var fpnFeatures = EnsureNeck.Forward(backboneFeatures);

        // Cascade R-CNN (Cai & Vasconcelos 2018) on an FPN (Lin et al. 2017; detectron2, torchvision): the shared
        // RPN head runs on every level P2-P5 plus P6 (P5 subsampled by 2), each with its own anchor size, and each
        // RoI is pooled from the level matching its size.
        var rpnLevels = new List<Tensor<T>>(fpnFeatures) { CvTensorOps<T>.MaxPoolPadded(fpnFeatures[^1], 1, 2, 0) };
        var (rpnObjectness, rpnDeltas, levelAnchors, levelAnchorCounts) = _rpn.ForwardLevels(rpnLevels);
        objectness = rpnObjectness;
        bboxDeltas = rpnDeltas;
        anchors = levelAnchors;

        // Generate initial proposals: top 1000 per level, NMS within each level, best 1000 overall.
        var initialProposals = _rpn.GenerateProposals(
            rpnObjectness, rpnDeltas, levelAnchors,
            imageHeight, imageWidth,
            preNmsTopK: 1000,
            postNmsTopK: 1000,
            nmsThreshold: 0.7,
            levelAnchorCounts: levelAnchorCounts);

        var currentBoxes = initialProposals.Count == 0 ? new Tensor<T>(new[] { 0, 4 }) : initialProposals[0].boxes;
        if (extraProposals is { Count: > 0 })
            currentBoxes = TwoStageTargets.AppendBoxes(currentBoxes, extraProposals);
        if (currentBoxes.Shape[0] == 0)
            return null;

        var stages = new List<CascadeStageOutput>(_numStages);
        for (int stageIdx = 0; stageIdx < _numStages; stageIdx++)
        {
            // Extract RoI features for current boxes, each from its size-matched pyramid level (the level can
            // change between stages as refinement resizes the boxes)
            var roiFeatures = FpnRoIPooler<T>.Pool(_roiAlign, fpnFeatures, EnsureBackbone.Strides, currentBoxes);
            var flattenedFeatures = FlattenRoIFeatures(roiFeatures);
            var (classLogits, boxDeltas) = _stages[stageIdx].Forward(flattenedFeatures);
            if (boxDeltas is null)
                throw new InvalidOperationException("Cascade stage did not produce box deltas.");
            stages.Add(new CascadeStageOutput(classLogits, boxDeltas, currentBoxes));

            // Refine boxes for the next stage. The boxes are constants to RoIAlign, so refinement carries no
            // gradient; each stage trains through its own logits and deltas above.
            if (stageIdx < _numStages - 1)
                currentBoxes = RefineBoxes(currentBoxes, boxDeltas, imageWidth, imageHeight);
        }
        return stages;
    }

    /// <summary>One cascade stage's raw heads and the boxes it classified.</summary>
    private sealed class CascadeStageOutput
    {
        internal CascadeStageOutput(Tensor<T> classLogits, Tensor<T> boxDeltas, Tensor<T> boxes)
        {
            ClassLogits = classLogits;
            BoxDeltas = boxDeltas;
            Boxes = boxes;
        }

        internal Tensor<T> ClassLogits { get; }
        internal Tensor<T> BoxDeltas { get; }
        internal Tensor<T> Boxes { get; }
    }

    /// <inheritdoc/>
    protected override List<Detection<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold,
        double nmsThreshold)
    {
        if (outputs.Count < 3 || outputs[0].Shape[0] == 0)
        {
            return new List<Detection<T>>();
        }

        var classLogits = outputs[0];
        var boxDeltas = outputs[1];
        var proposalBoxes = outputs[2];

        int numProposals = classLogits.Shape[0];
        int numClasses = Options.NumClasses + 1;

        var candidateDetections = new List<Detection<T>>();

        for (int i = 0; i < numProposals; i++)
        {
            // Apply softmax
            double maxLogit = double.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                maxLogit = Math.Max(maxLogit, NumOps.ToDouble(classLogits[i, c]));
            }

            double sumExp = 0;
            var probs = new double[numClasses];
            for (int c = 0; c < numClasses; c++)
            {
                probs[c] = Math.Exp(NumOps.ToDouble(classLogits[i, c]) - maxLogit);
                sumExp += probs[c];
            }

            for (int c = 0; c < numClasses; c++)
            {
                probs[c] /= sumExp;
            }

            // Find best non-background class
            int bestClass = 0;
            double bestScore = 0;
            for (int c = 1; c < numClasses; c++)
            {
                if (probs[c] > bestScore)
                {
                    bestScore = probs[c];
                    bestClass = c;
                }
            }

            if (bestScore < confidenceThreshold) continue;

            // The boxes have already been refined through cascade stages
            // Apply final delta for this class
            double px1 = NumOps.ToDouble(proposalBoxes[i, 0]);
            double py1 = NumOps.ToDouble(proposalBoxes[i, 1]);
            double px2 = NumOps.ToDouble(proposalBoxes[i, 2]);
            double py2 = NumOps.ToDouble(proposalBoxes[i, 3]);

            double pw = px2 - px1;
            double ph = py2 - py1;
            double pcx = px1 + pw / 2;
            double pcy = py1 + ph / 2;

            int deltaOffset = bestClass * 4;
            double dx = NumOps.ToDouble(boxDeltas[i, deltaOffset]);
            double dy = NumOps.ToDouble(boxDeltas[i, deltaOffset + 1]);
            double dw = NumOps.ToDouble(boxDeltas[i, deltaOffset + 2]);
            double dh = NumOps.ToDouble(boxDeltas[i, deltaOffset + 3]);

            double predCx = pcx + dx * pw;
            double predCy = pcy + dy * ph;
            double predW = pw * Math.Exp(Math.Min(dw, 4.0));
            double predH = ph * Math.Exp(Math.Min(dh, 4.0));

            // Decoded in network-input coordinates; map to the source image before clipping.
            // (RefineBoxes does NOT do this: it works on proposals in the input frame on purpose.)
            var (scaleX, scaleY) = InputToImageScale(imageWidth, imageHeight);
            double x1 = Math.Max(0, (predCx - predW / 2) * scaleX);
            double y1 = Math.Max(0, (predCy - predH / 2) * scaleY);
            double x2 = Math.Min(imageWidth, (predCx + predW / 2) * scaleX);
            double y2 = Math.Min(imageHeight, (predCy + predH / 2) * scaleY);

            if (x2 <= x1 || y2 <= y1) continue;

            int classId = bestClass - 1;

            var box = new BoundingBox<T>(
                NumOps.FromDouble(x1),
                NumOps.FromDouble(y1),
                NumOps.FromDouble(x2),
                NumOps.FromDouble(y2));

            candidateDetections.Add(new Detection<T>(
                box,
                classId,
                NumOps.FromDouble(bestScore),
                classId < ClassNames.Length ? ClassNames[classId] : null));
        }

        // Apply NMS
        var nmsResults = _nms.Apply(candidateDetections, nmsThreshold);

        if (nmsResults.Count > Options.MaxDetections)
        {
            return nmsResults.Take(Options.MaxDetections).ToList();
        }

        return nmsResults;
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        long count = _rpn.GetParameterCount();
        foreach (var stage in _stages)
        {
            count += stage.GetParameterCount();
        }
        return count;
    }

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using var stream = File.OpenRead(pathOrUrl);
        using var reader = new BinaryReader(stream);

        // Read and verify magic number and version
        int magic = reader.ReadInt32();
        if (magic != 0x43524E4E) // "CRNN" in ASCII (Cascade RCNN)
        {
            throw new InvalidDataException("Invalid weight file format: incorrect magic number.");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported weight file version: {version}.");
        }

        // Read model configuration
        string modelName = reader.ReadString();
        if (!modelName.StartsWith("Cascade-RCNN"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not Cascade-RCNN.");
        }

        // Read backbone parameters
        EnsureBackbone.ReadParameters(reader);

        // Read neck parameters
        EnsureNeck.ReadParameters(reader);

        // Read RPN parameters
        _rpn.ReadParameters(reader);

        // Read cascade stage parameters
        int stageCount = reader.ReadInt32();
        if (stageCount != _stages.Count)
        {
            throw new InvalidDataException($"Stage count mismatch: expected {_stages.Count}, got {stageCount}.");
        }
        foreach (var stage in _stages)
        {
            stage.ReadParameters(reader);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write magic number and version for identification
        writer.Write(0x43524E4E); // "CRNN" in ASCII (Cascade RCNN)
        writer.Write(1); // Version 1

        // Write model configuration
        writer.Write(Name);

        // Write backbone parameters
        EnsureBackbone.WriteParameters(writer);

        // Write neck parameters
        EnsureNeck.WriteParameters(writer);

        // Write RPN parameters
        _rpn.WriteParameters(writer);

        // Write cascade stage parameters
        writer.Write(_stages.Count);
        foreach (var stage in _stages)
        {
            stage.WriteParameters(writer);
        }
    }

    private Tensor<T> FlattenRoIFeatures(Tensor<T> roiFeatures)
        => AiDotNetEngine.Current.Reshape(
            roiFeatures, new[] { roiFeatures.Shape[0], roiFeatures.Shape[1] * roiFeatures.Shape[2] * roiFeatures.Shape[3] });

    /// <summary>
    /// Applies one stage's box deltas to its input boxes, giving the next stage's boxes.
    /// </summary>
    /// <param name="boxes">Input boxes <c>[N, 4]</c> as (x1, y1, x2, y2) in network-input coordinates.</param>
    /// <param name="deltas">The stage's regression output <c>[N, 4 * numClasses]</c>; the first
    /// foreground class's (dx, dy, dw, dh) are applied.</param>
    /// <param name="imageWidth">Right clip bound.</param>
    /// <param name="imageHeight">Bottom clip bound.</param>
    /// <returns>The refined boxes <c>[N, 4]</c>, still in network-input coordinates.</returns>
    /// <remarks>
    /// Engine ops over whole columns rather than a per-box scalar loop. The refined boxes only tell
    /// the next stage's RoIAlign WHERE to sample, and RoIAlign treats box coordinates as data, so no
    /// gradient flows back through them - the same "detached proposals" rule as Cai and Vasconcelos
    /// (2018) and detectron2's cascade head. The deltas themselves still reach the loss through the
    /// stage outputs <see cref="Forward"/> returns.
    /// </remarks>
    internal static Tensor<T> RefineBoxes(Tensor<T> boxes, Tensor<T> deltas, int imageWidth, int imageHeight)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        Tensor<T> Column(Tensor<T> source, int index) => engine.TensorNarrow(source, 1, index, 1);
        var half = ops.FromDouble(0.5);
        var unbounded = ops.FromDouble(double.MinValue);

        var px1 = Column(boxes, 0);
        var py1 = Column(boxes, 1);
        var pw = engine.TensorSubtract(Column(boxes, 2), px1);
        var ph = engine.TensorSubtract(Column(boxes, 3), py1);
        var pcx = engine.TensorAdd(px1, engine.TensorMultiplyScalar(pw, half));
        var pcy = engine.TensorAdd(py1, engine.TensorMultiplyScalar(ph, half));

        // Deltas of the first foreground class (columns 4..7; class 0 is background). The scale
        // deltas are capped at 4 before exponentiating, as in the per-box version this replaces.
        const int deltaOffset = 4;
        var predCx = engine.TensorAdd(pcx, engine.TensorMultiply(Column(deltas, deltaOffset), pw));
        var predCy = engine.TensorAdd(pcy, engine.TensorMultiply(Column(deltas, deltaOffset + 1), ph));
        var cap = ops.FromDouble(4.0);
        var predW = engine.TensorMultiply(pw, engine.TensorExp(engine.TensorClamp(Column(deltas, deltaOffset + 2), unbounded, cap)));
        var predH = engine.TensorMultiply(ph, engine.TensorExp(engine.TensorClamp(Column(deltas, deltaOffset + 3), unbounded, cap)));
        var halfW = engine.TensorMultiplyScalar(predW, half);
        var halfH = engine.TensorMultiplyScalar(predH, half);

        // Clip each edge on its own side only: x1/y1 at zero, x2/y2 at the image extent.
        var x1 = engine.TensorClampMin(engine.TensorSubtract(predCx, halfW), ops.Zero);
        var y1 = engine.TensorClampMin(engine.TensorSubtract(predCy, halfH), ops.Zero);
        var x2 = engine.TensorClamp(engine.TensorAdd(predCx, halfW), unbounded, ops.FromDouble(imageWidth));
        var y2 = engine.TensorClamp(engine.TensorAdd(predCy, halfH), unbounded, ops.FromDouble(imageHeight));

        return engine.TensorConcatenate(new[] { x1, y1, x2, y2 }, 1);
    }
}

/// <summary>
/// A single stage in the Cascade R-CNN pipeline.
/// </summary>
internal class CascadeStage<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Dense<T> _fc1;
    private readonly Dense<T> _fc2;
    private readonly Dense<T> _clsHead;
    private readonly Dense<T> _regHead;
    private readonly int _hiddenDim;
    private readonly int _numClasses;

    public CascadeStage(int inputSize, int hiddenDim, int numClasses)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numClasses = numClasses;

        // Shared FC layers
        _fc1 = new Dense<T>(inputSize, hiddenDim);
        _fc2 = new Dense<T>(hiddenDim, hiddenDim);

        // Heads
        _clsHead = new Dense<T>(hiddenDim, numClasses);
        _regHead = new Dense<T>(hiddenDim, numClasses * 4);
    }

    public (Tensor<T> classLogits, Tensor<T> boxDeltas) Forward(Tensor<T> features)
    {
        // Apply shared FC layers with ReLU
        var x = _fc1.Forward(features);
        x = ApplyReLU(x);

        x = _fc2.Forward(x);
        x = ApplyReLU(x);

        // Classification and regression heads
        var classLogits = _clsHead.Forward(x);
        var boxDeltas = _regHead.Forward(x);

        return (classLogits, boxDeltas);
    }

    public long GetParameterCount()
    {
        return _fc1.GetParameterCount() +
               _fc2.GetParameterCount() +
               _clsHead.GetParameterCount() +
               _regHead.GetParameterCount();
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numClasses);
        _fc1.WriteParameters(writer);
        _fc2.WriteParameters(writer);
        _clsHead.WriteParameters(writer);
        _regHead.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numClasses = reader.ReadInt32();

        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidDataException($"CascadeStage hiddenDim mismatch: expected {_hiddenDim}, got {hiddenDim}.");
        }

        if (numClasses != _numClasses)
        {
            throw new InvalidDataException($"CascadeStage numClasses mismatch: expected {_numClasses}, got {numClasses}.");
        }

        _fc1.ReadParameters(reader);
        _fc2.ReadParameters(reader);
        _clsHead.ReadParameters(reader);
        _regHead.ReadParameters(reader);
    }

    /// <summary>
    /// Elementwise ReLU, delegated to the engine.
    /// </summary>
    /// <remarks>
    /// This was a scalar loop that read each element out to <c>double</c> and wrote a fresh
    /// tensor. Arithmetically identical, but it severed the autodiff tape: the gradient chain
    /// stopped here, so every trainable layer UPSTREAM of this call received no gradient and
    /// silently never trained. The engine op records itself on the tape.
    /// </remarks>
    private Tensor<T> ApplyReLU(Tensor<T> x) => AiDotNetEngine.Current.ReLU(x);

    /// <inheritdoc />
    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _fc1;
        yield return _fc2;
        yield return _clsHead;
        yield return _regHead;
    }
}
