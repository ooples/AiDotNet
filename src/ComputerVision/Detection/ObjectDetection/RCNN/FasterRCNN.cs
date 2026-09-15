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

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;

/// <summary>
/// Faster R-CNN - Two-stage object detection with region proposal network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Faster R-CNN is a foundational object detection model that
/// works in two stages: first proposing regions that might contain objects (RPN),
/// then classifying and refining those proposals. It's highly accurate but slower
/// than single-stage detectors like YOLO.</para>
///
/// <para>Key features:
/// - Two-stage detection: RPN + Fast R-CNN
/// - End-to-end trainable
/// - High accuracy through refined proposals
/// - RoI pooling/align for fixed-size feature extraction
/// </para>
///
/// <para>Reference: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with
/// Region Proposal Networks", NeurIPS 2015</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
    "https://arxiv.org/abs/1506.01497",
    Year = 2015,
    Authors = "Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun")]
public partial class FasterRCNN<T> : ObjectDetectorBase<T>, IDetectionTrainingModel<T>
{
    private readonly AiDotNet.ComputerVision.Detection.Losses.TwoStageDetectionLoss<T> _detectionLoss;

    [AiDotNet.Attributes.Scratch]
    private Random? _trainingRandom;
    private readonly RPN<T> _rpn;
    private readonly RoIAlign<T> _roiAlign;
    private readonly Dense<T> _fcClassifier;
    private readonly Dense<T> _fcBoxRegressor;
    private readonly int _roiOutputSize;
    private readonly int _hiddenDim;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"Faster-RCNN-{Options.Size}";

    /// <summary>
    /// Creates a new Faster R-CNN detector with default options derived from the architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="size">Model size variant (default: Small).</param>
    /// <param name="numClasses">Number of detection classes (default: 80 for COCO).</param>
    public FasterRCNN(
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
    /// Creates a new Faster R-CNN detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public FasterRCNN(ObjectDetectionOptions<T> options) : base(options)
    {
        var (hiddenDim, roiOutputSize) = GetSizeConfig(options.Size);
        _hiddenDim = hiddenDim;
        _roiOutputSize = roiOutputSize;

        // Backbone: ResNet-50 with FPN
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);
        Neck = new FPN<T>(Backbone.OutputChannels.ToArray(), outputChannels: 256);

        // Region Proposal Network
        _rpn = new RPN<T>(256, hiddenDim);

        // RoI feature extraction
        _roiAlign = new RoIAlign<T>(roiOutputSize, samplingRatio: 2);

        // Detection head: classification + bounding box regression
        int roiFeatureSize = 256 * roiOutputSize * roiOutputSize;
        _fcClassifier = new Dense<T>(roiFeatureSize, options.NumClasses + 1); // +1 for background
        _fcBoxRegressor = new Dense<T>(roiFeatureSize, (options.NumClasses + 1) * 4);

        _nms = new NMS<T>();
        _detectionLoss = new AiDotNet.ComputerVision.Detection.Losses.TwoStageDetectionLoss<T>(options.NumClasses, 1,
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
    protected override List<Tensor<T>> Forward(Tensor<T> input) => ForwardDetection(input, null, out _);

    /// <summary>Trains the proposal network and the detection head with the Faster R-CNN objectives.</summary>
    /// <remarks>
    /// <para>
    /// One update sums the region proposal loss (Ren et al. 2015: IoU above 0.7 or best anchor positive, below 0.3
    /// negative, 256 anchors at up to 1:1, lambda = 10 over the anchor locations) and the region-of-interest loss
    /// (Girshick 2015: 64 RoIs with 25% foreground at IoU of at least 0.5, background in [0.1, 0.5), smooth-L1
    /// regression). As in the reference implementation, the object boxes are added to the proposals the head
    /// learns from. Override the settings with <see cref="ObjectDetectionOptions{T}.TwoStageLoss"/>.
    /// </para>
    /// <para>
    /// Inputs are model-ready NCHW tensors, as for Predict, and targets are normalized against that input size. The
    /// detection head here classifies the proposals of a single image per forward pass, so each step takes one image.
    /// </para>
    /// </remarks>
    public void TrainDetections(Tensor<T> input, DetectionTrainingBatch<T> targets)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (input.Rank != 4 || input.Shape[1] != 3 || input.Shape[2] <= 0 || input.Shape[3] <= 0)
            throw new ArgumentException("Faster R-CNN training requires a three-channel NCHW image batch.", nameof(input));
        if (input.Shape[0] != 1)
            throw new ArgumentException("Faster R-CNN's detection head classifies one image's proposals per forward pass; train one image per step.", nameof(input));
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
            image => ForwardDetection(image, gold, out anchors),
            (heads, batch) => Engine.TensorAdd(
                _detectionLoss.ComputeProposalLoss(TwoStageTargets.FirstImage(heads[3]), TwoStageTargets.FirstImage(heads[4]),
                    anchors, gold, _rpn.AnchorsPerLocation, random),
                _detectionLoss.ComputeStageLoss(heads[0], heads[1], heads[2], gold, goldClasses, 0, random)));
    }

    /// <summary>The detection forward, optionally adding boxes to the proposals the detection head classifies.</summary>
    private List<Tensor<T>> ForwardDetection(Tensor<T> input, IReadOnlyList<double[]>? extraProposals,
        out List<BoundingBox<T>> anchors)
    {
        int imageHeight = input.Shape[2];
        int imageWidth = input.Shape[3];

        // Extract backbone features
        var backboneFeatures = EnsureBackbone.ExtractFeatures(input);

        // Apply FPN neck to get multi-scale features
        var fpnFeatures = EnsureNeck.Forward(backboneFeatures);

        // Faster R-CNN with FPN (Lin et al. 2017; detectron2, torchvision): the shared RPN head runs on
        // every level P2-P5 plus P6 (P5 subsampled by 2), each with its own anchor size, and each RoI
        // is pooled from the level matching its size.
        var rpnLevels = new List<Tensor<T>>(fpnFeatures) { CvTensorOps<T>.MaxPoolPadded(fpnFeatures[^1], 1, 2, 0) };
        var (objectness, bboxDeltas, levelAnchors, levelAnchorCounts) = _rpn.ForwardLevels(rpnLevels);
        anchors = levelAnchors;

        // Generate proposals: top 1000 per level, NMS within each level, best 1000 overall.
        var proposals = _rpn.GenerateProposals(
            objectness, bboxDeltas, levelAnchors,
            imageHeight, imageWidth,
            preNmsTopK: 1000,
            postNmsTopK: 1000,
            nmsThreshold: 0.7,
            levelAnchorCounts: levelAnchorCounts);

        var proposalBoxes = proposals.Count == 0 ? new Tensor<T>(new[] { 0, 4 }) : proposals[0].boxes;
        if (extraProposals is { Count: > 0 })
            proposalBoxes = TwoStageTargets.AppendBoxes(proposalBoxes, extraProposals);

        if (proposalBoxes.Shape[0] == 0)
        {
            // No proposals, return empty result
            return new List<Tensor<T>>
            {
                new Tensor<T>(new[] { 0, Options.NumClasses + 1 }),
                new Tensor<T>(new[] { 0, (Options.NumClasses + 1) * 4 }),
                new Tensor<T>(new[] { 0, 4 }),
                objectness,
                bboxDeltas
            };
        }

        // Stage 2: RoI feature extraction from the size-matched pyramid level, then classification
        var roiFeatures = FpnRoIPooler<T>.Pool(_roiAlign, fpnFeatures, EnsureBackbone.Strides, proposalBoxes);

        // Flatten RoI features: [num_rois, channels, H, W] -> [num_rois, channels*H*W]
        var flattenedFeatures = FlattenRoIFeatures(roiFeatures);

        // Classification and box regression
        var classLogits = _fcClassifier.Forward(flattenedFeatures);
        var boxDeltas = _fcBoxRegressor.Forward(flattenedFeatures);

        // The RPN's raw objectness and box deltas are outputs too: proposal selection is a
        // non-differentiable top-k, so without them nothing would train the RPN. PostProcess reads
        // only the first three entries.
        return new List<Tensor<T>> { classLogits, boxDeltas, proposalBoxes, objectness, bboxDeltas };
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
        int numClasses = Options.NumClasses + 1; // Including background

        var candidateDetections = new List<Detection<T>>();

        for (int i = 0; i < numProposals; i++)
        {
            // Apply softmax to get class probabilities
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
            for (int c = 1; c < numClasses; c++) // Skip class 0 (background)
            {
                if (probs[c] > bestScore)
                {
                    bestScore = probs[c];
                    bestClass = c;
                }
            }

            if (bestScore < confidenceThreshold) continue;

            // Decode box
            double px1 = NumOps.ToDouble(proposalBoxes[i, 0]);
            double py1 = NumOps.ToDouble(proposalBoxes[i, 1]);
            double px2 = NumOps.ToDouble(proposalBoxes[i, 2]);
            double py2 = NumOps.ToDouble(proposalBoxes[i, 3]);

            double pw = px2 - px1;
            double ph = py2 - py1;
            double pcx = px1 + pw / 2;
            double pcy = py1 + ph / 2;

            // Get box deltas for this class
            int deltaOffset = bestClass * 4;
            double dx = NumOps.ToDouble(boxDeltas[i, deltaOffset]);
            double dy = NumOps.ToDouble(boxDeltas[i, deltaOffset + 1]);
            double dw = NumOps.ToDouble(boxDeltas[i, deltaOffset + 2]);
            double dh = NumOps.ToDouble(boxDeltas[i, deltaOffset + 3]);

            // Apply deltas
            double predCx = pcx + dx * pw;
            double predCy = pcy + dy * ph;
            double predW = pw * Math.Exp(Math.Min(dw, 4.0));
            double predH = ph * Math.Exp(Math.Min(dh, 4.0));

            // Convert to (x1, y1, x2, y2) and clip
            // Decoded in network-input coordinates; map to the source image before clipping.
            var (scaleX, scaleY) = InputToImageScale(imageWidth, imageHeight);
            double x1 = Math.Max(0, (predCx - predW / 2) * scaleX);
            double y1 = Math.Max(0, (predCy - predH / 2) * scaleY);
            double x2 = Math.Min(imageWidth, (predCx + predW / 2) * scaleX);
            double y2 = Math.Min(imageHeight, (predCy + predH / 2) * scaleY);

            if (x2 <= x1 || y2 <= y1) continue;

            // Adjust class ID (subtract 1 to remove background class offset)
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
        return _rpn.GetParameterCount() +
               _fcClassifier.GetParameterCount() +
               _fcBoxRegressor.GetParameterCount();
    }

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using var stream = File.OpenRead(pathOrUrl);
        using var reader = new BinaryReader(stream);

        // Read and verify magic number and version
        int magic = reader.ReadInt32();
        if (magic != 0x52434E4E) // "RCNN" in ASCII
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
        if (!modelName.StartsWith("Faster-RCNN"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not Faster-RCNN.");
        }

        // Read backbone parameters
        EnsureBackbone.ReadParameters(reader);

        // Read neck parameters
        EnsureNeck.ReadParameters(reader);

        // Read RPN parameters
        _rpn.ReadParameters(reader);

        // Read detection head parameters
        _fcClassifier.ReadParameters(reader);
        _fcBoxRegressor.ReadParameters(reader);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write magic number and version for identification
        writer.Write(0x52434E4E); // "RCNN" in ASCII
        writer.Write(1); // Version 1

        // Write model configuration
        writer.Write(Name);

        // Write backbone parameters
        EnsureBackbone.WriteParameters(writer);

        // Write neck parameters
        EnsureNeck.WriteParameters(writer);

        // Write RPN parameters
        _rpn.WriteParameters(writer);

        // Write detection head parameters
        _fcClassifier.WriteParameters(writer);
        _fcBoxRegressor.WriteParameters(writer);
    }

    private Tensor<T> FlattenRoIFeatures(Tensor<T> roiFeatures)
        => AiDotNetEngine.Current.Reshape(
            roiFeatures, new[] { roiFeatures.Shape[0], roiFeatures.Shape[1] * roiFeatures.Shape[2] * roiFeatures.Shape[3] });
}
