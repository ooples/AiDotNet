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

    /// <summary>
    /// The paper's optimizer (Ren et al. 2015, Section 3.1.3): SGD with momentum 0.9 at learning rate 0.001, one
    /// image per mini-batch as here. The paper's weight decay (0.0005) is not applied.
    /// </summary>
    protected override AiDotNet.Interfaces.IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? CreateTrainingOptimizer()
        => PaperSgdMomentum(0.001, 0.9);

    [AiDotNet.Attributes.Scratch]
    private Random? _trainingRandom;
    private readonly RPN<T> _rpn;
    private readonly RoIAlign<T> _roiAlign;
    // The box head of Faster R-CNN with FPN (Lin et al. 2017, Sec. 4.2: "two hidden 1,024-d fully-connected (fc)
    // layers, each followed by ReLU, before the final classification and bounding box regression layers").
    private readonly Dense<T> _fc6;
    private readonly Dense<T> _fc7;
    private readonly Dense<T> _fcClassifier;
    private readonly Dense<T> _fcBoxRegressor;
    private readonly int _roiOutputSize;
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
        int roiOutputSize = GetRoIOutputSize(options.Size);
        _roiOutputSize = roiOutputSize;

        // Backbone: ResNet-50 with FPN
        Backbone = new ResNet<T>(options: new ResNetBackboneOptions { Variant = ResNetVariant.ResNet50 });
        Neck = new FPN<T>(Backbone.OutputChannels.ToArray(), outputChannels: 256);

        // Region Proposal Network
        // The RPN head's 3x3 conv keeps the pyramid's width (Ren et al. 2015 on FPN features; torchvision's RPNHead).
        _rpn = new RPN<T>(256, 256);

        // RoI feature extraction
        _roiAlign = new RoIAlign<T>(roiOutputSize, samplingRatio: 2);

        // Detection head: classification + bounding box regression
        int roiFeatureSize = 256 * roiOutputSize * roiOutputSize;
        _fc6 = new Dense<T>(roiFeatureSize, BoxHeadWidth);
        _fc7 = new Dense<T>(BoxHeadWidth, BoxHeadWidth);
        _fcClassifier = new Dense<T>(BoxHeadWidth, options.NumClasses + 1); // +1 for background
        _fcBoxRegressor = new Dense<T>(BoxHeadWidth, (options.NumClasses + 1) * 4);

        _nms = new NMS<T>();
        _detectionLoss = new AiDotNet.ComputerVision.Detection.Losses.TwoStageDetectionLoss<T>(options.NumClasses, 1,
            options.TwoStageLoss ?? new AiDotNet.ComputerVision.Detection.Losses.TwoStageDetectionLossOptions());
    }

    /// <summary>The width of the two fully connected layers of the box head (Lin et al. 2017, Sec. 4.2).</summary>
    internal const int BoxHeadWidth = 1024;

    /// <summary>RoIAlign's output grid: 7x7 as in the paper, 5x5 for the Nano size.</summary>
    private static int GetRoIOutputSize(ModelSize size) => size == ModelSize.Nano ? 5 : 7;

    /// <inheritdoc/>
    /// <remarks>The forward pass classifies one image's proposals, so each image is detected on its own.</remarks>
    public override BatchDetectionResult<T> DetectBatch(Tensor<T> images, double confidenceThreshold, double nmsThreshold)
        => DetectEachImage(images, confidenceThreshold, nmsThreshold);

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

    /// <summary>Proposals kept after the RPN's NMS (Ren et al. 2015 with FPN; torchvision's post_nms_top_n).</summary>
    internal const int PostNmsTopK = 1000;

    /// <inheritdoc/>
    /// <remarks>
    /// The per-RoI heads (logits, deltas and proposals) are padded with zero rows to <see cref="PostNmsTopK"/>, so
    /// the raw prediction's layout depends on the input size alone. How many proposals NMS keeps depends on the
    /// scores, which batch statistics change in training mode: without the padding a training forward and
    /// <c>Predict</c> on the same image disagreed on the output's length. A padding row's proposal is empty, so it
    /// decodes to an empty box that <see cref="PostProcess"/> drops.
    /// </remarks>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        var outputs = ForwardDetection(input, null, out _);
        for (int head = 0; head < 3; head++)
            outputs[head] = CvTensorOps<T>.PadRows(outputs[head], PostNmsTopK);
        return outputs;
    }

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
        var random = _trainingRandom ??= Options.Seed is int seed
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        List<BoundingBox<T>> anchors = new();
        TrainWithTargets(input, targets,
            image => ForwardDetection(image, gold, out anchors, random),
            (heads, batch) => Engine.TensorAdd(
                _detectionLoss.ComputeProposalLoss(TwoStageTargets.FirstImage(heads[3]), TwoStageTargets.FirstImage(heads[4]),
                    anchors, gold, _rpn.AnchorsPerLocation, random),
                _detectionLoss.ComputeStageLoss(heads[0], heads[1], heads[2], gold, goldClasses, 0, random)));
    }

    /// <summary>The detection forward, optionally adding boxes to the proposals the detection head classifies.</summary>
    /// <param name="input">The NCHW image batch.</param>
    /// <param name="extraProposals">Boxes added to the proposals (the object boxes, in training).</param>
    /// <param name="anchors">The RPN's anchors, for the proposal loss.</param>
    /// <param name="sampleRandom">
    /// In training, the generator the RoI sampler draws from: the head then runs only on the sampled RoIs (64 per
    /// image by default) instead of every proposal. Running it on all ~1,000 proposals and sampling afterwards, as
    /// this used to, did 16 times the box-head work and held two ~200 MB RoIAlign intermediates per step.
    /// </param>
    private List<Tensor<T>> ForwardDetection(Tensor<T> input, IReadOnlyList<double[]>? extraProposals,
        out List<BoundingBox<T>> anchors, Random? sampleRandom = null)
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
            postNmsTopK: PostNmsTopK,
            nmsThreshold: 0.7,
            levelAnchorCounts: levelAnchorCounts);

        var proposalBoxes = proposals.Count == 0 ? new Tensor<T>(new[] { 0, 4 }) : proposals[0].boxes;
        if (extraProposals is { Count: > 0 })
            proposalBoxes = TwoStageTargets.AppendBoxes(proposalBoxes, extraProposals);

        bool noRois = proposalBoxes.Shape[0] == 0;
        if (sampleRandom is not null && extraProposals is not null && !noRois)
        {
            var rows = _detectionLoss.SampleStageRois(proposalBoxes, extraProposals, 0, sampleRandom);
            noRois = rows.Length == 0;
            if (!noRois)
                proposalBoxes = CvTensorOps<T>.Select(proposalBoxes, rows, 0);
        }

        if (noRois)
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
        var hidden = AiDotNetEngine.Current.ReLU(_fc7.Forward(AiDotNetEngine.Current.ReLU(_fc6.Forward(flattenedFeatures))));
        var classLogits = _fcClassifier.Forward(hidden);
        var boxDeltas = _fcBoxRegressor.Forward(hidden);

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

        int numClasses = Options.NumClasses + 1; // Including background

        // The detection loss multiplies the stage's regression targets by these weights, so the predicted deltas are
        // divided by them here; unit weights (the default) leave them raw.
        var regressionWeights = _detectionLoss.StageBoxRegressionWeights(0);

        // Softmax class scores, the best foreground class per RoI and that class's own deltas, all RoIs at once.
        var probabilities = TwoStageBoxDecoding<T>.ClassProbabilities(classLogits);
        var (bestClasses, bestScores) = TwoStageBoxDecoding<T>.BestForegroundClass(probabilities);
        var deltas = TwoStageBoxDecoding<T>.ClassDeltas(boxDeltas, bestClasses, numClasses);

        // Decoded in network-input coordinates; map to the source image before clipping.
        var (scaleX, scaleY) = InputToImageScale(imageWidth, imageHeight);
        var candidateDetections = TwoStageBoxDecoding<T>.BestClassCandidates(bestClasses, bestScores, proposalBoxes, deltas,
            regressionWeights, 4.0, scaleX, scaleY, imageWidth, imageHeight, confidenceThreshold, ClassNames);

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
               _fc6.GetParameterCount() +
               _fc7.GetParameterCount() +
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
        // Version 2 added the box head's two fully connected layers; a version 1 file has no weights for them.
        if (version != 2)
        {
            throw new InvalidDataException($"Unsupported weight file version: {version}; this build reads version 2.");
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
        _fc6.ReadParameters(reader);
        _fc7.ReadParameters(reader);
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
        writer.Write(2); // Version 2: the box head has its two fully connected layers

        // Write model configuration
        writer.Write(Name);

        // Write backbone parameters
        EnsureBackbone.WriteParameters(writer);

        // Write neck parameters
        EnsureNeck.WriteParameters(writer);

        // Write RPN parameters
        _rpn.WriteParameters(writer);

        // Write detection head parameters
        _fc6.WriteParameters(writer);
        _fc7.WriteParameters(writer);
        _fcClassifier.WriteParameters(writer);
        _fcBoxRegressor.WriteParameters(writer);
    }

    private Tensor<T> FlattenRoIFeatures(Tensor<T> roiFeatures)
        => AiDotNetEngine.Current.Reshape(
            roiFeatures, new[] { roiFeatures.Shape[0], roiFeatures.Shape[1] * roiFeatures.Shape[2] * roiFeatures.Shape[3] });
}
