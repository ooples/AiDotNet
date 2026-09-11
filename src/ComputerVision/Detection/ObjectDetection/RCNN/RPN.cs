using AiDotNet.Tensors.Engines;
using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Anchors;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;

/// <summary>
/// Region Proposal Network (RPN) - Generates object proposals for two-stage detectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> RPN is a neural network that scans an image and proposes
/// regions that likely contain objects. It's the first stage in two-stage detectors like
/// Faster R-CNN, enabling end-to-end training.</para>
///
/// <para>Key features:
/// - Slides a small network over the feature map
/// - Generates proposals at multiple scales and aspect ratios via anchors
/// - Predicts objectness scores and bounding box refinements
/// - Shared features with detection network for efficiency
/// </para>
///
/// <para>Reference: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with
/// Region Proposal Networks", NeurIPS 2015</para>
/// </remarks>
public class RPN<T> : IParameterSource<T>, AiDotNet.Models.Parameters.IParameterChunkSource<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Conv2D<T> _conv;
    private readonly Conv2D<T> _clsHead;
    private readonly Conv2D<T> _regHead;
    private readonly AnchorGenerator<T> _anchorGenerator;
    private readonly int _hiddenDim;
    private readonly int _numAnchors;
    private readonly int _featureStride;
    private readonly double _baseAnchorSize;
    private readonly int[] _levelStrides;
    private readonly double[] _levelBaseSizes;

    /// <summary>
    /// Gets the anchor generator used by this RPN.
    /// </summary>
    public AnchorGenerator<T> AnchorGenerator => _anchorGenerator;

    /// <summary>
    /// Creates a new Region Proposal Network.
    /// </summary>
    /// <param name="inChannels">Number of input feature channels.</param>
    /// <param name="hiddenDim">Hidden dimension for the intermediate convolution.</param>
    /// <param name="anchorSizes">Sizes of anchors in pixels.</param>
    /// <param name="aspectRatios">Aspect ratios for anchors.</param>
    /// <param name="featureLevel">Feature pyramid level to use (0-indexed). Determines stride and base size. Default is middle level.</param>
    public RPN(int inChannels, int hiddenDim = 256, int[]? anchorSizes = null, double[]? aspectRatios = null, int? featureLevel = null)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;

        anchorSizes ??= new[] { 32, 64, 128, 256, 512 };
        aspectRatios ??= new[] { 0.5, 1.0, 2.0 };

        // Define scales - this makes _numAnchors calculation robust if scales change
        var scales = new double[] { 1.0 };
        _numAnchors = aspectRatios.Length * scales.Length;

        // Shared 3x3 convolution
        _conv = new Conv2D<T>(inChannels, hiddenDim, kernelSize: 3, padding: 1);

        // Classification head: objectness (2 classes per anchor: object/not-object)
        _clsHead = new Conv2D<T>(hiddenDim, _numAnchors * 2, kernelSize: 1);

        // Regression head: bbox deltas (4 values per anchor: dx, dy, dw, dh)
        _regHead = new Conv2D<T>(hiddenDim, _numAnchors * 4, kernelSize: 1);

        // Create anchor generator with specified sizes and aspect ratios
        var baseSizes = anchorSizes.Select(s => (double)s).ToArray();
        var strides = anchorSizes.Select((_, i) => (int)Math.Pow(2, i + 2)).ToArray();
        _anchorGenerator = new AnchorGenerator<T>(
            baseSizes: baseSizes,
            aspectRatios: aspectRatios,
            scales: scales,
            strides: strides);

        // Use specified feature level or default to middle level (index 2 for default config: stride=16, baseSize=128)
        int level = featureLevel ?? Math.Min(2, anchorSizes.Length - 1);
        if (level < 0 || level >= anchorSizes.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(featureLevel),
                $"Feature level must be between 0 and {anchorSizes.Length - 1}.");
        }
        _featureStride = strides[level];
        _baseAnchorSize = baseSizes[level];
        _levelStrides = strides;
        _levelBaseSizes = baseSizes;
    }

    /// <summary>
    /// Gets the number of pyramid levels the RPN has anchors for (one per anchor size).
    /// </summary>
    public int LevelCount => _levelStrides.Length;

    /// <summary>
    /// Forward pass through the RPN.
    /// </summary>
    /// <param name="features">Feature map from backbone [batch, channels, height, width].</param>
    /// <returns>Tuple of (objectness logits, bbox deltas, anchors as list of BoundingBox).</returns>
    public (Tensor<T> objectness, Tensor<T> bboxDeltas, List<BoundingBox<T>> anchors) Forward(Tensor<T> features)
    {
        var (objectness, bboxDeltas) = Head(features);

        // Generate anchors for this feature map size using configured stride and base size
        var anchors = _anchorGenerator.GenerateAnchorsForLevel(
            features.Shape[2], features.Shape[3], stride: _featureStride, baseSize: _baseAnchorSize);

        return (objectness, bboxDeltas, anchors);
    }

    /// <summary>
    /// Runs the shared RPN head over every level of a feature pyramid.
    /// </summary>
    /// <param name="levels">Pyramid levels, finest first (P2, P3, ...), one per anchor size.</param>
    /// <returns>
    /// Objectness <c>[batch, totalAnchors, 2]</c> and deltas <c>[batch, totalAnchors, 4]</c> with the
    /// levels concatenated finest first, the matching anchors, and how many anchors each level has.
    /// </returns>
    /// <remarks>
    /// The FPN form of the RPN (Lin et al. 2017; detectron2, torchvision): ONE head, shared across
    /// levels, with anchors of a single size per level - level <c>i</c> uses the i-th anchor size at
    /// the i-th stride. Each level's anchors are laid out at that level's own stride.
    /// </remarks>
    public (Tensor<T> objectness, Tensor<T> bboxDeltas, List<BoundingBox<T>> anchors, int[] levelAnchorCounts) ForwardLevels(
        IReadOnlyList<Tensor<T>> levels)
    {
        if (levels is null || levels.Count == 0)
        {
            throw new ArgumentException("At least one pyramid level is required.", nameof(levels));
        }

        if (levels.Count > _levelStrides.Length)
        {
            throw new ArgumentException(
                $"The RPN has anchors for {_levelStrides.Length} levels but received {levels.Count}.", nameof(levels));
        }

        var objectness = new Tensor<T>[levels.Count];
        var deltas = new Tensor<T>[levels.Count];
        var anchors = new List<BoundingBox<T>>();
        var counts = new int[levels.Count];
        for (int l = 0; l < levels.Count; l++)
        {
            (objectness[l], deltas[l]) = Head(levels[l]);
            var levelAnchors = _anchorGenerator.GenerateAnchorsForLevel(
                levels[l].Shape[2], levels[l].Shape[3], stride: _levelStrides[l], baseSize: _levelBaseSizes[l]);
            anchors.AddRange(levelAnchors);
            counts[l] = levelAnchors.Count;
        }

        var engine = AiDotNetEngine.Current;
        return levels.Count == 1
            ? (objectness[0], deltas[0], anchors, counts)
            : (engine.TensorConcatenate(objectness, 1), engine.TensorConcatenate(deltas, 1), anchors, counts);
    }

    private (Tensor<T> Objectness, Tensor<T> Deltas) Head(Tensor<T> features)
    {
        int batch = features.Shape[0];
        int height = features.Shape[2];
        int width = features.Shape[3];

        // Shared convolution with ReLU
        var x = ApplyReLU(_conv.Forward(features));

        // [B, numAnchors*2, H, W] -> [B, H*W*numAnchors, 2] and [B, numAnchors*4, H, W] -> [B, H*W*numAnchors, 4]
        var objectness = ReshapeRPNOutput(_clsHead.Forward(x), batch, height, width, 2);
        var bboxDeltas = ReshapeRPNOutput(_regHead.Forward(x), batch, height, width, 4);
        return (objectness, bboxDeltas);
    }

    /// <summary>
    /// Generates proposals from RPN outputs.
    /// </summary>
    /// <param name="objectness">Objectness logits [batch, num_anchors, 2].</param>
    /// <param name="bboxDeltas">Bbox deltas [batch, num_anchors, 4].</param>
    /// <param name="anchors">Anchor boxes as list of BoundingBox.</param>
    /// <param name="imageHeight">Original image height.</param>
    /// <param name="imageWidth">Original image width.</param>
    /// <param name="preNmsTopK">Maximum proposals before NMS.</param>
    /// <param name="postNmsTopK">Maximum proposals after NMS.</param>
    /// <param name="nmsThreshold">IoU threshold for NMS.</param>
    /// <param name="levelAnchorCounts">
    /// Anchors per pyramid level, as returned by <see cref="ForwardLevels"/>. When given, the top
    /// <paramref name="preNmsTopK"/> are taken and NMS is applied WITHIN each level, then the best
    /// <paramref name="postNmsTopK"/> are kept across levels - the FPN proposal rule, which stops the
    /// many fine-level anchors from crowding out the coarse levels. When null, all anchors form one
    /// group.
    /// </param>
    /// <returns>Proposal boxes [num_proposals, 4] as (x1, y1, x2, y2).</returns>
    public List<(Tensor<T> boxes, Tensor<T> scores)> GenerateProposals(
        Tensor<T> objectness,
        Tensor<T> bboxDeltas,
        List<BoundingBox<T>> anchors,
        int imageHeight,
        int imageWidth,
        int preNmsTopK = 2000,
        int postNmsTopK = 1000,
        double nmsThreshold = 0.7,
        int[]? levelAnchorCounts = null)
    {
        int batch = objectness.Shape[0];
        int objectnessAnchors = objectness.Shape[1];
        int anchorCount = anchors.Count;

        // Validate shape consistency - objectness and anchors must match
        if (objectnessAnchors != anchorCount)
        {
            throw new ArgumentException(
                $"Shape mismatch: objectness tensor has {objectnessAnchors} anchor positions " +
                $"but anchor list contains {anchorCount} anchors. " +
                $"Ensure Forward() and GenerateProposals() use the same feature map dimensions.",
                nameof(anchors));
        }

        int numAnchors = objectnessAnchors;

        var proposals = new List<(Tensor<T> boxes, Tensor<T> scores)>();

        for (int b = 0; b < batch; b++)
        {
            // Apply softmax to get objectness scores
            var scores = new double[numAnchors];
            for (int i = 0; i < numAnchors; i++)
            {
                double notObj = _numOps.ToDouble(objectness[b, i, 0]);
                double obj = _numOps.ToDouble(objectness[b, i, 1]);
                double maxVal = Math.Max(notObj, obj);
                double sumExp = Math.Exp(notObj - maxVal) + Math.Exp(obj - maxVal);
                scores[i] = Math.Exp(obj - maxVal) / sumExp;
            }

            var groups = levelAnchorCounts ?? new[] { numAnchors };
            if (groups.Sum() != numAnchors)
            {
                throw new ArgumentException(
                    $"Level anchor counts sum to {groups.Sum()}, but there are {numAnchors} anchors.",
                    nameof(levelAnchorCounts));
            }

            var kept = new List<(double x1, double y1, double x2, double y2, double score)>();
            int groupStart = 0;
            foreach (int groupSize in groups)
            {
                int start = groupStart;
                groupStart += groupSize;

                // Get top-k proposals before NMS
                var indices = Enumerable.Range(start, groupSize)
                    .OrderByDescending(i => scores[i])
                    .Take(preNmsTopK)
                    .ToList();

                // Decode boxes
                var decodedBoxes = new List<(double x1, double y1, double x2, double y2, double score, int idx)>();
                foreach (int i in indices)
                {
                    // Get anchor - BoundingBox stores (x1, y1, x2, y2) in XYXY format
                    var anchor = anchors[i];
                    double ax1 = _numOps.ToDouble(anchor.X1);
                    double ay1 = _numOps.ToDouble(anchor.Y1);
                    double ax2 = _numOps.ToDouble(anchor.X2);
                    double ay2 = _numOps.ToDouble(anchor.Y2);
                    double aw = ax2 - ax1;
                    double ah = ay2 - ay1;

                    // Get deltas
                    double dx = _numOps.ToDouble(bboxDeltas[b, i, 0]);
                    double dy = _numOps.ToDouble(bboxDeltas[b, i, 1]);
                    double dw = _numOps.ToDouble(bboxDeltas[b, i, 2]);
                    double dh = _numOps.ToDouble(bboxDeltas[b, i, 3]);

                    // Anchor center
                    double cx = ax1 + aw / 2;
                    double cy = ay1 + ah / 2;

                    // Apply deltas (standard bbox encoding)
                    double predCx = cx + dx * aw;
                    double predCy = cy + dy * ah;
                    double predW = aw * Math.Exp(Math.Min(dw, 4.0)); // Clip to prevent explosion
                    double predH = ah * Math.Exp(Math.Min(dh, 4.0));

                    // Convert to (x1, y1, x2, y2)
                    double x1 = Math.Max(0, predCx - predW / 2);
                    double y1 = Math.Max(0, predCy - predH / 2);
                    double x2 = Math.Min(imageWidth, predCx + predW / 2);
                    double y2 = Math.Min(imageHeight, predCy + predH / 2);

                    if (x2 > x1 && y2 > y1)
                    {
                        decodedBoxes.Add((x1, y1, x2, y2, scores[i], i));
                    }
                }

                // Apply NMS
                kept.AddRange(ApplyNMS(decodedBoxes, nmsThreshold, postNmsTopK));
            }

            var nmsBoxes = kept.OrderByDescending(box => box.score).Take(postNmsTopK).ToList();

            // Convert to tensors
            int numProposals = nmsBoxes.Count;
            var boxTensor = new Tensor<T>(new[] { numProposals, 4 });
            var scoreTensor = new Tensor<T>(new[] { numProposals });

            for (int i = 0; i < numProposals; i++)
            {
                boxTensor[i, 0] = _numOps.FromDouble(nmsBoxes[i].x1);
                boxTensor[i, 1] = _numOps.FromDouble(nmsBoxes[i].y1);
                boxTensor[i, 2] = _numOps.FromDouble(nmsBoxes[i].x2);
                boxTensor[i, 3] = _numOps.FromDouble(nmsBoxes[i].y2);
                scoreTensor[i] = _numOps.FromDouble(nmsBoxes[i].score);
            }

            proposals.Add((boxTensor, scoreTensor));
        }

        return proposals;
    }

    /// <summary>
    /// Gets the total parameter count for this RPN.
    /// </summary>
    public long GetParameterCount()
    {
        return _conv.GetParameterCount() +
               _clsHead.GetParameterCount() +
               _regHead.GetParameterCount();
    }

    /// <summary>
    /// Writes the RPN parameters to a binary stream.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numAnchors);
        _conv.WriteParameters(writer);
        _clsHead.WriteParameters(writer);
        _regHead.WriteParameters(writer);
    }

    /// <summary>
    /// Reads the RPN parameters from a binary stream.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numAnchors = reader.ReadInt32();

        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidDataException($"RPN hiddenDim mismatch: expected {_hiddenDim}, got {hiddenDim}.");
        }

        if (numAnchors != _numAnchors)
        {
            throw new InvalidDataException($"RPN numAnchors mismatch: expected {_numAnchors}, got {numAnchors}.");
        }

        _conv.ReadParameters(reader);
        _clsHead.ReadParameters(reader);
        _regHead.ReadParameters(reader);
    }

    internal static Tensor<T> ReshapeRPNOutput(Tensor<T> x, int batch, int height, int width, int outputDim)
    {
        int channelDim = x.Shape[1];

        // Validate divisibility before integer division
        if (channelDim % outputDim != 0)
        {
            throw new InvalidOperationException(
                $"Cannot reshape RPN output: channel dimension {channelDim} is not divisible by outputDim {outputDim}. " +
                $"Expected channel dimension to be numAnchors * {outputDim}.");
        }

        // [B, A*D, H, W] -> [B, A, D, H, W] -> [B, H, W, A, D] -> [B, H*W*A, D], as engine ops so the
        // RPN heads stay on the gradient tape.
        int numAnchors = channelDim / outputDim;
        var engine = AiDotNetEngine.Current;
        var split = engine.Reshape(x, new[] { batch, numAnchors, outputDim, height, width });
        var ordered = engine.TensorPermute(split, new[] { 0, 3, 4, 1, 2 });
        return engine.Reshape(ordered, new[] { batch, height * width * numAnchors, outputDim });
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

    private List<(double x1, double y1, double x2, double y2, double score)> ApplyNMS(
        List<(double x1, double y1, double x2, double y2, double score, int idx)> boxes,
        double iouThreshold,
        int maxBoxes)
    {
        var sorted = boxes.OrderByDescending(b => b.score).ToList();
        var selected = new List<(double x1, double y1, double x2, double y2, double score)>();
        var used = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count && selected.Count < maxBoxes; i++)
        {
            if (used[i]) continue;

            var current = sorted[i];
            selected.Add((current.x1, current.y1, current.x2, current.y2, current.score));
            used[i] = true;

            // Suppress overlapping boxes
            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (used[j]) continue;

                double iou = ComputeIoU(current, sorted[j]);
                if (iou > iouThreshold)
                {
                    used[j] = true;
                }
            }
        }

        return selected;
    }

    private double ComputeIoU(
        (double x1, double y1, double x2, double y2, double score, int idx) a,
        (double x1, double y1, double x2, double y2, double score, int idx) b)
    {
        double intersectX1 = Math.Max(a.x1, b.x1);
        double intersectY1 = Math.Max(a.y1, b.y1);
        double intersectX2 = Math.Min(a.x2, b.x2);
        double intersectY2 = Math.Min(a.y2, b.y2);

        double intersectW = Math.Max(0, intersectX2 - intersectX1);
        double intersectH = Math.Max(0, intersectY2 - intersectY1);
        double intersect = intersectW * intersectH;

        double areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        double areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        double union = areaA + areaB - intersect;

        return union > 0 ? intersect / union : 0;
    }

    // The shared convolution and both heads, registered as live chunks. RPN is public, so it forwards
    // the parameter interfaces to an internal module instead of deriving from one. Before this the
    // generator could not see anything inside the RPN at all.
    private DelegatingCvParameterModule<T>? _parameters;

    private DelegatingCvParameterModule<T> Parameters
        => _parameters ??= new DelegatingCvParameterModule<T>(() => new IParameterSource<T>?[] { _conv, _clsHead, _regHead });

    /// <inheritdoc />
    long IParameterSource<T>.ParameterCount => Parameters.ParameterCount;

    /// <inheritdoc />
    Vector<T> IParameterSource<T>.GetParameters() => Parameters.GetParameters();

    /// <inheritdoc />
    void IParameterSource<T>.SetParameters(Vector<T> parameters) => Parameters.SetParameters(parameters);

    /// <inheritdoc />
    IEnumerable<AiDotNet.Models.Parameters.ParameterChunk<T>> AiDotNet.Models.Parameters.IParameterChunkSource<T>.GetParameterStateChunks()
        => Parameters.GetParameterStateChunks();
}

/// <summary>
/// RoI (Region of Interest) Align - Extracts fixed-size features from proposals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>RoI Align uses bilinear interpolation to extract features from arbitrary-sized
/// regions without quantization, improving detection accuracy compared to RoI Pooling.</para>
///
/// <para>Reference: He et al., "Mask R-CNN", ICCV 2017</para>
/// </remarks>
internal class RoIAlign<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _outputSize;
    private readonly int _samplingRatio;

    /// <summary>
    /// Creates a new RoI Align layer.
    /// </summary>
    /// <param name="outputSize">Size of output feature map (outputSize x outputSize).</param>
    /// <param name="samplingRatio">Number of sampling points per bin.</param>
    public RoIAlign(int outputSize = 7, int samplingRatio = 2)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _outputSize = outputSize;
        _samplingRatio = samplingRatio;
    }

    /// <summary>
    /// Extracts RoI features from feature maps.
    /// </summary>
    /// <param name="features">Feature map [batch, channels, height, width].</param>
    /// <param name="rois">Region of interest boxes [num_rois, 4] as (x1, y1, x2, y2).</param>
    /// <param name="spatialScale">Ratio of feature map size to input image size.</param>
    /// <param name="batchIndices">Optional batch index for each RoI. If null, all RoIs use batch 0.</param>
    /// <returns>Pooled features [num_rois, channels, outputSize, outputSize].</returns>
    public Tensor<T> Forward(Tensor<T> features, Tensor<T> rois, double spatialScale = 1.0 / 16.0, int[]? batchIndices = null)
    {
        int batchSize = features.Shape[0];
        int numRois = rois.Shape[0];

        // The boxes are constants to the gradient (as in standard RoIAlign), so they are read out once;
        // the pooling itself is a tape-visible gather over the feature map.
        var boxes = new double[numRois * 4];
        for (int i = 0; i < boxes.Length; i++)
        {
            boxes[i] = _numOps.ToDouble(rois[i]);
        }

        var indices = new int[numRois];
        for (int roiIdx = 0; roiIdx < numRois; roiIdx++)
        {
            indices[roiIdx] = batchIndices is not null && roiIdx < batchIndices.Length
                ? Math.Min(batchIndices[roiIdx], batchSize - 1)
                : 0;
        }

        return CvTensorOps<T>.RoIAlign(features, boxes, indices, spatialScale, _outputSize, _samplingRatio);
    }


}
