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
public class RPN<T>
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
    }

    /// <summary>
    /// Forward pass through the RPN.
    /// </summary>
    /// <param name="features">Feature map from backbone [batch, channels, height, width].</param>
    /// <returns>Tuple of (objectness logits, bbox deltas, anchors as list of BoundingBox).</returns>
    public (Tensor<T> objectness, Tensor<T> bboxDeltas, List<BoundingBox<T>> anchors) Forward(Tensor<T> features)
    {
        int batch = features.Shape[0];
        int height = features.Shape[2];
        int width = features.Shape[3];

        // Shared convolution with ReLU
        var x = _conv.Forward(features);
        x = ApplyReLU(x);

        // Get objectness scores
        var objectness = _clsHead.Forward(x);
        // Reshape: [B, numAnchors*2, H, W] -> [B, H*W*numAnchors, 2]
        objectness = ReshapeRPNOutput(objectness, batch, height, width, 2);

        // Get bbox deltas
        var bboxDeltas = _regHead.Forward(x);
        // Reshape: [B, numAnchors*4, H, W] -> [B, H*W*numAnchors, 4]
        bboxDeltas = ReshapeRPNOutput(bboxDeltas, batch, height, width, 4);

        // Generate anchors for this feature map size using configured stride and base size
        var anchors = _anchorGenerator.GenerateAnchorsForLevel(height, width, stride: _featureStride, baseSize: _baseAnchorSize);

        return (objectness, bboxDeltas, anchors);
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
    /// <returns>Proposal boxes [num_proposals, 4] as (x1, y1, x2, y2).</returns>
    public List<(Tensor<T> boxes, Tensor<T> scores)> GenerateProposals(
        Tensor<T> objectness,
        Tensor<T> bboxDeltas,
        List<BoundingBox<T>> anchors,
        int imageHeight,
        int imageWidth,
        int preNmsTopK = 2000,
        int postNmsTopK = 1000,
        double nmsThreshold = 0.7)
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

            // Get top-k proposals before NMS
            var indices = Enumerable.Range(0, numAnchors)
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
            var nmsBoxes = ApplyNMS(decodedBoxes, nmsThreshold, postNmsTopK);

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

    private Tensor<T> ReshapeRPNOutput(Tensor<T> x, int batch, int height, int width, int outputDim)
    {
        int channelDim = x.Shape[1];

        // Validate divisibility before integer division
        if (channelDim % outputDim != 0)
        {
            throw new InvalidOperationException(
                $"Cannot reshape RPN output: channel dimension {channelDim} is not divisible by outputDim {outputDim}. " +
                $"Expected channel dimension to be numAnchors * {outputDim}.");
        }

        int numAnchors = channelDim / outputDim;
        var result = new Tensor<T>(new[] { batch, height * width * numAnchors, outputDim });

        for (int b = 0; b < batch; b++)
        {
            int idx = 0;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int a = 0; a < numAnchors; a++)
                    {
                        for (int d = 0; d < outputDim; d++)
                        {
                            int channelIdx = a * outputDim + d;
                            result[b, idx, d] = x[b, channelIdx, h, w];
                        }
                        idx++;
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ApplyReLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            result[i] = _numOps.FromDouble(Math.Max(0, val));
        }
        return result;
    }

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
        int channels = features.Shape[1];
        int featureH = features.Shape[2];
        int featureW = features.Shape[3];
        int numRois = rois.Shape[0];

        var output = new Tensor<T>(new[] { numRois, channels, _outputSize, _outputSize });

        for (int roiIdx = 0; roiIdx < numRois; roiIdx++)
        {
            // Get batch index for this RoI (default to 0 if not provided)
            int batchIdx = batchIndices is not null && roiIdx < batchIndices.Length
                ? Math.Min(batchIndices[roiIdx], batchSize - 1)
                : 0;

            // Scale RoI to feature map coordinates
            double x1 = _numOps.ToDouble(rois[roiIdx, 0]) * spatialScale;
            double y1 = _numOps.ToDouble(rois[roiIdx, 1]) * spatialScale;
            double x2 = _numOps.ToDouble(rois[roiIdx, 2]) * spatialScale;
            double y2 = _numOps.ToDouble(rois[roiIdx, 3]) * spatialScale;

            double roiW = x2 - x1;
            double roiH = y2 - y1;

            double binW = roiW / _outputSize;
            double binH = roiH / _outputSize;

            for (int c = 0; c < channels; c++)
            {
                for (int ph = 0; ph < _outputSize; ph++)
                {
                    for (int pw = 0; pw < _outputSize; pw++)
                    {
                        // Compute bin boundaries
                        double binStartY = y1 + ph * binH;
                        double binStartX = x1 + pw * binW;

                        double sum = 0;
                        int count = 0;

                        // Sample points within the bin
                        for (int iy = 0; iy < _samplingRatio; iy++)
                        {
                            for (int ix = 0; ix < _samplingRatio; ix++)
                            {
                                double y = binStartY + (iy + 0.5) * binH / _samplingRatio;
                                double x = binStartX + (ix + 0.5) * binW / _samplingRatio;

                                // Bilinear interpolation
                                if (y >= 0 && y < featureH && x >= 0 && x < featureW)
                                {
                                    sum += BilinearInterpolate(features, batchIdx, c, y, x, featureH, featureW);
                                    count++;
                                }
                            }
                        }

                        output[roiIdx, c, ph, pw] = _numOps.FromDouble(count > 0 ? sum / count : 0);
                    }
                }
            }
        }

        return output;
    }

    private double BilinearInterpolate(Tensor<T> features, int batch, int channel, double y, double x, int height, int width)
    {
        int y0 = (int)Math.Floor(y);
        int x0 = (int)Math.Floor(x);
        int y1 = Math.Min(y0 + 1, height - 1);
        int x1 = Math.Min(x0 + 1, width - 1);

        double wy1 = y - y0;
        double wy0 = 1.0 - wy1;
        double wx1 = x - x0;
        double wx0 = 1.0 - wx1;

        double v00 = _numOps.ToDouble(features[batch, channel, y0, x0]);
        double v01 = _numOps.ToDouble(features[batch, channel, y0, x1]);
        double v10 = _numOps.ToDouble(features[batch, channel, y1, x0]);
        double v11 = _numOps.ToDouble(features[batch, channel, y1, x1]);

        return wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
    }
}
