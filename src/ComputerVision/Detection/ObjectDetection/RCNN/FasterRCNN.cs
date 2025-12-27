using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

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
public class FasterRCNN<T> : ObjectDetectorBase<T>
{
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
        Neck = new FPN<T>(Backbone.OutputChannels, outputChannels: 256);

        // Region Proposal Network
        _rpn = new RPN<T>(256, hiddenDim);

        // RoI feature extraction
        _roiAlign = new RoIAlign<T>(roiOutputSize, samplingRatio: 2);

        // Detection head: classification + bounding box regression
        int roiFeatureSize = 256 * roiOutputSize * roiOutputSize;
        _fcClassifier = new Dense<T>(roiFeatureSize, options.NumClasses + 1); // +1 for background
        _fcBoxRegressor = new Dense<T>(roiFeatureSize, (options.NumClasses + 1) * 4);

        _nms = new NMS<T>();
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
        int imageHeight = input.Shape[2];
        int imageWidth = input.Shape[3];

        // Extract backbone features
        var backboneFeatures = Backbone!.ExtractFeatures(input);

        // Apply FPN neck to get multi-scale features
        var fpnFeatures = Neck!.Forward(backboneFeatures);

        // Use P4 level for RPN (good balance of resolution and receptive field)
        var rpnFeatures = fpnFeatures.Count > 1 ? fpnFeatures[1] : fpnFeatures[0];

        // Stage 1: Region Proposal Network
        var (objectness, bboxDeltas, anchors) = _rpn.Forward(rpnFeatures);

        // Generate proposals
        var proposals = _rpn.GenerateProposals(
            objectness, bboxDeltas, anchors,
            imageHeight, imageWidth,
            preNmsTopK: 2000,
            postNmsTopK: 1000,
            nmsThreshold: 0.7);

        if (proposals.Count == 0 || proposals[0].boxes.Shape[0] == 0)
        {
            // No proposals, return empty result
            return new List<Tensor<T>>
            {
                new Tensor<T>(new[] { 0, Options.NumClasses + 1 }),
                new Tensor<T>(new[] { 0, (Options.NumClasses + 1) * 4 }),
                new Tensor<T>(new[] { 0, 4 })
            };
        }

        var proposalBoxes = proposals[0].boxes;

        // Stage 2: RoI feature extraction and classification
        // Use P4 features for RoI Align
        var p4Features = fpnFeatures.Count > 1 ? fpnFeatures[1] : fpnFeatures[0];
        double spatialScale = 1.0 / 16.0; // P4 is typically 1/16 resolution

        var roiFeatures = _roiAlign.Forward(p4Features, proposalBoxes, spatialScale);

        // Flatten RoI features: [num_rois, channels, H, W] -> [num_rois, channels*H*W]
        var flattenedFeatures = FlattenRoIFeatures(roiFeatures);

        // Classification and box regression
        var classLogits = _fcClassifier.Forward(flattenedFeatures);
        var boxDeltas = _fcBoxRegressor.Forward(flattenedFeatures);

        return new List<Tensor<T>> { classLogits, boxDeltas, proposalBoxes };
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
            double x1 = Math.Max(0, predCx - predW / 2);
            double y1 = Math.Max(0, predCy - predH / 2);
            double x2 = Math.Min(imageWidth, predCx + predW / 2);
            double y2 = Math.Min(imageHeight, predCy + predH / 2);

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
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        throw new NotImplementedException("Weight saving not yet implemented");
    }

    private Tensor<T> FlattenRoIFeatures(Tensor<T> roiFeatures)
    {
        int numRois = roiFeatures.Shape[0];
        int channels = roiFeatures.Shape[1];
        int h = roiFeatures.Shape[2];
        int w = roiFeatures.Shape[3];
        int flattenedSize = channels * h * w;

        var result = new Tensor<T>(new[] { numRois, flattenedSize });

        for (int roi = 0; roi < numRois; roi++)
        {
            int idx = 0;
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        result[roi, idx++] = roiFeatures[roi, c, y, x];
                    }
                }
            }
        }

        return result;
    }
}
