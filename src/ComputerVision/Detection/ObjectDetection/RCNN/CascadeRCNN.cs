using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

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
public class CascadeRCNN<T> : ObjectDetectorBase<T>
{
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
        Neck = new FPN<T>(Backbone.OutputChannels, outputChannels: 256);

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

        // Apply FPN neck
        var fpnFeatures = Neck!.Forward(backboneFeatures);

        // Use P4 level for RPN
        var rpnFeatures = fpnFeatures.Count > 1 ? fpnFeatures[1] : fpnFeatures[0];

        // Stage 1: Region Proposal Network
        var (objectness, bboxDeltas, anchors) = _rpn.Forward(rpnFeatures);

        // Generate initial proposals
        var initialProposals = _rpn.GenerateProposals(
            objectness, bboxDeltas, anchors,
            imageHeight, imageWidth,
            preNmsTopK: 2000,
            postNmsTopK: 1000,
            nmsThreshold: 0.7);

        if (initialProposals.Count == 0 || initialProposals[0].boxes.Shape[0] == 0)
        {
            return new List<Tensor<T>>
            {
                new Tensor<T>(new[] { 0, Options.NumClasses + 1 }),
                new Tensor<T>(new[] { 0, (Options.NumClasses + 1) * 4 }),
                new Tensor<T>(new[] { 0, 4 })
            };
        }

        // Get P4 features for RoI Align
        var p4Features = fpnFeatures.Count > 1 ? fpnFeatures[1] : fpnFeatures[0];
        double spatialScale = 1.0 / 16.0;

        // Current boxes to refine
        var currentBoxes = initialProposals[0].boxes;
        Tensor<T>? classLogits = null;
        Tensor<T>? boxDeltas = null;

        // Cascade through stages
        for (int stageIdx = 0; stageIdx < _numStages; stageIdx++)
        {
            // Extract RoI features for current boxes
            var roiFeatures = _roiAlign.Forward(p4Features, currentBoxes, spatialScale);

            // Flatten RoI features
            var flattenedFeatures = FlattenRoIFeatures(roiFeatures);

            // Run cascade stage
            var stage = _stages[stageIdx];
            (classLogits, boxDeltas) = stage.Forward(flattenedFeatures);

            // Refine boxes for next stage (except for last stage)
            if (stageIdx < _numStages - 1)
            {
                currentBoxes = RefineBoxes(currentBoxes, boxDeltas!, imageWidth, imageHeight);
            }
        }

        return new List<Tensor<T>> { classLogits!, boxDeltas!, currentBoxes };
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

            double x1 = Math.Max(0, predCx - predW / 2);
            double y1 = Math.Max(0, predCy - predH / 2);
            double x2 = Math.Min(imageWidth, predCx + predW / 2);
            double y2 = Math.Min(imageHeight, predCy + predH / 2);

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

    private Tensor<T> RefineBoxes(Tensor<T> boxes, Tensor<T> deltas, int imageWidth, int imageHeight)
    {
        int numBoxes = boxes.Shape[0];
        int numClasses = deltas.Shape[1] / 4;

        var refinedBoxes = new Tensor<T>(new[] { numBoxes, 4 });

        for (int i = 0; i < numBoxes; i++)
        {
            double px1 = NumOps.ToDouble(boxes[i, 0]);
            double py1 = NumOps.ToDouble(boxes[i, 1]);
            double px2 = NumOps.ToDouble(boxes[i, 2]);
            double py2 = NumOps.ToDouble(boxes[i, 3]);

            double pw = px2 - px1;
            double ph = py2 - py1;
            double pcx = px1 + pw / 2;
            double pcy = py1 + ph / 2;

            // Use class-agnostic refinement (average across all classes)
            // or use the most likely class - here we use first non-background class
            int deltaOffset = 4; // Skip background class
            double dx = NumOps.ToDouble(deltas[i, deltaOffset]);
            double dy = NumOps.ToDouble(deltas[i, deltaOffset + 1]);
            double dw = NumOps.ToDouble(deltas[i, deltaOffset + 2]);
            double dh = NumOps.ToDouble(deltas[i, deltaOffset + 3]);

            double predCx = pcx + dx * pw;
            double predCy = pcy + dy * ph;
            double predW = pw * Math.Exp(Math.Min(dw, 4.0));
            double predH = ph * Math.Exp(Math.Min(dh, 4.0));

            double x1 = Math.Max(0, predCx - predW / 2);
            double y1 = Math.Max(0, predCy - predH / 2);
            double x2 = Math.Min(imageWidth, predCx + predW / 2);
            double y2 = Math.Min(imageHeight, predCy + predH / 2);

            refinedBoxes[i, 0] = NumOps.FromDouble(x1);
            refinedBoxes[i, 1] = NumOps.FromDouble(y1);
            refinedBoxes[i, 2] = NumOps.FromDouble(x2);
            refinedBoxes[i, 3] = NumOps.FromDouble(y2);
        }

        return refinedBoxes;
    }
}

/// <summary>
/// A single stage in the Cascade R-CNN pipeline.
/// </summary>
internal class CascadeStage<T>
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
}
