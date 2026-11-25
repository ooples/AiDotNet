using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// The post-processing the segmentation models share once they have per-pixel class logits or a prompt score map:
/// panoptic assembly from a semantic map, instance extraction by connected components, and the prompt-conditioned
/// multi-threshold masks of the promptable models. Each routine runs as whole-tensor engine operations.
/// </summary>
/// <remarks>
/// <para>
/// Pixel counts are taken as integers and probabilities are summed in double, whatever the element type: a count
/// held in half precision stops growing at 2048, and a per-class sum of probabilities accumulated in single
/// precision drifts on a large image. The per-class and per-instance sums are scatter-additions in pixel order, the
/// order a row-by-row scan visits the pixels in.
/// </para>
/// <para><b>For Beginners:</b> a segmentation network produces a score for every class at every pixel. These helpers
/// turn those scores into the things a caller asks for: a list of segments with their area and confidence, a mask
/// per object, or a mask for the region a click or a box points at.</para>
/// </remarks>
internal static class SegmentationPostProcessing
{
    /// <summary>The factor a class id is multiplied by in a panoptic label: <c>classId * 1000 + instanceId</c>.</summary>
    internal const int PanopticLabelDivisor = 1000;

    /// <summary>
    /// The per-pixel class <c>[HW]</c> (int) and the probability each pixel gives that class <c>[HW]</c> (double),
    /// from class logits <c>[C, H, W]</c> and their softmax.
    /// </summary>
    private static (Tensor<int> Labels, Tensor<double> OwnProbability) LabelsAndOwnProbability<T>(
        Tensor<T> logits, Tensor<T> probabilities)
    {
        var engine = AiDotNetEngine.Current;
        int channels = logits.Shape[0], hw = logits.Shape[1] * logits.Shape[2];
        var labels = engine.Reshape(engine.TensorArgMax(logits, 0), [hw]);
        var perPixel = engine.TensorPermute(engine.Reshape(probabilities, [channels, hw]), [1, 0]);
        var own = engine.TensorGatherClassValues(perPixel, labels.Cast<T>()).Cast<double>();
        return (labels, engine.Reshape(own, [hw]));
    }

    /// <summary>
    /// Builds a panoptic result from semantic class logits <c>[C, H, W]</c>: every stuff class present becomes one
    /// segment, and every connected component of a thing class becomes one instance.
    /// </summary>
    /// <param name="logits">Class logits <c>[C, H, W]</c>.</param>
    /// <param name="numStuffClasses">Classes <c>0 .. numStuffClasses - 1</c> are stuff; the rest are things.</param>
    /// <param name="numClasses">The model's class count; classes at or above it are left unlabelled.</param>
    /// <returns>
    /// <c>SemanticMap</c> is the argmax class; <c>InstanceMap</c> the instance id of thing pixels (ids count up
    /// from 1 across the thing classes in class order, then component order) and 0 elsewhere; <c>PanopticMap</c>
    /// <c>classId * 1000</c> for stuff and <c>classId * 1000 + instanceId</c> for things. Each segment's
    /// <c>SegmentId</c> is its value in <c>PanopticMap</c>, its confidence the mean probability of its class over
    /// its pixels.
    /// </returns>
    internal static PanopticSegmentationResult<T> PanopticFromSemanticLogits<T>(
        Tensor<T> logits, int numStuffClasses, int numClasses)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (logits.Rank != 3)
            throw new ArgumentException("Logits must be rank 3 [C, H, W].", nameof(logits));

        var engine = AiDotNetEngine.Current;
        int channels = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2], hw = h * w;
        var probabilities = SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var semanticMap = SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        var (labels, own) = LabelsAndOwnProbability(logits, probabilities);

        int[] classArea = engine.TensorBinCount(labels, channels).ToArray();
        double[] classProbability = engine.TensorScatterAdd(new Tensor<double>([channels]), labels, own, 0).ToArray();

        var segments = new List<PanopticSegment<T>>();
        // The panoptic value of each stuff class; thing classes get theirs per instance below.
        var stuffLabel = new int[channels];
        int stuffEnd = Math.Min(numStuffClasses, channels);
        for (int cls = 0; cls < stuffEnd; cls++)
        {
            stuffLabel[cls] = cls * PanopticLabelDivisor;
            if (classArea[cls] == 0) continue;
            segments.Add(new PanopticSegment<T>
            {
                SegmentId = stuffLabel[cls],
                ClassId = cls,
                IsThing = false,
                Confidence = classProbability[cls] / classArea[cls],
                Area = classArea[cls],
            });
        }

        // One id map for all instances: a class's components are numbered after those of the classes before it.
        var instanceIds = new Tensor<int>([hw]);
        var instanceClass = new List<int>();
        int thingEnd = Math.Min(numClasses, channels);
        for (int cls = Math.Max(numStuffClasses, 0); cls < thingEnd; cls++)
        {
            if (classArea[cls] == 0) continue;
            var (labelMap, count) = SegmentationTensorOps.LabelConnectedComponents(semanticMap, cls);
            if (count == 0) continue;

            var local = engine.Reshape(labelMap.Cast<int>(), [hw]);
            var member = engine.TensorGreaterThan(local, 0);
            instanceIds = engine.TensorAdd(
                instanceIds, engine.TensorAdd(local, engine.TensorMultiplyScalar(member, instanceClass.Count)));
            instanceClass.AddRange(Enumerable.Repeat(cls, count));
        }

        int instances = instanceClass.Count;
        int[] instanceArea = engine.TensorBinCount(instanceIds, instances + 1).ToArray();
        double[] instanceProbability = engine.TensorScatterAdd(
            new Tensor<double>([instances + 1]), instanceIds, own, 0).ToArray();
        var instanceLabel = new int[instances + 1];
        for (int id = 1; id <= instances; id++)
        {
            int cls = instanceClass[id - 1];
            instanceLabel[id] = cls * PanopticLabelDivisor + id;
            segments.Add(new PanopticSegment<T>
            {
                SegmentId = instanceLabel[id],
                ClassId = cls,
                IsThing = true,
                Confidence = instanceProbability[id] / instanceArea[id],
                Area = instanceArea[id],
                Mask = engine.Reshape(engine.TensorEquals(instanceIds, id).Cast<T>(), [h, w]),
            });
        }

        // A thing pixel takes its instance's label and a stuff pixel its class's; each table is 0 for the other kind.
        var panoptic = engine.TensorAdd(
            engine.TensorIndexSelect(new Tensor<int>(instanceLabel, [instances + 1]), instanceIds, 0),
            engine.TensorIndexSelect(new Tensor<int>(stuffLabel, [channels]), labels, 0));

        return new PanopticSegmentationResult<T>
        {
            SemanticMap = semanticMap,
            InstanceMap = engine.Reshape(instanceIds.Cast<T>(), [h, w]),
            PanopticMap = engine.Reshape(panoptic.Cast<T>(), [h, w]),
            Segments = segments,
        };
    }

    /// <summary>
    /// Extracts instances from class logits <c>[C, H, W]</c> as the connected components of every non-background
    /// class (class 0 is background), taken in class order then component order.
    /// </summary>
    /// <param name="logits">Class logits <c>[C, H, W]</c>.</param>
    /// <param name="numClasses">The model's class count; classes at or above it are skipped.</param>
    /// <param name="maxInstances">Extraction stops once this many instances are kept.</param>
    /// <param name="minArea">Components with fewer pixels are dropped as noise.</param>
    /// <param name="confidenceThreshold">Components whose mean class probability is below this are dropped.</param>
    /// <returns>
    /// One mask per kept component, with its tight box (<c>[minX, minY, maxX + 1, maxY + 1]</c>) and the mean
    /// probability of its class over its pixels as the confidence, ordered by descending confidence (equal
    /// confidences keep their extraction order), ready for a greedy mask NMS.
    /// </returns>
    internal static List<InstanceMask<T>> ConnectedComponentInstances<T>(
        Tensor<T> logits, int numClasses, int maxInstances, int minArea, double confidenceThreshold)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (logits.Rank != 3)
            throw new ArgumentException("Logits must be rank 3 [C, H, W].", nameof(logits));

        var engine = AiDotNetEngine.Current;
        int channels = logits.Shape[0], h = logits.Shape[1], w = logits.Shape[2], hw = h * w;
        var classMap = SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        var (labels, own) = LabelsAndOwnProbability(logits, SegmentationTensorOps.SoftmaxAlongClassDim(logits));
        int[] classArea = engine.TensorBinCount(labels, channels).ToArray();

        // Column and row of every pixel, for the per-component bounding boxes.
        var column = engine.Reshape(
            engine.TensorTile(engine.Reshape(engine.TensorArange<int>(0, w, 1), [1, w]), [h, 1]), [hw]);
        var row = engine.Reshape(
            engine.TensorTile(engine.Reshape(engine.TensorArange<int>(0, h, 1), [h, 1]), [1, w]), [hw]);

        var instances = new List<InstanceMask<T>>();
        // Each instance's confidence as stored (rounded to T), the key the instances are ordered by.
        var storedConfidence = new List<double>();
        int classEnd = Math.Min(numClasses, channels);
        for (int cls = 1; cls < classEnd && instances.Count < maxInstances; cls++)
        {
            if (classArea[cls] == 0) continue;
            var (labelMap, count) = SegmentationTensorOps.LabelConnectedComponents(classMap, cls);
            if (count == 0) continue;

            var local = engine.Reshape(labelMap.Cast<int>(), [hw]);
            var areaTensor = engine.TensorBinCount(local, count + 1);
            int[] area = areaTensor.ToArray();
            // Mean class probability per component: the sum over its pixels divided by its area (1 for label 0,
            // whose entry is never read).
            var meanProbability = engine.TensorDivide(
                engine.TensorScatterAdd(new Tensor<double>([count + 1]), local, own, 0),
                engine.TensorClampMin(areaTensor, 1).Cast<double>());
            double[] confidence = meanProbability.ToArray();
            var roundedProbability = meanProbability.Cast<T>();
            T[] confidenceValue = roundedProbability.ToArray();
            double[] confidenceKey = roundedProbability.Cast<double>().ToArray();
            // Tight boxes [minX, minY, maxX + 1, maxY + 1], converted to T once per class.
            T[] left = ComponentExtreme(local, column, count, ScatterReduceMode.AMin).Cast<T>().ToArray();
            T[] top = ComponentExtreme(local, row, count, ScatterReduceMode.AMin).Cast<T>().ToArray();
            T[] right = engine.TensorAddScalar(
                ComponentExtreme(local, column, count, ScatterReduceMode.AMax), 1).Cast<T>().ToArray();
            T[] bottom = engine.TensorAddScalar(
                ComponentExtreme(local, row, count, ScatterReduceMode.AMax), 1).Cast<T>().ToArray();

            for (int comp = 1; comp <= count && instances.Count < maxInstances; comp++)
            {
                if (area[comp] < minArea || confidence[comp] < confidenceThreshold) continue;

                var mask = engine.Reshape(engine.TensorEquals(local, comp).Cast<T>(), [h, w]);
                var box = new BoundingBox<T>(left[comp], top[comp], right[comp], bottom[comp], BoundingBoxFormat.XYXY, cls);
                instances.Add(new InstanceMask<T>(box, mask, cls, confidenceValue[comp]));
                storedConfidence.Add(confidenceKey[comp]);
            }
        }

        return Enumerable.Range(0, instances.Count)
            .OrderByDescending(index => storedConfidence[index])
            .Select(index => instances[index])
            .ToList();
    }

    /// <summary>The smallest or largest <paramref name="coordinate"/> over the pixels of each component label.</summary>
    private static Tensor<int> ComponentExtreme(Tensor<int> componentLabels, Tensor<int> coordinate, int count, ScatterReduceMode mode)
        => AiDotNetEngine.Current.TensorScatterReduce(
            new Tensor<int>([count + 1]), 0, componentLabels, coordinate, mode, includeSelf: false);

    /// <summary>
    /// The attention map <c>[H, W]</c> of a set of point prompts: a Gaussian bump at every point, added for a
    /// positive point and subtracted for a negative one, in point order.
    /// </summary>
    /// <param name="points">Point coordinates <c>[N, 2]</c> as (x, y).</param>
    /// <param name="labels">One label per point <c>[N]</c>.</param>
    /// <param name="height">Map height.</param>
    /// <param name="width">Map width.</param>
    /// <param name="sigma">Standard deviation of each bump, in pixels.</param>
    /// <param name="positiveAtOrAbove">
    /// When null a point is positive only if its label is exactly 1; otherwise it is positive when its label is at
    /// least this value.
    /// </param>
    internal static Tensor<T> PointAttention<T>(
        Tensor<T> points, Tensor<T> labels, int height, int width, double sigma, double? positiveAtOrAbove = null)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        if (labels is null) throw new ArgumentNullException(nameof(labels));

        var engine = AiDotNetEngine.Current;
        int count = points.Shape[0];
        var attention = new Tensor<T>([height, width]);
        if (count == 0) return attention;

        int stride = points.Length / count;
        double[] coordinates = points.Cast<double>().ToArray();
        double[] label = labels.Cast<double>().ToArray();
        for (int i = 0; i < count; i++)
        {
            bool positive = positiveAtOrAbove is { } floor ? label[i] >= floor : label[i] == 1.0;
            var bump = SegmentationTensorOps.GaussianMask<T>(
                height, width, coordinates[i * stride], coordinates[i * stride + 1], sigma);
            attention = positive ? engine.TensorAdd(attention, bump) : engine.TensorSubtract(attention, bump);
        }

        return attention;
    }

    /// <summary>
    /// The prompt score map <c>[H, W]</c>: the per-pixel channel mean of <paramref name="features"/>
    /// <c>[C, H, W]</c> times <paramref name="gate"/> <c>[H, W]</c>, evaluated in double and stored in T.
    /// </summary>
    internal static Tensor<T> GatedChannelMean<T>(Tensor<T> features, Tensor<T> gate)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (gate is null) throw new ArgumentNullException(nameof(gate));

        var engine = AiDotNetEngine.Current;
        var sum = engine.ReduceSum(features.Cast<double>(), [0], keepDims: false);
        // A true division by the channel count (a tensor divisor): scaling by the reciprocal rounds differently.
        var mean = engine.TensorDivide(sum, Tensor<double>.CreateDefault(sum.Shape.ToArray(), features.Shape[0]));
        return engine.TensorMultiply(mean, gate.Cast<double>()).Cast<T>();
    }

    /// <summary>
    /// A <c>[height, width]</c> gate that is 1 where <paramref name="mask"/> <c>[Hm, Wm]</c> is positive and 0
    /// elsewhere, including every pixel the mask does not cover.
    /// </summary>
    internal static Tensor<T> PositiveMaskGate<T>(Tensor<T> mask, int height, int width)
    {
        if (mask is null) throw new ArgumentNullException(nameof(mask));

        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var positive = engine.TensorGreaterThan(mask, numOps.Zero);
        int maskHeight = mask.Shape[0], maskWidth = mask.Shape[1];
        if (maskHeight == height && maskWidth == width) return positive;

        var gate = new Tensor<T>([height, width]);
        int coveredHeight = Math.Min(maskHeight, height), coveredWidth = Math.Min(maskWidth, width);
        if (coveredHeight == 0 || coveredWidth == 0) return gate;
        return engine.TensorSetSlice(
            gate, engine.TensorSlice(positive, [0, 0], [coveredHeight, coveredWidth]), [0, 0]);
    }

    /// <summary>
    /// Turns a prompt score map <c>[H, W]</c> into the three candidate masks of a promptable model: the sigmoid
    /// of the scores thresholded at 0.3, 0.5 and 0.7.
    /// </summary>
    /// <returns>
    /// <c>Masks</c> <c>[3, H, W]</c>; each mask's score is the mean probability over its pixels (0 for an empty
    /// mask) and its stability the ratio of the areas at threshold + 0.05 and threshold - 0.05;
    /// <c>LowResLogits</c> <c>[1, max(1, H/4), max(1, W/4)]</c> is every fourth score.
    /// </returns>
    internal static PromptedSegmentationResult<T> PromptMasksFromScores<T>(Tensor<T> scoreMap, int height, int width)
    {
        if (scoreMap is null) throw new ArgumentNullException(nameof(scoreMap));

        var engine = AiDotNetEngine.Current;
        var probability = engine.Reshape(SegmentationTensorOps.Sigmoid(scoreMap).Cast<double>(), [height, width]);
        double[] thresholds = [0.3, 0.5, 0.7];
        var masks = new Tensor<T>[thresholds.Length];
        var scores = new double[thresholds.Length];
        var stability = new double[thresholds.Length];
        for (int m = 0; m < thresholds.Length; m++)
        {
            var inside = AtLeast(probability, thresholds[m]);
            double area = engine.TensorSum(inside);
            double areaLow = engine.TensorSum(AtLeast(probability, thresholds[m] - 0.05));
            double areaHigh = engine.TensorSum(AtLeast(probability, thresholds[m] + 0.05));
            masks[m] = inside.Cast<T>();
            scores[m] = area > 0 ? engine.TensorSum(engine.TensorMultiply(probability, inside)) / area : 0;
            stability[m] = areaLow > 0 ? areaHigh / areaLow : 0;
        }

        int lowHeight = Math.Max(1, height / 4), lowWidth = Math.Max(1, width / 4);
        var lowRes = CvTensorOps<T>.Select(
            CvTensorOps<T>.Select(scoreMap, EveryFourth(lowHeight, height), 0), EveryFourth(lowWidth, width), 1);
        return new PromptedSegmentationResult<T>
        {
            Masks = engine.TensorStack(masks, 0),
            Scores = scores,
            LowResLogits = engine.Reshape(lowRes, [1, lowHeight, lowWidth]),
            StabilityScores = stability,
        };
    }

    /// <summary>1 where <paramref name="values"/> is at least <paramref name="threshold"/>, 0 elsewhere (and for NaN).</summary>
    private static Tensor<double> AtLeast(Tensor<double> values, double threshold)
        => AiDotNetEngine.Current.TensorGreaterEqual(values, Tensor<double>.CreateDefault(values.Shape.ToArray(), threshold));

    /// <summary>Source indices <c>0, 4, 8, ...</c>, clamped to the last valid index.</summary>
    private static int[] EveryFourth(int count, int size)
    {
        var indices = new int[count];
        for (int i = 0; i < count; i++) indices[i] = Math.Min(i * 4, size - 1);
        return indices;
    }

    /// <summary>
    /// "Segment everything" for a promptable model with per-class features <c>[C, H, W]</c>: one result per class
    /// that wins at least <paramref name="minArea"/> pixels, in class order, at most
    /// <paramref name="maxResults"/> of them.
    /// </summary>
    /// <param name="features">Per-class scores <c>[C, H, W]</c>; a pixel belongs to its highest-scoring class.</param>
    /// <param name="probabilities">The softmax of <paramref name="features"/> over the class axis.</param>
    /// <param name="maxResults">The most results returned.</param>
    /// <param name="minArea">Classes with fewer pixels are skipped.</param>
    /// <returns>
    /// Each result's mask <c>[1, H, W]</c>, its score (the mean probability of the class over its pixels) and its
    /// stability (0.95 when the score is above 0.7, the score otherwise). When no class qualifies, one empty mask
    /// with score 0.
    /// </returns>
    internal static List<PromptedSegmentationResult<T>> SegmentEveryClass<T>(
        Tensor<T> features, Tensor<T> probabilities, int maxResults = 100, int minArea = 4)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (probabilities is null) throw new ArgumentNullException(nameof(probabilities));

        var engine = AiDotNetEngine.Current;
        int channels = features.Shape[0], h = features.Shape[1], w = features.Shape[2];
        var (labels, own) = LabelsAndOwnProbability(features, probabilities);
        int[] area = engine.TensorBinCount(labels, channels).ToArray();
        double[] probability = engine.TensorScatterAdd(new Tensor<double>([channels]), labels, own, 0).ToArray();

        var results = new List<PromptedSegmentationResult<T>>();
        for (int cls = 0; cls < channels && results.Count < maxResults; cls++)
        {
            if (area[cls] < minArea) continue;
            double confidence = probability[cls] / area[cls];
            results.Add(new PromptedSegmentationResult<T>
            {
                Masks = engine.Reshape(engine.TensorEquals(labels, cls).Cast<T>(), [1, h, w]),
                Scores = [confidence],
                StabilityScores = [confidence > 0.7 ? 0.95 : confidence],
            });
        }

        if (results.Count == 0)
        {
            results.Add(new PromptedSegmentationResult<T>
            {
                Masks = new Tensor<T>([1, h, w]),
                Scores = [0.0],
                StabilityScores = [0.0],
            });
        }

        return results;
    }
}
