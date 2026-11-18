using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>Task-aligned assignment with BCE, CIoU and distribution focal losses for anchor-free YOLO heads.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Assignment (Feng et al. 2021, TOOD): an anchor point is a candidate for an object when it lies inside
/// the object's box. Candidates are ranked by t = s^alpha * IoU^beta, where s is the predicted score of
/// the object's class and IoU is between the anchor's predicted box and the object. The top-k candidates
/// become positives; an anchor selected by several objects keeps the one with the highest IoU. Each
/// positive's classification target is t normalized so that, per object, the largest target equals
/// the largest IoU among that object's positives. Assignment uses detached predictions.
/// </para>
/// <para>
/// Losses: BCE of every class logit against those targets; CIoU of each positive's decoded box; and the
/// distribution focal loss of Li et al. (2020), DFL = -((y_{i+1} - y) log S_i + (y - y_i) log S_{i+1}),
/// averaged over the four box sides. Box and DFL terms are weighted per positive by its target. All
/// three sums are divided by the total target mass (at least 1) and scaled by the configured gains.
/// Boxes and DFL distances are measured in units of each level's stride, as the head predicts them.
/// </para>
/// <para><b>For Beginners:</b> This is the training objective of YOLOv8-style detectors. It picks the grid
/// cells best placed to detect each object, teaches their class scores to reflect how well they detect
/// it, and pulls their predicted box edges toward the real ones.</para>
/// </remarks>
public sealed class TaskAlignedDetectionLoss<T>
{
    private const double InsideEpsilon = 1e-9;
    private const double MetricEpsilon = 1e-9;
    private static readonly INumericOperations<T> NumOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
    private readonly TaskAlignedLossOptions _options;
    private readonly int _numClasses;
    private readonly int _regMax;

    /// <summary>Creates the objective for a head with the given class count and distribution bins.</summary>
    /// <param name="numClasses">Foreground classes scored by independent sigmoids.</param>
    /// <param name="regMax">Distribution bins per box side (16 in YOLOv8-family heads).</param>
    /// <param name="options">Assignment exponents, top-k and loss gains; copied on construction.</param>
    public TaskAlignedDetectionLoss(int numClasses, int regMax, TaskAlignedLossOptions options)
    {
        if (numClasses < 1) throw new ArgumentOutOfRangeException(nameof(numClasses));
        if (regMax < 2) throw new ArgumentOutOfRangeException(nameof(regMax), "A distribution needs at least two bins.");
        if (options is null) throw new ArgumentNullException(nameof(options));
        _options = options.Snapshot();
        _numClasses = numClasses;
        _regMax = regMax;
    }

    /// <summary>Anchors selected per object by the one-to-many assignment.</summary>
    public int TopK => _options.TopK;

    /// <summary>Anchors selected per object by a one-to-one head.</summary>
    public int OneToOneTopK => _options.OneToOneTopK;

    /// <summary>Evaluates the objective without recording gradients.</summary>
    public T CalculateLoss(IReadOnlyList<Tensor<T>> classLevels, IReadOnlyList<Tensor<T>> distributionLevels,
        IReadOnlyList<int> strides, int imageHeight, int imageWidth, DetectionTrainingBatch<T> targets, int topK)
    {
        using var noGrad = new NoGradScope<T>();
        using var objective = ComputeTapeLoss(classLevels, distributionLevels, strides, imageHeight, imageWidth, targets, topK);
        return objective[0];
    }

    /// <summary>Builds the differentiable objective over the live head outputs.</summary>
    public Tensor<T> ComputeTapeLoss(IReadOnlyList<Tensor<T>> classLevels, IReadOnlyList<Tensor<T>> distributionLevels,
        IReadOnlyList<int> strides, int imageHeight, int imageWidth, DetectionTrainingBatch<T> targets, int topK)
    {
        Validate(classLevels, distributionLevels, strides, imageHeight, imageWidth, targets, topK);
        var engine = AiDotNetEngine.Current;
        int levels = classLevels.Count;
        int batch = classLevels[0].Shape[0];
        var cells = new int[levels];
        var widths = new int[levels];
        var levelStart = new int[levels + 1];
        var classOffsets = new int[levels + 1];
        var distOffsets = new int[levels + 1];
        var flatClasses = new Tensor<T>[levels];
        var flatDists = new Tensor<T>[levels];
        for (int level = 0; level < levels; level++)
        {
            widths[level] = classLevels[level].Shape[3];
            cells[level] = classLevels[level].Shape[2] * widths[level];
            levelStart[level + 1] = levelStart[level] + cells[level];
            classOffsets[level + 1] = classOffsets[level] + classLevels[level].Length;
            distOffsets[level + 1] = distOffsets[level] + distributionLevels[level].Length;
            flatClasses[level] = engine.Reshape(classLevels[level], new[] { classLevels[level].Length });
            flatDists[level] = engine.Reshape(distributionLevels[level], new[] { distributionLevels[level].Length });
        }

        var allClassFlat = levels == 1 ? flatClasses[0] : engine.TensorConcatenate(flatClasses, 0);
        var allDistFlat = levels == 1 ? flatDists[0] : engine.TensorConcatenate(flatDists, 0);
        int totalClassLen = classOffsets[levels];
        int totalDistLen = distOffsets[levels];

        if (targets.TargetCount == 0)
        {
            var bgLoss = engine.TensorMultiplyScalar(engine.ReduceSum(engine.Softplus(allClassFlat), null),
                NumOps.FromDouble(_options.ClassGain));
            var zeroDistBg = engine.TensorMultiplyScalar(engine.ReduceSum(allDistFlat, null), NumOps.Zero);
            return engine.TensorAdd(bgLoss, zeroDistBg);
        }

        int anchors = levelStart[levels];
        int maxPositives = Math.Max(1, Math.Min(checked(batch * anchors), checked(targets.TargetCount * topK)));
        int bins = _regMax;
        int numClasses = _numClasses;
        double alpha = _options.Alpha;
        double beta = _options.Beta;
        double classGain = _options.ClassGain;
        double boxGain = _options.BoxGain;
        double dflGain = _options.DflGain;

        // Packed layout:
        // [0 .. totalClassLen): classTargets
        // [totalClassLen]: classGain / normalizer
        // [.. + maxPositives * 4 * bins]: distGatherIndices
        // [.. + maxPositives]: anchorX
        // [.. + maxPositives]: anchorY
        // [.. + maxPositives * 4]: goldBoxes
        // [.. + maxPositives]: boxWeights (positive.Target * boxGain / normalizer)
        // [.. + maxPositives * 8]: dflIndices
        // [.. + maxPositives * 8]: dflWeights (sideWeight * interp * dflGain / normalizer)
        int oScale = totalClassLen;
        int oDistIdx = oScale + 1;
        int distIdxLen = checked(maxPositives * 4 * bins);
        int oAnchorX = oDistIdx + distIdxLen;
        int oAnchorY = oAnchorX + maxPositives;
        int oGold = oAnchorY + maxPositives;
        int goldLen = checked(maxPositives * 4);
        int oBoxW = oGold + goldLen;
        int oDflIdx = oBoxW + maxPositives;
        int dflLen = checked(maxPositives * 8);
        int oDflW = oDflIdx + dflLen;
        int totalPackedLen = oDflW + dflLen;

        var packedSupervision = PackSupervision(
            batch, levels, strides, cells, widths, levelStart, classOffsets, distOffsets,
            imageHeight, imageWidth, topK, targets);
        var supervision = AiDotNet.Training.CompiledStepHostValue<T>.Compute(
            "TaskAlignedSupervision",
            new[] { allClassFlat, allDistFlat, packedSupervision },
            new[] { totalPackedLen },
            (inputs, dst) => FillSupervision(
                inputs[0], inputs[1], inputs[2], numClasses, bins, maxPositives,
                alpha, beta, classGain, boxGain, dflGain, totalClassLen, dst));

        var classTargets = engine.TensorSlice(supervision, new[] { 0 }, new[] { totalClassLen });
        var classScale = engine.TensorSlice(supervision, new[] { oScale }, new[] { 1 });
        var distGatherIndices = engine.TensorSlice(supervision, new[] { oDistIdx }, new[] { distIdxLen });
        var x = engine.Reshape(engine.TensorSlice(supervision, new[] { oAnchorX }, new[] { maxPositives }), new[] { maxPositives, 1 });
        var y = engine.Reshape(engine.TensorSlice(supervision, new[] { oAnchorY }, new[] { maxPositives }), new[] { maxPositives, 1 });
        var goldBoxes = engine.Reshape(engine.TensorSlice(supervision, new[] { oGold }, new[] { goldLen }), new[] { maxPositives, 4 });
        var boxWeights = engine.TensorSlice(supervision, new[] { oBoxW }, new[] { maxPositives });
        var dflIndices = engine.TensorSlice(supervision, new[] { oDflIdx }, new[] { dflLen });
        var dflWeights = engine.TensorSlice(supervision, new[] { oDflW }, new[] { dflLen });

        var bceElements = engine.TensorSubtract(engine.Softplus(allClassFlat), engine.TensorMultiply(classTargets, allClassFlat));
        var bceSum = engine.Reshape(engine.ReduceSum(bceElements, new[] { 0 }, keepDims: true), new[] { 1 });
        var loss = engine.ReduceSum(engine.TensorMultiply(bceSum, classScale), null);

        var gatheredDist = engine.TensorEmbeddingLookupFromFloatIndices(
            engine.Reshape(allDistFlat, new[] { totalDistLen, 1 }), distGatherIndices);
        var rows = engine.Reshape(gatheredDist, new[] { maxPositives * 4, bins });

        var binValues = new double[maxPositives * 4 * bins];
        for (int row = 0; row < maxPositives * 4; row++)
            for (int bin = 0; bin < bins; bin++)
                binValues[row * bins + bin] = bin;
        var distances = engine.Reshape(engine.ReduceSum(
            engine.TensorMultiply(engine.TensorSoftmax(rows, 1), new Tensor<double>(binValues, new[] { maxPositives * 4, bins }).Cast<T>()),
            new[] { 1 }, false), new[] { maxPositives, 4 });

        var predicted = engine.TensorConcatenate(new[]
        {
            engine.TensorSubtract(x, engine.TensorNarrow(distances, 1, 0, 1)),
            engine.TensorSubtract(y, engine.TensorNarrow(distances, 1, 1, 1)),
            engine.TensorAdd(x, engine.TensorNarrow(distances, 1, 2, 1)),
            engine.TensorAdd(y, engine.TensorNarrow(distances, 1, 3, 1))
        }, 1);
        var ciou = engine.TensorCIoULoss(predicted, goldBoxes);
        var boxTerm = engine.ReduceSum(engine.TensorMultiply(engine.Reshape(ciou, new[] { maxPositives }), boxWeights), null);
        loss = engine.TensorAdd(loss, boxTerm);

        var logProbabilities = engine.Reshape(engine.TensorLogSoftmax(rows, 1), new[] { maxPositives * 4 * bins, 1 });
        var selectedLogProbs = engine.Reshape(
            engine.TensorEmbeddingLookupFromFloatIndices(logProbabilities, dflIndices),
            new[] { dflLen });
        var dflTerm = engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(selectedLogProbs, dflWeights), null));
        loss = engine.TensorAdd(loss, dflTerm);

        // Keep every distribution level connected even when positives do not touch it.
        loss = engine.TensorAdd(loss, engine.TensorMultiplyScalar(engine.ReduceSum(allDistFlat, null), NumOps.Zero));
        return loss;
    }

    private static Tensor<T> PackSupervision(
        int batch, int levels, IReadOnlyList<int> strides, int[] cells, int[] widths, int[] levelStart,
        int[] classOffsets, int[] distOffsets, int imageHeight, int imageWidth, int topK, DetectionTrainingBatch<T> targets)
    {
        var values = new List<double> { batch, levels, imageHeight, imageWidth, topK };
        for (int l = 0; l < levels; l++)
        {
            values.Add(strides[l]);
            values.Add(cells[l]);
            values.Add(widths[l]);
            values.Add(levelStart[l]);
            values.Add(classOffsets[l]);
            values.Add(distOffsets[l]);
        }
        values.Add(levelStart[levels]);
        values.Add(classOffsets[levels]);
        values.Add(distOffsets[levels]);

        int totalObjects = targets.TargetCount;
        var rawCoords = new T[totalObjects * 4];
        int coordWrite = 0;
        for (int image = 0; image < batch; image++)
        {
            var imgTargets = targets[image];
            for (int t = 0; t < imgTargets.Count; t++)
            {
                var item = imgTargets[t];
                rawCoords[coordWrite++] = item.CenterX;
                rawCoords[coordWrite++] = item.CenterY;
                rawCoords[coordWrite++] = item.Width;
                rawCoords[coordWrite++] = item.Height;
            }
        }
        var doubleCoords = totalObjects > 0
            ? new Tensor<T>(rawCoords, new[] { rawCoords.Length }).Cast<double>().ToArray()
            : Array.Empty<double>();
        int coordRead = 0;
        for (int image = 0; image < batch; image++)
        {
            var imgTargets = targets[image];
            values.Add(imgTargets.Count);
            for (int t = 0; t < imgTargets.Count; t++)
            {
                values.Add(imgTargets[t].ClassId);
                values.Add(doubleCoords[coordRead++]);
                values.Add(doubleCoords[coordRead++]);
                values.Add(doubleCoords[coordRead++]);
                values.Add(doubleCoords[coordRead++]);
            }
        }

        return new Tensor<double>(values.ToArray(), new[] { values.Count }).Cast<T>();
    }

    private static void FillSupervision(
        Tensor<T> allClassFlat, Tensor<T> allDistFlat, Tensor<T> packedSupervision,
        int numClasses, int bins, int maxPositives,
        double alpha, double beta, double classGain, double boxGain, double dflGain,
        int totalClassLen, Tensor<T> dst)
    {
        var classData = allClassFlat.Contiguous().Cast<double>().ToArray();
        var distData = allDistFlat.Contiguous().Cast<double>().ToArray();
        var sup = packedSupervision.Contiguous().Cast<double>().ToArray();

        int cursor = 0;
        int batch = (int)Math.Round(sup[cursor++]);
        int levels = (int)Math.Round(sup[cursor++]);
        int imageHeight = (int)Math.Round(sup[cursor++]);
        int imageWidth = (int)Math.Round(sup[cursor++]);
        int topK = (int)Math.Round(sup[cursor++]);

        var strides = new int[levels];
        var cells = new int[levels];
        var widths = new int[levels];
        var levelStart = new int[levels + 1];
        var classOffsets = new int[levels + 1];
        var distOffsets = new int[levels + 1];
        for (int l = 0; l < levels; l++)
        {
            strides[l] = (int)Math.Round(sup[cursor++]);
            cells[l] = (int)Math.Round(sup[cursor++]);
            widths[l] = (int)Math.Round(sup[cursor++]);
            levelStart[l] = (int)Math.Round(sup[cursor++]);
            classOffsets[l] = (int)Math.Round(sup[cursor++]);
            distOffsets[l] = (int)Math.Round(sup[cursor++]);
        }
        levelStart[levels] = (int)Math.Round(sup[cursor++]);
        classOffsets[levels] = (int)Math.Round(sup[cursor++]);
        distOffsets[levels] = (int)Math.Round(sup[cursor++]);

        var positives = new List<Positive>();
        double targetMass = 0;
        for (int image = 0; image < batch; image++)
        {
            int objCount = (int)Math.Round(sup[cursor++]);
            var objects = new (int ClassId, double Cx, double Cy, double W, double H)[objCount];
            for (int t = 0; t < objCount; t++)
            {
                int cid = (int)Math.Round(sup[cursor++]);
                double cx = sup[cursor++];
                double cy = sup[cursor++];
                double w = sup[cursor++];
                double h = sup[cursor++];
                objects[t] = (cid, cx, cy, w, h);
            }
            targetMass += AssignStatic(
                image, objects, classData, distData, strides, cells, widths, levelStart,
                classOffsets, distOffsets, imageHeight, imageWidth, numClasses, bins, alpha, beta, topK, positives);
        }

        double normalizer = Math.Max(1.0, targetMass);
        var buffer = new double[dst.Length];
        int oScale = totalClassLen;
        int oDistIdx = oScale + 1;
        int distIdxLen = maxPositives * 4 * bins;
        int oAnchorX = oDistIdx + distIdxLen;
        int oAnchorY = oAnchorX + maxPositives;
        int oGold = oAnchorY + maxPositives;
        int oBoxW = oGold + (maxPositives * 4);
        int oDflIdx = oBoxW + maxPositives;
        int oDflW = oDflIdx + (maxPositives * 8);

        buffer[oScale] = classGain / normalizer;

        var ordered = positives.OrderBy(p => p.Level).ToArray();
        foreach (var p in ordered)
        {
            int clsIdx = classOffsets[p.Level] + ((p.Image * numClasses + p.ClassId) * cells[p.Level] + p.Cell);
            buffer[clsIdx] = p.Target;
        }

        int activeCount = Math.Min(ordered.Length, maxPositives);
        for (int index = 0; index < maxPositives; index++)
        {
            if (index < activeCount)
            {
                var p = ordered[index];
                int cellCount = cells[p.Level];
                int distBase = distOffsets[p.Level];
                int write = oDistIdx + (index * 4 * bins);
                for (int side = 0; side < 4; side++)
                    for (int bin = 0; bin < bins; bin++)
                        buffer[write++] = distBase + ((p.Image * 4 * bins + side * bins + bin) * cellCount + p.Cell);

                double stride = strides[p.Level];
                double gridX = p.AnchorX / stride;
                double gridY = p.AnchorY / stride;
                buffer[oAnchorX + index] = gridX;
                buffer[oAnchorY + index] = gridY;
                double g0 = p.Gold[0] / stride, g1 = p.Gold[1] / stride, g2 = p.Gold[2] / stride, g3 = p.Gold[3] / stride;
                buffer[oGold + (index * 4)] = g0;
                buffer[oGold + (index * 4) + 1] = g1;
                buffer[oGold + (index * 4) + 2] = g2;
                buffer[oGold + (index * 4) + 3] = g3;
                buffer[oBoxW + index] = p.Target * (boxGain / normalizer);

                var sides = new[] { gridX - g0, gridY - g1, g2 - gridX, g3 - gridY };
                double sideScale = (p.Target / 4.0) * (dflGain / normalizer);
                for (int side = 0; side < 4; side++)
                {
                    double distance = Math.Min(Math.Max(sides[side], 0.0), bins - 1 - 0.01);
                    int lower = (int)Math.Floor(distance);
                    int row = (index * 4) + side;
                    buffer[oDflIdx + (row * 2)] = (row * bins) + lower;
                    buffer[oDflIdx + (row * 2) + 1] = (row * bins) + lower + 1;
                    buffer[oDflW + (row * 2)] = (lower + 1 - distance) * sideScale;
                    buffer[oDflW + (row * 2) + 1] = (distance - lower) * sideScale;
                }
            }
            else
            {
                // Non-degenerate zero-weight padding slot so CIoU and DFL stay strictly finite.
                buffer[oAnchorX + index] = 0.5;
                buffer[oAnchorY + index] = 0.5;
                buffer[oGold + (index * 4)] = 0.0;
                buffer[oGold + (index * 4) + 1] = 0.0;
                buffer[oGold + (index * 4) + 2] = 1.0;
                buffer[oGold + (index * 4) + 3] = 1.0;
                for (int side = 0; side < 4; side++)
                {
                    int row = (index * 4) + side;
                    buffer[oDflIdx + (row * 2)] = row * bins;
                    buffer[oDflIdx + (row * 2) + 1] = (row * bins) + 1;
                }
            }
        }

        var typed = new Tensor<double>(dst._shape, new Vector<double>(buffer)).Cast<T>();
        typed.AsSpan().CopyTo(dst.AsWritableSpan());
    }

    private static double AssignStatic(
        int image, (int ClassId, double Cx, double Cy, double W, double H)[] objects,
        double[] classData, double[] distData,
        int[] strides, int[] cells, int[] widths, int[] levelStart,
        int[] classOffsets, int[] distOffsets,
        int imageHeight, int imageWidth, int numClasses, int regMax,
        double alpha, double beta, int topK, List<Positive> positives)
    {
        if (objects.Length == 0) return 0;
        int anchors = levelStart[levelStart.Length - 1];
        var anchorX = new double[anchors];
        var anchorY = new double[anchors];
        var predicted = new double[anchors, 4];
        var anchorLevel = new int[anchors];
        for (int level = 0; level < cells.Length; level++)
        {
            int stride = strides[level];
            int distBase = distOffsets[level];
            for (int cell = 0; cell < cells[level]; cell++)
            {
                int anchor = levelStart[level] + cell;
                anchorLevel[anchor] = level;
                int row = cell / widths[level];
                int column = cell % widths[level];
                anchorX[anchor] = (column + 0.5) * stride;
                anchorY[anchor] = (row + 0.5) * stride;
                for (int side = 0; side < 4; side++)
                {
                    double expectation = ExpectedBinStatic(distData, distBase, image, side, cells[level], cell, regMax) * stride;
                    predicted[anchor, side] = side < 2
                        ? (side == 0 ? anchorX[anchor] : anchorY[anchor]) - expectation
                        : (side == 2 ? anchorX[anchor] : anchorY[anchor]) + expectation;
                }
            }
        }

        var gold = new double[objects.Length][];
        var assigned = new int[anchors];
        var assignedIoU = new double[anchors];
        var assignedMetric = new double[anchors];
        for (int anchor = 0; anchor < anchors; anchor++) assigned[anchor] = -1;
        for (int index = 0; index < objects.Length; index++)
        {
            var target = objects[index];
            double cx = target.Cx * imageWidth;
            double cy = target.Cy * imageHeight;
            double halfWidth = target.W * imageWidth / 2;
            double halfHeight = target.H * imageHeight / 2;
            gold[index] = new[] { cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight };

            var candidates = new List<(int Anchor, double Metric, double IoU)>();
            for (int anchor = 0; anchor < anchors; anchor++)
            {
                double inside = Math.Min(Math.Min(anchorX[anchor] - gold[index][0], anchorY[anchor] - gold[index][1]),
                    Math.Min(gold[index][2] - anchorX[anchor], gold[index][3] - anchorY[anchor]));
                if (inside <= InsideEpsilon) continue;
                int level = anchorLevel[anchor];
                int cell = anchor - levelStart[level];
                double logit = classData[classOffsets[level] + ((image * numClasses + target.ClassId) * cells[level] + cell)];
                double score = Logistic(logit);
                double iou = IoU(predicted, anchor, gold[index]);
                candidates.Add((anchor, Math.Pow(score, alpha) * Math.Pow(iou, beta), iou));
            }
            foreach (var candidate in candidates.OrderByDescending(item => item.Metric).ThenBy(item => item.Anchor).Take(topK))
            {
                if (assigned[candidate.Anchor] >= 0 && candidate.IoU <= assignedIoU[candidate.Anchor]) continue;
                assigned[candidate.Anchor] = index;
                assignedIoU[candidate.Anchor] = candidate.IoU;
                assignedMetric[candidate.Anchor] = candidate.Metric;
            }
        }

        var maximumMetric = new double[objects.Length];
        var maximumIoU = new double[objects.Length];
        for (int anchor = 0; anchor < anchors; anchor++)
        {
            int index = assigned[anchor];
            if (index < 0) continue;
            maximumMetric[index] = Math.Max(maximumMetric[index], assignedMetric[anchor]);
            maximumIoU[index] = Math.Max(maximumIoU[index], assignedIoU[anchor]);
        }

        double mass = 0;
        for (int anchor = 0; anchor < anchors; anchor++)
        {
            int index = assigned[anchor];
            if (index < 0) continue;
            double normalized = assignedMetric[anchor] * maximumIoU[index] / (maximumMetric[index] + MetricEpsilon);
            int level = anchorLevel[anchor];
            positives.Add(new Positive(image, level, anchor - levelStart[level], objects[index].ClassId,
                anchorX[anchor], anchorY[anchor], gold[index], normalized));
            mass += normalized;
        }
        return mass;
    }

    private static double ExpectedBinStatic(double[] distData, int distBase, int image, int side, int cells, int cell, int regMax)
    {
        double maximum = double.NegativeInfinity;
        for (int bin = 0; bin < regMax; bin++)
            maximum = Math.Max(maximum, distData[distBase + ((image * 4 * regMax + side * regMax + bin) * cells + cell)]);
        double total = 0;
        double weighted = 0;
        for (int bin = 0; bin < regMax; bin++)
        {
            double value = Math.Exp(distData[distBase + ((image * 4 * regMax + side * regMax + bin) * cells + cell)] - maximum);
            total += value;
            weighted += value * bin;
        }
        return weighted / total;
    }

    private static double IoU(double[,] predicted, int anchor, double[] gold)
    {
        double width = Math.Max(0, Math.Min(predicted[anchor, 2], gold[2]) - Math.Max(predicted[anchor, 0], gold[0]));
        double height = Math.Max(0, Math.Min(predicted[anchor, 3], gold[3]) - Math.Max(predicted[anchor, 1], gold[1]));
        double intersection = width * height;
        double union = (predicted[anchor, 2] - predicted[anchor, 0]) * (predicted[anchor, 3] - predicted[anchor, 1])
            + (gold[2] - gold[0]) * (gold[3] - gold[1]) - intersection;
        return union > 0 ? Math.Max(0, intersection / union) : 0;
    }

    private static double Logistic(double x) => x >= 0 ? 1 / (1 + Math.Exp(-x)) : Math.Exp(x) / (1 + Math.Exp(x));

    private void Validate(IReadOnlyList<Tensor<T>> classLevels, IReadOnlyList<Tensor<T>> distributionLevels,
        IReadOnlyList<int> strides, int imageHeight, int imageWidth, DetectionTrainingBatch<T> targets, int topK)
    {
        if (classLevels is null) throw new ArgumentNullException(nameof(classLevels));
        if (distributionLevels is null) throw new ArgumentNullException(nameof(distributionLevels));
        if (strides is null) throw new ArgumentNullException(nameof(strides));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (classLevels.Count == 0 || distributionLevels.Count != classLevels.Count || strides.Count != classLevels.Count)
            throw new ArgumentException("Class levels, distribution levels and strides must be nonempty and of equal count.", nameof(classLevels));
        if (imageHeight <= 0 || imageWidth <= 0) throw new ArgumentOutOfRangeException(nameof(imageHeight), "Image dimensions must be positive.");
        if (topK < 1) throw new ArgumentOutOfRangeException(nameof(topK));
        int batch = classLevels[0].Rank == 4 ? classLevels[0].Shape[0] : 0;
        for (int level = 0; level < classLevels.Count; level++)
        {
            var logits = classLevels[level];
            var distribution = distributionLevels[level];
            if (logits.Rank != 4 || logits.Shape[0] != batch || batch <= 0 || logits.Shape[1] != _numClasses || logits.Shape[2] <= 0 || logits.Shape[3] <= 0)
                throw new ArgumentException($"Class level {level} must be [batch, {_numClasses}, height, width].", nameof(classLevels));
            if (distribution.Rank != 4 || distribution.Shape[0] != batch || distribution.Shape[1] != 4 * _regMax
                || distribution.Shape[2] != logits.Shape[2] || distribution.Shape[3] != logits.Shape[3])
                throw new ArgumentException($"Distribution level {level} must be [batch, {4 * _regMax}, height, width] matching its class level.", nameof(distributionLevels));
            if (strides[level] <= 0) throw new ArgumentOutOfRangeException(nameof(strides), "Strides must be positive.");
        }
        targets.ValidateForModel(batch, _numClasses, int.MaxValue);
    }

    private sealed class Positive
    {
        internal Positive(int image, int level, int cell, int classId, double anchorX, double anchorY, double[] gold, double target)
        {
            Image = image;
            Level = level;
            Cell = cell;
            ClassId = classId;
            AnchorX = anchorX;
            AnchorY = anchorY;
            Gold = gold;
            Target = target;
        }

        internal int Image { get; }
        internal int Level { get; }
        internal int Cell { get; }
        internal int ClassId { get; }
        internal double AnchorX { get; }
        internal double AnchorY { get; }
        internal double[] Gold { get; }
        internal double Target { get; }
    }
}
