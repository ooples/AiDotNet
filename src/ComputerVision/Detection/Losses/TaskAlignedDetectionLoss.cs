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
    /// <param name="classLevels">Raw class logits per level, [batch, classes, height, width].</param>
    /// <param name="distributionLevels">Raw box-distribution logits per level, [batch, 4 * regMax, height, width].</param>
    /// <param name="strides">Input pixels per feature cell for each level.</param>
    /// <param name="imageHeight">Height in pixels of the network input the targets are normalized against.</param>
    /// <param name="imageWidth">Width in pixels of the network input the targets are normalized against.</param>
    /// <param name="targets">Normalized center-format targets, one list per image; empty lists are valid.</param>
    /// <param name="topK">Anchors selected per object: <see cref="TopK"/> or <see cref="OneToOneTopK"/>.</param>
    /// <returns>A scalar connected to every class and distribution level on the active tape.</returns>
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
        for (int level = 0; level < levels; level++)
        {
            widths[level] = classLevels[level].Shape[3];
            cells[level] = classLevels[level].Shape[2] * widths[level];
            levelStart[level + 1] = levelStart[level] + cells[level];
        }
        int anchors = levelStart[levels];

        var classData = new T[levels][];
        var distributionData = new T[levels][];
        for (int level = 0; level < levels; level++)
        {
            classData[level] = classLevels[level].ToArray();
            distributionData[level] = distributionLevels[level].ToArray();
        }

        var positives = new List<Positive>();
        double targetMass = 0;
        for (int image = 0; image < batch; image++)
            targetMass += Assign(image, targets[image], classData, distributionData, strides, cells, widths, levelStart,
                imageHeight, imageWidth, topK, positives);
        double normalizer = Math.Max(1.0, targetMass);

        // Classification: BCE-with-logits, softplus(x) - t x, over every class logit of every anchor.
        Tensor<T>? classification = null;
        var byLevel = positives.GroupBy(positive => positive.Level).ToDictionary(group => group.Key, group => group.ToArray());
        for (int level = 0; level < levels; level++)
        {
            var targetValues = new T[classLevels[level].Length];
            if (byLevel.TryGetValue(level, out var levelPositives))
                foreach (var positive in levelPositives)
                    targetValues[(positive.Image * _numClasses + positive.ClassId) * cells[level] + positive.Cell] = NumOps.FromDouble(positive.Target);
            var logits = classLevels[level];
            var targetTensor = new Tensor<T>(targetValues, logits.Shape.ToArray());
            var bce = engine.ReduceSum(engine.TensorSubtract(engine.Softplus(logits), engine.TensorMultiply(targetTensor, logits)), null);
            classification = classification is null ? bce : engine.TensorAdd(classification, bce);
        }
        var loss = engine.TensorMultiplyScalar(classification ?? throw new InvalidOperationException("No class levels were supplied."),
            NumOps.FromDouble(_options.ClassGain / normalizer));

        if (positives.Count == 0)
        {
            // Background classification is still trained. Connect each distribution level with an exact zero derivative.
            foreach (var distribution in distributionLevels)
                loss = engine.TensorAdd(loss, engine.TensorMultiplyScalar(engine.ReduceSum(distribution, null), NumOps.Zero));
            return loss;
        }

        var ordered = Enumerable.Range(0, levels).Where(byLevel.ContainsKey).SelectMany(level => byLevel[level]).ToArray();
        int count = ordered.Length;
        int bins = _regMax;
        var gathered = new List<Tensor<T>>();
        foreach (int level in Enumerable.Range(0, levels).Where(byLevel.ContainsKey))
        {
            var levelPositives = byLevel[level];
            var indices = new int[checked(levelPositives.Length * 4 * bins)];
            int write = 0;
            foreach (var positive in levelPositives)
                for (int side = 0; side < 4; side++)
                    for (int bin = 0; bin < bins; bin++)
                        indices[write++] = (positive.Image * 4 * bins + side * bins + bin) * cells[level] + positive.Cell;
            var flat = engine.Reshape(distributionLevels[level], new[] { distributionLevels[level].Length });
            gathered.Add(CvTensorOps<T>.Select(flat, indices, 0));
        }
        var rows = engine.Reshape(gathered.Count == 1 ? gathered[0] : engine.TensorConcatenate(gathered.ToArray(), 0),
            new[] { count * 4, bins });

        // Decoded distances: the expectation of each side's softmax distribution, in stride units.
        var binValues = new T[count * 4 * bins];
        for (int row = 0; row < count * 4; row++)
            for (int bin = 0; bin < bins; bin++)
                binValues[row * bins + bin] = NumOps.FromDouble(bin);
        var distances = engine.Reshape(engine.ReduceSum(
            engine.TensorMultiply(engine.TensorSoftmax(rows, 1), new Tensor<T>(binValues, new[] { count * 4, bins })),
            new[] { 1 }, false), new[] { count, 4 });

        var anchorX = new T[count];
        var anchorY = new T[count];
        var goldBoxes = new T[count * 4];
        var weights = new T[count];
        var dflIndices = new int[count * 8];
        var dflWeights = new T[count * 8];
        for (int index = 0; index < count; index++)
        {
            var positive = ordered[index];
            double stride = strides[positive.Level];
            double gridX = positive.AnchorX / stride;
            double gridY = positive.AnchorY / stride;
            anchorX[index] = NumOps.FromDouble(gridX);
            anchorY[index] = NumOps.FromDouble(gridY);
            var gold = new[] { positive.Gold[0] / stride, positive.Gold[1] / stride, positive.Gold[2] / stride, positive.Gold[3] / stride };
            for (int coordinate = 0; coordinate < 4; coordinate++) goldBoxes[index * 4 + coordinate] = NumOps.FromDouble(gold[coordinate]);
            weights[index] = NumOps.FromDouble(positive.Target);

            var sides = new[] { gridX - gold[0], gridY - gold[1], gold[2] - gridX, gold[3] - gridY };
            for (int side = 0; side < 4; side++)
            {
                double distance = Math.Min(Math.Max(sides[side], 0), bins - 1 - 0.01);
                int lower = (int)Math.Floor(distance);
                int row = index * 4 + side;
                double sideWeight = positive.Target / 4;
                dflIndices[row * 2] = row * bins + lower;
                dflIndices[row * 2 + 1] = row * bins + lower + 1;
                dflWeights[row * 2] = NumOps.FromDouble((lower + 1 - distance) * sideWeight);
                dflWeights[row * 2 + 1] = NumOps.FromDouble((distance - lower) * sideWeight);
            }
        }

        var column = new[] { count, 1 };
        var x = new Tensor<T>(anchorX, column);
        var y = new Tensor<T>(anchorY, column);
        var predicted = engine.TensorConcatenate(new[]
        {
            engine.TensorSubtract(x, engine.TensorNarrow(distances, 1, 0, 1)),
            engine.TensorSubtract(y, engine.TensorNarrow(distances, 1, 1, 1)),
            engine.TensorAdd(x, engine.TensorNarrow(distances, 1, 2, 1)),
            engine.TensorAdd(y, engine.TensorNarrow(distances, 1, 3, 1))
        }, 1);
        var ciou = engine.TensorCIoULoss(predicted, new Tensor<T>(goldBoxes, new[] { count, 4 }));
        var box = engine.ReduceSum(engine.TensorMultiply(engine.Reshape(ciou, new[] { count }), new Tensor<T>(weights, new[] { count })), null);
        loss = engine.TensorAdd(loss, engine.TensorMultiplyScalar(box, NumOps.FromDouble(_options.BoxGain / normalizer)));

        // Gather only the two neighbouring bins, so an extreme finite logit elsewhere cannot turn 0 * -inf into NaN.
        var logProbabilities = engine.Reshape(engine.TensorLogSoftmax(rows, 1), new[] { count * 4 * bins });
        var dfl = engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(
            CvTensorOps<T>.Select(logProbabilities, dflIndices, 0), new Tensor<T>(dflWeights, new[] { dflWeights.Length })), null));
        loss = engine.TensorAdd(loss, engine.TensorMultiplyScalar(dfl, NumOps.FromDouble(_options.DflGain / normalizer)));

        // A level can hold no positives (always true for most levels under top-1). Its distribution logits
        // still belong to the objective, with an exact zero derivative, so every head output has a gradient.
        foreach (int level in Enumerable.Range(0, levels).Where(level => !byLevel.ContainsKey(level)))
            loss = engine.TensorAdd(loss, engine.TensorMultiplyScalar(engine.ReduceSum(distributionLevels[level], null), NumOps.Zero));
        return loss;
    }

    private double Assign(int image, IReadOnlyList<DetectionTrainingTarget<T>> objects, T[][] classData, T[][] distributionData,
        IReadOnlyList<int> strides, int[] cells, int[] widths, int[] levelStart, int imageHeight, int imageWidth, int topK,
        List<Positive> positives)
    {
        if (objects.Count == 0) return 0;
        int anchors = levelStart[levelStart.Length - 1];
        var anchorX = new double[anchors];
        var anchorY = new double[anchors];
        var predicted = new double[anchors, 4];
        var anchorLevel = new int[anchors];
        for (int level = 0; level < cells.Length; level++)
        {
            int stride = strides[level];
            for (int cell = 0; cell < cells[level]; cell++)
            {
                int anchor = levelStart[level] + cell;
                anchorLevel[anchor] = level;
                // Row-major grid: the integer quotient is the row, the remainder the column.
                int row = cell / widths[level];
                int column = cell % widths[level];
                anchorX[anchor] = (column + 0.5) * stride;
                anchorY[anchor] = (row + 0.5) * stride;
                for (int side = 0; side < 4; side++)
                {
                    double expectation = ExpectedBin(distributionData[level], image, side, cells[level], cell) * stride;
                    predicted[anchor, side] = side < 2
                        ? (side == 0 ? anchorX[anchor] : anchorY[anchor]) - expectation
                        : (side == 2 ? anchorX[anchor] : anchorY[anchor]) + expectation;
                }
            }
        }

        var gold = new double[objects.Count][];
        var assigned = new int[anchors];
        var assignedIoU = new double[anchors];
        var assignedMetric = new double[anchors];
        for (int anchor = 0; anchor < anchors; anchor++) assigned[anchor] = -1;
        for (int index = 0; index < objects.Count; index++)
        {
            var target = objects[index];
            double cx = NumOps.ToDouble(target.CenterX) * imageWidth;
            double cy = NumOps.ToDouble(target.CenterY) * imageHeight;
            double halfWidth = NumOps.ToDouble(target.Width) * imageWidth / 2;
            double halfHeight = NumOps.ToDouble(target.Height) * imageHeight / 2;
            gold[index] = new[] { cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight };

            var candidates = new List<(int Anchor, double Metric, double IoU)>();
            for (int anchor = 0; anchor < anchors; anchor++)
            {
                double inside = Math.Min(Math.Min(anchorX[anchor] - gold[index][0], anchorY[anchor] - gold[index][1]),
                    Math.Min(gold[index][2] - anchorX[anchor], gold[index][3] - anchorY[anchor]));
                if (inside <= InsideEpsilon) continue;
                int level = anchorLevel[anchor];
                int cell = anchor - levelStart[level];
                double score = Logistic(NumOps.ToDouble(classData[level][(image * _numClasses + target.ClassId) * cells[level] + cell]));
                double iou = IoU(predicted, anchor, gold[index]);
                candidates.Add((anchor, Math.Pow(score, _options.Alpha) * Math.Pow(iou, _options.Beta), iou));
            }
            foreach (var candidate in candidates.OrderByDescending(item => item.Metric).ThenBy(item => item.Anchor).Take(topK))
            {
                if (assigned[candidate.Anchor] >= 0 && candidate.IoU <= assignedIoU[candidate.Anchor]) continue;
                assigned[candidate.Anchor] = index;
                assignedIoU[candidate.Anchor] = candidate.IoU;
                assignedMetric[candidate.Anchor] = candidate.Metric;
            }
        }

        var maximumMetric = new double[objects.Count];
        var maximumIoU = new double[objects.Count];
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

    private double ExpectedBin(T[] distribution, int image, int side, int cells, int cell)
    {
        double maximum = double.NegativeInfinity;
        for (int bin = 0; bin < _regMax; bin++)
            maximum = Math.Max(maximum, NumOps.ToDouble(distribution[(image * 4 * _regMax + side * _regMax + bin) * cells + cell]));
        double total = 0;
        double weighted = 0;
        for (int bin = 0; bin < _regMax; bin++)
        {
            double value = Math.Exp(NumOps.ToDouble(distribution[(image * 4 * _regMax + side * _regMax + bin) * cells + cell]) - maximum);
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
