using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Solvers.Assignment;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>DETR-family set prediction loss with exact Hungarian assignment.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Foreground queries are assigned with a weighted sum of a classification cost, center-format L1
/// distance and negative GIoU. The matched boxes receive L1 and GIoU losses normalized by the
/// total foreground target count across the local batch. Every query receives classification
/// supervision, including unmatched queries and empty images.
/// </para>
/// <para>
/// Three classification forms are supported (<see cref="DetrSetLossOptions.ClassificationLoss"/>):
/// DETR's softmax cross-entropy with a down-weighted no-object class (Carion et al. 2020); the
/// sigmoid focal loss of DINO (Zhang et al. 2022), normalized by the target count; and RT-DETR's
/// IoU-aware varifocal loss (Zhao et al. 2023; Zhang et al. 2021), whose matched class target is the
/// IoU of the matched predicted box. Sigmoid heads are matched with the focal classification cost
/// of Deformable DETR's reference matcher.
/// </para>
/// <para>
/// Only the supplied final prediction heads are supervised; intermediate decoder, query-selection
/// and denoising losses are not fabricated. More targets than queries in an image are rejected
/// rather than silently dropped.
/// </para>
/// </remarks>
public class DETRSetLoss<T> : LossFunctionBase<T>
{
    private readonly NMS<T> _nms = new();
    private readonly DetrSetLossOptions _options;
    private readonly int _numClasses;

    /// <summary>Creates a DETR objective with the standard final-head loss weights.</summary>
    /// <param name="numClasses">Number of output classes, including the final no-object class.</param>
    /// <param name="classWeight">Nonnegative classification and matching cost weight.</param>
    /// <param name="boxL1Weight">Nonnegative center-format L1 loss and matching cost weight.</param>
    /// <param name="boxGIoUWeight">Nonnegative GIoU loss and matching cost weight.</param>
    public DETRSetLoss(int numClasses = 91, double classWeight = 1.0,
        double boxL1Weight = 5.0, double boxGIoUWeight = 2.0)
        : this(numClasses, SoftmaxOptions(classWeight, boxL1Weight, boxGIoUWeight))
    {
    }

    /// <summary>Creates a DETR-family objective with explicit classification form and weights.</summary>
    /// <param name="numClasses">
    /// Width of the class head: foreground classes plus the final no-object class for softmax
    /// cross-entropy, or foreground classes only for the sigmoid focal and varifocal forms.
    /// </param>
    /// <param name="options">Classification form, matching costs and loss weights; copied on construction.</param>
    public DETRSetLoss(int numClasses, DetrSetLossOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        _options = options.Snapshot();
        int minimumClasses = UsesNoObjectClass ? 2 : 1;
        if (numClasses < minimumClasses) throw new ArgumentOutOfRangeException(nameof(numClasses));
        _numClasses = numClasses;
    }

    /// <summary>The classification form this objective trains.</summary>
    public SetPredictionClassificationLoss ClassificationLoss => _options.ClassificationLoss;

    private bool UsesNoObjectClass => _options.ClassificationLoss == SetPredictionClassificationLoss.SoftmaxCrossEntropy;

    private int ForegroundClasses => UsesNoObjectClass ? _numClasses - 1 : _numClasses;

    /// <summary>Calculates the documented element-wise MAE compatibility objective.</summary>
    /// <remarks>This vector overload does not describe detection targets or perform matching.</remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);
        double total = 0;
        for (int i = 0; i < predicted.Length; i++)
            total += Math.Abs(NumOps.ToDouble(predicted[i]) - NumOps.ToDouble(actual[i]));
        return NumOps.FromDouble(total / predicted.Length);
    }

    /// <summary>Evaluates structured predictions against -1-padded DETR targets.</summary>
    /// <remarks>
    /// Predictions are [batch, queries, classes + 4], with logits followed by normalized cxcywh.
    /// Targets are [batch, objects, at least 5], containing class and normalized cxcywh.
    /// A -1 class starts padding. COCO loader xywh targets must be converted explicitly.
    /// </remarks>
    public T CalculateLoss(Tensor<T> predicted, Tensor<T> targets)
    {
        using var noGrad = new NoGradScope<T>();
        ValidateStructuredLayout(predicted, targets);
        using var objective = ComputeStructuredLoss(predicted, targets);
        return objective[0];
    }

    /// <summary>Evaluates the actual class and normalized cxcywh heads against typed targets.</summary>
    public T CalculateLoss(Tensor<T> classLogits, Tensor<T> boxes, DetectionTrainingBatch<T> targets)
    {
        using var noGrad = new NoGradScope<T>();
        using var objective = ComputeTapeLoss(classLogits, boxes, targets);
        return objective[0];
    }

    /// <inheritdoc />
    /// <remarks>
    /// Structured tensor inputs use the same objective as the typed-head overload. Identical
    /// non-structured shapes retain the documented element-wise MAE compatibility objective.
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (HasStructuredLayout(predicted, target))
            return ComputeStructuredLoss(predicted, target);

        bool sameShape = predicted.Rank == target.Rank;
        for (int axis = 0; axis < predicted.Rank && sameShape; axis++)
            sameShape = predicted.Shape[axis] == target.Shape[axis];
        if (sameShape && !IsStructuredPrediction(predicted))
        {
            var difference = Engine.TensorAbs(Engine.TensorSubtract(predicted, target));
            return Engine.TensorMultiplyScalar(Engine.ReduceSum(difference, null),
                NumOps.FromDouble(1.0 / Math.Max(1, predicted.Length)));
        }

        ValidateStructuredLayout(predicted, target);
        throw new InvalidOperationException("Structured layout validation did not reject an incompatible shape.");
    }

    /// <summary>Builds differentiable final-head classification, L1 and GIoU losses after discrete assignment.</summary>
    /// <param name="classLogits">Raw class logits [batch, queries, class-head width].</param>
    /// <param name="boxes">Sigmoid box predictions [batch, queries, 4] in normalized cxcywh.</param>
    /// <param name="targets">Immutable, unpadded foreground targets; empty images are valid.</param>
    /// <returns>A scalar connected to both prediction heads on the active gradient tape.</returns>
    /// <remarks>
    /// Assignment, and the varifocal IoU targets and weights, use detached host values and the exact
    /// shared Hungarian solver. Loss and gradient calculations use engine operations and retain the
    /// active CPU/GPU backend. Input tensors are borrowed, never mutated or disposed. The returned
    /// scalar belongs to the active tensor/tape lifetime and must be consumed before that lifetime ends.
    /// </remarks>
    public Tensor<T> ComputeTapeLoss(Tensor<T> classLogits, Tensor<T> boxes, DetectionTrainingBatch<T> targets)
    {
        ValidateHeads(classLogits, boxes, targets);
        int batch = classLogits.Shape[0];
        int queries = classLogits.Shape[1];

        // Materialize detached host snapshots once for discrete matching, not once per pair.
        // These reads do not replace either live head in the differentiable objective below.
        var logitsData = classLogits.ToArray();
        var boxData = boxes.ToArray();
        ValidateFinitePredictions(logitsData, boxData);
        var assignments = Match(logitsData, boxData, targets, queries);

        var classification = UsesNoObjectClass
            ? SoftmaxClassification(classLogits, assignments, targets)
            : SigmoidClassification(classLogits, logitsData, boxData, assignments, targets);

        if (targets.TargetCount == 0)
        {
            // Background classification is still nonzero. Connect the box head with an exact zero derivative.
            var zeroBoxes = Engine.TensorMultiplyScalar(Engine.ReduceSum(boxes, null), NumOps.Zero);
            return Engine.TensorAdd(classification, zeroBoxes);
        }

        var matchedRows = new int[targets.TargetCount];
        var targetBoxes = new T[checked(targets.TargetCount * 4)];
        int matched = 0;
        for (int image = 0; image < batch; image++)
        {
            for (int targetIndex = 0; targetIndex < targets[image].Count; targetIndex++)
            {
                matchedRows[matched] = image * queries + assignments[image][targetIndex];
                WriteBox(targetBoxes, matched * 4, targets[image][targetIndex]);
                matched++;
            }
        }

        var flatBoxes = Engine.Reshape(boxes, new[] { checked(batch * queries), 4 });
        var matchedBoxes = CvTensorOps<T>.Select(flatBoxes, matchedRows, 0);
        var actualBoxes = new Tensor<T>(targetBoxes, new[] { matched, 4 });
        var l1Sum = Engine.ReduceSum(Engine.TensorAbs(Engine.TensorSubtract(matchedBoxes, actualBoxes)), null);
        var giouSum = Engine.ReduceSum(
            Engine.TensorGIoULoss(ToCorners(matchedBoxes), ToCorners(actualBoxes)), null);
        var weightedL1 = Engine.TensorMultiplyScalar(l1Sum, NumOps.FromDouble(_options.L1LossWeight / targets.TargetCount));
        var weightedGIoU = Engine.TensorMultiplyScalar(giouSum, NumOps.FromDouble(_options.GIoULossWeight / targets.TargetCount));
        return Engine.TensorAdd(classification, Engine.TensorAdd(weightedL1, weightedGIoU));
    }

    private Tensor<T> SoftmaxClassification(Tensor<T> classLogits, int[][] assignments, DetectionTrainingBatch<T> targets)
    {
        int batch = classLogits.Shape[0];
        int queries = classLogits.Shape[1];
        var selectedEntries = new int[checked(batch * queries)];
        var weights = new T[selectedEntries.Length];
        double denominator = 0;
        for (int image = 0; image < batch; image++)
        {
            var assignedClasses = new int[queries];
            for (int query = 0; query < queries; query++) assignedClasses[query] = _numClasses - 1;
            for (int targetIndex = 0; targetIndex < targets[image].Count; targetIndex++)
                assignedClasses[assignments[image][targetIndex]] = targets[image][targetIndex].ClassId;
            for (int query = 0; query < queries; query++)
            {
                int label = assignedClasses[query];
                double weight = label == _numClasses - 1 ? _options.NoObjectWeight : 1;
                int row = image * queries + query;
                selectedEntries[row] = row * _numClasses + label;
                weights[row] = NumOps.FromDouble(weight);
                denominator += weight;
            }
        }

        // Gather only each query's assigned log-probability. A dense one-hot product would multiply
        // the -infinity log-probability of an extreme finite unselected logit by zero and yield NaN.
        var logProbabilities = Engine.TensorLogSoftmax(classLogits, axis: 2);
        var flatLogProbabilities = Engine.Reshape(logProbabilities, new[] { checked(batch * queries * _numClasses) });
        var selected = CvTensorOps<T>.Select(flatLogProbabilities, selectedEntries, 0);
        var negativeLogLikelihood = Engine.TensorNegate(Engine.ReduceSum(
            Engine.TensorMultiply(selected, new Tensor<T>(weights, new[] { weights.Length })), null));
        // An all-background batch with no-object weight zero has nothing to classify.
        double scale = denominator > 0 ? _options.ClassLossWeight / denominator : 0;
        return Engine.TensorMultiplyScalar(negativeLogLikelihood, NumOps.FromDouble(scale));
    }

    private Tensor<T> SigmoidClassification(Tensor<T> classLogits, T[] logitsData, T[] boxData,
        int[][] assignments, DetectionTrainingBatch<T> targets)
    {
        int batch = classLogits.Shape[0];
        int queries = classLogits.Shape[1];
        int length = checked(batch * queries * _numClasses);
        var positive = new bool[length];
        var soft = new double[length];
        for (int image = 0; image < batch; image++)
        {
            for (int targetIndex = 0; targetIndex < targets[image].Count; targetIndex++)
            {
                var target = targets[image][targetIndex];
                int query = assignments[image][targetIndex];
                int index = (image * queries + query) * _numClasses + target.ClassId;
                positive[index] = true;
                soft[index] = _options.ClassificationLoss == SetPredictionClassificationLoss.VariFocal
                    ? _nms.ComputeIoU(PredictedBox(boxData, image, queries, query), TargetBox(target))
                    : 1.0;
            }
        }

        // Deformable DETR, DINO and RT-DETR sum the per-element loss over queries and classes and
        // divide by the number of target boxes (at least one).
        double scale = _options.ClassLossWeight / Math.Max(1, targets.TargetCount);
        var shape = classLogits.Shape.ToArray();
        var targetTensor = new Tensor<T>(soft.Select(value => NumOps.FromDouble(value)).ToArray(), shape);
        var complement = new Tensor<T>(soft.Select(value => NumOps.FromDouble(1 - value)).ToArray(), shape);

        // log(p) = -softplus(-x) and log(1 - p) = -softplus(x) avoid evaluating log(sigmoid(x)).
        var logProbability = Engine.TensorNegate(Engine.Softplus(Engine.TensorNegate(classLogits)));
        var logComplement = Engine.TensorNegate(Engine.Softplus(classLogits));
        var crossEntropy = Engine.TensorNegate(Engine.TensorAdd(
            Engine.TensorMultiply(targetTensor, logProbability),
            Engine.TensorMultiply(complement, logComplement)));

        Tensor<T> perElement;
        if (_options.ClassificationLoss == SetPredictionClassificationLoss.SigmoidFocal)
        {
            // FL = -alpha_t (1 - p_t)^gamma log(p_t) with binary targets (Lin et al. 2017); the
            // modulating factor stays on the tape, as in the reference sigmoid_focal_loss.
            double alpha = _options.FocalAlpha;
            var alphaT = new Tensor<T>(positive.Select(isPositive => NumOps.FromDouble(isPositive ? alpha : 1 - alpha)).ToArray(), shape);
            perElement = Engine.TensorMultiply(alphaT, crossEntropy);
            if (_options.FocalGamma > 0)
            {
                // 1 - p_t = p + t - 2 p t for a binary target t.
                var signs = new Tensor<T>(positive.Select(isPositive => NumOps.FromDouble(isPositive ? -1 : 1)).ToArray(), shape);
                var oneMinusPt = Engine.TensorAdd(Engine.TensorMultiply(Engine.Sigmoid(classLogits), signs), targetTensor);
                perElement = Engine.TensorMultiply(perElement,
                    Engine.TensorPower(oneMinusPt, NumOps.FromDouble(_options.FocalGamma)));
            }
        }
        else
        {
            // VFL(p, q) = -q (q log p + (1 - q) log(1 - p)) for the matched class and
            // -alpha p^gamma log(1 - p) otherwise (Zhang et al. 2021). The weights use the detached
            // score, as the RT-DETR and VarifocalNet reference implementations do.
            var weights = new T[length];
            for (int index = 0; index < length; index++)
            {
                double probability = Logistic(NumOps.ToDouble(logitsData[index]));
                weights[index] = NumOps.FromDouble(positive[index]
                    ? soft[index]
                    : _options.FocalAlpha * Math.Pow(probability, _options.FocalGamma));
            }
            perElement = Engine.TensorMultiply(new Tensor<T>(weights, shape), crossEntropy);
        }

        return Engine.TensorMultiplyScalar(Engine.ReduceSum(perElement, null), NumOps.FromDouble(scale));
    }

    private Tensor<T> ComputeStructuredLoss(Tensor<T> predicted, Tensor<T> targets)
    {
        int batch = predicted.Shape[0];
        int queries = predicted.Shape[1];
        var typedTargets = DetectionTrainingBatch<T>.FromPaddedDetr(targets);
        // Validate before allocating loss intermediates, including before slicing the predictions.
        typedTargets.ValidateForModel(batch, ForegroundClasses, queries);
        var logits = Engine.TensorSlice(predicted, new[] { 0, 0, 0 }, new[] { batch, queries, _numClasses });
        var boxes = Engine.TensorSlice(predicted, new[] { 0, 0, _numClasses }, new[] { batch, queries, 4 });
        return ComputeTapeLoss(logits, boxes, typedTargets);
    }

    private int[][] Match(T[] logits, T[] boxes, DetectionTrainingBatch<T> targets, int queries)
    {
        var result = new int[targets.ImageCount][];
        var solver = new LinearAssignmentSolver<double>();
        for (int image = 0; image < targets.ImageCount; image++)
        {
            var imageTargets = targets[image];
            if (imageTargets.Count == 0)
            {
                result[image] = Array.Empty<int>();
                continue;
            }

            var classCosts = UsesNoObjectClass
                ? SoftmaxClassCosts(logits, image, queries)
                : FocalClassCosts(logits, image, queries);
            var predictedBoxes = new BoundingBox<T>[queries];
            for (int query = 0; query < queries; query++)
                predictedBoxes[query] = PredictedBox(boxes, image, queries, query);
            var costs = new Matrix<double>(imageTargets.Count, queries);
            for (int targetIndex = 0; targetIndex < imageTargets.Count; targetIndex++)
            {
                var target = imageTargets[targetIndex];
                var actual = TargetBox(target);
                for (int query = 0; query < queries; query++)
                {
                    int offset = (image * queries + query) * 4;
                    double l1 = Math.Abs(NumOps.ToDouble(boxes[offset]) - NumOps.ToDouble(target.CenterX))
                        + Math.Abs(NumOps.ToDouble(boxes[offset + 1]) - NumOps.ToDouble(target.CenterY))
                        + Math.Abs(NumOps.ToDouble(boxes[offset + 2]) - NumOps.ToDouble(target.Width))
                        + Math.Abs(NumOps.ToDouble(boxes[offset + 3]) - NumOps.ToDouble(target.Height));
                    costs[targetIndex, query] = _options.ClassCostWeight * classCosts[query * _numClasses + target.ClassId]
                        + _options.L1CostWeight * l1 - _options.GIoUCostWeight * _nms.ComputeGIoU(predictedBoxes[query], actual);
                }
            }
            var assignment = solver.Solve(costs);
            var rows = new int[imageTargets.Count];
            for (int targetIndex = 0; targetIndex < rows.Length; targetIndex++)
            {
                int query = assignment[targetIndex];
                if (query < 0 || query >= queries)
                    throw new InvalidOperationException("The assignment solver did not match every validated target.");
                rows[targetIndex] = query;
            }
            result[image] = rows;
        }
        return result;
    }

    /// <summary>DETR's class cost: the negative softmax probability of the target class.</summary>
    private double[] SoftmaxClassCosts(T[] logits, int image, int queries)
    {
        var costs = new double[checked(queries * _numClasses)];
        for (int query = 0; query < queries; query++)
        {
            int offset = (image * queries + query) * _numClasses;
            double maximum = double.NegativeInfinity;
            for (int label = 0; label < _numClasses; label++)
                maximum = Math.Max(maximum, NumOps.ToDouble(logits[offset + label]));
            double sum = 0;
            for (int label = 0; label < _numClasses; label++)
            {
                double value = Math.Exp(NumOps.ToDouble(logits[offset + label]) - maximum);
                costs[query * _numClasses + label] = value;
                sum += value;
            }
            for (int label = 0; label < _numClasses; label++)
                costs[query * _numClasses + label] = -costs[query * _numClasses + label] / sum;
        }
        return costs;
    }

    /// <summary>
    /// Deformable DETR's focal class cost: alpha (1 - p)^gamma (-log p) - (1 - alpha) p^gamma (-log(1 - p)).
    /// </summary>
    private double[] FocalClassCosts(T[] logits, int image, int queries)
    {
        double alpha = _options.MatchingFocalAlpha;
        double gamma = _options.MatchingFocalGamma;
        var costs = new double[checked(queries * _numClasses)];
        for (int index = 0; index < costs.Length; index++)
        {
            double logit = NumOps.ToDouble(logits[image * queries * _numClasses + index]);
            double probability = Logistic(logit);
            double positive = alpha * Math.Pow(1 - probability, gamma) * Softplus(-logit);
            double negative = (1 - alpha) * Math.Pow(probability, gamma) * Softplus(logit);
            costs[index] = positive - negative;
        }
        return costs;
    }

    private static double Logistic(double x) => x >= 0 ? 1 / (1 + Math.Exp(-x)) : Math.Exp(x) / (1 + Math.Exp(x));

    private static double Softplus(double x) => x > 0 ? x + Math.Log(1 + Math.Exp(-x)) : Math.Log(1 + Math.Exp(x));

    private static BoundingBox<T> PredictedBox(T[] boxes, int image, int queries, int query)
    {
        int offset = (image * queries + query) * 4;
        return new BoundingBox<T>(boxes[offset], boxes[offset + 1], boxes[offset + 2], boxes[offset + 3], BoundingBoxFormat.CXCYWH);
    }

    private static BoundingBox<T> TargetBox(DetectionTrainingTarget<T> target) =>
        new(target.CenterX, target.CenterY, target.Width, target.Height, BoundingBoxFormat.CXCYWH);

    private Tensor<T> ToCorners(Tensor<T> boxes)
    {
        int count = boxes.Shape[0];
        var centers = Engine.TensorSlice(boxes, new[] { 0, 0 }, new[] { count, 2 });
        var halfExtents = Engine.TensorMultiplyScalar(
            Engine.TensorSlice(boxes, new[] { 0, 2 }, new[] { count, 2 }), NumOps.FromDouble(0.5));
        return Engine.TensorConcatenate(
            new[] { Engine.TensorSubtract(centers, halfExtents), Engine.TensorAdd(centers, halfExtents) }, 1);
    }

    private void ValidateHeads(Tensor<T> logits, Tensor<T> boxes, DetectionTrainingBatch<T> targets)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (boxes is null) throw new ArgumentNullException(nameof(boxes));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (logits.Rank != 3 || logits.Shape[0] <= 0 || logits.Shape[1] <= 0 || logits.Shape[2] != _numClasses)
            throw new ArgumentException("Class logits must be [positive batch, positive queries, configured classes].", nameof(logits));
        if (boxes.Rank != 3 || boxes.Shape[0] != logits.Shape[0] || boxes.Shape[1] != logits.Shape[1] || boxes.Shape[2] != 4)
            throw new ArgumentException("Boxes must match the logit batch and query dimensions, with four cxcywh coordinates.", nameof(boxes));
        targets.ValidateForModel(logits.Shape[0], ForegroundClasses, logits.Shape[1]);
    }

    private void ValidateFinitePredictions(T[] logits, T[] boxes)
    {
        foreach (T value in logits)
        {
            double number = NumOps.ToDouble(value);
            if (double.IsNaN(number) || double.IsInfinity(number))
                throw new ArgumentException("Class logits must be finite.", nameof(logits));
        }
        foreach (T value in boxes)
        {
            double number = NumOps.ToDouble(value);
            if (double.IsNaN(number) || double.IsInfinity(number) || number < 0 || number > 1)
                throw new ArgumentException("Predicted sigmoid cxcywh coordinates must be finite and in [0, 1].", nameof(boxes));
        }
    }

    private bool IsStructuredPrediction(Tensor<T> predicted) =>
        predicted.Rank == 3 && predicted.Shape[2] == _numClasses + 4;

    private bool HasStructuredLayout(Tensor<T> predicted, Tensor<T> target)
    {
        if (!IsStructuredPrediction(predicted) || predicted.Shape[0] <= 0 || predicted.Shape[1] <= 0)
            return false;
        return target.Rank == 3 && target.Shape[0] == predicted.Shape[0] && target.Shape[2] >= 5;
    }

    private void ValidateStructuredLayout(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (!HasStructuredLayout(predicted, target))
            throw new ArgumentException(
                $"DETR loss requires predicted [batch, queries, {_numClasses + 4}] and target [the same batch, objects, at least 5]. " +
                $"Received predicted [{string.Join(", ", predicted.Shape)}] and target [{string.Join(", ", target.Shape)}].",
                nameof(target));
    }

    private static void WriteBox(T[] destination, int offset, DetectionTrainingTarget<T> target)
    {
        destination[offset] = target.CenterX;
        destination[offset + 1] = target.CenterY;
        destination[offset + 2] = target.Width;
        destination[offset + 3] = target.Height;
    }

    private static DetrSetLossOptions SoftmaxOptions(double classWeight, double boxL1Weight, double boxGIoUWeight)
    {
        ValidateWeight(classWeight, nameof(classWeight));
        ValidateWeight(boxL1Weight, nameof(boxL1Weight));
        ValidateWeight(boxGIoUWeight, nameof(boxGIoUWeight));
        return new DetrSetLossOptions
        {
            ClassificationLoss = SetPredictionClassificationLoss.SoftmaxCrossEntropy,
            ClassLossWeight = classWeight,
            ClassCostWeight = classWeight,
            L1LossWeight = boxL1Weight,
            L1CostWeight = boxL1Weight,
            GIoULossWeight = boxGIoUWeight,
            GIoUCostWeight = boxGIoUWeight
        };
    }

    private static void ValidateWeight(double weight, string parameterName)
    {
        if (double.IsNaN(weight) || double.IsInfinity(weight) || weight < 0)
            throw new ArgumentOutOfRangeException(parameterName, "Loss weights must be finite and nonnegative.");
    }
}
