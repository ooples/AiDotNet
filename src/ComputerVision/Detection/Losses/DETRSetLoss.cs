using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.LossFunctions;
using AiDotNet.Solvers.Assignment;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>DETR set prediction loss with exact Hungarian assignment.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Foreground queries are assigned using negative class probability, center-format L1 distance and
/// negative GIoU. Every query receives cross-entropy supervision, including unmatched queries and
/// empty images. The no-object class has weight 0.1. Classification uses a weighted mean; matched
/// L1 and GIoU sums are normalized by the total foreground target count across the local batch.
/// </para>
/// <para>
/// Reference: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020.
/// Only the supplied final prediction heads are supervised; intermediate decoder losses are not
/// fabricated. More targets than queries in an image are rejected rather than silently dropped.
/// </para>
/// </remarks>
public class DETRSetLoss<T> : LossFunctionBase<T>
{
    private const double NoObjectWeight = 0.1;
    private readonly NMS<T> _nms = new();
    private readonly double _classWeight;
    private readonly double _boxL1Weight;
    private readonly double _boxGIoUWeight;
    private readonly int _numClasses;

    /// <summary>Creates a DETR objective with the standard final-head loss weights.</summary>
    /// <param name="numClasses">Number of output classes, including the final no-object class.</param>
    /// <param name="classWeight">Nonnegative classification and matching cost weight.</param>
    /// <param name="boxL1Weight">Nonnegative center-format L1 loss and matching cost weight.</param>
    /// <param name="boxGIoUWeight">Nonnegative GIoU loss and matching cost weight.</param>
    public DETRSetLoss(int numClasses = 91, double classWeight = 1.0,
        double boxL1Weight = 5.0, double boxGIoUWeight = 2.0)
    {
        if (numClasses < 2) throw new ArgumentOutOfRangeException(nameof(numClasses));
        ValidateWeight(classWeight, nameof(classWeight));
        ValidateWeight(boxL1Weight, nameof(boxL1Weight));
        ValidateWeight(boxGIoUWeight, nameof(boxGIoUWeight));
        _numClasses = numClasses;
        _classWeight = classWeight;
        _boxL1Weight = boxL1Weight;
        _boxGIoUWeight = boxGIoUWeight;
    }

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

    /// <summary>Builds differentiable final-head CE, L1 and GIoU losses after discrete assignment.</summary>
    /// <param name="classLogits">Raw class logits [batch, queries, classes including no-object].</param>
    /// <param name="boxes">Sigmoid box predictions [batch, queries, 4] in normalized cxcywh.</param>
    /// <param name="targets">Immutable, unpadded foreground targets; empty images are valid.</param>
    /// <returns>A scalar connected to both prediction heads on the active gradient tape.</returns>
    /// <remarks>
    /// Assignment alone uses detached host values and the exact shared Hungarian solver. Loss and
    /// gradient calculations use engine operations and retain the active CPU/GPU backend. Input
    /// tensors are borrowed, never mutated or disposed. The returned scalar belongs to the active
    /// tensor/tape lifetime and must be consumed before that lifetime ends.
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
        var weightedTargets = new Tensor<T>(classLogits.Shape.ToArray());
        var matchedRows = new int[targets.TargetCount];
        var targetBoxes = new T[checked(targets.TargetCount * 4)];
        double classificationDenominator = 0;
        int matched = 0;

        for (int image = 0; image < batch; image++)
        {
            var assignedClasses = new int[queries];
            for (int query = 0; query < queries; query++) assignedClasses[query] = _numClasses - 1;
            for (int targetIndex = 0; targetIndex < targets[image].Count; targetIndex++)
            {
                int query = assignments[image][targetIndex];
                var target = targets[image][targetIndex];
                assignedClasses[query] = target.ClassId;
                matchedRows[matched] = image * queries + query;
                WriteBox(targetBoxes, matched * 4, target);
                matched++;
            }
            for (int query = 0; query < queries; query++)
            {
                int label = assignedClasses[query];
                double weight = label == _numClasses - 1 ? NoObjectWeight : 1;
                weightedTargets[image, query, label] = NumOps.FromDouble(weight);
                classificationDenominator += weight;
            }
        }

        var logProbabilities = Engine.TensorLogSoftmax(classLogits, axis: 2);
        var negativeLogLikelihood = Engine.TensorNegate(
            Engine.ReduceSum(Engine.TensorMultiply(logProbabilities, weightedTargets), null));
        var classification = Engine.TensorMultiplyScalar(negativeLogLikelihood,
            NumOps.FromDouble(_classWeight / classificationDenominator));

        if (matched == 0)
        {
            // Background CE is still nonzero. Connect the box head with an exact zero derivative.
            var zeroBoxes = Engine.TensorMultiplyScalar(Engine.ReduceSum(boxes, null), NumOps.Zero);
            return Engine.TensorAdd(classification, zeroBoxes);
        }

        var flatBoxes = Engine.Reshape(boxes, new[] { checked(batch * queries), 4 });
        var matchedBoxes = CvTensorOps<T>.Select(flatBoxes, matchedRows, 0);
        var actualBoxes = new Tensor<T>(targetBoxes, new[] { matched, 4 });
        var l1Sum = Engine.ReduceSum(Engine.TensorAbs(Engine.TensorSubtract(matchedBoxes, actualBoxes)), null);
        var giouSum = Engine.ReduceSum(
            Engine.TensorGIoULoss(ToCorners(matchedBoxes), ToCorners(actualBoxes)), null);
        var weightedL1 = Engine.TensorMultiplyScalar(l1Sum, NumOps.FromDouble(_boxL1Weight / targets.TargetCount));
        var weightedGIoU = Engine.TensorMultiplyScalar(giouSum, NumOps.FromDouble(_boxGIoUWeight / targets.TargetCount));
        return Engine.TensorAdd(classification, Engine.TensorAdd(weightedL1, weightedGIoU));
    }

    private Tensor<T> ComputeStructuredLoss(Tensor<T> predicted, Tensor<T> targets)
    {
        int batch = predicted.Shape[0];
        int queries = predicted.Shape[1];
        var typedTargets = DetectionTrainingBatch<T>.FromPaddedDetr(targets);
        // Validate before allocating loss intermediates, including before slicing the predictions.
        typedTargets.ValidateForModel(batch, _numClasses - 1, queries);
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

            var probabilities = ClassProbabilities(logits, image, queries);
            var predictedBoxes = new BoundingBox<T>[queries];
            for (int query = 0; query < queries; query++)
            {
                int offset = (image * queries + query) * 4;
                predictedBoxes[query] = new BoundingBox<T>(boxes[offset], boxes[offset + 1],
                    boxes[offset + 2], boxes[offset + 3], BoundingBoxFormat.CXCYWH);
            }
            var costs = new Matrix<double>(imageTargets.Count, queries);
            for (int targetIndex = 0; targetIndex < imageTargets.Count; targetIndex++)
            {
                var target = imageTargets[targetIndex];
                var actual = new BoundingBox<T>(target.CenterX, target.CenterY, target.Width, target.Height, BoundingBoxFormat.CXCYWH);
                for (int query = 0; query < queries; query++)
                {
                    int offset = (image * queries + query) * 4;
                    double l1 = Math.Abs(NumOps.ToDouble(boxes[offset]) - NumOps.ToDouble(target.CenterX))
                        + Math.Abs(NumOps.ToDouble(boxes[offset + 1]) - NumOps.ToDouble(target.CenterY))
                        + Math.Abs(NumOps.ToDouble(boxes[offset + 2]) - NumOps.ToDouble(target.Width))
                        + Math.Abs(NumOps.ToDouble(boxes[offset + 3]) - NumOps.ToDouble(target.Height));
                    costs[targetIndex, query] = -_classWeight * probabilities[query * _numClasses + target.ClassId]
                        + _boxL1Weight * l1 - _boxGIoUWeight * _nms.ComputeGIoU(predictedBoxes[query], actual);
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

    private double[] ClassProbabilities(T[] logits, int image, int queries)
    {
        var probabilities = new double[checked(queries * _numClasses)];
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
                probabilities[query * _numClasses + label] = value;
                sum += value;
            }
            for (int label = 0; label < _numClasses; label++)
                probabilities[query * _numClasses + label] /= sum;
        }
        return probabilities;
    }

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
        targets.ValidateForModel(logits.Shape[0], _numClasses - 1, logits.Shape[1]);
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

    private static void ValidateWeight(double weight, string parameterName)
    {
        if (double.IsNaN(weight) || double.IsInfinity(weight) || weight < 0)
            throw new ArgumentOutOfRangeException(parameterName, "Loss weights must be finite and nonnegative.");
    }
}
