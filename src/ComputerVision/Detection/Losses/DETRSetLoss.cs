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
        var difference = Engine.TensorAbs(Engine.TensorSubtract(
            new Tensor<T>(new[] { predicted.Length }, predicted), new Tensor<T>(new[] { actual.Length }, actual)));
        return CvTensorOps<T>.Mean(difference);
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
        return ComputeTapeLossCore(classLogits, boxes, targets, fixedAssignments: null);
    }

    /// <summary>
    /// The set loss for a FIXED assignment: target <c>j</c> of image <c>i</c> is matched to query
    /// <c>assignments[i][j]</c>, and every other query is background. DINO's contrastive denoising uses it:
    /// each positive denoising query is pinned to the target it was noised from, and negatives and padding
    /// are pushed to background.
    /// </summary>
    public Tensor<T> ComputeTapeLoss(Tensor<T> classLogits, Tensor<T> boxes, DetectionTrainingBatch<T> targets, int[][] assignments)
    {
        ValidateHeads(classLogits, boxes, targets);
        if (assignments is null) throw new ArgumentNullException(nameof(assignments));
        int queries = classLogits.Shape[1];
        if (assignments.Length != classLogits.Shape[0])
            throw new ArgumentException("One assignment row per image is required.", nameof(assignments));
        for (int image = 0; image < assignments.Length; image++)
        {
            if (assignments[image] is null || assignments[image].Length != targets[image].Count)
                throw new ArgumentException($"Image {image} needs one query per target.", nameof(assignments));
            if (assignments[image].Any(query => query < 0 || query >= queries) || assignments[image].Distinct().Count() != assignments[image].Length)
                throw new ArgumentException($"Image {image} assigns a query out of range or twice.", nameof(assignments));
        }

        return ComputeTapeLossCore(classLogits, boxes, targets, assignments);
    }

    private Tensor<T> ComputeTapeLossCore(Tensor<T> classLogits, Tensor<T> boxes, DetectionTrainingBatch<T> targets,
        int[][]? fixedAssignments)
    {
        int batch = classLogits.Shape[0];
        int queries = classLogits.Shape[1];
        var classification = UsesNoObjectClass
            ? SoftmaxClassification(classLogits, boxes, fixedAssignments, targets)
            : SigmoidClassification(classLogits, boxes, fixedAssignments, targets);

        if (targets.TargetCount == 0)
        {
            // Background classification is still nonzero. Connect the box head with an exact zero derivative.
            var zeroBoxes = Engine.TensorMultiplyScalar(Engine.ReduceSum(boxes, null), NumOps.Zero);
            return Engine.TensorAdd(classification, zeroBoxes);
        }

        int numClasses = _numClasses;
        double classCostWeight = _options.ClassCostWeight;
        double l1CostWeight = _options.L1CostWeight;
        double giouCostWeight = _options.GIoUCostWeight;
        double matchingAlpha = _options.MatchingFocalAlpha;
        double matchingGamma = _options.MatchingFocalGamma;
        bool usesNoObject = UsesNoObjectClass;
        int totalTargets = targets.TargetCount;
        var packedTargets = PackTargets(targets, fixedAssignments);
        var matchedRowIndices = AiDotNet.Training.CompiledStepHostValue<T>.Compute(
            "DetrMatchedQueries",
            new[] { classLogits, boxes, packedTargets },
            new[] { totalTargets },
            (inputs, outputRows) => FillMatchedQueries(
                inputs[0], inputs[1], inputs[2], batch, queries, numClasses, usesNoObject,
                classCostWeight, l1CostWeight, giouCostWeight, matchingAlpha, matchingGamma, outputRows));

        var targetBoxes = new T[checked(targets.TargetCount * 4)];
        int matched = 0;
        for (int image = 0; image < batch; image++)
        {
            for (int targetIndex = 0; targetIndex < targets[image].Count; targetIndex++)
            {
                WriteBox(targetBoxes, matched * 4, targets[image][targetIndex]);
                matched++;
            }
        }

        var flatBoxes = Engine.Reshape(boxes, new[] { checked(batch * queries), 4 });
        var matchedBoxes = Engine.TensorEmbeddingLookupFromFloatIndices(flatBoxes, matchedRowIndices);
        var actualBoxes = new Tensor<T>(targetBoxes, new[] { matched, 4 });
        var l1Sum = Engine.ReduceSum(Engine.TensorAbs(Engine.TensorSubtract(matchedBoxes, actualBoxes)), null);
        var giouSum = Engine.ReduceSum(
            Engine.TensorGIoULoss(ToCorners(matchedBoxes), ToCorners(actualBoxes)), null);
        var weightedL1 = Engine.TensorMultiplyScalar(l1Sum, NumOps.FromDouble(_options.L1LossWeight / targets.TargetCount));
        var weightedGIoU = Engine.TensorMultiplyScalar(giouSum, NumOps.FromDouble(_options.GIoULossWeight / targets.TargetCount));
        return Engine.TensorAdd(classification, Engine.TensorAdd(weightedL1, weightedGIoU));
    }

    private Tensor<T> SoftmaxClassification(Tensor<T> classLogits, Tensor<T> boxes, int[][]? fixedAssignments, DetectionTrainingBatch<T> targets)
    {
        int batch = classLogits.Shape[0];
        int queries = classLogits.Shape[1];
        int rowCount = checked(batch * queries);
        int numClasses = _numClasses;
        double noObjectWeight = _options.NoObjectWeight;
        double classLossWeight = _options.ClassLossWeight;
        double classCostWeight = _options.ClassCostWeight;
        double l1CostWeight = _options.L1CostWeight;
        double giouCostWeight = _options.GIoUCostWeight;
        double matchingAlpha = _options.MatchingFocalAlpha;
        double matchingGamma = _options.MatchingFocalGamma;
        bool usesNoObject = UsesNoObjectClass;
        var packedTargets = PackTargets(targets, fixedAssignments);
        var packed = AiDotNet.Training.CompiledStepHostValue<T>.Compute(
            "DetrSoftmaxSupervision",
            new[] { classLogits, boxes, packedTargets },
            new[] { 2, rowCount },
            (inputs, outputPacked) => FillSoftmaxSupervision(
                inputs[0], inputs[1], inputs[2], batch, queries, numClasses, usesNoObject,
                noObjectWeight, classLossWeight, classCostWeight, l1CostWeight, giouCostWeight,
                matchingAlpha, matchingGamma, outputPacked));

        var selectedEntries = Engine.Reshape(Engine.TensorSlice(packed, new[] { 0, 0 }, new[] { 1, rowCount }), new[] { rowCount });
        var scaledWeights = Engine.Reshape(Engine.TensorSlice(packed, new[] { 1, 0 }, new[] { 1, rowCount }), new[] { rowCount });

        // Gather only each query's assigned log-probability. A dense one-hot product would multiply
        // the -infinity log-probability of an extreme finite unselected logit by zero and yield NaN.
        var logProbabilities = Engine.TensorLogSoftmax(classLogits, axis: 2);
        var flatLogProbabilities = Engine.Reshape(logProbabilities, new[] { checked(rowCount * _numClasses), 1 });
        var selected = Engine.Reshape(Engine.TensorEmbeddingLookupFromFloatIndices(flatLogProbabilities, selectedEntries), new[] { rowCount });
        return Engine.TensorNegate(Engine.ReduceSum(Engine.TensorMultiply(selected, scaledWeights), null));
    }

    private Tensor<T> SigmoidClassification(Tensor<T> classLogits, Tensor<T> boxes,
        int[][]? fixedAssignments, DetectionTrainingBatch<T> targets)
    {
        int batch = classLogits.Shape[0];
        int queries = classLogits.Shape[1];
        int length = checked(batch * queries * _numClasses);
        var shape = classLogits.Shape.ToArray();

        int numClasses = _numClasses;
        bool variFocal = _options.ClassificationLoss == SetPredictionClassificationLoss.VariFocal;
        double classCostWeight = _options.ClassCostWeight;
        double l1CostWeight = _options.L1CostWeight;
        double giouCostWeight = _options.GIoUCostWeight;
        double matchingAlpha = _options.MatchingFocalAlpha;
        double matchingGamma = _options.MatchingFocalGamma;
        bool usesNoObject = UsesNoObjectClass;
        var packedTargets = PackTargets(targets, fixedAssignments);
        var packed = AiDotNet.Training.CompiledStepHostValue<T>.Compute(
            "DetrSigmoidSupervision",
            new[] { classLogits, boxes, packedTargets },
            new[] { 2, length },
            (inputs, outputPacked) => FillSigmoidSupervision(
                inputs[0], inputs[1], inputs[2], batch, queries, numClasses, usesNoObject, variFocal,
                classCostWeight, l1CostWeight, giouCostWeight, matchingAlpha, matchingGamma, outputPacked));

        // Deformable DETR, DINO and RT-DETR sum the per-element loss over queries and classes and
        // divide by the number of target boxes (at least one).
        double scale = _options.ClassLossWeight / Math.Max(1, targets.TargetCount);
        var isPositive = Engine.Reshape(Engine.TensorSlice(packed, new[] { 0, 0 }, new[] { 1, length }), shape);
        var targetTensor = Engine.Reshape(Engine.TensorSlice(packed, new[] { 1, 0 }, new[] { 1, length }), shape);
        var complement = Engine.ScalarMinusTensor(NumOps.One, targetTensor);

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
            var alphaT = Engine.TensorWhere(isPositive, CvTensorOps<T>.Fill(shape, alpha), CvTensorOps<T>.Fill(shape, 1 - alpha));
            perElement = Engine.TensorMultiply(alphaT, crossEntropy);
            if (_options.FocalGamma > 0)
            {
                // 1 - p_t = p + t - 2 p t for a binary target t.
                var signs = Engine.TensorWhere(isPositive, CvTensorOps<T>.Fill(shape, -1.0), CvTensorOps<T>.Fill(shape, 1.0));
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
            var detachedProbability = Engine.Sigmoid(Engine.StopGradient(classLogits));
            var negativeWeight = Engine.TensorMultiplyScalar(
                Engine.TensorPower(detachedProbability, NumOps.FromDouble(_options.FocalGamma)), NumOps.FromDouble(_options.FocalAlpha));
            var weights = Engine.TensorWhere(isPositive, targetTensor, negativeWeight);
            perElement = Engine.TensorMultiply(weights, crossEntropy);
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

    private static Tensor<T> PackTargets(DetectionTrainingBatch<T> targets, int[][]? fixedAssignments)
    {
        bool hasFixed = fixedAssignments is not null;
        int totalObjects = targets.TargetCount;
        var rawCoords = new T[totalObjects * 4];
        int coordWrite = 0;
        for (int image = 0; image < targets.ImageCount; image++)
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
        var values = new List<double> { targets.ImageCount, hasFixed ? 1.0 : 0.0 };
        for (int image = 0; image < targets.ImageCount; image++)
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
                values.Add(fixedAssignments is not null ? fixedAssignments[image][t] : -1);
            }
        }

        return new Tensor<double>(values.ToArray(), new[] { values.Count }).Cast<T>();
    }

    private static void UnpackTargets(
        Tensor<T> packed,
        out (int ClassId, double Cx, double Cy, double W, double H)[][] targets,
        out int[][]? fixedAssignments)
    {
        var span = packed.Contiguous().Cast<double>().ToArray();
        int cursor = 0;
        int imageCount = (int)Math.Round(span[cursor++]);
        bool hasFixed = span[cursor++] > 0.5;
        targets = new (int ClassId, double Cx, double Cy, double W, double H)[imageCount][];
        fixedAssignments = hasFixed ? new int[imageCount][] : null;
        for (int image = 0; image < imageCount; image++)
        {
            int count = (int)Math.Round(span[cursor++]);
            targets[image] = new (int ClassId, double Cx, double Cy, double W, double H)[count];
            if (fixedAssignments is not null) fixedAssignments[image] = new int[count];
            for (int t = 0; t < count; t++)
            {
                int classId = (int)Math.Round(span[cursor++]);
                double cx = span[cursor++];
                double cy = span[cursor++];
                double w = span[cursor++];
                double h = span[cursor++];
                int fixedQuery = (int)Math.Round(span[cursor++]);
                targets[image][t] = (classId, cx, cy, w, h);
                if (fixedAssignments is not null) fixedAssignments[image][t] = fixedQuery;
            }
        }
    }

    private static void FillMatchedQueries(
        Tensor<T> classLogits, Tensor<T> boxes, Tensor<T> packedTargets,
        int batch, int queries, int numClasses, bool usesNoObject,
        double classCostWeight, double l1CostWeight, double giouCostWeight,
        double matchingAlpha, double matchingGamma, Tensor<T> outputRows)
    {
        var logitsData = classLogits.Contiguous().Cast<double>().ToArray();
        var boxData = boxes.Contiguous().Cast<double>().ToArray();
        ValidateFinitePredictionsStatic(logitsData, boxData);
        UnpackTargets(packedTargets, out var targets, out var fixedAssignments);
        var assignments = fixedAssignments ?? MatchStatic(
            logitsData, boxData, targets, queries, numClasses, usesNoObject,
            classCostWeight, l1CostWeight, giouCostWeight, matchingAlpha, matchingGamma);
        var rows = new double[outputRows.Length];
        int idx = 0;
        for (int image = 0; image < batch; image++)
        {
            for (int targetIndex = 0; targetIndex < targets[image].Length; targetIndex++)
                rows[idx++] = image * queries + assignments[image][targetIndex];
        }

        var typed = new Tensor<double>(outputRows._shape, new Vector<double>(rows)).Cast<T>();
        typed.AsSpan().CopyTo(outputRows.AsWritableSpan());
    }

    private static void FillSoftmaxSupervision(
        Tensor<T> classLogits, Tensor<T> boxes, Tensor<T> packedTargets,
        int batch, int queries, int numClasses, bool usesNoObject,
        double noObjectWeight, double classLossWeight,
        double classCostWeight, double l1CostWeight, double giouCostWeight,
        double matchingAlpha, double matchingGamma, Tensor<T> outputPacked)
    {
        int rowCount = checked(batch * queries);
        var logitsData = classLogits.Contiguous().Cast<double>().ToArray();
        var boxData = boxes.Contiguous().Cast<double>().ToArray();
        ValidateFinitePredictionsStatic(logitsData, boxData);
        UnpackTargets(packedTargets, out var targets, out var fixedAssignments);
        var assignments = fixedAssignments ?? MatchStatic(
            logitsData, boxData, targets, queries, numClasses, usesNoObject,
            classCostWeight, l1CostWeight, giouCostWeight, matchingAlpha, matchingGamma);
        var buffer = new double[2 * rowCount];
        double denominator = 0;
        for (int image = 0; image < batch; image++)
        {
            var assignedClasses = new int[queries];
            for (int query = 0; query < queries; query++) assignedClasses[query] = numClasses - 1;
            for (int targetIndex = 0; targetIndex < targets[image].Length; targetIndex++)
                assignedClasses[assignments[image][targetIndex]] = targets[image][targetIndex].ClassId;
            for (int query = 0; query < queries; query++)
            {
                int label = assignedClasses[query];
                double weight = label == numClasses - 1 ? noObjectWeight : 1.0;
                int row = image * queries + query;
                buffer[row] = row * numClasses + label;
                buffer[rowCount + row] = weight;
                denominator += weight;
            }
        }

        double invScale = denominator > 0 ? classLossWeight / denominator : 0.0;
        for (int row = 0; row < rowCount; row++)
            buffer[rowCount + row] *= invScale;

        var typed = new Tensor<double>(outputPacked._shape, new Vector<double>(buffer)).Cast<T>();
        typed.AsSpan().CopyTo(outputPacked.AsWritableSpan());
    }

    private static void FillSigmoidSupervision(
        Tensor<T> classLogits, Tensor<T> boxes, Tensor<T> packedTargets,
        int batch, int queries, int numClasses, bool usesNoObject, bool variFocal,
        double classCostWeight, double l1CostWeight, double giouCostWeight,
        double matchingAlpha, double matchingGamma, Tensor<T> outputPacked)
    {
        int length = checked(batch * queries * numClasses);
        var logitsData = classLogits.Contiguous().Cast<double>().ToArray();
        var boxData = boxes.Contiguous().Cast<double>().ToArray();
        ValidateFinitePredictionsStatic(logitsData, boxData);
        UnpackTargets(packedTargets, out var targets, out var fixedAssignments);
        var assignments = fixedAssignments ?? MatchStatic(
            logitsData, boxData, targets, queries, numClasses, usesNoObject,
            classCostWeight, l1CostWeight, giouCostWeight, matchingAlpha, matchingGamma);

        var entries = new List<int>();
        var matchedQueries = new List<int>();
        var matchedTargetCoords = new List<double>();
        for (int image = 0; image < batch; image++)
        {
            for (int targetIndex = 0; targetIndex < targets[image].Length; targetIndex++)
            {
                var target = targets[image][targetIndex];
                int query = assignments[image][targetIndex];
                entries.Add((image * queries + query) * numClasses + target.ClassId);
                matchedQueries.Add(image * queries + query);
                matchedTargetCoords.Add(target.Cx);
                matchedTargetCoords.Add(target.Cy);
                matchedTargetCoords.Add(target.W);
                matchedTargetCoords.Add(target.H);
            }
        }

        var buffer = new double[2 * length];
        if (entries.Count > 0)
        {
            double[] softVals;
            if (variFocal)
            {
                using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
                var predRows = CvTensorOps<double>.Select(new Tensor<double>(boxData, new[] { boxData.Length / 4, 4 }), matchedQueries.ToArray(), 0);
                var tgtRows = new Tensor<double>(matchedTargetCoords.ToArray(), new[] { entries.Count, 4 });
                softVals = BoxOverlap<double>.IoU(CornersOf<double>(predRows), CornersOf<double>(tgtRows)).ToArray();
            }
            else
            {
                softVals = Enumerable.Repeat(1.0, entries.Count).ToArray();
            }

            for (int m = 0; m < entries.Count; m++)
            {
                int idx = entries[m];
                buffer[idx] = 1.0;
                buffer[length + idx] = softVals[m];
            }
        }

        var typed = new Tensor<double>(outputPacked._shape, new Vector<double>(buffer)).Cast<T>();
        typed.AsSpan().CopyTo(outputPacked.AsWritableSpan());
    }

    private static int[][] MatchStatic(
        double[] logits, double[] boxes,
        (int ClassId, double Cx, double Cy, double W, double H)[][] targets, int queries,
        int numClasses, bool usesNoObject,
        double classCostWeight, double l1CostWeight, double giouCostWeight,
        double matchingAlpha, double matchingGamma)
    {
        var result = new int[targets.Length][];
        var solver = new LinearAssignmentSolver<double>();
        var nms = new NMS<double>();
        for (int image = 0; image < targets.Length; image++)
        {
            var imageTargets = targets[image];
            if (imageTargets.Length == 0)
            {
                result[image] = Array.Empty<int>();
                continue;
            }

            var classCosts = usesNoObject
                ? SoftmaxClassCostsStatic(logits, image, queries, numClasses)
                : FocalClassCostsStatic(logits, image, queries, numClasses, matchingAlpha, matchingGamma);
            var predictedBoxes = new BoundingBox<double>[queries];
            for (int query = 0; query < queries; query++)
            {
                int off = (image * queries + query) * 4;
                predictedBoxes[query] = new BoundingBox<double>(boxes[off], boxes[off + 1], boxes[off + 2], boxes[off + 3], BoundingBoxFormat.CXCYWH);
            }

            var costs = new Matrix<double>(imageTargets.Length, queries);
            var costsSpan = costs.AsWritableSpan();
            for (int targetIndex = 0; targetIndex < imageTargets.Length; targetIndex++)
            {
                var target = imageTargets[targetIndex];
                var actual = new BoundingBox<double>(target.Cx, target.Cy, target.W, target.H, BoundingBoxFormat.CXCYWH);
                for (int query = 0; query < queries; query++)
                {
                    int offset = (image * queries + query) * 4;
                    double l1 = Math.Abs(boxes[offset] - target.Cx)
                        + Math.Abs(boxes[offset + 1] - target.Cy)
                        + Math.Abs(boxes[offset + 2] - target.W)
                        + Math.Abs(boxes[offset + 3] - target.H);
                    costsSpan[(targetIndex * queries) + query] = classCostWeight * classCosts[query * numClasses + target.ClassId]
                        + l1CostWeight * l1 - giouCostWeight * nms.ComputeGIoU(predictedBoxes[query], actual);
                }
            }
            var assignment = solver.Solve(costs);
            var assignSpan = assignment.AsSpan();
            var rows = new int[imageTargets.Length];
            for (int targetIndex = 0; targetIndex < rows.Length; targetIndex++)
            {
                int query = assignSpan[targetIndex];
                if (query < 0 || query >= queries)
                    throw new InvalidOperationException("The assignment solver did not match every validated target.");
                rows[targetIndex] = query;
            }
            result[image] = rows;
        }
        return result;
    }

    private static double[] SoftmaxClassCostsStatic(double[] logits, int image, int queries, int numClasses)
    {
        var costs = new double[checked(queries * numClasses)];
        for (int query = 0; query < queries; query++)
        {
            int offset = (image * queries + query) * numClasses;
            double maximum = double.NegativeInfinity;
            for (int label = 0; label < numClasses; label++)
                maximum = Math.Max(maximum, logits[offset + label]);
            double sum = 0;
            for (int label = 0; label < numClasses; label++)
            {
                double value = Math.Exp(logits[offset + label] - maximum);
                costs[query * numClasses + label] = value;
                sum += value;
            }
            for (int label = 0; label < numClasses; label++)
                costs[query * numClasses + label] = -costs[query * numClasses + label] / sum;
        }
        return costs;
    }

    private static double[] FocalClassCostsStatic(double[] logits, int image, int queries, int numClasses, double alpha, double gamma)
    {
        var costs = new double[checked(queries * numClasses)];
        for (int index = 0; index < costs.Length; index++)
        {
            double logit = logits[image * queries * numClasses + index];
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

    /// <summary>The (cx, cy, w, h) of each target as a <c>[N, 4]</c> tensor.</summary>
    private static Tensor<T> TargetRows(IEnumerable<DetectionTrainingTarget<T>> targets)
        => CvTensorOps<T>.FromRows(targets.Select(target => new[] { target.CenterX, target.CenterY, target.Width, target.Height }).ToArray(), 4);

    private Tensor<T> ToCorners(Tensor<T> boxes) => CornersOf(boxes);

    /// <summary>cxcywh rows <c>[N, 4]</c> as corners <c>(cx - w/2, cy - h/2, cx + w/2, cy + h/2)</c>.</summary>
    private static Tensor<T> CornersOf(Tensor<T> boxes) => CornersOf<T>(boxes);

    private static Tensor<TBox> CornersOf<TBox>(Tensor<TBox> boxes)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TBox>();
        int count = boxes.Shape[0];
        var centers = engine.TensorSlice(boxes, new[] { 0, 0 }, new[] { count, 2 });
        var halfExtents = engine.TensorDivideScalar(
            engine.TensorSlice(boxes, new[] { 0, 2 }, new[] { count, 2 }), ops.FromDouble(2));
        return engine.TensorConcatenate(
            new[] { engine.TensorSubtract(centers, halfExtents), engine.TensorAdd(centers, halfExtents) }, 1);
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

    private static void ValidateFinitePredictionsStatic(double[] logits, double[] boxes)
    {
        foreach (double number in logits)
        {
            if (double.IsNaN(number) || double.IsInfinity(number))
                throw new ArgumentException("Class logits must be finite.", nameof(logits));
        }
        foreach (double number in boxes)
        {
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
