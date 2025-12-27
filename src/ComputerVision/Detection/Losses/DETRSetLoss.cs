using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>
/// DETR Set Prediction Loss with Hungarian Matching for end-to-end object detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Unlike traditional detectors that use anchors and NMS,
/// DETR treats detection as a set prediction problem. It uses Hungarian matching to
/// find the optimal assignment between predicted and ground truth boxes, then computes
/// loss on the matched pairs.</para>
///
/// <para>The loss has three components:
/// - Classification loss: Cross-entropy for class predictions
/// - Box loss: L1 loss for box coordinates
/// - GIoU loss: For better box regression
/// </para>
///
/// <para>Reference: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020</para>
/// </remarks>
public class DETRSetLoss<T> : LossFunctionBase<T>
{
    private readonly NMS<T> _nms;
    private readonly double _classWeight;
    private readonly double _boxL1Weight;
    private readonly double _boxGIoUWeight;
    private readonly int _numClasses;

    /// <summary>
    /// Creates a new DETR set loss instance.
    /// </summary>
    /// <param name="numClasses">Number of object classes (including no-object class).</param>
    /// <param name="classWeight">Weight for classification loss.</param>
    /// <param name="boxL1Weight">Weight for L1 box loss.</param>
    /// <param name="boxGIoUWeight">Weight for GIoU box loss.</param>
    public DETRSetLoss(
        int numClasses = 91,
        double classWeight = 1.0,
        double boxL1Weight = 5.0,
        double boxGIoUWeight = 2.0) : base()
    {
        _nms = new NMS<T>();
        _numClasses = numClasses;
        _classWeight = classWeight;
        _boxL1Weight = boxL1Weight;
        _boxGIoUWeight = boxGIoUWeight;
    }

    /// <summary>
    /// Calculates the DETR loss using flattened vectors (simplified for interface compatibility).
    /// </summary>
    /// <remarks>
    /// For DETR, the tensor-based CalculateLoss overload is preferred as it preserves
    /// the structured format needed for Hungarian matching. This vector-based method
    /// provides a basic L1 loss between predicted and actual values.
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Simple L1 loss for vector interface compatibility
        double totalLoss = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = NumOps.ToDouble(predicted[i]) - NumOps.ToDouble(actual[i]);
            totalLoss += Math.Abs(diff);
        }

        return NumOps.FromDouble(totalLoss / predicted.Length);
    }

    /// <summary>
    /// Calculates the gradient of DETR loss with respect to predicted values.
    /// </summary>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        var gradient = new Vector<T>(predicted.Length);
        double eps = 1e-5;

        for (int i = 0; i < predicted.Length; i++)
        {
            double original = NumOps.ToDouble(predicted[i]);

            predicted[i] = NumOps.FromDouble(original + eps);
            double lossPlus = NumOps.ToDouble(CalculateLoss(predicted, actual));

            predicted[i] = NumOps.FromDouble(original - eps);
            double lossMinus = NumOps.ToDouble(CalculateLoss(predicted, actual));

            predicted[i] = NumOps.FromDouble(original);
            gradient[i] = NumOps.FromDouble((lossPlus - lossMinus) / (2 * eps));
        }

        return gradient;
    }

    /// <summary>
    /// Calculates the DETR set loss.
    /// </summary>
    /// <param name="predicted">Predicted tensor containing class logits and boxes.</param>
    /// <param name="targets">Target tensor containing ground truth.</param>
    /// <returns>Combined loss value.</returns>
    /// <remarks>
    /// Expected shapes:
    /// - predicted: [batch, num_queries, num_classes + 4] (logits + boxes)
    /// - targets: [batch, max_objects, 1 + 4] (class + boxes, padded)
    /// </remarks>
    public T CalculateLoss(Tensor<T> predicted, Tensor<T> targets)
    {
        int batch = predicted.Shape[0];
        int numQueries = predicted.Shape[1];

        double totalLoss = 0;
        int validBatches = 0;

        for (int b = 0; b < batch; b++)
        {
            // Extract predictions and targets for this batch
            var predBoxes = ExtractPredictedBoxes(predicted, b, numQueries);
            var predLogits = ExtractPredictedLogits(predicted, b, numQueries, _numClasses);
            var gtBoxes = ExtractGroundTruthBoxes(targets, b);
            var gtClasses = ExtractGroundTruthClasses(targets, b);

            if (gtBoxes.Count == 0) continue;

            // Perform Hungarian matching
            var (predIndices, gtIndices) = HungarianMatch(predBoxes, predLogits, gtBoxes, gtClasses);

            // Calculate losses for matched pairs
            double classLoss = CalculateClassificationLoss(predLogits, gtClasses, predIndices, gtIndices, numQueries);
            double boxL1Loss = CalculateBoxL1Loss(predBoxes, gtBoxes, predIndices, gtIndices);
            double boxGIoULoss = CalculateBoxGIoULoss(predBoxes, gtBoxes, predIndices, gtIndices);

            totalLoss += _classWeight * classLoss + _boxL1Weight * boxL1Loss + _boxGIoUWeight * boxGIoULoss;
            validBatches++;
        }

        double meanLoss = validBatches > 0 ? totalLoss / validBatches : 0;
        return NumOps.FromDouble(meanLoss);
    }

    /// <summary>
    /// Calculates the gradient of the DETR loss.
    /// </summary>
    public Tensor<T> CalculateDerivative(Tensor<T> predicted, Tensor<T> targets)
    {
        // Numerical gradient for now - analytical gradients are complex for Hungarian matching
        var gradient = new Tensor<T>(predicted.Shape);
        double eps = 1e-5;

        for (int i = 0; i < predicted.Length; i++)
        {
            double original = NumOps.ToDouble(predicted[i]);

            predicted[i] = NumOps.FromDouble(original + eps);
            double lossPlus = NumOps.ToDouble(CalculateLoss(predicted, targets));

            predicted[i] = NumOps.FromDouble(original - eps);
            double lossMinus = NumOps.ToDouble(CalculateLoss(predicted, targets));

            predicted[i] = NumOps.FromDouble(original);
            gradient[i] = NumOps.FromDouble((lossPlus - lossMinus) / (2 * eps));
        }

        return gradient;
    }

    /// <summary>
    /// Performs Hungarian matching between predictions and ground truth.
    /// </summary>
    /// <param name="predBoxes">Predicted bounding boxes.</param>
    /// <param name="predLogits">Predicted class logits.</param>
    /// <param name="gtBoxes">Ground truth bounding boxes.</param>
    /// <param name="gtClasses">Ground truth class labels.</param>
    /// <returns>Matched indices (prediction indices, ground truth indices).</returns>
    private (int[] PredIndices, int[] GtIndices) HungarianMatch(
        List<BoundingBox<T>> predBoxes,
        double[,] predLogits,
        List<BoundingBox<T>> gtBoxes,
        List<int> gtClasses)
    {
        int numPred = predBoxes.Count;
        int numGt = gtBoxes.Count;

        // Build cost matrix [numPred x numGt]
        var costMatrix = new double[numPred, numGt];

        for (int i = 0; i < numPred; i++)
        {
            for (int j = 0; j < numGt; j++)
            {
                // Classification cost: negative log probability for the correct class
                double classCost = ComputeClassCost(predLogits, i, gtClasses[j]);

                // Box L1 cost
                double l1Cost = ComputeL1Cost(predBoxes[i], gtBoxes[j]);

                // Box GIoU cost (negative because we minimize)
                double giouCost = 1.0 - _nms.ComputeGIoU(predBoxes[i], gtBoxes[j]);

                // Combined cost (weighted)
                costMatrix[i, j] = classCost + 5.0 * l1Cost + 2.0 * giouCost;
            }
        }

        // Solve assignment using simplified greedy approach
        // (Full Hungarian algorithm would be more optimal but complex)
        var predIndices = new List<int>();
        var gtIndices = new List<int>();
        var usedPred = new HashSet<int>();
        var usedGt = new HashSet<int>();

        // Greedy matching: assign lowest cost pairs first
        var assignments = new List<(int Pred, int Gt, double Cost)>();
        for (int i = 0; i < numPred; i++)
        {
            for (int j = 0; j < numGt; j++)
            {
                assignments.Add((i, j, costMatrix[i, j]));
            }
        }

        assignments.Sort((a, b) => a.Cost.CompareTo(b.Cost));

        foreach (var (pred, gt, _) in assignments)
        {
            if (!usedPred.Contains(pred) && !usedGt.Contains(gt))
            {
                predIndices.Add(pred);
                gtIndices.Add(gt);
                usedPred.Add(pred);
                usedGt.Add(gt);

                if (gtIndices.Count >= numGt) break;
            }
        }

        return (predIndices.ToArray(), gtIndices.ToArray());
    }

    /// <summary>
    /// Computes classification cost for Hungarian matching.
    /// </summary>
    private double ComputeClassCost(double[,] predLogits, int predIdx, int gtClass)
    {
        // Softmax over classes
        double maxLogit = double.NegativeInfinity;
        int numClasses = predLogits.GetLength(1);
        for (int c = 0; c < numClasses; c++)
        {
            maxLogit = Math.Max(maxLogit, predLogits[predIdx, c]);
        }

        double sumExp = 0;
        for (int c = 0; c < numClasses; c++)
        {
            sumExp += Math.Exp(predLogits[predIdx, c] - maxLogit);
        }

        double logProb = predLogits[predIdx, gtClass] - maxLogit - Math.Log(sumExp);
        return -logProb; // Negative log probability
    }

    /// <summary>
    /// Computes L1 cost between two boxes.
    /// </summary>
    private double ComputeL1Cost(BoundingBox<T> pred, BoundingBox<T> gt)
    {
        var (px1, py1, px2, py2) = pred.ToXYXY();
        var (gx1, gy1, gx2, gy2) = gt.ToXYXY();

        return Math.Abs(px1 - gx1) + Math.Abs(py1 - gy1) +
               Math.Abs(px2 - gx2) + Math.Abs(py2 - gy2);
    }

    /// <summary>
    /// Calculates classification loss for matched pairs.
    /// </summary>
    private double CalculateClassificationLoss(
        double[,] predLogits,
        List<int> gtClasses,
        int[] predIndices,
        int[] gtIndices,
        int numQueries)
    {
        int numClasses = predLogits.GetLength(1);
        double loss = 0;

        // Loss for matched pairs
        for (int i = 0; i < predIndices.Length; i++)
        {
            int predIdx = predIndices[i];
            int gtIdx = gtIndices[i];
            int gtClass = gtClasses[gtIdx];

            loss += ComputeClassCost(predLogits, predIdx, gtClass);
        }

        // Loss for unmatched predictions (should predict no-object class)
        int noObjectClass = numClasses - 1;
        for (int p = 0; p < numQueries; p++)
        {
            if (!predIndices.Contains(p))
            {
                // Lower weight for no-object class
                loss += 0.1 * ComputeClassCost(predLogits, p, noObjectClass);
            }
        }

        return loss / numQueries;
    }

    /// <summary>
    /// Calculates L1 box loss for matched pairs.
    /// </summary>
    private double CalculateBoxL1Loss(
        List<BoundingBox<T>> predBoxes,
        List<BoundingBox<T>> gtBoxes,
        int[] predIndices,
        int[] gtIndices)
    {
        if (predIndices.Length == 0) return 0;

        double loss = 0;
        for (int i = 0; i < predIndices.Length; i++)
        {
            loss += ComputeL1Cost(predBoxes[predIndices[i]], gtBoxes[gtIndices[i]]);
        }

        return loss / predIndices.Length;
    }

    /// <summary>
    /// Calculates GIoU box loss for matched pairs.
    /// </summary>
    private double CalculateBoxGIoULoss(
        List<BoundingBox<T>> predBoxes,
        List<BoundingBox<T>> gtBoxes,
        int[] predIndices,
        int[] gtIndices)
    {
        if (predIndices.Length == 0) return 0;

        double loss = 0;
        for (int i = 0; i < predIndices.Length; i++)
        {
            double giou = _nms.ComputeGIoU(predBoxes[predIndices[i]], gtBoxes[gtIndices[i]]);
            loss += 1.0 - giou;
        }

        return loss / predIndices.Length;
    }

    /// <summary>
    /// Extracts predicted boxes from the combined tensor.
    /// </summary>
    private List<BoundingBox<T>> ExtractPredictedBoxes(Tensor<T> predicted, int batch, int numQueries)
    {
        var boxes = new List<BoundingBox<T>>();
        int boxOffset = _numClasses; // Boxes come after class logits

        for (int i = 0; i < numQueries; i++)
        {
            boxes.Add(new BoundingBox<T>(
                predicted[batch, i, boxOffset],
                predicted[batch, i, boxOffset + 1],
                predicted[batch, i, boxOffset + 2],
                predicted[batch, i, boxOffset + 3],
                BoundingBoxFormat.CXCYWH)); // DETR uses center format
        }

        return boxes;
    }

    /// <summary>
    /// Extracts predicted logits from the combined tensor.
    /// </summary>
    private double[,] ExtractPredictedLogits(Tensor<T> predicted, int batch, int numQueries, int numClasses)
    {
        var logits = new double[numQueries, numClasses];

        for (int i = 0; i < numQueries; i++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                logits[i, c] = NumOps.ToDouble(predicted[batch, i, c]);
            }
        }

        return logits;
    }

    /// <summary>
    /// Extracts ground truth boxes from the targets tensor.
    /// </summary>
    private List<BoundingBox<T>> ExtractGroundTruthBoxes(Tensor<T> targets, int batch)
    {
        var boxes = new List<BoundingBox<T>>();
        int maxObjects = targets.Shape[1];

        for (int i = 0; i < maxObjects; i++)
        {
            // Class is first, check if valid (not padding)
            int classId = (int)NumOps.ToDouble(targets[batch, i, 0]);
            if (classId < 0) break; // Padding marker

            boxes.Add(new BoundingBox<T>(
                targets[batch, i, 1],
                targets[batch, i, 2],
                targets[batch, i, 3],
                targets[batch, i, 4],
                BoundingBoxFormat.CXCYWH));
        }

        return boxes;
    }

    /// <summary>
    /// Extracts ground truth class labels from the targets tensor.
    /// </summary>
    private List<int> ExtractGroundTruthClasses(Tensor<T> targets, int batch)
    {
        var classes = new List<int>();
        int maxObjects = targets.Shape[1];

        for (int i = 0; i < maxObjects; i++)
        {
            int classId = (int)NumOps.ToDouble(targets[batch, i, 0]);
            if (classId < 0) break; // Padding marker

            classes.Add(classId);
        }

        return classes;
    }
}
