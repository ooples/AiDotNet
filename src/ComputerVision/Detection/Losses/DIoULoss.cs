using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>
/// Distance Intersection over Union (DIoU) loss for bounding box regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DIoU loss adds a center distance penalty to GIoU loss.
/// This helps the model converge faster by explicitly minimizing the distance between
/// predicted and target box centers.</para>
///
/// <para>DIoU = IoU - d²/c², where d is the center distance and c is the enclosing diagonal.
/// DIoU Loss = 1 - DIoU</para>
///
/// <para>Reference: Zheng et al., "Distance-IoU Loss: Faster and Better Learning for
/// Bounding Box Regression", AAAI 2020</para>
/// </remarks>
public class DIoULoss<T> : LossFunctionBase<T>
{
    private readonly NMS<T> _nms;

    /// <summary>
    /// Creates a new DIoU loss instance.
    /// </summary>
    public DIoULoss() : base()
    {
        _nms = new NMS<T>();
    }

    /// <summary>
    /// Calculates the DIoU loss between predicted and target bounding box vectors.
    /// </summary>
    /// <param name="predicted">Flattened predicted boxes [x1,y1,x2,y2, x1,y1,x2,y2, ...]</param>
    /// <param name="actual">Flattened target boxes [x1,y1,x2,y2, x1,y1,x2,y2, ...]</param>
    /// <returns>Mean DIoU loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        if (predicted.Length % 4 != 0)
        {
            throw new ArgumentException("Vector length must be a multiple of 4 (x1,y1,x2,y2 per box)");
        }

        int numBoxes = predicted.Length / 4;
        double totalLoss = 0;

        for (int i = 0; i < numBoxes; i++)
        {
            int offset = i * 4;
            var predBox = new BoundingBox<T>(
                predicted[offset], predicted[offset + 1],
                predicted[offset + 2], predicted[offset + 3],
                BoundingBoxFormat.XYXY);

            var targetBox = new BoundingBox<T>(
                actual[offset], actual[offset + 1],
                actual[offset + 2], actual[offset + 3],
                BoundingBoxFormat.XYXY);

            double diou = _nms.ComputeDIoU(predBox, targetBox);
            totalLoss += 1.0 - diou;
        }

        return NumOps.FromDouble(numBoxes > 0 ? totalLoss / numBoxes : 0);
    }

    /// <summary>
    /// Calculates the gradient of DIoU loss with respect to predicted boxes.
    /// </summary>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        var gradient = new Vector<T>(predicted.Length);
        double eps = 1e-7;

        // Create a copy of predicted for perturbation to avoid mutating the input
        var perturbedPredicted = new Vector<T>(predicted.Length);
        for (int j = 0; j < predicted.Length; j++)
        {
            perturbedPredicted[j] = predicted[j];
        }

        for (int i = 0; i < predicted.Length; i++)
        {
            double original = NumOps.ToDouble(predicted[i]);

            // Forward perturbation (on copy)
            perturbedPredicted[i] = NumOps.FromDouble(original + eps);
            double lossPlus = NumOps.ToDouble(CalculateLoss(perturbedPredicted, actual));

            // Backward perturbation (on copy)
            perturbedPredicted[i] = NumOps.FromDouble(original - eps);
            double lossMinus = NumOps.ToDouble(CalculateLoss(perturbedPredicted, actual));

            // Restore copy and compute gradient
            perturbedPredicted[i] = NumOps.FromDouble(original);
            gradient[i] = NumOps.FromDouble((lossPlus - lossMinus) / (2 * eps));
        }

        return gradient;
    }

    /// <summary>
    /// Calculates the DIoU loss between predicted and target bounding boxes.
    /// </summary>
    /// <param name="predicted">Predicted boxes tensor [batch, num_boxes, 4] in XYXY format.</param>
    /// <param name="targets">Target boxes tensor [batch, num_boxes, 4] in XYXY format.</param>
    /// <returns>Mean DIoU loss value.</returns>
    public T CalculateLoss(Tensor<T> predicted, Tensor<T> targets)
    {
        if (predicted.Rank != 3 || targets.Rank != 3)
        {
            throw new ArgumentException("Expected 3D tensors [batch, num_boxes, 4]");
        }

        int batch = predicted.Shape[0];
        int numBoxes = predicted.Shape[1];
        double totalLoss = 0;
        int count = 0;

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < numBoxes; i++)
            {
                var predBox = ExtractBox(predicted, b, i);
                var targetBox = ExtractBox(targets, b, i);

                double diou = _nms.ComputeDIoU(predBox, targetBox);
                double loss = 1.0 - diou;
                totalLoss += loss;
                count++;
            }
        }

        double meanLoss = count > 0 ? totalLoss / count : 0;
        return NumOps.FromDouble(meanLoss);
    }

    /// <summary>
    /// Calculates DIoU loss for a pair of bounding boxes.
    /// </summary>
    /// <param name="predicted">Predicted bounding box.</param>
    /// <param name="target">Target bounding box.</param>
    /// <returns>DIoU loss value (1 - DIoU).</returns>
    public double CalculateLossForBox(BoundingBox<T> predicted, BoundingBox<T> target)
    {
        double diou = _nms.ComputeDIoU(predicted, target);
        return 1.0 - diou;
    }

    /// <summary>
    /// Extracts a bounding box from a tensor at the specified indices.
    /// </summary>
    private BoundingBox<T> ExtractBox(Tensor<T> tensor, int batch, int boxIdx)
    {
        return new BoundingBox<T>(
            tensor[batch, boxIdx, 0],
            tensor[batch, boxIdx, 1],
            tensor[batch, boxIdx, 2],
            tensor[batch, boxIdx, 3],
            BoundingBoxFormat.XYXY);
    }
}
