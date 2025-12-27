using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>
/// Generalized Intersection over Union (GIoU) loss for bounding box regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> GIoU loss improves upon standard IoU loss by providing
/// gradients even when boxes don't overlap. This helps the model learn to move boxes
/// towards their targets even when they start far apart.</para>
///
/// <para>GIoU = IoU - (|C - U|) / |C|, where C is the smallest enclosing box and U is the union.
/// GIoU Loss = 1 - GIoU</para>
///
/// <para>Reference: Rezatofighi et al., "Generalized Intersection over Union: A Metric and A Loss
/// for Bounding Box Regression", CVPR 2019</para>
/// </remarks>
public class GIoULoss<T> : LossFunctionBase<T>
{
    private readonly NMS<T> _nms;

    /// <summary>
    /// Creates a new GIoU loss instance.
    /// </summary>
    public GIoULoss() : base()
    {
        _nms = new NMS<T>();
    }

    /// <summary>
    /// Calculates the GIoU loss between predicted and target bounding box vectors.
    /// </summary>
    /// <param name="predicted">Flattened predicted boxes [x1,y1,x2,y2, x1,y1,x2,y2, ...]</param>
    /// <param name="actual">Flattened target boxes [x1,y1,x2,y2, x1,y1,x2,y2, ...]</param>
    /// <returns>Mean GIoU loss value.</returns>
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

            double giou = _nms.ComputeGIoU(predBox, targetBox);
            totalLoss += 1.0 - giou;
        }

        return NumOps.FromDouble(numBoxes > 0 ? totalLoss / numBoxes : 0);
    }

    /// <summary>
    /// Calculates the gradient of GIoU loss with respect to predicted boxes.
    /// </summary>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        var gradient = new Vector<T>(predicted.Length);
        double eps = 1e-7;

        for (int i = 0; i < predicted.Length; i++)
        {
            double original = NumOps.ToDouble(predicted[i]);

            // Forward perturbation
            predicted[i] = NumOps.FromDouble(original + eps);
            double lossPlus = NumOps.ToDouble(CalculateLoss(predicted, actual));

            // Backward perturbation
            predicted[i] = NumOps.FromDouble(original - eps);
            double lossMinus = NumOps.ToDouble(CalculateLoss(predicted, actual));

            // Restore and compute gradient
            predicted[i] = NumOps.FromDouble(original);
            gradient[i] = NumOps.FromDouble((lossPlus - lossMinus) / (2 * eps));
        }

        return gradient;
    }

    /// <summary>
    /// Calculates GIoU loss for tensor inputs.
    /// </summary>
    /// <param name="predicted">Predicted boxes tensor [batch, num_boxes, 4] in XYXY format.</param>
    /// <param name="targets">Target boxes tensor [batch, num_boxes, 4] in XYXY format.</param>
    /// <returns>Mean GIoU loss value.</returns>
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
                var predBox = new BoundingBox<T>(
                    predicted[b, i, 0], predicted[b, i, 1],
                    predicted[b, i, 2], predicted[b, i, 3],
                    BoundingBoxFormat.XYXY);

                var targetBox = new BoundingBox<T>(
                    targets[b, i, 0], targets[b, i, 1],
                    targets[b, i, 2], targets[b, i, 3],
                    BoundingBoxFormat.XYXY);

                double giou = _nms.ComputeGIoU(predBox, targetBox);
                totalLoss += 1.0 - giou;
                count++;
            }
        }

        return NumOps.FromDouble(count > 0 ? totalLoss / count : 0);
    }

    /// <summary>
    /// Calculates GIoU loss for a pair of bounding boxes.
    /// </summary>
    /// <param name="predicted">Predicted bounding box.</param>
    /// <param name="target">Target bounding box.</param>
    /// <returns>GIoU loss value (1 - GIoU).</returns>
    public double CalculateLossForBox(BoundingBox<T> predicted, BoundingBox<T> target)
    {
        return 1.0 - _nms.ComputeGIoU(predicted, target);
    }
}
