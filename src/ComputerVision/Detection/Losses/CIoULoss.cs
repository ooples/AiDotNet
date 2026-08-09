using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.Losses;

/// <summary>
/// Complete Intersection over Union (CIoU) loss for bounding box regression.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> CIoU loss extends DIoU by also considering aspect ratio.
/// This provides the most accurate bounding box regression and is used in modern
/// YOLO versions (v5, v7, v8, etc.).</para>
///
/// <para>CIoU = IoU - d²/c² - αv, where:
/// - d is the center distance
/// - c is the enclosing diagonal
/// - v measures aspect ratio consistency
/// - α is a balancing factor
/// CIoU Loss = 1 - CIoU</para>
///
/// <para>Reference: Zheng et al., "Distance-IoU Loss: Faster and Better Learning for
/// Bounding Box Regression", AAAI 2020</para>
/// </remarks>
public class CIoULoss<T> : LossFunctionBase<T>
{
    private readonly NMS<T> _nms;

    /// <summary>
    /// Creates a new CIoU loss instance.
    /// </summary>
    public CIoULoss() : base()
    {
        _nms = new NMS<T>();
    }

    /// <summary>
    /// Calculates the CIoU loss between predicted and target bounding box vectors.
    /// </summary>
    /// <param name="predicted">Flattened predicted boxes [x1,y1,x2,y2, x1,y1,x2,y2, ...]</param>
    /// <param name="actual">Flattened target boxes [x1,y1,x2,y2, x1,y1,x2,y2, ...]</param>
    /// <returns>Mean CIoU loss value.</returns>
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

            double ciou = _nms.ComputeCIoU(predBox, targetBox);
            totalLoss += 1.0 - ciou;
        }

        return NumOps.FromDouble(numBoxes > 0 ? totalLoss / numBoxes : 0);
    }

    /// <summary>
    /// Calculates the CIoU loss between predicted and target bounding boxes.
    /// </summary>
    /// <param name="predicted">Predicted boxes tensor [batch, num_boxes, 4] in XYXY format.</param>
    /// <param name="targets">Target boxes tensor [batch, num_boxes, 4] in XYXY format.</param>
    /// <returns>Mean CIoU loss value.</returns>
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

                double ciou = _nms.ComputeCIoU(predBox, targetBox);
                double loss = 1.0 - ciou;
                totalLoss += loss;
                count++;
            }
        }

        double meanLoss = count > 0 ? totalLoss / count : 0;
        return NumOps.FromDouble(meanLoss);
    }

    /// <summary>
    /// Calculates CIoU loss for a pair of bounding boxes.
    /// </summary>
    /// <param name="predicted">Predicted bounding box.</param>
    /// <param name="target">Target bounding box.</param>
    /// <returns>CIoU loss value (1 - CIoU).</returns>
    public double CalculateLossForBox(BoundingBox<T> predicted, BoundingBox<T> target)
    {
        double ciou = _nms.ComputeCIoU(predicted, target);
        return 1.0 - ciou;
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

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // Accept the FLAT layout CalculateLoss(Vector, Vector) takes -- a rank-1 run of boxes whose
        // length is a multiple of 4 -- by folding it to [N, 4]. Without this the two forwards
        // disagreed about what they accept: the vector API validated "multiple of 4" and the tape
        // API rejected anything that was not already rank 2, so the same input was valid for the
        // loss and invalid for its gradient.
        predicted = NormalizeBoxes(predicted, nameof(predicted));
        target = NormalizeBoxes(target, nameof(target));

        if (target.Shape[0] != predicted.Shape[0])
        {
            throw new ArgumentException(
                $"Target batch size ({target.Shape[0]}) must match predicted ({predicted.Shape[0]}).",
                nameof(target));
        }

        var perBoxLoss = Engine.TensorCIoULoss(predicted, target);
        var allAxes = Enumerable.Range(0, perBoxLoss.Shape.Length).ToArray();
        return Engine.ReduceMean(perBoxLoss, allAxes, keepDims: false);
    }

    /// <summary>
    /// Folds a flat run of box coordinates into the [N, 4] layout the IoU kernels require.
    /// </summary>
    /// <param name="boxes">Either an [N, 4] tensor or a rank-1 run of 4N coordinates.</param>
    /// <param name="parameterName">Name used when reporting an invalid shape.</param>
    /// <returns>The tensor as [N, 4].</returns>
    /// <exception cref="ArgumentException">Thrown when the shape is neither form.</exception>
    private Tensor<T> NormalizeBoxes(Tensor<T> boxes, string parameterName)
    {
        if (boxes.Shape.Length == 2 && boxes.Shape[1] == 4)
        {
            return boxes;
        }

        if (boxes.Shape.Length == 1 && boxes.Length % 4 == 0 && boxes.Length > 0)
        {
            return Engine.Reshape(boxes, new[] { boxes.Length / 4, 4 });
        }

        throw new ArgumentException(
            $"Boxes must be [N, 4] in (x1, y1, x2, y2) format, or a flat run of 4N coordinates, "
            + $"but were [{string.Join(", ", boxes.Shape.ToArray())}].",
            parameterName);
    }
}
