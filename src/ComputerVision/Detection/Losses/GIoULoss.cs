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
        if (numBoxes == 0) return NumOps.Zero;

        var predictedBoxes = new Tensor<T>(new[] { numBoxes, 4 }, predicted);
        var targetBoxes = new Tensor<T>(new[] { numBoxes, 4 }, actual);
        return MeanLoss(predictedBoxes, targetBoxes);
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
        if (batch * numBoxes == 0) return NumOps.Zero;

        // Box (b, i) of each tensor is its first four coordinates at [b, i, ..].
        var predictedBoxes = BoxOverlap<T>.AsRows(Engine.TensorSlice(predicted, new[] { 0, 0, 0 }, new[] { batch, numBoxes, predicted.Shape[2] }));
        var targetBoxes = BoxOverlap<T>.AsRows(Engine.TensorSlice(targets, new[] { 0, 0, 0 }, new[] { batch, numBoxes, targets.Shape[2] }));
        return MeanLoss(predictedBoxes, targetBoxes);
    }

    /// <summary>The mean of <c>1 - GIoU</c> over the rows of two <c>[N, 4]</c> XYXY box sets.</summary>
    private T MeanLoss(Tensor<T> predictedBoxes, Tensor<T> targetBoxes)
    {
        var perBox = Engine.ScalarMinusTensor(NumOps.One, BoxOverlap<T>.GIoU(predictedBoxes, targetBoxes));
        return CvTensorOps<T>.Mean(perBox);
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

        var perBoxLoss = Engine.TensorGIoULoss(predicted, target);
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
