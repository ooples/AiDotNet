using AiDotNet.ComputerVision.Detection;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// One object to learn: its class and box, and its binary mask over the whole input image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Instance segmentation learns three things per object: what it is (the class),
/// where it is (the box) and exactly which pixels belong to it (the mask). The mask has one value per input
/// pixel: 1 inside the object, 0 outside.</para>
/// </remarks>
public sealed class InstanceSegmentationTrainingTarget<T>
{
    /// <summary>
    /// Creates a training target.
    /// </summary>
    /// <param name="box">The object's class and box, normalized against the input size.</param>
    /// <param name="mask">The object's mask [inputHeight, inputWidth]: 1 inside the object, 0 outside.</param>
    public InstanceSegmentationTrainingTarget(DetectionTrainingTarget<T> box, Tensor<T> mask)
    {
        Box = box ?? throw new ArgumentNullException(nameof(box));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (mask.Rank != 2 || mask.Shape[0] <= 0 || mask.Shape[1] <= 0)
            throw new ArgumentException("A mask must be [height, width].", nameof(mask));
        var ops = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        foreach (var value in mask.ToArray())
        {
            double v = ops.ToDouble(value);
            if (double.IsNaN(v) || v < 0 || v > 1)
                throw new ArgumentException("Mask values must be in [0, 1].", nameof(mask));
        }

        Mask = mask;
    }

    /// <summary>The object's class and box.</summary>
    public DetectionTrainingTarget<T> Box { get; }

    /// <summary>The object's mask over the input image [height, width].</summary>
    public Tensor<T> Mask { get; }

    /// <summary>The object's foreground class.</summary>
    public int ClassId => Box.ClassId;
}
