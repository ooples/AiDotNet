namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Converts a flat array to a Tensor with the specified shape.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This transform reshapes a 1D array of values into a multi-dimensional tensor.
/// The total number of elements must match the product of the shape dimensions.
/// </para>
/// <para><b>For Beginners:</b> When you load image data as a flat array of pixels,
/// you need to reshape it into the correct dimensions (height x width x channels)
/// for neural network input.
/// </para>
/// </remarks>
public class ToTensorTransform<T> : ITransform<T[], Tensor<T>>
{
    private readonly int[] _shape;
    private readonly int _expectedLength;

    /// <summary>
    /// Creates a transform that converts arrays to tensors with the specified shape.
    /// </summary>
    /// <param name="shape">The target tensor shape.</param>
    public ToTensorTransform(params int[] shape)
    {
        if (shape is null)
        {
            throw new ArgumentNullException(nameof(shape));
        }

        if (shape.Length == 0)
        {
            throw new ArgumentException("Shape must have at least one dimension.", nameof(shape));
        }

        int total = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] <= 0)
            {
                throw new ArgumentException($"Shape dimension {i} must be positive, got {shape[i]}.", nameof(shape));
            }

            total *= shape[i];
        }

        _shape = (int[])shape.Clone();
        _expectedLength = total;
    }

    /// <inheritdoc/>
    public Tensor<T> Apply(T[] input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        if (input.Length != _expectedLength)
        {
            throw new ArgumentException(
                $"Input array length ({input.Length}) does not match expected tensor size ({_expectedLength}).",
                nameof(input));
        }

        return new Tensor<T>(input, _shape);
    }
}
