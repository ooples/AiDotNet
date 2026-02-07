namespace AiDotNet.Data.Transforms.Numeric;

/// <summary>
/// Converts a class index (integer label) to a one-hot encoded vector.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Given an integer class index and the total number of classes, produces a vector
/// of length numClasses where all elements are zero except the element at the class index.
/// </para>
/// <para><b>For Beginners:</b> One-hot encoding converts a category number into a vector.
/// For example, with 3 classes: class 0 becomes [1, 0, 0], class 1 becomes [0, 1, 0],
/// and class 2 becomes [0, 0, 1]. This is required by many neural network loss functions.
/// </para>
/// </remarks>
public class OneHotEncodeTransform<T> : ITransform<int, T[]>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _numClasses;

    /// <summary>
    /// Creates a one-hot encoder.
    /// </summary>
    /// <param name="numClasses">The total number of classes.</param>
    public OneHotEncodeTransform(int numClasses)
    {
        if (numClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
        }

        _numClasses = numClasses;
    }

    /// <summary>
    /// Gets the number of classes.
    /// </summary>
    public int NumClasses => _numClasses;

    /// <inheritdoc/>
    public T[] Apply(int input)
    {
        if (input < 0 || input >= _numClasses)
        {
            throw new ArgumentOutOfRangeException(
                nameof(input),
                $"Class index {input} is out of range [0, {_numClasses}).");
        }

        var result = new T[_numClasses];
        for (int i = 0; i < _numClasses; i++)
        {
            result[i] = NumOps.Zero;
        }

        result[input] = NumOps.One;
        return result;
    }
}
