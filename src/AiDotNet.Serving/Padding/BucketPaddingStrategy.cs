using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Padding;

/// <summary>
/// Bucket-based padding strategy that pads sequences to predefined bucket sizes.
/// This reduces padding overhead while maintaining consistent batch shapes for better hardware utilization.
/// </summary>
public class BucketPaddingStrategy : IPaddingStrategy
{
    private static readonly Lazy<object> _one = new Lazy<object>(() => 1);
    private static readonly Lazy<object> _zero = new Lazy<object>(() => 0);

    private readonly int[] _bucketSizes;

    /// <summary>
    /// Initializes a new instance of the BucketPaddingStrategy.
    /// </summary>
    /// <param name="bucketSizes">Array of bucket sizes (e.g., [32, 64, 128, 256, 512])</param>
    public BucketPaddingStrategy(int[] bucketSizes)
    {
        if (bucketSizes == null || bucketSizes.Length == 0)
            throw new ArgumentException("Bucket sizes cannot be null or empty", nameof(bucketSizes));

        _bucketSizes = bucketSizes.OrderBy(x => x).ToArray();
    }

    public string Name => "Bucket";

    /// <summary>
    /// Gets the appropriate bucket size for a given length.
    /// </summary>
    /// <param name="length">The length to find a bucket for</param>
    /// <returns>The bucket size</returns>
    private int GetBucketSize(int length)
    {
        foreach (var bucketSize in _bucketSizes.Where(bucketSize => length <= bucketSize))
        {
            return bucketSize;
        }

        // If larger than all buckets, return the next power of 2
        return (int)Math.Pow(2, Math.Ceiling(MathHelper.Log2(length)));
    }

    public Matrix<T> PadBatch<T>(Vector<T>[] vectors, out Matrix<T>? attentionMask)
    {
        if (vectors == null || vectors.Length == 0)
            throw new ArgumentException("Vectors array cannot be null or empty", nameof(vectors));

        // Validate no null vectors
        for (int i = 0; i < vectors.Length; i++)
        {
            if (vectors[i] == null)
                throw new ArgumentException($"Vector at index {i} cannot be null.", nameof(vectors));
        }

        var batchSize = vectors.Length;
        var maxLength = vectors.Max(v => v.Length);
        var paddedLength = GetBucketSize(maxLength);

        // Create padded matrix
        var paddedMatrix = new Matrix<T>(batchSize, paddedLength);

        // Create attention mask (1 for actual data, 0 for padding)
        attentionMask = new Matrix<T>(batchSize, paddedLength);

        var one = (T)Convert.ChangeType(_one.Value, typeof(T));
        var zero = (T)Convert.ChangeType(_zero.Value, typeof(T));

        for (int i = 0; i < batchSize; i++)
        {
            var vector = vectors[i];
            for (int j = 0; j < paddedLength; j++)
            {
                if (j < vector.Length)
                {
                    paddedMatrix[i, j] = vector[j];
                    attentionMask[i, j] = one;
                }
                else
                {
                    paddedMatrix[i, j] = default(T)!;
                    attentionMask[i, j] = zero;
                }
            }
        }

        return paddedMatrix;
    }

    public Vector<T>[] UnpadBatch<T>(Matrix<T> paddedMatrix, int[] originalLengths)
    {
        if (paddedMatrix == null)
            throw new ArgumentNullException(nameof(paddedMatrix));
        if (originalLengths == null)
            throw new ArgumentNullException(nameof(originalLengths));
        if (paddedMatrix.Rows != originalLengths.Length)
            throw new ArgumentException("Number of rows must match number of original lengths");

        // Validate original lengths are non-negative
        for (int i = 0; i < originalLengths.Length; i++)
        {
            if (originalLengths[i] < 0)
                throw new ArgumentException($"Original length at index {i} cannot be negative, but was {originalLengths[i]}.", nameof(originalLengths));
        }

        var result = new Vector<T>[paddedMatrix.Rows];

        for (int i = 0; i < paddedMatrix.Rows; i++)
        {
            var length = originalLengths[i];
            var vector = new Vector<T>(length);

            for (int j = 0; j < length; j++)
            {
                vector[j] = paddedMatrix[i, j];
            }

            result[i] = vector;
        }

        return result;
    }
}
