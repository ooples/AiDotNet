using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Padding;

/// <summary>
/// Minimal padding strategy that pads sequences to the length of the longest sequence in the batch.
/// This minimizes padding overhead but may result in variable batch shapes.
/// </summary>
public class MinimalPaddingStrategy : IPaddingStrategy
{
    private static readonly Lazy<object> _one = new Lazy<object>(() => 1);
    private static readonly Lazy<object> _zero = new Lazy<object>(() => 0);

    public string Name => "Minimal";

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

        // Create padded matrix
        var paddedMatrix = new Matrix<T>(batchSize, maxLength);

        // Create attention mask (1 for actual data, 0 for padding)
        attentionMask = new Matrix<T>(batchSize, maxLength);

        var one = (T)Convert.ChangeType(_one.Value, typeof(T));
        var zero = (T)Convert.ChangeType(_zero.Value, typeof(T));

        for (int i = 0; i < batchSize; i++)
        {
            var vector = vectors[i];
            for (int j = 0; j < maxLength; j++)
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
