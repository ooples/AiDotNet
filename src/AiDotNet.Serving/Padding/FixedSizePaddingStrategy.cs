using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Padding;

/// <summary>
/// Fixed-size padding strategy that always pads to a predetermined length.
/// This provides consistent batch shapes but may result in more padding overhead.
/// </summary>
public class FixedSizePaddingStrategy : IPaddingStrategy
{
    private static readonly Lazy<object> _one = new Lazy<object>(() => 1);
    private static readonly Lazy<object> _zero = new Lazy<object>(() => 0);

    private readonly int _fixedLength;

    /// <summary>
    /// Initializes a new instance of the FixedSizePaddingStrategy.
    /// </summary>
    /// <param name="fixedLength">The fixed length to pad all sequences to</param>
    public FixedSizePaddingStrategy(int fixedLength)
    {
        if (fixedLength <= 0)
            throw new ArgumentException("Fixed length must be positive", nameof(fixedLength));

        _fixedLength = fixedLength;
    }

    public string Name => "Fixed";

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

        // Validate that no vector exceeds fixed length
        var maxLength = vectors.Max(v => v.Length);
        if (maxLength > _fixedLength)
            throw new ArgumentException($"Vector length {maxLength} exceeds fixed length {_fixedLength}");

        // Create padded matrix
        var paddedMatrix = new Matrix<T>(batchSize, _fixedLength);

        // Create attention mask (1 for actual data, 0 for padding)
        attentionMask = new Matrix<T>(batchSize, _fixedLength);

        var one = (T)Convert.ChangeType(_one.Value, typeof(T));
        var zero = (T)Convert.ChangeType(_zero.Value, typeof(T));

        for (int i = 0; i < batchSize; i++)
        {
            var vector = vectors[i];
            for (int j = 0; j < _fixedLength; j++)
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
            var length = Math.Min(originalLengths[i], _fixedLength);
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
