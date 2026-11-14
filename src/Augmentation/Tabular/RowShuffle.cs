using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Shuffles rows within a batch of tabular data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Row shuffling randomly reorders the samples in your data.
/// While this doesn't create new data, it ensures the model doesn't learn from the
/// order of samples, which is especially important when data has natural ordering.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When data has natural temporal or sequential ordering</item>
/// <item>As part of mini-batch training to randomize batch composition</item>
/// <item>When consecutive samples might be correlated</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RowShuffle<T> : TabularAugmenterBase<T>
{
    /// <summary>
    /// Creates a new row shuffle augmentation.
    /// </summary>
    /// <param name="probability">Probability of applying this augmentation (default: 1.0).</param>
    public RowShuffle(double probability = 1.0) : base(probability)
    {
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);

        if (rows <= 1)
        {
            return data.Clone();
        }

        // Create shuffled indices
        var indices = Enumerable.Range(0, rows).ToArray();
        ShuffleArray(indices, context.Random);

        // Create result with shuffled rows
        var order = new Tensor<int>(indices, [rows]);
        return Engine.TensorIndexSelect(Tensor<T>.FromMatrix(data), order, 0).ToMatrix();
    }

    /// <summary>
    /// Shuffles data and labels together, maintaining row correspondence.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="labels">The label vector.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Tuple of (shuffled data, shuffled labels).</returns>
    public (Matrix<T> Data, Vector<T> Labels) ShuffleWithLabels(
        Matrix<T> data,
        Vector<T> labels,
        AugmentationContext<T> context)
    {
        int rows = GetSampleCount(data);
        int cols = GetFeatureCount(data);

        if (rows <= 1)
        {
            return (data.Clone(), labels.Clone());
        }

        if (labels.Length != rows)
        {
            throw new ArgumentException("Labels must have the same length as data rows.");
        }

        // Create shuffled indices
        var indices = Enumerable.Range(0, rows).ToArray();
        ShuffleArray(indices, context.Random);

        // Create results with shuffled rows
        var engine = Engine;
        var order = new Tensor<int>(indices, [rows]);
        var shuffledData = engine.TensorIndexSelect(Tensor<T>.FromMatrix(data), order, 0).ToMatrix();
        var shuffledLabels = engine.TensorIndexSelect(labels.AsTensor(), order, 0).ToVector();

        return (shuffledData, shuffledLabels);
    }

    private static void ShuffleArray(int[] array, Random random)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
