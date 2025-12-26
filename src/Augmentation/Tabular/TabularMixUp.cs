using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Applies MixUp augmentation to tabular data by linearly interpolating between samples.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MixUp creates new training samples by blending two existing
/// samples together. If you have sample A and sample B, MixUp creates a new sample that's
/// (λ × A) + ((1-λ) × B), where λ is randomly chosen. The labels are blended the same way.</para>
/// <para><b>Benefits:</b>
/// <list type="bullet">
/// <item>Regularizes the model by creating "virtual" training examples</item>
/// <item>Encourages smooth decision boundaries</item>
/// <item>Reduces overconfidence in predictions</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Classification tasks with numerical features</item>
/// <item>When you want to reduce overfitting</item>
/// <item>When decision boundaries should be smooth</item>
/// </list>
/// </para>
/// <para><b>Reference:</b> Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2017)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TabularMixUp<T> : TabularMixingAugmenterBase<T>
{
    /// <summary>
    /// Creates a new MixUp augmentation for tabular data.
    /// </summary>
    /// <param name="alpha">Alpha parameter for Beta distribution (default: 0.2).
    /// Smaller values keep samples closer to originals; larger values create more blending.</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    public TabularMixUp(double alpha = 0.2, double probability = 0.5)
        : base(probability, alpha)
    {
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // For single-sample augmentation, we need at least 2 samples
        int rows = data.Rows;
        if (rows < 2)
        {
            return data.Clone();
        }

        var result = data.Clone();
        int cols = data.Columns;

        // Sample lambda from Beta(alpha, alpha) distribution
        double lambda = SampleLambda(context);
        LastMixingLambda = NumOps.FromDouble(lambda);

        // Shuffle indices to pair samples for mixing
        var indices = Enumerable.Range(0, rows).ToArray();
        ShuffleArray(indices, context.Random);

        for (int i = 0; i < rows; i++)
        {
            int j = indices[i];

            // Avoid self-mixing by using neighbor instead - ensures consistent augmentation
            if (i == j)
            {
                j = (i + 1) % rows;
            }

            for (int c = 0; c < cols; c++)
            {
                double val1 = NumOps.ToDouble(data[i, c]);
                double val2 = NumOps.ToDouble(data[j, c]);
                double mixed = lambda * val1 + (1 - lambda) * val2;
                result[i, c] = NumOps.FromDouble(mixed);
            }
        }

        return result;
    }

    /// <summary>
    /// Applies MixUp to two data matrices and their labels, returning mixed results.
    /// </summary>
    /// <param name="data1">The first data matrix.</param>
    /// <param name="data2">The second data matrix.</param>
    /// <param name="labels1">Labels for the first data matrix.</param>
    /// <param name="labels2">Labels for the second data matrix.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Tuple of (mixed data, mixed labels, lambda value).</returns>
    public (Matrix<T> Data, Vector<T> Labels, double Lambda) MixWithLabels(
        Matrix<T> data1,
        Matrix<T> data2,
        Vector<T> labels1,
        Vector<T> labels2,
        AugmentationContext<T> context)
    {
        int rows1 = data1.Rows;
        int rows2 = data2.Rows;
        int cols = data1.Columns;

        if (cols != data2.Columns)
        {
            throw new ArgumentException("Both data matrices must have the same number of features.");
        }

        // Use minimum row count for mixing
        int rows = Math.Min(rows1, rows2);
        var mixedData = new Matrix<T>(rows, cols);
        var mixedLabels = new Vector<T>(rows);

        double lambda = SampleLambda(context);
        LastMixingLambda = NumOps.FromDouble(lambda);

        for (int i = 0; i < rows; i++)
        {
            for (int c = 0; c < cols; c++)
            {
                double val1 = NumOps.ToDouble(data1[i, c]);
                double val2 = NumOps.ToDouble(data2[i, c]);
                double mixed = lambda * val1 + (1 - lambda) * val2;
                mixedData[i, c] = NumOps.FromDouble(mixed);
            }

            // Mix labels the same way
            double label1 = NumOps.ToDouble(labels1[i]);
            double label2 = NumOps.ToDouble(labels2[i]);
            double mixedLabel = lambda * label1 + (1 - lambda) * label2;
            mixedLabels[i] = NumOps.FromDouble(mixedLabel);
        }

        return (mixedData, mixedLabels, lambda);
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
