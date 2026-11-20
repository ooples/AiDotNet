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
    /// <remarks>
    /// Each row i becomes lambda * x_i + (1 - lambda) * x_perm(i) with perm a uniform random permutation and
    /// lambda ~ Beta(alpha, alpha), exactly as in Zhang et al.'s reference implementation (index = randperm(n)).
    /// A row the permutation maps to itself stays unchanged; every row is used as a partner exactly once.
    /// </remarks>
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // For single-sample augmentation, we need at least 2 samples
        int rows = data.Rows;
        if (rows < 2)
        {
            return data.Clone();
        }

        // Sample lambda from Beta(alpha, alpha) distribution
        T lambda = NumOps.FromDouble(SampleLambda(context));
        LastMixingLambda = lambda;

        // Shuffle indices to pair samples for mixing
        var indices = Enumerable.Range(0, rows).ToArray();
        ShuffleArray(indices, context.Random);

        var engine = AiDotNetEngine.Current;
        var source = Tensor<T>.FromMatrix(data);
        var partners = engine.TensorIndexSelect(source, new Tensor<int>(indices, new[] { rows }), 0);
        return Mix(source, partners, lambda).ToMatrix();
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
    public (Matrix<T> Data, Vector<T> Labels, T Lambda) MixWithLabels(
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

        if (labels1.Length < rows1)
        {
            throw new ArgumentException("labels1 length must match data1 row count.", nameof(labels1));
        }

        if (labels2.Length < rows2)
        {
            throw new ArgumentException("labels2 length must match data2 row count.", nameof(labels2));
        }

        // Use minimum row count for mixing
        int rows = Math.Min(rows1, rows2);

        T lambda = NumOps.FromDouble(SampleLambda(context));
        LastMixingLambda = lambda;

        var engine = AiDotNetEngine.Current;
        var first = engine.TensorSlice(Tensor<T>.FromMatrix(data1), new[] { 0, 0 }, new[] { rows, cols });
        var second = engine.TensorSlice(Tensor<T>.FromMatrix(data2), new[] { 0, 0 }, new[] { rows, cols });
        var mixedData = Mix(first, second, lambda).ToMatrix();

        // Mix labels the same way
        var mixedLabels = Mix(labels1.GetSubVector(0, rows).AsTensor(), labels2.GetSubVector(0, rows).AsTensor(), lambda).ToVector();

        return (mixedData, mixedLabels, lambda);
    }

    /// <summary>lambda * a + (1 - lambda) * b.</summary>
    private static Tensor<T> Mix(Tensor<T> a, Tensor<T> b, T lambda)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorAdd(
            engine.TensorMultiplyScalar(a, lambda),
            engine.TensorMultiplyScalar(b, NumOps.Subtract(NumOps.One, lambda)));
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
