namespace AiDotNet.Initialization;

/// <summary>
/// Orthogonal initialization strategy for RNNs, LSTMs, and deep networks.
/// </summary>
/// <remarks>
/// <para>
/// Orthogonal initialization creates a random orthogonal matrix, which preserves
/// gradient norms across layers. This prevents vanishing/exploding gradients in
/// deep networks and recurrent architectures.
/// </para>
/// <para><b>For Beginners:</b> Use this for RNNs, LSTMs, and very deep networks.
/// It helps gradients flow smoothly through many layers without vanishing or exploding.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OrthogonalInitializationStrategy<T> : InitializationStrategyBase<T>
{
    private readonly double _gain;

    /// <summary>
    /// Creates an orthogonal initialization strategy.
    /// </summary>
    /// <param name="gain">Scaling factor. Default 1.0. Use sqrt(2) for ReLU.</param>
    public OrthogonalInitializationStrategy(double gain = 1.0)
    {
        _gain = gain;
    }

    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        // Generate random matrix and orthogonalize via Gram-Schmidt
        int rows = Math.Max(inputSize, outputSize);
        int cols = Math.Min(inputSize, outputSize);
        var flat = new double[rows * cols];
        for (int i = 0; i < flat.Length; i++)
            flat[i] = SampleGaussian(0, 1.0);

        // Simple Gram-Schmidt orthogonalization
        for (int c = 0; c < cols; c++)
        {
            // Subtract projections onto previous columns
            for (int prev = 0; prev < c; prev++)
            {
                double dot = 0, normSq = 0;
                for (int r = 0; r < rows; r++)
                {
                    dot += flat[r * cols + c] * flat[r * cols + prev];
                    normSq += flat[r * cols + prev] * flat[r * cols + prev];
                }
                if (normSq > 1e-10)
                {
                    double scale = dot / normSq;
                    for (int r = 0; r < rows; r++)
                        flat[r * cols + c] -= scale * flat[r * cols + prev];
                }
            }

            // Normalize column
            double colNorm = 0;
            for (int r = 0; r < rows; r++)
                colNorm += flat[r * cols + c] * flat[r * cols + c];
            colNorm = Math.Sqrt(colNorm);
            if (colNorm > 1e-10)
            {
                for (int r = 0; r < rows; r++)
                    flat[r * cols + c] *= _gain / colNorm;
            }
        }

        // Copy to tensor
        var span = weights.AsWritableSpan();
        int len = Math.Min(span.Length, flat.Length);
        for (int i = 0; i < len; i++)
            span[i] = NumOps.FromDouble(flat[i]);
        // Zero any remaining elements
        for (int i = len; i < span.Length; i++)
            span[i] = NumOps.Zero;
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
