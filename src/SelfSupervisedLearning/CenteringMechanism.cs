using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Centering mechanism for preventing collapse in self-distillation methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Centering is a crucial technique in DINO and similar methods
/// that prevents the teacher network from collapsing to a trivial solution where it
/// outputs the same constant for all inputs.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="bullet">
/// <item>Maintains a running mean (center) of teacher outputs</item>
/// <item>Subtracts the center from teacher outputs before computing loss</item>
/// <item>Updates the center with exponential moving average (EMA)</item>
/// </list>
///
/// <para><b>Why it prevents collapse:</b> Without centering, the teacher could learn
/// to output a constant vector for all inputs (trivial solution). By subtracting the
/// running mean, we ensure the outputs are zero-centered on average, forcing the
/// network to produce varied outputs.</para>
///
/// <para><b>Reference:</b> Caron et al., "Emerging Properties in Self-Supervised Vision
/// Transformers" (ICCV 2021)</para>
/// </remarks>
public class CenteringMechanism<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _dimension;
    private readonly double _momentum;
    private T[] _center;

    /// <summary>
    /// Gets the dimension of the center vector.
    /// </summary>
    public int Dimension => _dimension;

    /// <summary>
    /// Gets the momentum for EMA updates.
    /// </summary>
    public double Momentum => _momentum;

    /// <summary>
    /// Initializes a new instance of the CenteringMechanism class.
    /// </summary>
    /// <param name="dimension">Dimension of the output space to center.</param>
    /// <param name="momentum">Momentum for EMA center updates (default: 0.9).</param>
    public CenteringMechanism(int dimension, double momentum = 0.9)
    {
        if (dimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be positive");
        if (momentum < 0 || momentum > 1)
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be between 0 and 1");

        _dimension = dimension;
        _momentum = momentum;
        _center = new T[dimension];

        Reset();
    }

    /// <summary>
    /// Applies centering to the input tensor.
    /// </summary>
    /// <param name="input">Input tensor [batch_size, dim].</param>
    /// <returns>Centered tensor.</returns>
    public Tensor<T> ApplyCenter(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var batchSize = input.Shape[0];
        var dim = input.Shape[1];

        if (dim != _dimension)
            throw new ArgumentException($"Input dimension {dim} doesn't match center dimension {_dimension}");

        var result = new T[batchSize * dim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                result[b * dim + d] = NumOps.Subtract(input[b, d], _center[d]);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }

    /// <summary>
    /// Updates the center using EMA with the given batch.
    /// </summary>
    /// <param name="batchOutput">Batch of outputs from teacher network [batch_size, dim].</param>
    public void Update(Tensor<T> batchOutput)
    {
        if (batchOutput is null) throw new ArgumentNullException(nameof(batchOutput));

        var batchSize = batchOutput.Shape[0];
        var dim = batchOutput.Shape[1];

        if (dim != _dimension)
            throw new ArgumentException($"Output dimension {dim} doesn't match center dimension {_dimension}");

        var momentum = NumOps.FromDouble(_momentum);
        var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);
        var invBatch = NumOps.FromDouble(1.0 / batchSize);

        for (int d = 0; d < dim; d++)
        {
            // Compute batch mean for this dimension
            T batchMean = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                batchMean = NumOps.Add(batchMean, batchOutput[b, d]);
            }
            batchMean = NumOps.Multiply(batchMean, invBatch);

            // EMA update: center = m * center + (1-m) * batch_mean
            _center[d] = NumOps.Add(
                NumOps.Multiply(momentum, _center[d]),
                NumOps.Multiply(oneMinusMomentum, batchMean));
        }
    }

    /// <summary>
    /// Updates the center using multiple batches of outputs.
    /// </summary>
    /// <param name="outputs">List of output tensors.</param>
    public void UpdateFromMultiple(IList<Tensor<T>> outputs)
    {
        if (outputs is null || outputs.Count == 0)
            throw new ArgumentException("Must provide at least one output tensor", nameof(outputs));

        var momentum = NumOps.FromDouble(_momentum);
        var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);

        // Compute mean across all outputs
        var meanValues = new T[_dimension];
        int totalSamples = 0;

        foreach (var output in outputs)
        {
            var batchSize = output.Shape[0];
            totalSamples += batchSize;

            for (int d = 0; d < _dimension; d++)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    meanValues[d] = NumOps.Add(meanValues[d], output[b, d]);
                }
            }
        }

        var invTotal = NumOps.FromDouble(1.0 / totalSamples);

        for (int d = 0; d < _dimension; d++)
        {
            var batchMean = NumOps.Multiply(meanValues[d], invTotal);

            _center[d] = NumOps.Add(
                NumOps.Multiply(momentum, _center[d]),
                NumOps.Multiply(oneMinusMomentum, batchMean));
        }
    }

    /// <summary>
    /// Applies centering and updates in one step (common usage pattern).
    /// </summary>
    /// <param name="teacherOutput">Teacher network output.</param>
    /// <returns>Centered output.</returns>
    public Tensor<T> CenterAndUpdate(Tensor<T> teacherOutput)
    {
        var centered = ApplyCenter(teacherOutput);
        Update(teacherOutput);
        return centered;
    }

    /// <summary>
    /// Gets the current center values.
    /// </summary>
    /// <returns>Copy of the center vector.</returns>
    public T[] GetCenter() => (T[])_center.Clone();

    /// <summary>
    /// Sets the center values directly.
    /// </summary>
    /// <param name="center">New center values.</param>
    public void SetCenter(T[] center)
    {
        if (center is null) throw new ArgumentNullException(nameof(center));
        if (center.Length != _dimension)
            throw new ArgumentException($"Center length {center.Length} doesn't match dimension {_dimension}");

        Array.Copy(center, _center, _dimension);
    }

    /// <summary>
    /// Resets the center to zeros.
    /// </summary>
    public void Reset()
    {
        for (int i = 0; i < _dimension; i++)
        {
            _center[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Computes the L2 norm of the center (useful for monitoring).
    /// </summary>
    public T CenterNorm()
    {
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < _dimension; i++)
        {
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(_center[i], _center[i]));
        }
        return NumOps.Sqrt(sumSquared);
    }

    /// <summary>
    /// Computes statistics about the center (useful for debugging).
    /// </summary>
    public (T mean, T std, T min, T max) CenterStatistics()
    {
        if (_dimension == 0)
            return (NumOps.Zero, NumOps.Zero, NumOps.Zero, NumOps.Zero);

        T mean = NumOps.Zero;
        T min = _center[0];
        T max = _center[0];

        for (int i = 0; i < _dimension; i++)
        {
            mean = NumOps.Add(mean, _center[i]);
            if (NumOps.GreaterThan(min, _center[i])) min = _center[i];
            if (NumOps.GreaterThan(_center[i], max)) max = _center[i];
        }
        mean = NumOps.Divide(mean, NumOps.FromDouble(_dimension));

        T variance = NumOps.Zero;
        for (int i = 0; i < _dimension; i++)
        {
            var diff = NumOps.Subtract(_center[i], mean);
            variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(_dimension));
        var std = NumOps.Sqrt(variance);

        return (mean, std, min, max);
    }
}
