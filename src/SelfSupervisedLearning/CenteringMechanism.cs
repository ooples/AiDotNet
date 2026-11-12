using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

using AiDotNet.Models.Parameters;

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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Emerging Properties in Self-Supervised Vision Transformers", "https://arxiv.org/abs/2104.14294", Year = 2021, Authors = "Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin")]
public partial class CenteringMechanism<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{

    /// <inheritdoc />
    /// <remarks>The running centre this mechanism maintains, which is a genuine learned buffer: DINO-style centering subtracts it from teacher logits, so a checkpoint that dropped it would restore a differently-behaved model.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new DelegatingParameterSource<T>(
            () => _dimension,
            () => new Vector<T>(_center),
            value =>
            {
                // The hand-written restore copied min(length, _dimension) so a short vector was
                // tolerated. The base fold now rejects a wrong length outright, and this stays
                // defensive for the equal-length case it is handed.
                int count = Math.Min(value.Length, _dimension);
                if (count > 0) value.AsSpan().Slice(0, count).CopyTo(_center.AsSpan(0, count));
            }));
    }
    // NumOps inherited from ModelBase

    private readonly int _dimension;
    private readonly double _momentum;
    private T[] _center;

    /// <summary>
    /// Gets the dimension of the center vector.
    /// </summary>
    public int Dimension => _dimension;

    /// <summary>
    /// Zero-copy rank-1 tensor view of the center buffer.
    /// </summary>
    private Tensor<T> CenterTensor => Vector<T>.Wrap(_center).AsTensor();

    /// <summary>
    /// Sums the first <see cref="Dimension"/> columns of a [batch, dim] tensor over the batch axis.
    /// </summary>
    private Tensor<T> ColumnSums(Tensor<T> batch)
    {
        var batchSize = batch.Shape[0];
        var columns = batch.Shape[1] == _dimension
            ? batch
            : Engine.TensorSlice(batch, new[] { 0, 0 }, new[] { batchSize, _dimension });
        return Engine.ReduceSum(columns, new[] { 0 }, keepDims: false);
    }

    /// <summary>
    /// EMA update of the center: center = m * center + (1 - m) * batchMean, written into the center buffer.
    /// </summary>
    private void BlendIntoCenter(Tensor<T> batchMean)
    {
        var momentum = NumOps.FromDouble(_momentum);
        var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);
        var blended = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(CenterTensor, momentum),
            Engine.TensorMultiplyScalar(batchMean, oneMinusMomentum));
        blended.AsSpan().CopyTo(_center);
    }

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

        // input[b, d] - center[d], the center broadcast over the batch
        var rows = input.Rank == 2 ? input : input.Reshape(batchSize, dim);
        return Engine.TensorSubtract(rows, CenterTensor.Reshape(1, dim));
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

        var invBatch = NumOps.FromDouble(1.0 / batchSize);

        // Batch mean per dimension, then EMA update: center = m * center + (1-m) * batch_mean
        var batchMean = Engine.TensorMultiplyScalar(ColumnSums(batchOutput), invBatch);
        BlendIntoCenter(batchMean);
    }

    /// <summary>
    /// Updates the center using multiple batches of outputs.
    /// </summary>
    /// <param name="outputs">List of output tensors.</param>
    public void UpdateFromMultiple(IList<Tensor<T>> outputs)
    {
        if (outputs is null || outputs.Count == 0)
            throw new ArgumentException("Must provide at least one output tensor", nameof(outputs));

        // Compute mean across all outputs
        Tensor<T>? columnTotals = null;
        int totalSamples = 0;

        foreach (var output in outputs)
        {
            totalSamples += output.Shape[0];
            var sums = ColumnSums(output);
            columnTotals = columnTotals is null ? sums : Engine.TensorAdd(columnTotals, sums);
        }

        var invTotal = NumOps.FromDouble(1.0 / totalSamples);
        var totals = columnTotals ?? new Tensor<T>(new[] { _dimension });
        BlendIntoCenter(Engine.TensorMultiplyScalar(totals, invTotal));
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
    public T[] GetCenter() => Vector<T>.Wrap(_center).ToArray();

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
        Engine.Fill(_dimension, NumOps.Zero).AsSpan().CopyTo(_center);
    }

    /// <summary>
    /// Computes the L2 norm of the center (useful for monitoring).
    /// </summary>
    public T CenterNorm()
    {
        T sumSquared = Engine.TensorSumOfSquares(CenterTensor);
        return NumOps.Sqrt(sumSquared);
    }

    /// <summary>
    /// Computes statistics about the center (useful for debugging).
    /// </summary>
    public (T mean, T std, T min, T max) CenterStatistics()
    {
        if (_dimension == 0)
            return (NumOps.Zero, NumOps.Zero, NumOps.Zero, NumOps.Zero);

        var center = CenterTensor;
        T min = Engine.TensorMinValue(center);
        T max = Engine.TensorMaxValue(center);
        T mean = NumOps.Divide(Engine.TensorSum(center), NumOps.FromDouble(_dimension));

        // Population variance of the center values
        var diff = Engine.TensorSubtractScalar(center, mean);
        T variance = NumOps.Divide(Engine.TensorSumOfSquares(diff), NumOps.FromDouble(_dimension));
        var std = NumOps.Sqrt(variance);

        return (mean, std, min, max);
    }

    #region ModelBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input) => ApplyCenter(input);

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput) => Update(input);

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        ((IParameterizable<T, Tensor<T>, Tensor<T>>)copy).SetParameters(parameters);
        return copy;
    }

    #endregion
}
