using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Entmoid activation function for NODE architecture.
/// </summary>
/// <remarks>
/// <para>
/// Entmoid is a sparse, differentiable activation that generalizes softmax to produce
/// sparse outputs. It's the element-wise version of entmax (sparse attention).
/// The function is defined as: entmoid(x) = max(0, (alpha * x + 1) / (2 * alpha))^(1/(alpha-1))
/// </para>
/// <para>
/// <b>For Beginners:</b> Entmoid is like a "smart sigmoid":
/// - Regular sigmoid smoothly goes from 0 to 1
/// - Entmoid can produce exact zeros (sparse output)
/// - This helps the model focus on important features and ignore noise
///
/// The alpha parameter controls sparsity:
/// - alpha = 1.5 (default): moderately sparse
/// - alpha = 2: sparsemax (can produce exact zeros)
/// - alpha → 1: approaches softmax (no sparsity)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class EntmoidActivation<T> : ActivationFunctionBase<T>
{
    private readonly double _alpha;
    private readonly T _alphaT;
    private readonly T _alphaMinusOneInverse;
    private readonly T _twoAlpha;

    /// <summary>
    /// Gets the alpha parameter controlling sparsity.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Initializes the entmoid activation.
    /// </summary>
    /// <param name="alpha">
    /// Alpha parameter controlling sparsity (default 1.5).
    /// - alpha = 1.5: moderately sparse (good default for NODE)
    /// - alpha = 2.0: sparsemax behavior
    /// - alpha closer to 1: less sparse, more like softmax
    /// </param>
    public EntmoidActivation(double alpha = 1.5)
    {
        if (alpha <= 1.0)
        {
            throw new ArgumentException("Alpha must be greater than 1", nameof(alpha));
        }

        _alpha = alpha;
        _alphaT = NumOps.FromDouble(alpha);
        _alphaMinusOneInverse = NumOps.FromDouble(1.0 / (alpha - 1.0));
        _twoAlpha = NumOps.FromDouble(2.0 * alpha);
    }

    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    protected override bool SupportsScalarOperations() => true;

    /// <summary>
    /// Applies the entmoid activation function to a single input value.
    /// </summary>
    /// <param name="input">The input value to activate.</param>
    /// <returns>The activated output value.</returns>
    public override T Activate(T input)
    {
        // entmoid(x) = max(0, (alpha * x + 1) / (2 * alpha))^(1/(alpha-1))
        var scaledInput = NumOps.Multiply(_alphaT, input);
        var numerator = NumOps.Add(scaledInput, NumOps.One);
        var fraction = NumOps.Divide(numerator, _twoAlpha);

        // Clamp to [0, 1] before power operation
        if (NumOps.Compare(fraction, NumOps.Zero) <= 0)
        {
            return NumOps.Zero;
        }
        else if (NumOps.Compare(fraction, NumOps.One) >= 0)
        {
            return NumOps.One;
        }
        else
        {
            // Apply power: x^(1/(alpha-1))
            var fractionDouble = NumOps.ToDouble(fraction);
            var exponentDouble = 1.0 / (_alpha - 1.0);
            return NumOps.FromDouble(Math.Pow(fractionDouble, exponentDouble));
        }
    }

    /// <summary>
    /// Computes the derivative of the entmoid activation.
    /// </summary>
    /// <param name="input">Input value.</param>
    /// <returns>Derivative value.</returns>
    public override T Derivative(T input)
    {
        var scaledInput = NumOps.Multiply(_alphaT, input);
        var numerator = NumOps.Add(scaledInput, NumOps.One);
        var fraction = NumOps.Divide(numerator, _twoAlpha);

        // Derivative is 0 outside [0, 1] region
        if (NumOps.Compare(fraction, NumOps.Zero) <= 0 ||
            NumOps.Compare(fraction, NumOps.One) >= 0)
        {
            return NumOps.Zero;
        }
        else
        {
            // d/dx entmoid(x) = (alpha / (2 * alpha * (alpha - 1))) * (fraction)^((2-alpha)/(alpha-1))
            var derivScale = 1.0 / (2.0 * (_alpha - 1.0));
            var exponent = (2.0 - _alpha) / (_alpha - 1.0);
            var fractionDouble = NumOps.ToDouble(fraction);
            var basePow = Math.Pow(fractionDouble, exponent);
            return NumOps.FromDouble(derivScale * basePow);
        }
    }

    /// <summary>
    /// Applies the entmoid activation function to a vector.
    /// </summary>
    /// <param name="input">Input vector.</param>
    /// <returns>Activated vector.</returns>
    public override Vector<T> Activate(Vector<T> input)
    {
        return input.Transform(Activate);
    }

    /// <summary>
    /// Computes the Jacobian matrix of entmoid for a vector.
    /// </summary>
    /// <param name="input">Input vector.</param>
    /// <returns>Diagonal Jacobian matrix.</returns>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        // Entmoid is element-wise, so its Jacobian is diagonal.
        return Matrix<T>.CreateDiagonal(Derivative(Tensor<T>.FromVector(input)).ToVector());
    }

    /// <summary>
    /// Applies the entmoid activation function to a tensor.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Activated tensor.</returns>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        var engine = AiDotNetEngine.Current;

        // Fresh, detached output as before. fraction = (alpha * x + 1) / (2 * alpha); values at or
        // below 0 (and NaN) map to 0, values at or above 1 to 1, the rest to fraction^(1/(alpha-1)).
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var fraction = engine.TensorDivideScalar(
                engine.TensorAddScalar(engine.TensorMultiplyScalar(input, _alphaT), NumOps.One), _twoAlpha);
            var powered = engine.TensorPower(engine.TensorClamp(fraction, NumOps.Zero, NumOps.One), _alphaMinusOneInverse);
            return engine.TensorWhere(engine.TensorGreaterThan(fraction, NumOps.Zero), powered, new Tensor<T>(input._shape));
        }
    }

    /// <summary>
    /// Computes the derivative of entmoid for a tensor.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Derivative tensor.</returns>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        var engine = AiDotNetEngine.Current;

        // Fresh, detached output as before: 1 / (2 (alpha - 1)) * fraction^((2 - alpha) / (alpha - 1))
        // strictly inside (0, 1), 0 elsewhere (and for NaN).
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var fraction = engine.TensorDivideScalar(
                engine.TensorAddScalar(engine.TensorMultiplyScalar(input, _alphaT), NumOps.One), _twoAlpha);
            var inside = engine.TensorMultiply(
                engine.TensorGreaterThan(fraction, NumOps.Zero), engine.TensorLessThan(fraction, NumOps.One));
            var safeFraction = engine.TensorWhere(inside, fraction, engine.TensorAddScalar(new Tensor<T>(input._shape), NumOps.One));
            var slope = engine.TensorMultiplyScalar(
                engine.TensorPower(safeFraction, NumOps.FromDouble((2.0 - _alpha) / (_alpha - 1.0))),
                NumOps.FromDouble(1.0 / (2.0 * (_alpha - 1.0))));
            return engine.TensorWhere(inside, slope, new Tensor<T>(input._shape));
        }
    }

    /// <summary>
    /// Applies this activation function to a computation graph node.
    /// </summary>
    /// <param name="input">The computation node to apply the activation to.</param>
    /// <returns>A new computation node with entmoid activation applied.</returns>
    public override ComputationNode<T> ApplyToGraph(ComputationNode<T> input)
    {
        // Entmoid is not yet supported in computation graphs
        // Fall back to element-wise application
        throw new NotSupportedException("Entmoid activation does not yet support JIT compilation.");
    }

    /// <summary>
    /// Gets whether entmoid supports GPU-resident training.
    /// </summary>
    public override bool SupportsGpuTraining => false;

    /// <summary>
    /// Applies the entmoid activation function on GPU.
    /// </summary>
    public override void ForwardGpu(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer output, int size)
    {
        throw new NotSupportedException("Entmoid activation does not yet support GPU training.");
    }
}

/// <summary>
/// Entmax sparse attention function for NODE architecture.
/// </summary>
/// <remarks>
/// <para>
/// Entmax is a sparse alternative to softmax that can produce exact zeros in the output
/// distribution. This is useful for attention mechanisms where you want to focus on
/// only a few important elements.
/// </para>
/// <para>
/// <b>For Beginners:</b> Entmax is like softmax but can completely ignore some inputs:
/// - Softmax: Every input gets some attention (even if tiny)
/// - Entmax: Unimportant inputs get exactly zero attention
///
/// This helps the model focus better and makes the attention patterns easier to interpret.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Optimizer)]
[PipelineStage(PipelineStage.Training)]
public class EntmaxAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _alpha;

    /// <summary>
    /// Initializes entmax attention.
    /// </summary>
    /// <param name="alpha">Alpha parameter (default 1.5, use 2.0 for sparsemax).</param>
    public EntmaxAttention(double alpha = 1.5)
    {
        if (alpha < 1.0)
        {
            throw new ArgumentException("Alpha must be >= 1", nameof(alpha));
        }
        _alpha = alpha;
    }

    /// <summary>
    /// Applies entmax to convert scores to a sparse probability distribution.
    /// </summary>
    /// <param name="scores">Input scores [batchSize, sequenceLength].</param>
    /// <returns>Sparse attention weights summing to 1.</returns>
    public Tensor<T> Forward(Tensor<T> scores)
    {
        int batchSize = scores.Shape[0];
        int seqLen = scores.Shape[1];

        // Special case: alpha = 1 is softmax
        if (Math.Abs(_alpha - 1.0) < 1e-6)
        {
            return ApplySoftmax(scores, batchSize, seqLen);
        }

        // Special case: alpha = 2 is sparsemax
        if (Math.Abs(_alpha - 2.0) < 1e-6)
        {
            return ApplySparsemax(scores, batchSize, seqLen);
        }

        // General entmax via bisection (slower but works for any alpha > 1)
        return ApplyEntmaxBisection(scores, batchSize, seqLen);
    }

    private Tensor<T> ApplySoftmax(Tensor<T> scores, int batchSize, int seqLen)
    {
        var reshaped = scores.Reshape([batchSize, seqLen]);
        var result = AiDotNetEngine.Current.Softmax(reshaped, -1);
        return result.Reshape(scores._shape);
    }

    private Tensor<T> ApplySparsemax(Tensor<T> scores, int batchSize, int seqLen)
    {
        var engine = AiDotNetEngine.Current;

        // Fresh, detached output as before. Per row: sort descending, threshold_k = (cumsum_k - 1) / k,
        // tau = the threshold at the last k with sorted_k > threshold_k (0 if none), p = max(0, z - tau).
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var z = engine.Reshape(scores, new[] { batchSize, seqLen });
            var sorted = engine.TensorSort(z, 1, true).Item1;
            var k = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorArange<T>(1, seqLen + 1, 1), new[] { 1, seqLen }), new[] { batchSize, seqLen });
            var threshold = engine.TensorDivide(engine.TensorSubtractScalar(engine.TensorCumSum(sorted, 1), NumOps.One), k);
            var inSupport = engine.TensorGreaterThan(sorted, threshold);
            var supportRank = engine.TensorMultiply(inSupport, k);
            var lastRank = engine.TensorBroadcastTo(engine.ReduceMax(supportRank, new[] { 1 }, true), new[] { batchSize, seqLen });
            var isLast = engine.TensorMultiply(inSupport, engine.TensorEquals(supportRank, lastRank));
            var tau = engine.ReduceSum(engine.TensorMultiply(threshold, isLast), new[] { 1 }, true);

            var shifted = engine.TensorSubtract(z, engine.TensorBroadcastTo(tau, new[] { batchSize, seqLen }));
            var output = engine.TensorWhere(engine.TensorGreaterThan(shifted, NumOps.Zero), shifted, new Tensor<T>(new[] { batchSize, seqLen }));
            return engine.Reshape(output, scores._shape);
        }
    }

    private Tensor<T> ApplyEntmaxBisection(Tensor<T> scores, int batchSize, int seqLen)
    {
        var engine = AiDotNetEngine.Current;
        var alphaMinusOne = NumOps.FromDouble(_alpha - 1.0);
        var invAlphaMinusOne = NumOps.FromDouble(1.0 / (_alpha - 1.0));
        var half = NumOps.FromDouble(0.5);
        var rowShape = new[] { batchSize, seqLen };

        // Fresh, detached output as before. entmax_alpha(z) = [(alpha - 1) z - tau]_+^(1/(alpha-1))
        // (Peters et al., 2019, eq. 9), with tau found per row by bisection so the output sums to 1.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var z = engine.TensorMultiplyScalar(engine.Reshape(scores, rowShape), alphaMinusOne);
            var zeros = new Tensor<T>(rowShape);

            Tensor<T> Powered(Tensor<T> tau)
            {
                var diff = engine.TensorSubtract(z, engine.TensorBroadcastTo(tau, rowShape));
                var positive = engine.TensorWhere(engine.TensorGreaterThan(diff, NumOps.Zero), diff, zeros);
                return engine.TensorPower(positive, invAlphaMinusOne);
            }

            // tau lies in [min - 1, max] for every row.
            var tauHigh = engine.ReduceMax(z, new[] { 1 }, true);
            var tauLow = engine.TensorSubtractScalar(
                engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(z), new[] { 1 }, true)), NumOps.One);

            for (int iter = 0; iter < 50; iter++)
            {
                var tauMid = engine.TensorMultiplyScalar(engine.TensorAdd(tauLow, tauHigh), half);
                var sum = engine.ReduceSum(Powered(tauMid), new[] { 1 }, true);
                var tooLow = engine.TensorGreaterThan(sum, NumOps.One);
                tauLow = engine.TensorWhere(tooLow, tauMid, tauLow);
                tauHigh = engine.TensorWhere(tooLow, tauHigh, tauMid);
            }

            var tauFinal = engine.TensorMultiplyScalar(engine.TensorAdd(tauLow, tauHigh), half);
            return engine.Reshape(Powered(tauFinal), scores._shape);
        }
    }
}
