using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
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
        int n = input.Length;
        Matrix<T> jacobian = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            jacobian[i, i] = Derivative(input[i]);
        }

        return jacobian;
    }

    /// <summary>
    /// Applies the entmoid activation function to a tensor.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Activated tensor.</returns>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            output[i] = Activate(input[i]);
        }

        return output;
    }

    /// <summary>
    /// Computes the derivative of entmoid for a tensor.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Derivative tensor.</returns>
    public override Tensor<T> Derivative(Tensor<T> input)
    {
        var derivative = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            derivative[i] = Derivative(input[i]);
        }

        return derivative;
    }

    /// <summary>
    /// Computes the backward pass gradient for entmoid.
    /// </summary>
    /// <param name="input">Input tensor from forward pass.</param>
    /// <param name="outputGradient">Gradient flowing back from next layer.</param>
    /// <returns>Gradient with respect to input.</returns>
    public override Tensor<T> Backward(Tensor<T> input, Tensor<T> outputGradient)
    {
        var deriv = Derivative(input);
        var result = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.Multiply(deriv[i], outputGradient[i]);
        }

        return result;
    }

    /// <summary>
    /// Gets whether this activation function supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => false;

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

    /// <summary>
    /// Calculates the entmoid backward pass gradient on GPU.
    /// </summary>
    public override void BackwardGpu(IDirectGpuBackend backend, IGpuBuffer gradOutput, IGpuBuffer? input, IGpuBuffer? output, IGpuBuffer gradInput, int size)
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
        var output = new Tensor<T>(scores.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Find max for numerical stability
            var maxVal = scores[b * seqLen];
            for (int i = 1; i < seqLen; i++)
            {
                if (NumOps.Compare(scores[b * seqLen + i], maxVal) > 0)
                    maxVal = scores[b * seqLen + i];
            }

            // Compute exp and sum
            var sumExp = NumOps.Zero;
            for (int i = 0; i < seqLen; i++)
            {
                output[b * seqLen + i] = NumOps.Exp(NumOps.Subtract(scores[b * seqLen + i], maxVal));
                sumExp = NumOps.Add(sumExp, output[b * seqLen + i]);
            }

            // Normalize
            for (int i = 0; i < seqLen; i++)
            {
                output[b * seqLen + i] = NumOps.Divide(output[b * seqLen + i], sumExp);
            }
        }

        return output;
    }

    private Tensor<T> ApplySparsemax(Tensor<T> scores, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(scores.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Sort scores in descending order
            var sortedScores = new T[seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                sortedScores[i] = scores[b * seqLen + i];
            }
            Array.Sort(sortedScores, (a, c) => NumOps.Compare(c, a));

            // Find threshold tau using cumulative sum
            var cumSum = NumOps.Zero;
            var tau = NumOps.Zero;

            for (int i = 0; i < seqLen; i++)
            {
                cumSum = NumOps.Add(cumSum, sortedScores[i]);
                var threshold = NumOps.Divide(
                    NumOps.Subtract(cumSum, NumOps.One),
                    NumOps.FromDouble(i + 1));

                if (NumOps.Compare(sortedScores[i], threshold) > 0)
                {
                    tau = threshold;
                }
            }

            // Apply sparsemax: max(0, x - tau)
            for (int i = 0; i < seqLen; i++)
            {
                var val = NumOps.Subtract(scores[b * seqLen + i], tau);
                output[b * seqLen + i] = NumOps.Compare(val, NumOps.Zero) > 0 ? val : NumOps.Zero;
            }
        }

        return output;
    }

    private Tensor<T> ApplyEntmaxBisection(Tensor<T> scores, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(scores.Shape);
        var alphaMinusOne = _alpha - 1.0;
        var invAlphaMinusOne = 1.0 / alphaMinusOne;

        for (int b = 0; b < batchSize; b++)
        {
            // Find threshold via bisection
            var maxScore = scores[b * seqLen];
            var minScore = scores[b * seqLen];
            for (int i = 1; i < seqLen; i++)
            {
                if (NumOps.Compare(scores[b * seqLen + i], maxScore) > 0)
                    maxScore = scores[b * seqLen + i];
                if (NumOps.Compare(scores[b * seqLen + i], minScore) < 0)
                    minScore = scores[b * seqLen + i];
            }

            var tauLow = NumOps.ToDouble(minScore) - 1.0;
            var tauHigh = NumOps.ToDouble(maxScore);

            // Bisection to find tau such that sum of p = 1
            for (int iter = 0; iter < 50; iter++)
            {
                var tauMid = (tauLow + tauHigh) / 2.0;
                var sum = 0.0;

                for (int i = 0; i < seqLen; i++)
                {
                    var diff = NumOps.ToDouble(scores[b * seqLen + i]) - tauMid;
                    if (diff > 0)
                    {
                        sum += Math.Pow(diff, invAlphaMinusOne);
                    }
                }

                if (sum > 1.0)
                    tauLow = tauMid;
                else
                    tauHigh = tauMid;
            }

            var tau = (tauLow + tauHigh) / 2.0;

            // Compute output
            for (int i = 0; i < seqLen; i++)
            {
                var diff = NumOps.ToDouble(scores[b * seqLen + i]) - tau;
                if (diff > 0)
                {
                    output[b * seqLen + i] = NumOps.FromDouble(Math.Pow(diff, invAlphaMinusOne));
                }
                else
                {
                    output[b * seqLen + i] = NumOps.Zero;
                }
            }
        }

        return output;
    }
}
