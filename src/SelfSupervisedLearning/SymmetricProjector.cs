using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Symmetric Projector Head for BYOL and SimSiam-style methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The symmetric projector is used in BYOL and SimSiam.
/// It consists of a projector MLP followed by a predictor MLP. The predictor
/// creates asymmetry between online and target branches, which is key to avoiding collapse.</para>
///
/// <para><b>Architecture:</b></para>
/// <list type="bullet">
/// <item><b>Projector:</b> Linear → BN → ReLU → Linear → BN</item>
/// <item><b>Predictor:</b> Linear → BN → ReLU → Linear</item>
/// </list>
///
/// <para><b>Key insight:</b> The predictor is only applied to the online branch,
/// creating asymmetry. The target branch only uses the projector.</para>
/// </remarks>
public class SymmetricProjector<T> : IProjectorHead<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _projectionDim;
    private readonly int _predictorHiddenDim;
    private readonly bool _hasPredictor;

    // Projector parameters
    private T[] _projWeight1;
    private T[] _projBias1;
    private T[] _projBn1Gamma;
    private T[] _projBn1Beta;
    private T[] _projWeight2;
    private T[] _projBias2;
    private T[] _projBn2Gamma;
    private T[] _projBn2Beta;

    // Predictor parameters (optional)
    private T[]? _predWeight1;
    private T[]? _predBias1;
    private T[]? _predBn1Gamma;
    private T[]? _predBn1Beta;
    private T[]? _predWeight2;
    private T[]? _predBias2;

    // Cached activations for backward pass
    private Tensor<T>? _cachedInput;
    private Tensor<T>? _cachedH1;
    private Tensor<T>? _cachedH1Bn;
    private Tensor<T>? _cachedH1Relu;
    private Tensor<T>? _cachedProjection;
    private Tensor<T>? _cachedPredH1;
    private Tensor<T>? _cachedPredH1Bn;
    private Tensor<T>? _cachedPredH1Relu;
    private Vector<T>? _gradients;

    // BatchNorm cached statistics for full backward pass
    private T[]? _projBn1Mean;
    private T[]? _projBn1Var;
    private Tensor<T>? _projBn1Normalized;
    private T[]? _projBn2Mean;
    private T[]? _projBn2Var;
    private Tensor<T>? _projBn2Normalized;
    private T[]? _predBn1Mean;
    private T[]? _predBn1Var;
    private Tensor<T>? _predBn1Normalized;

    // BatchNorm gradients
    private T[]? _projBn1GammaGrad;
    private T[]? _projBn1BetaGrad;
    private T[]? _projBn2GammaGrad;
    private T[]? _projBn2BetaGrad;
    private T[]? _predBn1GammaGrad;
    private T[]? _predBn1BetaGrad;

    /// <inheritdoc />
    public int InputDimension => _inputDim;

    /// <inheritdoc />
    public int OutputDimension => _projectionDim;

    /// <inheritdoc />
    public int? HiddenDimension => _hiddenDim;

    /// <inheritdoc />
    public int ParameterCount => ComputeParameterCount();

    private bool _isTraining = true;

    /// <summary>
    /// Gets whether this projector has a predictor head.
    /// </summary>
    public bool HasPredictor => _hasPredictor;

    /// <summary>
    /// Initializes a new instance of the SymmetricProjector class.
    /// </summary>
    /// <param name="inputDim">Input dimension from encoder.</param>
    /// <param name="hiddenDim">Hidden dimension of the projector (default: 4096).</param>
    /// <param name="projectionDim">Output dimension (default: 256).</param>
    /// <param name="predictorHiddenDim">Hidden dimension of predictor (default: 4096). Set to 0 to disable predictor.</param>
    /// <param name="seed">Random seed for initialization.</param>
    public SymmetricProjector(
        int inputDim,
        int hiddenDim = 4096,
        int projectionDim = 256,
        int predictorHiddenDim = 4096,
        int? seed = null)
    {
        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _projectionDim = projectionDim;
        _predictorHiddenDim = predictorHiddenDim;
        _hasPredictor = predictorHiddenDim > 0;

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.Shared;

        // Initialize projector
        _projWeight1 = InitializeWeight(inputDim, hiddenDim, rng);
        _projBias1 = new T[hiddenDim];
        _projBn1Gamma = InitializeOnes(hiddenDim);
        _projBn1Beta = new T[hiddenDim];
        _projWeight2 = InitializeWeight(hiddenDim, projectionDim, rng);
        _projBias2 = new T[projectionDim];
        _projBn2Gamma = InitializeOnes(projectionDim);
        _projBn2Beta = new T[projectionDim];

        // Initialize predictor if needed
        if (_hasPredictor)
        {
            _predWeight1 = InitializeWeight(projectionDim, predictorHiddenDim, rng);
            _predBias1 = new T[predictorHiddenDim];
            _predBn1Gamma = InitializeOnes(predictorHiddenDim);
            _predBn1Beta = new T[predictorHiddenDim];
            _predWeight2 = InitializeWeight(predictorHiddenDim, projectionDim, rng);
            _predBias2 = new T[projectionDim];
        }
    }

    /// <inheritdoc />
    public Tensor<T> Project(Tensor<T> input)
    {
        _cachedInput = input;

        // Projector: Linear → BN → ReLU → Linear → BN
        _cachedH1 = Linear(input, _projWeight1, _projBias1, _inputDim, _hiddenDim);
        _cachedH1Bn = BatchNorm(_cachedH1, _projBn1Gamma, _projBn1Beta,
            out _projBn1Mean, out _projBn1Var, out _projBn1Normalized);
        _cachedH1Relu = ReLU(_cachedH1Bn);
        _cachedProjection = Linear(_cachedH1Relu, _projWeight2, _projBias2, _hiddenDim, _projectionDim);
        var projNorm = BatchNorm(_cachedProjection, _projBn2Gamma, _projBn2Beta,
            out _projBn2Mean, out _projBn2Var, out _projBn2Normalized);

        return projNorm;
    }

    /// <summary>
    /// Applies the predictor head (for online branch only).
    /// </summary>
    /// <param name="projection">Output from the projector.</param>
    /// <returns>Prediction output.</returns>
    public Tensor<T> Predict(Tensor<T> projection)
    {
        if (!_hasPredictor)
            return projection;

        // Predictor: Linear → BN → ReLU → Linear
        _cachedPredH1 = Linear(projection, _predWeight1!, _predBias1!, _projectionDim, _predictorHiddenDim);
        _cachedPredH1Bn = BatchNorm(_cachedPredH1, _predBn1Gamma!, _predBn1Beta!,
            out _predBn1Mean, out _predBn1Var, out _predBn1Normalized);
        _cachedPredH1Relu = ReLU(_cachedPredH1Bn);
        var output = Linear(_cachedPredH1Relu, _predWeight2!, _predBias2!, _predictorHiddenDim, _projectionDim);

        return output;
    }

    /// <summary>
    /// Projects and predicts in one call (convenience method).
    /// </summary>
    public Tensor<T> ProjectAndPredict(Tensor<T> input)
    {
        var projection = Project(input);
        return Predict(projection);
    }

    /// <inheritdoc />
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        // Backward through predictor if present
        if (_hasPredictor && _cachedPredH1Relu is not null)
        {
            // Backward through final linear layer of predictor
            grad = LinearBackward(grad, _predWeight2!, _predictorHiddenDim, _projectionDim);
            // Backward through ReLU
            grad = ReLUBackward(grad, _cachedPredH1Bn!);
            // Backward through BN with full gradient computation
            grad = BatchNormBackward(grad, _predBn1Gamma!, _predBn1Var, _predBn1Normalized,
                out _predBn1GammaGrad, out _predBn1BetaGrad);
            // Backward through first linear layer of predictor
            grad = LinearBackward(grad, _predWeight1!, _projectionDim, _predictorHiddenDim);
        }

        // Backward through projector BN2 with full gradient computation
        grad = BatchNormBackward(grad, _projBn2Gamma, _projBn2Var, _projBn2Normalized,
            out _projBn2GammaGrad, out _projBn2BetaGrad);
        // Backward through linear2
        grad = LinearBackward(grad, _projWeight2, _hiddenDim, _projectionDim);
        // Backward through ReLU
        grad = ReLUBackward(grad, _cachedH1Bn!);
        // Backward through BN1 with full gradient computation
        grad = BatchNormBackward(grad, _projBn1Gamma, _projBn1Var, _projBn1Normalized,
            out _projBn1GammaGrad, out _projBn1BetaGrad);
        // Backward through linear1
        grad = LinearBackward(grad, _projWeight1, _inputDim, _hiddenDim);

        // Compute and store parameter gradients
        _gradients = ComputeParameterGradients(gradOutput);

        return grad;
    }

    private Tensor<T> LinearBackward(Tensor<T> gradOutput, T[] weight, int inDim, int outDim)
    {
        var batchSize = gradOutput.Shape[0];
        var gradInput = new T[batchSize * inDim];

        // gradInput = gradOutput @ weight.T
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inDim; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < outDim; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(gradOutput[b, j], weight[i * outDim + j]));
                }
                gradInput[b * inDim + i] = sum;
            }
        }

        return new Tensor<T>(gradInput, [batchSize, inDim]);
    }

    private Tensor<T> ReLUBackward(Tensor<T> gradOutput, Tensor<T> preActivation)
    {
        var batchSize = gradOutput.Shape[0];
        var dim = gradOutput.Shape[1];
        var gradInput = new T[batchSize * dim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < dim; i++)
            {
                // Gradient is passed through only where input was positive
                var wasPositive = NumOps.GreaterThan(preActivation[b, i], NumOps.Zero);
                gradInput[b * dim + i] = wasPositive ? gradOutput[b, i] : NumOps.Zero;
            }
        }

        return new Tensor<T>(gradInput, [batchSize, dim]);
    }

    /// <summary>
    /// Full BatchNorm backward pass computing gradients for input, gamma, and beta.
    /// </summary>
    /// <remarks>
    /// The full BatchNorm backward follows these equations:
    /// dx = (gamma / std) * (dout - mean(dout) - xhat * mean(dout * xhat))
    /// dgamma = sum(dout * xhat, axis=batch)
    /// dbeta = sum(dout, axis=batch)
    /// where xhat is the normalized input and std = sqrt(var + eps)
    /// </remarks>
    private Tensor<T> BatchNormBackward(
        Tensor<T> gradOutput,
        T[] gamma,
        T[]? variance,
        Tensor<T>? normalizedInput,
        out T[] gammaGrad,
        out T[] betaGrad)
    {
        var batchSize = gradOutput.Shape[0];
        var dim = gradOutput.Shape[1];
        var gradInput = new T[batchSize * dim];
        gammaGrad = new T[dim];
        betaGrad = new T[dim];

        var eps = NumOps.FromDouble(1e-5);
        var invN = NumOps.FromDouble(1.0 / batchSize);

        for (int j = 0; j < dim; j++)
        {
            // Compute std = sqrt(variance + eps)
            var std = variance != null
                ? NumOps.Sqrt(NumOps.Add(variance[j], eps))
                : NumOps.One;
            var invStd = NumOps.Divide(NumOps.One, std);

            // Compute dgamma = sum(dout * xhat, axis=batch)
            // Compute dbeta = sum(dout, axis=batch)
            T dgamma = NumOps.Zero;
            T dbeta = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                var dout = gradOutput[b, j];
                var xhat = normalizedInput != null ? normalizedInput[b, j] : NumOps.Zero;
                dgamma = NumOps.Add(dgamma, NumOps.Multiply(dout, xhat));
                dbeta = NumOps.Add(dbeta, dout);
            }
            gammaGrad[j] = dgamma;
            betaGrad[j] = dbeta;

            // Compute mean(dout) and mean(dout * xhat)
            T meanDout = NumOps.Multiply(dbeta, invN);
            T meanDoutXhat = NumOps.Multiply(dgamma, invN);

            // Compute gradInput for each sample:
            // dx = (gamma / std) * (dout - mean(dout) - xhat * mean(dout * xhat))
            var gammaOverStd = NumOps.Multiply(gamma[j], invStd);
            for (int b = 0; b < batchSize; b++)
            {
                var dout = gradOutput[b, j];
                var xhat = normalizedInput != null ? normalizedInput[b, j] : NumOps.Zero;

                // dout - mean(dout) - xhat * mean(dout * xhat)
                var term = NumOps.Subtract(dout, meanDout);
                term = NumOps.Subtract(term, NumOps.Multiply(xhat, meanDoutXhat));

                gradInput[b * dim + j] = NumOps.Multiply(gammaOverStd, term);
            }
        }

        return new Tensor<T>(gradInput, [batchSize, dim]);
    }

    private Vector<T> ComputeParameterGradients(Tensor<T> gradOutput)
    {
        var grads = new T[ParameterCount];
        var batchSize = gradOutput.Shape[0];
        var invBatchSize = NumOps.FromDouble(1.0 / batchSize);
        int offset = 0;

        // If we don't have cached activations, we can't compute proper gradients
        if (_cachedInput is null || _cachedH1Relu is null || _cachedProjection is null)
        {
            return new Vector<T>(grads);
        }

        // Backpropagate gradOutput through BN2 to get gradients at projection output
        // Note: BN gradients are already computed in Backward() and stored in class fields
        var gradBeforeBn2 = BatchNormBackward(gradOutput, _projBn2Gamma, _projBn2Var, _projBn2Normalized,
            out _, out _);

        // Compute gradients for projWeight2: _cachedH1Relu.T @ gradBeforeBn2
        // dL/dW2[i,j] = sum_b(_cachedH1Relu[b,i] * gradBeforeBn2[b,j])
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _projectionDim; j++)
            {
                T sum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_cachedH1Relu![b, i], gradBeforeBn2[b, j]));
                }
                grads[offset + _projWeight1.Length + _projBias1.Length + _projBn1Gamma.Length + _projBn1Beta.Length + i * _projectionDim + j] =
                    NumOps.Multiply(sum, invBatchSize);
            }
        }

        // Compute gradients for projBias2: sum of gradBeforeBn2 across batch
        int bias2Offset = offset + _projWeight1.Length + _projBias1.Length + _projBn1Gamma.Length + _projBn1Beta.Length + _projWeight2.Length;
        for (int j = 0; j < _projectionDim; j++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, gradBeforeBn2[b, j]);
            }
            grads[bias2Offset + j] = NumOps.Multiply(sum, invBatchSize);
        }

        // Backprop through linear2 to get gradients at H1Relu
        var gradAtH1Relu = LinearBackward(gradBeforeBn2, _projWeight2, _hiddenDim, _projectionDim);

        // Backprop through ReLU
        var gradAtH1Bn = ReLUBackward(gradAtH1Relu, _cachedH1Bn!);

        // Backprop through BN1
        var gradAtH1 = BatchNormBackward(gradAtH1Bn, _projBn1Gamma, _projBn1Var, _projBn1Normalized,
            out _, out _);

        // Compute gradients for projWeight1: _cachedInput.T @ gradAtH1
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                T sum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_cachedInput![b, i], gradAtH1[b, j]));
                }
                grads[offset + i * _hiddenDim + j] = NumOps.Multiply(sum, invBatchSize);
            }
        }

        // Compute gradients for projBias1
        int bias1Offset = offset + _projWeight1.Length;
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, gradAtH1[b, j]);
            }
            grads[bias1Offset + j] = NumOps.Multiply(sum, invBatchSize);
        }

        // Use the properly computed BN gradients from BatchNormBackward
        int bn1GammaOffset = bias1Offset + _projBias1.Length;
        int bn1BetaOffset = bn1GammaOffset + _projBn1Gamma.Length;
        if (_projBn1GammaGrad != null && _projBn1BetaGrad != null)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                grads[bn1GammaOffset + j] = _projBn1GammaGrad[j];
                grads[bn1BetaOffset + j] = _projBn1BetaGrad[j];
            }
        }

        int bn2GammaOffset = bias2Offset + _projBias2.Length;
        int bn2BetaOffset = bn2GammaOffset + _projBn2Gamma.Length;
        if (_projBn2GammaGrad != null && _projBn2BetaGrad != null)
        {
            for (int j = 0; j < _projectionDim; j++)
            {
                grads[bn2GammaOffset + j] = _projBn2GammaGrad[j];
                grads[bn2BetaOffset + j] = _projBn2BetaGrad[j];
            }
        }

        // Add predictor BN gradients if predictor is present
        if (_hasPredictor && _predBn1GammaGrad != null && _predBn1BetaGrad != null)
        {
            // Calculate predictor offset
            int predOffset = bn2BetaOffset + _projBn2Beta.Length;
            int predWeight1Offset = predOffset;
            int predBias1Offset = predWeight1Offset + _predWeight1!.Length;
            int predBn1GammaOffset = predBias1Offset + _predBias1!.Length;
            int predBn1BetaOffset = predBn1GammaOffset + _predBn1Gamma!.Length;

            for (int j = 0; j < _predictorHiddenDim; j++)
            {
                grads[predBn1GammaOffset + j] = _predBn1GammaGrad[j];
                grads[predBn1BetaOffset + j] = _predBn1BetaGrad[j];
            }
        }

        return new Vector<T>(grads);
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Projector parameters
        allParams.AddRange(_projWeight1);
        allParams.AddRange(_projBias1);
        allParams.AddRange(_projBn1Gamma);
        allParams.AddRange(_projBn1Beta);
        allParams.AddRange(_projWeight2);
        allParams.AddRange(_projBias2);
        allParams.AddRange(_projBn2Gamma);
        allParams.AddRange(_projBn2Beta);

        // Predictor parameters
        if (_hasPredictor)
        {
            allParams.AddRange(_predWeight1!);
            allParams.AddRange(_predBias1!);
            allParams.AddRange(_predBn1Gamma!);
            allParams.AddRange(_predBn1Beta!);
            allParams.AddRange(_predWeight2!);
            allParams.AddRange(_predBias2!);
        }

        return new Vector<T>([.. allParams]);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Projector parameters
        Array.Copy(parameters.ToArray(), offset, _projWeight1, 0, _projWeight1.Length);
        offset += _projWeight1.Length;
        Array.Copy(parameters.ToArray(), offset, _projBias1, 0, _projBias1.Length);
        offset += _projBias1.Length;
        Array.Copy(parameters.ToArray(), offset, _projBn1Gamma, 0, _projBn1Gamma.Length);
        offset += _projBn1Gamma.Length;
        Array.Copy(parameters.ToArray(), offset, _projBn1Beta, 0, _projBn1Beta.Length);
        offset += _projBn1Beta.Length;
        Array.Copy(parameters.ToArray(), offset, _projWeight2, 0, _projWeight2.Length);
        offset += _projWeight2.Length;
        Array.Copy(parameters.ToArray(), offset, _projBias2, 0, _projBias2.Length);
        offset += _projBias2.Length;
        Array.Copy(parameters.ToArray(), offset, _projBn2Gamma, 0, _projBn2Gamma.Length);
        offset += _projBn2Gamma.Length;
        Array.Copy(parameters.ToArray(), offset, _projBn2Beta, 0, _projBn2Beta.Length);
        offset += _projBn2Beta.Length;

        // Predictor parameters
        if (_hasPredictor)
        {
            Array.Copy(parameters.ToArray(), offset, _predWeight1!, 0, _predWeight1!.Length);
            offset += _predWeight1.Length;
            Array.Copy(parameters.ToArray(), offset, _predBias1!, 0, _predBias1!.Length);
            offset += _predBias1.Length;
            Array.Copy(parameters.ToArray(), offset, _predBn1Gamma!, 0, _predBn1Gamma!.Length);
            offset += _predBn1Gamma.Length;
            Array.Copy(parameters.ToArray(), offset, _predBn1Beta!, 0, _predBn1Beta!.Length);
            offset += _predBn1Beta.Length;
            Array.Copy(parameters.ToArray(), offset, _predWeight2!, 0, _predWeight2!.Length);
            offset += _predWeight2.Length;
            Array.Copy(parameters.ToArray(), offset, _predBias2!, 0, _predBias2!.Length);
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameterGradients()
    {
        return _gradients ?? new Vector<T>(new T[ParameterCount]);
    }

    /// <inheritdoc />
    public void ClearGradients()
    {
        _gradients = null;
    }

    /// <inheritdoc />
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _cachedInput = null;
        _cachedH1 = null;
        _cachedH1Bn = null;
        _cachedH1Relu = null;
        _cachedProjection = null;
        _cachedPredH1 = null;
        _cachedPredH1Bn = null;
        _cachedPredH1Relu = null;
        _gradients = null;

        // Clear BatchNorm cached statistics
        _projBn1Mean = null;
        _projBn1Var = null;
        _projBn1Normalized = null;
        _projBn2Mean = null;
        _projBn2Var = null;
        _projBn2Normalized = null;
        _predBn1Mean = null;
        _predBn1Var = null;
        _predBn1Normalized = null;

        // Clear BatchNorm gradients
        _projBn1GammaGrad = null;
        _projBn1BetaGrad = null;
        _projBn2GammaGrad = null;
        _projBn2BetaGrad = null;
        _predBn1GammaGrad = null;
        _predBn1BetaGrad = null;
    }

    private int ComputeParameterCount()
    {
        // Projector: 2 linear layers with bias + 2 BN layers
        int projCount = (_inputDim * _hiddenDim + _hiddenDim) +     // Linear1 + bias
                       (_hiddenDim * 2) +                            // BN1 gamma + beta
                       (_hiddenDim * _projectionDim + _projectionDim) + // Linear2 + bias
                       (_projectionDim * 2);                         // BN2 gamma + beta

        if (!_hasPredictor)
            return projCount;

        // Predictor: 2 linear layers with bias + 1 BN layer
        int predCount = (_projectionDim * _predictorHiddenDim + _predictorHiddenDim) + // Linear1 + bias
                       (_predictorHiddenDim * 2) +                   // BN1 gamma + beta
                       (_predictorHiddenDim * _projectionDim + _projectionDim); // Linear2 + bias

        return projCount + predCount;
    }

    private T[] InitializeWeight(int fanIn, int fanOut, Random rng)
    {
        var weights = new T[fanIn * fanOut];
        var scale = Math.Sqrt(2.0 / fanIn); // He initialization

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }

        return weights;
    }

    private T[] InitializeOnes(int size)
    {
        var ones = new T[size];
        for (int i = 0; i < size; i++)
        {
            ones[i] = NumOps.One;
        }
        return ones;
    }

    private Tensor<T> Linear(Tensor<T> input, T[] weight, T[] bias, int inDim, int outDim)
    {
        var batchSize = input.Shape[0];
        var output = new T[batchSize * outDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < outDim; j++)
            {
                T sum = bias[j];
                for (int i = 0; i < inDim; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(input[b, i], weight[i * outDim + j]));
                }
                output[b * outDim + j] = sum;
            }
        }

        return new Tensor<T>(output, [batchSize, outDim]);
    }

    private Tensor<T> BatchNorm(Tensor<T> input, T[] gamma, T[] beta,
        out T[] mean, out T[] variance, out Tensor<T> normalized)
    {
        var batchSize = input.Shape[0];
        var dim = input.Shape[1];
        var output = new T[batchSize * dim];
        var normData = new T[batchSize * dim];
        mean = new T[dim];
        variance = new T[dim];

        for (int j = 0; j < dim; j++)
        {
            // Compute mean
            T m = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                m = NumOps.Add(m, input[b, j]);
            }
            m = NumOps.Divide(m, NumOps.FromDouble(batchSize));
            mean[j] = m;

            // Compute variance
            T v = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                var diff = NumOps.Subtract(input[b, j], m);
                v = NumOps.Add(v, NumOps.Multiply(diff, diff));
            }
            v = NumOps.Divide(v, NumOps.FromDouble(batchSize));
            variance[j] = v;
            var std = NumOps.Sqrt(NumOps.Add(v, NumOps.FromDouble(1e-5)));

            // Normalize and scale
            for (int b = 0; b < batchSize; b++)
            {
                var norm = NumOps.Divide(NumOps.Subtract(input[b, j], m), std);
                normData[b * dim + j] = norm;
                output[b * dim + j] = NumOps.Add(NumOps.Multiply(gamma[j], norm), beta[j]);
            }
        }

        normalized = new Tensor<T>(normData, [batchSize, dim]);
        return new Tensor<T>(output, [batchSize, dim]);
    }

    private Tensor<T> ReLU(Tensor<T> input)
    {
        var size = input.Shape[0] * input.Shape[1];
        var output = new T[size];

        for (int i = 0; i < size; i++)
        {
            var idx0 = i / input.Shape[1];
            var idx1 = i % input.Shape[1];
            var val = input[idx0, idx1];
            output[i] = NumOps.GreaterThan(val, NumOps.Zero) ? val : NumOps.Zero;
        }

        return new Tensor<T>(output, input.Shape);
    }
}
