using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Multi-layer perceptron (MLP) projection head for self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> An MLP projector transforms encoder outputs into a lower-dimensional
/// space optimized for the SSL loss. This is the standard projector used in SimCLR, MoCo v2, BYOL.</para>
///
/// <para><b>Architecture:</b></para>
/// <code>
/// Input → Linear → BatchNorm → ReLU → Linear → [BatchNorm] → Output
/// [d_in]   [d_hid]              [d_hid]  [d_out]         [d_out]
/// </code>
///
/// <para><b>Why MLP over Linear?</b></para>
/// <list type="bullet">
/// <item>Non-linearity allows learning more complex projections</item>
/// <item>Extra capacity prevents encoder from being constrained by SSL loss</item>
/// <item>Empirically shown to significantly improve downstream performance</item>
/// </list>
/// </remarks>
public class MLPProjector<T> : IProjectorHead<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the global execution engine for vector operations and GPU/CPU acceleration.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _outputDim;
    private readonly bool _useBatchNormOnOutput;

    // Layer 1: Input → Hidden
    private Tensor<T> _weight1;
    private Tensor<T> _bias1;
    private Tensor<T>? _gradWeight1;
    private Tensor<T>? _gradBias1;

    // BatchNorm 1
    private Tensor<T> _gamma1;
    private Tensor<T> _beta1;
    private Tensor<T> _runningMean1;
    private Tensor<T> _runningVar1;
    private Tensor<T>? _gradGamma1;
    private Tensor<T>? _gradBeta1;

    // Layer 2: Hidden → Output
    private Tensor<T> _weight2;
    private Tensor<T> _bias2;
    private Tensor<T>? _gradWeight2;
    private Tensor<T>? _gradBias2;

    // BatchNorm 2 (optional)
    private Tensor<T>? _gamma2;
    private Tensor<T>? _beta2;
    private Tensor<T>? _runningMean2;
    private Tensor<T>? _runningVar2;
    private Tensor<T>? _gradGamma2;
    private Tensor<T>? _gradBeta2;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _preActivation1;
    private Tensor<T>? _postBatchNorm1;
    private Tensor<T>? _postRelu1;
    private Tensor<T>? _preActivation2;

    private bool _isTraining = true;
    private readonly double _batchNormMomentum = 0.1;
    private readonly double _batchNormEpsilon = 1e-5;

    /// <inheritdoc />
    public int InputDimension => _inputDim;

    /// <inheritdoc />
    public int OutputDimension => _outputDim;

    /// <inheritdoc />
    public int? HiddenDimension => _hiddenDim;

    /// <inheritdoc />
    public int ParameterCount
    {
        get
        {
            int count = 0;
            // Layer 1: weight + bias
            count += _inputDim * _hiddenDim + _hiddenDim;
            // BatchNorm 1: gamma + beta
            count += _hiddenDim * 2;
            // Layer 2: weight + bias
            count += _hiddenDim * _outputDim + _outputDim;
            // BatchNorm 2 (if used)
            if (_useBatchNormOnOutput)
            {
                count += _outputDim * 2;
            }
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the MLPProjector class.
    /// </summary>
    /// <param name="inputDim">Input dimension (encoder output size).</param>
    /// <param name="hiddenDim">Hidden dimension (typically 2048-4096).</param>
    /// <param name="outputDim">Output dimension (typically 128-256).</param>
    /// <param name="useBatchNormOnOutput">Whether to apply BatchNorm on the output layer.</param>
    /// <param name="seed">Optional random seed for initialization.</param>
    public MLPProjector(
        int inputDim,
        int hiddenDim = 2048,
        int outputDim = 128,
        bool useBatchNormOnOutput = false,
        int? seed = null)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _outputDim = outputDim;
        _useBatchNormOnOutput = useBatchNormOnOutput;

        // Initialize weights using Kaiming/He initialization
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.Shared;

        _weight1 = InitializeWeight(inputDim, hiddenDim, random);
        _bias1 = InitializeZeros(hiddenDim);

        _gamma1 = InitializeOnes(hiddenDim);
        _beta1 = InitializeZeros(hiddenDim);
        _runningMean1 = InitializeZeros(hiddenDim);
        _runningVar1 = InitializeOnes(hiddenDim);

        _weight2 = InitializeWeight(hiddenDim, outputDim, random);
        _bias2 = InitializeZeros(outputDim);

        if (_useBatchNormOnOutput)
        {
            _gamma2 = InitializeOnes(outputDim);
            _beta2 = InitializeZeros(outputDim);
            _runningMean2 = InitializeZeros(outputDim);
            _runningVar2 = InitializeOnes(outputDim);
        }
    }

    /// <inheritdoc />
    public Tensor<T> Project(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        _lastInput = input;

        // Layer 1: Linear
        _preActivation1 = LinearForward(input, _weight1, _bias1);

        // BatchNorm 1
        _postBatchNorm1 = BatchNormForward(_preActivation1, _gamma1, _beta1, _runningMean1, _runningVar1);

        // ReLU
        _postRelu1 = ReLUForward(_postBatchNorm1);

        // Layer 2: Linear
        _preActivation2 = LinearForward(_postRelu1, _weight2, _bias2);

        // Optional BatchNorm 2
        if (_useBatchNormOnOutput && _gamma2 is not null && _beta2 is not null &&
            _runningMean2 is not null && _runningVar2 is not null)
        {
            return BatchNormForward(_preActivation2, _gamma2, _beta2, _runningMean2, _runningVar2);
        }

        return _preActivation2;
    }

    /// <inheritdoc />
    public Tensor<T> Backward(Tensor<T> gradients)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (_lastInput is null) throw new InvalidOperationException("Forward must be called before Backward");

        var grad = gradients;

        // Backward through optional BatchNorm 2
        if (_useBatchNormOnOutput && _preActivation2 is not null)
        {
            (grad, _gradGamma2, _gradBeta2) = BatchNormBackward(grad, _preActivation2, _gamma2!);
        }

        // Backward through Layer 2
        (grad, _gradWeight2, _gradBias2) = LinearBackward(grad, _postRelu1!, _weight2);

        // Backward through ReLU
        grad = ReLUBackward(grad, _postBatchNorm1!);

        // Backward through BatchNorm 1
        (grad, _gradGamma1, _gradBeta1) = BatchNormBackward(grad, _preActivation1!, _gamma1);

        // Backward through Layer 1
        (grad, _gradWeight1, _gradBias1) = LinearBackward(grad, _lastInput, _weight1);

        return grad;
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var paramList = new List<T>();

        // Layer 1
        AddTensorToList(paramList, _weight1);
        AddTensorToList(paramList, _bias1);
        AddTensorToList(paramList, _gamma1);
        AddTensorToList(paramList, _beta1);

        // Layer 2
        AddTensorToList(paramList, _weight2);
        AddTensorToList(paramList, _bias2);

        if (_useBatchNormOnOutput)
        {
            AddTensorToList(paramList, _gamma2!);
            AddTensorToList(paramList, _beta2!);
        }

        return new Vector<T>([.. paramList]);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int offset = 0;

        // Layer 1
        _weight1 = ExtractTensor(parameters, ref offset, [_inputDim, _hiddenDim]);
        _bias1 = ExtractTensor(parameters, ref offset, [_hiddenDim]);
        _gamma1 = ExtractTensor(parameters, ref offset, [_hiddenDim]);
        _beta1 = ExtractTensor(parameters, ref offset, [_hiddenDim]);

        // Layer 2
        _weight2 = ExtractTensor(parameters, ref offset, [_hiddenDim, _outputDim]);
        _bias2 = ExtractTensor(parameters, ref offset, [_outputDim]);

        if (_useBatchNormOnOutput)
        {
            _gamma2 = ExtractTensor(parameters, ref offset, [_outputDim]);
            _beta2 = ExtractTensor(parameters, ref offset, [_outputDim]);
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameterGradients()
    {
        var gradList = new List<T>();

        // Layer 1
        AddTensorToList(gradList, _gradWeight1 ?? InitializeZeros(_inputDim * _hiddenDim, [_inputDim, _hiddenDim]));
        AddTensorToList(gradList, _gradBias1 ?? InitializeZeros(_hiddenDim));
        AddTensorToList(gradList, _gradGamma1 ?? InitializeZeros(_hiddenDim));
        AddTensorToList(gradList, _gradBeta1 ?? InitializeZeros(_hiddenDim));

        // Layer 2
        AddTensorToList(gradList, _gradWeight2 ?? InitializeZeros(_hiddenDim * _outputDim, [_hiddenDim, _outputDim]));
        AddTensorToList(gradList, _gradBias2 ?? InitializeZeros(_outputDim));

        if (_useBatchNormOnOutput)
        {
            AddTensorToList(gradList, _gradGamma2 ?? InitializeZeros(_outputDim));
            AddTensorToList(gradList, _gradBeta2 ?? InitializeZeros(_outputDim));
        }

        return new Vector<T>([.. gradList]);
    }

    /// <inheritdoc />
    public void ClearGradients()
    {
        _gradWeight1 = null;
        _gradBias1 = null;
        _gradGamma1 = null;
        _gradBeta1 = null;
        _gradWeight2 = null;
        _gradBias2 = null;
        _gradGamma2 = null;
        _gradBeta2 = null;
    }

    /// <inheritdoc />
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    /// <inheritdoc />
    public void Reset()
    {
        ClearGradients();
        _lastInput = null;
        _preActivation1 = null;
        _postBatchNorm1 = null;
        _postRelu1 = null;
        _preActivation2 = null;
    }

    #region Private Helper Methods

    private Tensor<T> InitializeWeight(int fanIn, int fanOut, Random random)
    {
        // He/Kaiming initialization for ReLU
        var stddev = Math.Sqrt(2.0 / fanIn);
        var data = new T[fanIn * fanOut];

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = NumOps.FromDouble(random.NextGaussian() * stddev);
        }

        return new Tensor<T>(data, [fanIn, fanOut]);
    }

    private Tensor<T> InitializeZeros(int size)
    {
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = NumOps.Zero;
        }
        return new Tensor<T>(data, [size]);
    }

    private Tensor<T> InitializeZeros(int size, int[] shape)
    {
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = NumOps.Zero;
        }
        return new Tensor<T>(data, shape);
    }

    private Tensor<T> InitializeOnes(int size)
    {
        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = NumOps.One;
        }
        return new Tensor<T>(data, [size]);
    }

    private Tensor<T> LinearForward(Tensor<T> input, Tensor<T> weight, Tensor<T> bias)
    {
        // input: [batch, inputDim], weight: [inputDim, outputDim], bias: [outputDim]
        var batchSize = input.Shape[0];
        var inputDim = input.Shape[1];
        var outputDim = weight.Shape[1];

        var result = new T[batchSize * outputDim];

        // Use Engine-accelerated dot products for each row x column combination
        for (int b = 0; b < batchSize; b++)
        {
            // Extract input row
            var inputRow = new T[inputDim];
            for (int i = 0; i < inputDim; i++)
            {
                inputRow[i] = input[b, i];
            }
            var inputVec = new Vector<T>(inputRow);

            for (int o = 0; o < outputDim; o++)
            {
                // Extract weight column
                var weightCol = new T[inputDim];
                for (int i = 0; i < inputDim; i++)
                {
                    weightCol[i] = weight[i, o];
                }
                var weightVec = new Vector<T>(weightCol);

                // Use engine for accelerated dot product
                var dot = Engine.DotProduct(inputVec, weightVec);
                result[b * outputDim + o] = NumOps.Add(bias[o], dot);
            }
        }

        return new Tensor<T>(result, [batchSize, outputDim]);
    }

    private (Tensor<T> inputGrad, Tensor<T> weightGrad, Tensor<T> biasGrad) LinearBackward(
        Tensor<T> outputGrad, Tensor<T> input, Tensor<T> weight)
    {
        var batchSize = outputGrad.Shape[0];
        var outputDim = outputGrad.Shape[1];
        var inputDim = weight.Shape[0];

        // Input gradient: outputGrad @ weight.T
        var inputGrad = new T[batchSize * inputDim];
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputDim; i++)
            {
                T sum = NumOps.Zero;
                for (int o = 0; o < outputDim; o++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(outputGrad[b, o], weight[i, o]));
                }
                inputGrad[b * inputDim + i] = sum;
            }
        }

        // Weight gradient: input.T @ outputGrad
        var weightGrad = new T[inputDim * outputDim];
        for (int i = 0; i < inputDim; i++)
        {
            for (int o = 0; o < outputDim; o++)
            {
                T sum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(input[b, i], outputGrad[b, o]));
                }
                weightGrad[i * outputDim + o] = sum;
            }
        }

        // Bias gradient: sum over batch
        var biasGrad = new T[outputDim];
        for (int o = 0; o < outputDim; o++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, outputGrad[b, o]);
            }
            biasGrad[o] = sum;
        }

        return (
            new Tensor<T>(inputGrad, [batchSize, inputDim]),
            new Tensor<T>(weightGrad, [inputDim, outputDim]),
            new Tensor<T>(biasGrad, [outputDim])
        );
    }

    private Tensor<T> BatchNormForward(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta,
        Tensor<T> runningMean, Tensor<T> runningVar)
    {
        var batchSize = input.Shape[0];
        var dim = input.Shape[1];
        var result = new T[batchSize * dim];

        if (_isTraining)
        {
            // Compute batch mean and variance
            var mean = new T[dim];
            var variance = new T[dim];

            for (int d = 0; d < dim; d++)
            {
                T sum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    sum = NumOps.Add(sum, input[b, d]);
                }
                mean[d] = NumOps.Divide(sum, NumOps.FromDouble(batchSize));
            }

            for (int d = 0; d < dim; d++)
            {
                T sum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    var diff = NumOps.Subtract(input[b, d], mean[d]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
                }
                variance[d] = NumOps.Divide(sum, NumOps.FromDouble(batchSize));
            }

            // Update running statistics
            for (int d = 0; d < dim; d++)
            {
                runningMean[d] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(1 - _batchNormMomentum), runningMean[d]),
                    NumOps.Multiply(NumOps.FromDouble(_batchNormMomentum), mean[d]));
                runningVar[d] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(1 - _batchNormMomentum), runningVar[d]),
                    NumOps.Multiply(NumOps.FromDouble(_batchNormMomentum), variance[d]));
            }

            // Normalize
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < dim; d++)
                {
                    var normalized = NumOps.Divide(
                        NumOps.Subtract(input[b, d], mean[d]),
                        NumOps.Sqrt(NumOps.Add(variance[d], NumOps.FromDouble(_batchNormEpsilon))));
                    result[b * dim + d] = NumOps.Add(NumOps.Multiply(gamma[d], normalized), beta[d]);
                }
            }
        }
        else
        {
            // Use running statistics
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < dim; d++)
                {
                    var normalized = NumOps.Divide(
                        NumOps.Subtract(input[b, d], runningMean[d]),
                        NumOps.Sqrt(NumOps.Add(runningVar[d], NumOps.FromDouble(_batchNormEpsilon))));
                    result[b * dim + d] = NumOps.Add(NumOps.Multiply(gamma[d], normalized), beta[d]);
                }
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }

    private (Tensor<T> inputGrad, Tensor<T> gammaGrad, Tensor<T> betaGrad) BatchNormBackward(
        Tensor<T> outputGrad, Tensor<T> input, Tensor<T> gamma)
    {
        var batchSize = outputGrad.Shape[0];
        var dim = outputGrad.Shape[1];

        var inputGrad = new T[batchSize * dim];
        var gammaGrad = new T[dim];
        var betaGrad = new T[dim];

        // Compute mean and variance for normalization
        var mean = new T[dim];
        var variance = new T[dim];
        var normalized = new T[batchSize * dim];

        for (int d = 0; d < dim; d++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, input[b, d]);
            }
            mean[d] = NumOps.Divide(sum, NumOps.FromDouble(batchSize));
        }

        for (int d = 0; d < dim; d++)
        {
            T sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                var diff = NumOps.Subtract(input[b, d], mean[d]);
                sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
            }
            variance[d] = NumOps.Divide(sum, NumOps.FromDouble(batchSize));
        }

        // Precompute normalized values: x_norm = (x - mean) / std
        for (int d = 0; d < dim; d++)
        {
            var std = NumOps.Sqrt(NumOps.Add(variance[d], NumOps.FromDouble(_batchNormEpsilon)));
            for (int b = 0; b < batchSize; b++)
            {
                normalized[b * dim + d] = NumOps.Divide(NumOps.Subtract(input[b, d], mean[d]), std);
            }
        }

        // Compute parameter gradients and input gradient using full BatchNorm formula:
        // dL/dx = (1/N) * γ * (1/σ) * [N * dL/dy - sum(dL/dy) - x_norm * sum(dL/dy * x_norm)]
        var n = NumOps.FromDouble(batchSize);

        for (int d = 0; d < dim; d++)
        {
            var std = NumOps.Sqrt(NumOps.Add(variance[d], NumOps.FromDouble(_batchNormEpsilon)));
            var invStd = NumOps.Divide(NumOps.One, std);

            // dL/dβ = sum(dL/dy)
            T sumDy = NumOps.Zero;
            // dL/dγ = sum(dL/dy * x_norm)
            T sumDyXnorm = NumOps.Zero;

            for (int b = 0; b < batchSize; b++)
            {
                var dy = outputGrad[b, d];
                var xn = normalized[b * dim + d];
                sumDy = NumOps.Add(sumDy, dy);
                sumDyXnorm = NumOps.Add(sumDyXnorm, NumOps.Multiply(dy, xn));
            }

            betaGrad[d] = sumDy;
            gammaGrad[d] = sumDyXnorm;

            // dL/dx = (γ / (N * σ)) * [N * dL/dy - sum(dL/dy) - x_norm * sum(dL/dy * x_norm)]
            var scale = NumOps.Multiply(gamma[d], NumOps.Divide(invStd, n));

            for (int b = 0; b < batchSize; b++)
            {
                var dy = outputGrad[b, d];
                var xn = normalized[b * dim + d];

                // N * dL/dy - sum(dL/dy) - x_norm * sum(dL/dy * x_norm)
                var term = NumOps.Subtract(
                    NumOps.Subtract(
                        NumOps.Multiply(n, dy),
                        sumDy),
                    NumOps.Multiply(xn, sumDyXnorm));

                inputGrad[b * dim + d] = NumOps.Multiply(scale, term);
            }
        }

        return (
            new Tensor<T>(inputGrad, [batchSize, dim]),
            new Tensor<T>(gammaGrad, [dim]),
            new Tensor<T>(betaGrad, [dim])
        );
    }

    private Tensor<T> ReLUForward(Tensor<T> input)
    {
        var size = input.Length;
        var result = new T[size];

        for (int i = 0; i < size; i++)
        {
            result[i] = NumOps.GreaterThan(input.Data[i], NumOps.Zero) ? input.Data[i] : NumOps.Zero;
        }

        return new Tensor<T>(result, input.Shape);
    }

    private Tensor<T> ReLUBackward(Tensor<T> outputGrad, Tensor<T> input)
    {
        var size = outputGrad.Length;
        var result = new T[size];

        for (int i = 0; i < size; i++)
        {
            result[i] = NumOps.GreaterThan(input.Data[i], NumOps.Zero) ? outputGrad.Data[i] : NumOps.Zero;
        }

        return new Tensor<T>(result, outputGrad.Shape);
    }

    private static void AddTensorToList(List<T> list, Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            list.Add(tensor.Data[i]);
        }
    }

    private static Tensor<T> ExtractTensor(Vector<T> parameters, ref int offset, int[] shape)
    {
        int size = 1;
        foreach (var dim in shape)
        {
            size *= dim;
        }

        var data = new T[size];
        for (int i = 0; i < size; i++)
        {
            data[i] = parameters[offset++];
        }

        return new Tensor<T>(data, shape);
    }

    #endregion
}
