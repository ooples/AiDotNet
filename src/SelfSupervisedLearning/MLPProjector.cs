using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709", Year = 2020, Authors = "Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton")]
public partial class MLPProjector<T> : IProjectorHead<T>
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

    // Layer 1: Input → Hidden. Every learnable tensor is written in place, never replaced: an optimizer keys its
    // state by tensor reference.
    private readonly Tensor<T> _weight1;
    private readonly Tensor<T> _bias1;

    // BatchNorm 1
    private readonly Tensor<T> _gamma1;
    private readonly Tensor<T> _beta1;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _runningMean1;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _runningVar1;

    // Layer 2: Hidden → Output
    private readonly Tensor<T> _weight2;
    private readonly Tensor<T> _bias2;

    // BatchNorm 2 (optional)
    private readonly Tensor<T>? _gamma2;
    private readonly Tensor<T>? _beta2;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _runningMean2;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _runningVar2;

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
    public long ParameterCount
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

        // Linear, BatchNorm, ReLU, Linear, [BatchNorm]: engine ops the gradient tape records.
        var hidden = Engine.ReLU(BatchNormForward(LinearForward(input, _weight1, _bias1), _gamma1, _beta1, _runningMean1, _runningVar1));
        var output = LinearForward(hidden, _weight2, _bias2);

        if (_useBatchNormOnOutput && _gamma2 is not null && _beta2 is not null &&
            _runningMean2 is not null && _runningVar2 is not null)
        {
            return BatchNormForward(output, _gamma2, _beta2, _runningMean2, _runningVar2);
        }

        return output;
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
            var g2 = _gamma2 ?? throw new InvalidOperationException("Gamma2 has not been initialized.");
            AddTensorToList(paramList, g2);
            var b2 = _beta2 ?? throw new InvalidOperationException("Beta2 has not been initialized.");
            AddTensorToList(paramList, b2);
        }

        return new Vector<T>([.. paramList]);
    }

    /// <inheritdoc />
    public IReadOnlyList<Tensor<T>> GetTrainableTensors()
    {
        var tensors = new List<Tensor<T>> { _weight1, _bias1, _gamma1, _beta1, _weight2, _bias2 };
        if (_useBatchNormOnOutput && _gamma2 is not null && _beta2 is not null)
        {
            tensors.Add(_gamma2);
            tensors.Add(_beta2);
        }

        return tensors;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters; got {parameters.Length}.", nameof(parameters));

        // Written into the live tensors, in GetParameters' order, so an optimizer keyed by tensor reference keeps its
        // state.
        int offset = 0;
        foreach (var tensor in GetTrainableTensors())
        {
            for (int i = 0; i < tensor.Length; i++) tensor[i] = parameters[offset++];
        }
    }

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">Always: the gradient tape differentiates this head.</exception>
    public Vector<T> GetParameterGradients() => throw new NotSupportedException(
        "MLPProjector keeps no gradients: record Project on a GradientTape and differentiate GetTrainableTensors().");

    /// <inheritdoc />
    /// <remarks>This head keeps no gradients, so there is nothing to clear.</remarks>
    public void ClearGradients()
    {
    }

    /// <inheritdoc />
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    /// <inheritdoc />
    /// <remarks>The head keeps no per-call state; Reset leaves the weights and running statistics as they are.</remarks>
    public void Reset()
    {
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
        // Vectorized: output = input @ weight + bias
        var output = Engine.TensorMatMul(input, weight);
        var bias2D = bias.Reshape(1, weight.Shape[1]);
        return Engine.TensorAdd(output, bias2D);
    }

    private Tensor<T> BatchNormForward(Tensor<T> input, Tensor<T> gamma, Tensor<T> beta,
        Tensor<T> runningMean, Tensor<T> runningVar)
    {
        var batchSize = input.Shape[0];
        var dim = input.Shape[1];

        if (_isTraining)
        {
            // Engine.BatchNorm records on the gradient tape, so the loss reaches gamma, beta and everything upstream.
            // The hand-written loops this replaces built the output from a fresh array, which cut the tape: no
            // gradient reached the projector's first layer or the encoder behind it.
            var output = Engine.BatchNorm(input, gamma, beta, _batchNormEpsilon, out var batchMean, out var batchVariance);
            // torch.nn.BatchNorm1d keeps the unbiased variance, batch / (batch - 1) times the batch's, as
            // SymmetricProjector does.
            double unbias = batchSize > 1 ? batchSize / (batchSize - 1.0) : 1.0;
            for (int d = 0; d < dim; d++)
            {
                runningMean[d] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(1 - _batchNormMomentum), runningMean[d]),
                    NumOps.Multiply(NumOps.FromDouble(_batchNormMomentum), batchMean[d]));
                runningVar[d] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(1 - _batchNormMomentum), runningVar[d]),
                    NumOps.Multiply(NumOps.FromDouble(_batchNormMomentum * unbias), batchVariance[d]));
            }

            return output;
        }

        var result = new T[batchSize * dim];
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

    private static void AddTensorToList(List<T> list, Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            list.Add(tensor.Data.Span[i]);
        }
    }

    #endregion
}
