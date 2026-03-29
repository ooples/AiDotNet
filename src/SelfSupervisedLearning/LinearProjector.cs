using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Linear projection head for self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A linear projector is the simplest projection head - just a single
/// linear transformation (matrix multiplication + bias). While simpler than MLP projectors,
/// linear projectors can still be effective in some scenarios.</para>
///
/// <para><b>Architecture:</b></para>
/// <code>
/// Input → Linear → Output
/// [d_in]   [d_out]
/// </code>
///
/// <para><b>When to use Linear vs MLP:</b></para>
/// <list type="bullet">
/// <item>Use <b>Linear</b> for simplicity, lower compute, or when encoder is already powerful</item>
/// <item>Use <b>MLP</b> for better downstream performance (recommended for most SSL methods)</item>
/// </list>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709", Year = 2020, Authors = "Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton")]
public class LinearProjector<T> : IProjectorHead<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly bool _useBias;

    private Tensor<T> _weight;
    private Tensor<T>? _bias;
    private Tensor<T>? _gradWeight;
    private Tensor<T>? _gradBias;
    private Tensor<T>? _lastInput;

    private bool _isTraining = true;

    /// <inheritdoc />
    public int InputDimension => _inputDim;

    /// <inheritdoc />
    public int OutputDimension => _outputDim;

    /// <inheritdoc />
    public int? HiddenDimension => null;

    /// <inheritdoc />
    public int ParameterCount => _inputDim * _outputDim + (_useBias ? _outputDim : 0);

    /// <summary>
    /// Initializes a new instance of the LinearProjector class.
    /// </summary>
    /// <param name="inputDim">Input dimension (encoder output size).</param>
    /// <param name="outputDim">Output dimension (projection size).</param>
    /// <param name="useBias">Whether to include a bias term.</param>
    /// <param name="seed">Optional random seed for initialization.</param>
    public LinearProjector(
        int inputDim,
        int outputDim = 128,
        bool useBias = true,
        int? seed = null)
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

        _inputDim = inputDim;
        _outputDim = outputDim;
        _useBias = useBias;

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.Shared;

        // Xavier/Glorot initialization for linear layer
        var stddev = Math.Sqrt(2.0 / (inputDim + outputDim));
        var weightData = new T[inputDim * outputDim];

        for (int i = 0; i < weightData.Length; i++)
        {
            weightData[i] = NumOps.FromDouble(random.NextGaussian() * stddev);
        }

        _weight = new Tensor<T>(weightData, [inputDim, outputDim]);

        if (_useBias)
        {
            var biasData = new T[outputDim];
            for (int i = 0; i < outputDim; i++)
            {
                biasData[i] = NumOps.Zero;
            }
            _bias = new Tensor<T>(biasData, [outputDim]);
        }
    }

    /// <inheritdoc />
    public Tensor<T> Project(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        _lastInput = input;

        // Vectorized: output = input @ weight + bias
        var output = Engine.TensorMatMul(input, _weight);
        if (_useBias && _bias is not null)
        {
            var bias2D = _bias.Reshape(1, _outputDim);
            output = Engine.TensorBroadcastAdd(output, bias2D);
        }
        return output;
    }

    /// <inheritdoc />
    public Tensor<T> Backward(Tensor<T> gradients)
    {
        if (gradients is null) throw new ArgumentNullException(nameof(gradients));
        if (_lastInput is null) throw new InvalidOperationException("Forward must be called before Backward");

        // Vectorized backward using Engine.TensorMatMul

        // Input gradient: dX = dY @ W^T
        var weightT = _weight.Transpose([1, 0]);
        var inputGradTensor = Engine.TensorMatMul(gradients, weightT);

        // Weight gradient: dW = X^T @ dY
        var inputT = _lastInput.Transpose([1, 0]);
        _gradWeight = Engine.TensorMatMul(inputT, gradients);

        // Bias gradient: dB = sum(dY, axis=0)
        if (_useBias)
        {
            _gradBias = Engine.ReduceSum(gradients, new[] { 0 });
        }

        return inputGradTensor;
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var paramList = new List<T>();

        for (int i = 0; i < _weight.Length; i++)
        {
            paramList.Add(_weight.Data.Span[i]);
        }

        if (_useBias && _bias is not null)
        {
            for (int i = 0; i < _bias.Length; i++)
            {
                paramList.Add(_bias.Data.Span[i]);
            }
        }

        return new Vector<T>([.. paramList]);
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int offset = 0;

        // Set weight
        var weightData = new T[_inputDim * _outputDim];
        for (int i = 0; i < weightData.Length; i++)
        {
            weightData[i] = parameters[offset++];
        }
        _weight = new Tensor<T>(weightData, [_inputDim, _outputDim]);

        // Set bias
        if (_useBias)
        {
            var biasData = new T[_outputDim];
            for (int i = 0; i < biasData.Length; i++)
            {
                biasData[i] = parameters[offset++];
            }
            _bias = new Tensor<T>(biasData, [_outputDim]);
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameterGradients()
    {
        var gradList = new List<T>();

        if (_gradWeight is not null)
        {
            for (int i = 0; i < _gradWeight.Length; i++)
            {
                gradList.Add(_gradWeight.Data.Span[i]);
            }
        }
        else
        {
            for (int i = 0; i < _inputDim * _outputDim; i++)
            {
                gradList.Add(NumOps.Zero);
            }
        }

        if (_useBias)
        {
            if (_gradBias is not null)
            {
                for (int i = 0; i < _gradBias.Length; i++)
                {
                    gradList.Add(_gradBias.Data.Span[i]);
                }
            }
            else
            {
                for (int i = 0; i < _outputDim; i++)
                {
                    gradList.Add(NumOps.Zero);
                }
            }
        }

        return new Vector<T>([.. gradList]);
    }

    /// <inheritdoc />
    public void ClearGradients()
    {
        _gradWeight = null;
        _gradBias = null;
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
    }
}
