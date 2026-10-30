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
[ResearchPaper("A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709", Year = 2020, Authors = "Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton")]
public class LinearProjector<T> : IProjectorHead<T>, AiDotNet.Models.Parameters.IParameterChunkSource<T>, AiDotNet.Models.Parameters.IParameterLayoutSource
{
    IEnumerable<AiDotNet.Models.Parameters.ParameterChunk<T>> AiDotNet.Models.Parameters.IParameterChunkSource<T>.GetParameterStateChunks()
        => ProjectorParameterTransfer<T>.StateChunks(this);

    IReadOnlyList<AiDotNet.Models.Parameters.ParameterSlotDescriptor> AiDotNet.Models.Parameters.IParameterLayoutSource.GetParameterLayout()
        => ProjectorParameterTransfer<T>.Layout(this);

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly bool _useBias;

    // Written in place, never replaced: an optimizer keys its state by tensor reference.
    private readonly Tensor<T> _weight;
    private readonly Tensor<T>? _bias;

    /// <inheritdoc />
    public int InputDimension => _inputDim;

    /// <inheritdoc />
    public int OutputDimension => _outputDim;

    /// <inheritdoc />
    public int? HiddenDimension => null;

    /// <inheritdoc />
    public long ParameterCount => _inputDim * _outputDim + (_useBias ? _outputDim : 0);

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
    public IReadOnlyList<Tensor<T>> GetTrainableTensors() =>
        _bias is null ? new[] { _weight } : new[] { _weight, _bias };

    /// <inheritdoc />
    public Tensor<T> Project(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        // Vectorized: output = input @ weight + bias
        var output = Engine.TensorMatMul(input, _weight);
        if (_useBias && _bias is not null)
        {
            var bias2D = Engine.Reshape(_bias, new[] { 1, _outputDim });
            output = Engine.TensorAdd(output, bias2D);
        }
        return output;
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var paramList = new List<T>();
        foreach (var tensor in GetTrainableTensors())
        {
            for (int i = 0; i < tensor.Length; i++) paramList.Add(tensor[i]);
        }

        return new Vector<T>([.. paramList]);
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
        "LinearProjector keeps no gradients: record Project on a GradientTape and differentiate GetTrainableTensors().");

    /// <inheritdoc />
    /// <remarks>This head keeps no gradients, so there is nothing to clear.</remarks>
    public void ClearGradients()
    {
    }

    /// <inheritdoc />
    /// <remarks>A linear head computes the same function in training and evaluation.</remarks>
    public void SetTrainingMode(bool isTraining)
    {
    }

    /// <inheritdoc />
    /// <remarks>The head keeps no per-call state; Reset leaves the weights as they are.</remarks>
    public void Reset()
    {
    }
}
