using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Transformer-based teacher model that provides logits from transformer architectures.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>Architecture Note:</b> This class supports two construction modes:</para>
/// <list type="bullet">
/// <item><description>Function delegate mode: Uses a Func&lt;&gt; for forward pass (not JIT-compilable)</description></item>
/// <item><description>IJitCompilable mode: Uses a JIT-compilable model for forward pass (JIT-compilable)</description></item>
/// </list>
///
/// <para>For attention-based distillation strategies that need attention weights, implement
/// a custom IDistillationStrategy that can extract attention from the underlying model.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Distilling the Knowledge in a Neural Network",
    "https://arxiv.org/abs/1503.02531",
    Year = 2015,
    Authors = "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")]
public class TransformerTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>>? _forwardFunc;
    private readonly IJitCompilable<T>? _jitCompilableModel;
    private readonly int _outputDim;
    private readonly int _inputDim;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public override int OutputDimension => _outputDim;

    /// <summary>
    /// Initializes a new instance of the TransformerTeacherModel class using a function delegate.
    /// </summary>
    /// <param name="forwardFunc">Function that performs forward pass and returns logits.</param>
    /// <param name="inputDimension">The number of input dimensions.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <exception cref="ArgumentNullException">Thrown when forwardFunc is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when dimensions are not positive.</exception>
    /// <remarks>
    /// <para><b>Note:</b> This constructor creates a non-JIT-compilable teacher.
    /// For JIT support, use the constructor that accepts an IJitCompilable model.</para>
    /// </remarks>
    public TransformerTeacherModel(
        Func<Vector<T>, Vector<T>> forwardFunc,
        int inputDimension,
        int outputDimension)
    {
        Guard.NotNull(forwardFunc);
        _forwardFunc = forwardFunc;
        if (inputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputDimension),
                "Input dimension must be positive.");
        if (outputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputDimension),
                "Output dimension must be positive.");
        _inputDim = inputDimension;
        _outputDim = outputDimension;
        _jitCompilableModel = null;
    }

    /// <summary>
    /// Initializes a new instance of the TransformerTeacherModel class using a JIT-compilable model.
    /// </summary>
    /// <param name="jitCompilableModel">A JIT-compilable model that performs forward pass.</param>
    /// <param name="inputDimension">The number of input dimensions.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <exception cref="ArgumentNullException">Thrown when jitCompilableModel is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when dimensions are not positive.</exception>
    /// <remarks>
    /// <para><b>JIT Support:</b> This constructor enables JIT compilation when the underlying
    /// model supports it. Use this constructor for optimal inference performance.</para>
    /// </remarks>
    public TransformerTeacherModel(
        IJitCompilable<T> jitCompilableModel,
        int inputDimension,
        int outputDimension)
    {
        Guard.NotNull(jitCompilableModel);
        _jitCompilableModel = jitCompilableModel;
        if (inputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputDimension),
                "Input dimension must be positive.");
        if (outputDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputDimension),
                "Output dimension must be positive.");
        _inputDim = inputDimension;
        _outputDim = outputDimension;
        _forwardFunc = null;
    }

    /// <summary>
    /// Gets logits from the transformer model.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>Raw logits from the transformer.</returns>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        if (_forwardFunc != null)
        {
            return _forwardFunc(input);
        }
        else if (_jitCompilableModel != null)
        {
            // For JIT-compilable models, we need to predict through the model
            // This is a fallback for non-JIT execution
            var inputNodes = new List<ComputationNode<T>>();
            var inputTensor = new Tensor<T>(new[] { _inputDim }, input);
            var inputNode = TensorOperations<T>.Variable(inputTensor, "transformer_input");
            inputNodes.Add(inputNode);

            var outputNode = _jitCompilableModel.ExportComputationGraph(inputNodes);
            return outputNode.Value.ToVector();
        }
        else
        {
            throw new InvalidOperationException("No forward function or JIT-compilable model available.");
        }
    }
}
