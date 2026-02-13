using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if constructed with a JIT-compilable model that supports JIT; otherwise, <c>false</c>.
    /// </value>
    public override bool SupportsJitCompilation =>
        _jitCompilableModel != null && _jitCompilableModel.SupportsJitCompilation;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when constructed with a function delegate instead of a JIT-compilable model.
    /// </exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_jitCompilableModel == null)
        {
            throw new NotSupportedException(
                "TransformerTeacherModel does not support JIT compilation because it was constructed " +
                "with a function delegate. To enable JIT compilation, use the constructor that accepts " +
                "an IJitCompilable<T> model.");
        }

        if (!_jitCompilableModel.SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"TransformerTeacherModel cannot export computation graph because the underlying model " +
                $"({_jitCompilableModel.GetType().Name}) does not support JIT compilation.");
        }

        return _jitCompilableModel.ExportComputationGraph(inputNodes);
    }
}
