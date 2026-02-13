using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Pretrained teacher model from external source (e.g., ImageNet, BERT).
/// </summary>
/// <remarks>
/// <para><b>Architecture Note:</b> This class supports two construction modes:</para>
/// <list type="bullet">
/// <item><description>Function delegate mode: Uses a Func&lt;&gt; for forward pass (not JIT-compilable)</description></item>
/// <item><description>IJitCompilable mode: Uses a JIT-compilable model for forward pass (JIT-compilable)</description></item>
/// </list>
/// </remarks>
public class PretrainedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>>? _pretrainedForward;
    private readonly IJitCompilable<T>? _jitCompilableModel;
    private readonly int _outputDim;
    private readonly int _inputDim;

    public override int OutputDimension => _outputDim;

    /// <summary>
    /// Initializes a new instance using a function delegate (not JIT-compilable).
    /// </summary>
    /// <param name="pretrainedForward">Function that performs forward pass.</param>
    /// <param name="inputDimension">The number of input dimensions.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    public PretrainedTeacherModel(
        Func<Vector<T>, Vector<T>> pretrainedForward,
        int inputDimension,
        int outputDimension)
    {
        Guard.NotNull(pretrainedForward);
        _pretrainedForward = pretrainedForward;
        _inputDim = inputDimension;
        _outputDim = outputDimension;
        _jitCompilableModel = null;
    }

    /// <summary>
    /// Initializes a new instance using a JIT-compilable model.
    /// </summary>
    /// <param name="jitCompilableModel">A JIT-compilable model for forward pass.</param>
    /// <param name="inputDimension">The number of input dimensions.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    /// <remarks>
    /// <para><b>JIT Support:</b> This constructor enables JIT compilation when the underlying
    /// model supports it. Use this constructor for optimal inference performance.</para>
    /// </remarks>
    public PretrainedTeacherModel(
        IJitCompilable<T> jitCompilableModel,
        int inputDimension,
        int outputDimension)
    {
        Guard.NotNull(jitCompilableModel);
        _jitCompilableModel = jitCompilableModel;
        _inputDim = inputDimension;
        _outputDim = outputDimension;
        _pretrainedForward = null;
    }

    /// <summary>
    /// Gets logits from the pretrained model.
    /// </summary>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        if (_pretrainedForward != null)
        {
            return _pretrainedForward(input);
        }
        else if (_jitCompilableModel != null)
        {
            var inputNodes = new List<ComputationNode<T>>();
            var inputTensor = new Tensor<T>(new[] { _inputDim }, input);
            var inputNode = TensorOperations<T>.Variable(inputTensor, "pretrained_input");
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
                "PretrainedTeacherModel does not support JIT compilation because it was constructed " +
                "with a function delegate. To enable JIT compilation, use the constructor that accepts " +
                "an IJitCompilable<T> model.");
        }

        if (!_jitCompilableModel.SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"PretrainedTeacherModel cannot export computation graph because the underlying model " +
                $"({_jitCompilableModel.GetType().Name}) does not support JIT compilation.");
        }

        return _jitCompilableModel.ExportComputationGraph(inputNodes);
    }
}
