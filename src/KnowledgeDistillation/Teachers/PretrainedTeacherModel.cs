using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Distilling the Knowledge in a Neural Network",
    "https://arxiv.org/abs/1503.02531",
    Year = 2015,
    Authors = "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")]
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
}
