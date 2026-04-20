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
/// <para>
/// This wrapper takes a <c>Func&lt;Vector&lt;T&gt;, Vector&lt;T&gt;&gt;</c> forward-pass
/// delegate and invokes it directly on every <see cref="GetLogits"/> call.
/// The wrapper itself performs no caching or graph compilation — any
/// optimizations (including Tensors' AutoTracer auto-compile) depend entirely
/// on what happens inside the supplied delegate. A delegate that wraps a
/// standard neural-network model's <c>Predict</c> path will pick up those
/// engine-level optimizations; a delegate that invokes external code
/// (pre-converted ONNX, a REST call, etc.) will not.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Distilling the Knowledge in a Neural Network",
    "https://arxiv.org/abs/1503.02531",
    Year = 2015,
    Authors = "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class PretrainedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>>? _pretrainedForward;
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
        else
        {
            throw new InvalidOperationException("No forward function available.");
        }
    }
}
