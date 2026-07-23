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
/// <para>
/// This wrapper takes a <c>Func&lt;Vector&lt;T&gt;, Vector&lt;T&gt;&gt;</c> forward-pass
/// delegate and invokes it directly on every <see cref="GetLogits"/> call.
/// The wrapper performs no caching or graph compilation itself — any
/// optimizations (including Tensors' AutoTracer auto-compile) depend on what
/// the supplied delegate does internally.
/// </para>
/// <para>For attention-based distillation strategies that need attention weights, implement
/// a custom IDistillationStrategy that can extract attention from the underlying model.</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Distilling the Knowledge in a Neural Network",
    "https://arxiv.org/abs/1503.02531",
    Year = 2015,
    Authors = "Geoffrey Hinton, Oriol Vinyals, Jeff Dean")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class TransformerTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>>? _forwardFunc;
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
        else
        {
            throw new InvalidOperationException("No forward function available.");
        }
    }
}
