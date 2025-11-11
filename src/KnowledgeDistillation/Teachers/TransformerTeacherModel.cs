using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Transformer-based teacher model that provides logits from transformer architectures.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>Architecture Note:</b> This class has been simplified to match the current architecture
/// where teachers only provide logits. Attention mechanism extraction and temperature scaling
/// belong in the strategy layer, not in teacher models.</para>
///
/// <para>For attention-based distillation strategies that need attention weights, implement
/// a custom IDistillationStrategy that can extract attention from the underlying model.</para>
/// </remarks>
public class TransformerTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>> _forwardFunc;
    private readonly int _outputDim;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public override int OutputDimension => _outputDim;

    /// <summary>
    /// Initializes a new instance of the TransformerTeacherModel class.
    /// </summary>
    /// <param name="forwardFunc">Function that performs forward pass and returns logits.</param>
    /// <param name="outputDimension">The number of output dimensions.</param>
    public TransformerTeacherModel(
        Func<Vector<T>, Vector<T>> forwardFunc,
        int outputDimension)
    {
        _forwardFunc = forwardFunc ?? throw new ArgumentNullException(nameof(forwardFunc));
        _outputDim = outputDimension;
    }

    /// <summary>
    /// Gets logits from the transformer model.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>Raw logits from the transformer.</returns>
    public override Vector<T> GetLogits(Vector<T> input) => _forwardFunc(input);
}
