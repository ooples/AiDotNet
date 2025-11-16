using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Pretrained teacher model from external source (e.g., ImageNet, BERT).
/// </summary>
public class PretrainedTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly Func<Vector<T>, Vector<T>> _pretrainedForward;
    private readonly int _outputDim;

    public override int OutputDimension => _outputDim;

    public PretrainedTeacherModel(
        Func<Vector<T>, Vector<T>> pretrainedForward,
        int outputDimension)
    {
        _pretrainedForward = pretrainedForward ?? throw new ArgumentNullException(nameof(pretrainedForward));
        _outputDim = outputDimension;
    }

    /// <summary>
    /// Gets logits from the pretrained model.
    /// </summary>
    /// <remarks>
    /// <para><b>Architecture Note:</b> Returns raw logits. Temperature scaling and softmax
    /// are handled by distillation strategies, not by the teacher model.</para>
    /// </remarks>
    public override Vector<T> GetLogits(Vector<T> input) => _pretrainedForward(input);
}
