using AiDotNet.Autodiff;
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

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>false</c>. PretrainedTeacherModel uses a function delegate which cannot be
    /// exported as a computation graph.
    /// </value>
    /// <remarks>
    /// <para>
    /// Function delegates are opaque to the JIT compiler - they can contain arbitrary code
    /// that cannot be represented as tensor operations. To enable JIT compilation, wrap
    /// an IFullModel directly instead of using a function delegate.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Not supported for PretrainedTeacherModel.
    /// </summary>
    /// <param name="inputNodes">Not used.</param>
    /// <returns>Never returns normally.</returns>
    /// <exception cref="NotSupportedException">Always thrown.</exception>
    /// <remarks>
    /// <para>
    /// PretrainedTeacherModel uses a function delegate which cannot be exported as a
    /// computation graph. To enable JIT compilation, use a model that implements
    /// IJitCompilable directly, or wrap an IFullModel instead of a function delegate.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "PretrainedTeacherModel does not support JIT compilation because it uses a function delegate. " +
            "Function delegates are opaque and cannot be exported as computation graphs. " +
            "To enable JIT compilation, wrap a model that implements IJitCompilable<T> instead.");
    }
}
