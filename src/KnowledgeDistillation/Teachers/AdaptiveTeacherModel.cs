using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Adaptive teacher model that wraps a base teacher and provides its logits.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>Architecture Note:</b> This class has been simplified to match the current architecture
/// where temperature scaling is handled by distillation strategies, not teachers. The adaptive
/// features (dynamic temperature adjustment based on student performance) have been removed as they
/// belong in the strategy layer.</para>
///
/// <para>For adaptive temperature scaling, implement a custom IDistillationStrategy that monitors
/// student performance and adjusts temperature accordingly.</para>
/// </remarks>
public class AdaptiveTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _baseTeacher;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public override int OutputDimension => _baseTeacher.OutputDimension;

    /// <summary>
    /// Initializes a new instance of the AdaptiveTeacherModel class.
    /// </summary>
    /// <param name="baseTeacher">The underlying teacher model.</param>
    public AdaptiveTeacherModel(ITeacherModel<Vector<T>, Vector<T>> baseTeacher)
    {
        Guard.NotNull(baseTeacher);
        _baseTeacher = baseTeacher;
    }

    /// <summary>
    /// Gets logits from the base teacher.
    /// </summary>
    public override Vector<T> GetLogits(Vector<T> input)
    {
        return _baseTeacher.GetLogits(input);
    }

    /// <summary>
    /// Gets whether this teacher supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if the base teacher implements IJitCompilable and supports JIT; otherwise, <c>false</c>.
    /// </value>
    public override bool SupportsJitCompilation =>
        _baseTeacher is IJitCompilable<T> jitCompilable && jitCompilable.SupportsJitCompilation;

    /// <summary>
    /// Exports the computation graph by delegating to the base teacher.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node from the base teacher.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the base teacher does not support JIT compilation.
    /// </exception>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_baseTeacher is not IJitCompilable<T> jitCompilable)
        {
            throw new NotSupportedException(
                $"AdaptiveTeacherModel cannot export computation graph because the base teacher " +
                $"({_baseTeacher.GetType().Name}) does not implement IJitCompilable<T>.");
        }

        if (!jitCompilable.SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"AdaptiveTeacherModel cannot export computation graph because the base teacher " +
                $"({_baseTeacher.GetType().Name}) does not support JIT compilation.");
        }

        return jitCompilable.ExportComputationGraph(inputNodes);
    }
}
