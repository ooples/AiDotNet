using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation.Teachers;

/// <summary>
/// Curriculum teacher that wraps a base teacher for curriculum learning scenarios.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>Architecture Note:</b> This class provides a simple wrapper around a base teacher.
/// Curriculum learning logic (adjusting difficulty over time) should be implemented in the
/// training loop or distillation strategy, not in the teacher model.</para>
///
/// <para>The teacher model's responsibility is only to provide predictions (logits).
/// Curriculum decisions (which samples to show, how to adjust temperature/alpha) belong
/// in the strategy or trainer layer.</para>
/// </remarks>
public class CurriculumTeacherModel<T> : TeacherModelBase<Vector<T>, Vector<T>, T>
{
    private readonly ITeacherModel<Vector<T>, Vector<T>> _baseTeacher;

    /// <summary>
    /// Gets the output dimension from the base teacher.
    /// </summary>
    public override int OutputDimension => _baseTeacher.OutputDimension;

    /// <summary>
    /// Initializes a new instance of the CurriculumTeacherModel class.
    /// </summary>
    /// <param name="baseTeacher">The underlying teacher model.</param>
    public CurriculumTeacherModel(ITeacherModel<Vector<T>, Vector<T>> baseTeacher)
    {
        Guard.NotNull(baseTeacher);
        _baseTeacher = baseTeacher;
        // Note: Curriculum logic is implemented in CurriculumDistillationStrategy,
        // not in the teacher model. This is a simple wrapper around the base teacher.
    }

    /// <summary>
    /// Gets logits from the base teacher.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>Raw logits from the base teacher.</returns>
    public override Vector<T> GetLogits(Vector<T> input) => _baseTeacher.GetLogits(input);

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
                $"CurriculumTeacherModel cannot export computation graph because the base teacher " +
                $"({_baseTeacher.GetType().Name}) does not implement IJitCompilable<T>.");
        }

        if (!jitCompilable.SupportsJitCompilation)
        {
            throw new NotSupportedException(
                $"CurriculumTeacherModel cannot export computation graph because the base teacher " +
                $"({_baseTeacher.GetType().Name}) does not support JIT compilation.");
        }

        return jitCompilable.ExportComputationGraph(inputNodes);
    }
}

/// <summary>
/// Defines the curriculum learning strategy direction.
/// </summary>
/// <remarks>
/// <para>Note: This enum is maintained for backward compatibility. Curriculum logic
/// should be implemented in custom distillation strategies or training loops.</para>
/// </remarks>
public enum CurriculumStrategy
{
    /// <summary>
    /// Start with easy examples and gradually increase difficulty.
    /// </summary>
    EasyToHard,

    /// <summary>
    /// Start with hard examples and gradually decrease difficulty.
    /// </summary>
    HardToEasy
}
