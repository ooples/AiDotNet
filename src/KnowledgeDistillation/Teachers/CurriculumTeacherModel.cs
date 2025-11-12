using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
        _baseTeacher = baseTeacher ?? throw new ArgumentNullException(nameof(baseTeacher));
        // Note: Curriculum logic is implemented in CurriculumDistillationStrategy,
        // not in the teacher model. This is a simple wrapper around the base teacher.
    }

    /// <summary>
    /// Gets logits from the base teacher.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>Raw logits from the base teacher.</returns>
    public override Vector<T> GetLogits(Vector<T> input) => _baseTeacher.GetLogits(input);
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
