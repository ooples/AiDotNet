namespace AiDotNet.Interfaces;

using AiDotNet.Compression.KnowledgeDistillation;

/// <summary>
/// Interface for models that can be students in knowledge distillation.
/// </summary>
/// <remarks>
/// <para>
/// This interface should be implemented by models that can learn from teacher models
/// in knowledge distillation.
/// </para>
/// <para><b>For Beginners:</b> This marks a model as able to learn from a teacher model.
///
/// Models that implement this interface know how to:
/// - Learn from a teacher model's soft targets
/// - Update their parameters based on distillation loss
/// - Balance soft targets with hard labels
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IDistillableStudent<T, TInput, TOutput>
    where T : unmanaged
{
    /// <summary>
    /// Learns from teacher outputs and optional hard targets.
    /// </summary>
    /// <param name="inputs">The input data.</param>
    /// <param name="teacherOutputs">The soft targets from the teacher model.</param>
    /// <param name="hardTargets">Optional hard targets (true labels).</param>
    /// <param name="parameters">Parameters for the distillation process.</param>
    /// <remarks>
    /// <para>
    /// This method updates the student model's parameters based on the teacher's outputs
    /// and optional hard targets.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the student learns from the teacher.
    /// 
    /// For each input:
    /// - The teacher has generated a probability distribution (soft target)
    /// - The student learns to match this distribution
    /// - If hard targets are provided, the student also learns from them
    /// - The parameters control how these learning signals are balanced
    /// </para>
    /// </remarks>
    void LearnFromTeacher(
        TInput[] inputs,
        TOutput[] teacherOutputs,
        TOutput[]? hardTargets,
        DistillationParameters parameters);
}