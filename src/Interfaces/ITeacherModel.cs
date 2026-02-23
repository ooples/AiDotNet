namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a trained teacher model for knowledge distillation.
/// </summary>
/// <typeparam name="TInput">The input data type (e.g., Vector, Matrix, Tensor).</typeparam>
/// <typeparam name="TOutput">The output data type (typically logits as Vector or Matrix).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Knowledge distillation is a technique where a smaller "student" model
/// learns from a larger "teacher" model. The teacher model acts as a guide, sharing not just the correct
/// answers but also its "knowledge" about the relationships between different classes.</para>
///
/// <para>Think of it like a master teaching an apprentice. The teacher doesn't just tell the apprentice
/// the final answer, but shares their reasoning and understanding, which helps the apprentice learn more effectively.</para>
///
/// <para><b>Architecture Note:</b> This interface defines a lightweight contract for teacher models
/// in knowledge distillation. Teachers are inference-only - they provide predictions but don't need
/// training capabilities. For wrapping a trained IFullModel as a teacher, use TeacherModelWrapper.</para>
///
/// <para><b>Design Principles:</b>
/// - Teachers are frozen/pre-trained - no training methods needed
/// - Temperature scaling handled by distillation strategy, not teacher
/// - Focuses on core functionality: get predictions and report output dimension
/// - Avoids type-unsafe methods (no object? returns)</para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("TeacherModel")]
public interface ITeacherModel<TInput, TOutput>
{
    /// <summary>
    /// Gets the teacher's raw logits (pre-softmax outputs) for the given input.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <returns>Raw logits before applying softmax activation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Logits are the raw numerical outputs from a neural network
    /// before they're converted to probabilities. They're used in distillation because they
    /// contain richer information than final probabilities.</para>
    ///
    /// <para><b>Architecture Note:</b> For teachers wrapped from IFullModel, this method
    /// simply calls the underlying model's Predict() method. Logits and predictions are
    /// equivalent in this architecture.</para>
    ///
    /// <para>Example: If the teacher is a 10-class classifier, GetLogits returns a Vector&lt;T&gt;
    /// of length 10 containing the raw pre-softmax scores for each class.</para>
    /// </remarks>
    TOutput GetLogits(TInput input);

    /// <summary>
    /// Gets the number of output dimensions (e.g., number of classes for classification).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many outputs the teacher produces.
    /// For a classification task with 10 classes (like CIFAR-10), this would return 10.</para>
    ///
    /// <para>The student model should typically have the same output dimension as the teacher,
    /// though some distillation techniques support dimension mismatch with projection layers.</para>
    /// </remarks>
    int OutputDimension { get; }
}
