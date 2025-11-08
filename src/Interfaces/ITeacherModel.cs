namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a trained teacher model for knowledge distillation.
/// </summary>
/// <typeparam name="TInput">The input data type (e.g., Vector, Matrix, Tensor).</typeparam>
/// <typeparam name="TOutput">The output data type (typically logits or probabilities as Vector or Matrix).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Knowledge distillation is a technique where a smaller "student" model
/// learns from a larger "teacher" model. The teacher model acts as a guide, sharing not just the correct
/// answers but also its "knowledge" about the relationships between different classes.</para>
///
/// <para>Think of it like a master teaching an apprentice. The teacher doesn't just tell the apprentice
/// the final answer, but shares their reasoning and understanding, which helps the apprentice learn more effectively.</para>
///
/// <para>This interface defines what a teacher model must be able to do:
/// - Provide raw predictions (logits) that can be used for distillation
/// - Provide "soft" predictions with adjustable confidence (using temperature)
/// - Extract intermediate layer features for deeper knowledge transfer
/// - Expose attention patterns in transformer models</para>
/// </remarks>
public interface ITeacherModel<in TInput, out TOutput>
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
    /// </remarks>
    TOutput GetLogits(TInput input);

    /// <summary>
    /// Gets the teacher's soft predictions (probabilities) for the given input with temperature scaling.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="temperature">Softmax temperature for softening predictions. Higher values (2-10) produce
    /// softer distributions that reveal more about the model's uncertainty.</param>
    /// <returns>Probability distribution with temperature scaling applied.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls how "soft" or "hard" the predictions are:
    /// - Low temperature (T=1): Sharp predictions (e.g., [0.95, 0.03, 0.02])
    /// - High temperature (T=5): Softer predictions (e.g., [0.60, 0.25, 0.15])</para>
    ///
    /// <para>Softer predictions are better for distillation because they reveal what the teacher
    /// thinks about relationships between classes. For example, if predicting "dog", a soft prediction
    /// might show that "cat" is more likely than "car", revealing semantic relationships.</para>
    /// </remarks>
    TOutput GetSoftPredictions(TInput input, double temperature = 1.0);

    /// <summary>
    /// Gets intermediate layer activations (features) from the teacher model.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="layerName">The name or identifier of the layer to extract features from.</param>
    /// <returns>Feature map from the specified layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Neural networks learn hierarchical features in their layers.
    /// Early layers might learn simple patterns (edges, colors), while deeper layers learn
    /// complex concepts (shapes, objects).</para>
    ///
    /// <para>By matching intermediate features, the student can learn to think more like the
    /// teacher, not just produce similar final outputs. This is called "feature distillation"
    /// or "hint learning".</para>
    /// </remarks>
    object? GetFeatures(TInput input, string layerName);

    /// <summary>
    /// Gets attention weights from the teacher model (for transformer architectures).
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <param name="layerName">The name of the attention layer to extract weights from.</param>
    /// <returns>Attention weight matrix from the specified layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Attention mechanisms help models focus on relevant parts
    /// of the input. For example, when translating "The cat sat on the mat", the model
    /// learns which words to pay attention to for each output word.</para>
    ///
    /// <para>By transferring attention patterns from teacher to student, the student learns
    /// what parts of the input are important, improving its understanding. This is especially
    /// useful for language models like BERT, GPT, and transformers.</para>
    /// </remarks>
    object? GetAttentionWeights(TInput input, string layerName);

    /// <summary>
    /// Gets the number of output dimensions (e.g., number of classes for classification).
    /// </summary>
    int OutputDimension { get; }
}
