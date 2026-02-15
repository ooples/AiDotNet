
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Wraps an existing trained IFullModel to act as a teacher for knowledge distillation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class takes any trained IFullModel and adapts it to work
/// as a teacher in knowledge distillation. The teacher model should already be trained and
/// perform well on your task.</para>
///
/// <para><b>Architecture Note:</b> This is a lightweight adapter that bridges IFullModel
/// to ITeacherModel. It simply delegates GetLogits() to the underlying model's Predict() method,
/// since in this architecture, predictions and logits are equivalent.</para>
///
/// <para><b>Real-world Example:</b>
/// Imagine you have a large, accurate neural network trained on your dataset. You can wrap it
/// with TeacherModelWrapper and use it to train a smaller, faster student model that retains
/// most of the accuracy but runs much faster.</para>
///
/// <para>Common teacher-student scenarios:
/// - Large neural network (teacher) → Smaller network (student): 40-60% smaller, 95-97% of performance
/// - Deep network (teacher) → Shallow network (student): 10x faster inference
/// - Ensemble (teacher) → Single model (student): Deployable on resource-constrained devices</para>
/// </remarks>
public class TeacherModelWrapper<T> : ITeacherModel<Vector<T>, Vector<T>>
{
    private readonly Func<Vector<T>, Vector<T>> _forwardFunc;

    /// <summary>
    /// Gets the number of output dimensions (e.g., number of classes for classification).
    /// </summary>
    public int OutputDimension { get; }

    /// <summary>
    /// Initializes a new instance of the TeacherModelWrapper class from a forward function.
    /// </summary>
    /// <param name="forwardFunc">Function that performs forward pass and returns logits.</param>
    /// <param name="outputDimension">The number of output dimensions (classes).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor lets you create a teacher from any prediction
    /// function. The forward function should take input and return logits (raw outputs).</para>
    ///
    /// <para>Example usage:
    /// <code>
    /// var teacher = new TeacherModelWrapper&lt;double&gt;(
    ///     forwardFunc: input => myTrainedModel.Predict(input),
    ///     outputDimension: 10 // 10 classes (e.g., CIFAR-10)
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public TeacherModelWrapper(
        Func<Vector<T>, Vector<T>> forwardFunc,
        int outputDimension)
    {
        if (outputDimension <= 0)
            throw new ArgumentException("Output dimension must be positive", nameof(outputDimension));

        Guard.NotNull(forwardFunc);
        _forwardFunc = forwardFunc;
        OutputDimension = outputDimension;
    }

    /// <summary>
    /// Gets the teacher's raw logits (pre-softmax outputs) for the given input.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <returns>Raw logits before applying softmax.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Logits are the raw numerical outputs from a neural network
    /// before converting them to probabilities. They're preferred for distillation because:
    /// 1. They preserve more information than probabilities
    /// 2. They're numerically more stable
    /// 3. Temperature scaling works better on logits</para>
    ///
    /// <para><b>Architecture Note:</b> This method simply delegates to the wrapped model's
    /// Predict() method. In this architecture, predictions and logits are equivalent.</para>
    /// </remarks>
    public Vector<T> GetLogits(Vector<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        return _forwardFunc(input);
    }
}
