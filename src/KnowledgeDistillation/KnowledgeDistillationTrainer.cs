using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Standard knowledge distillation trainer that uses a fixed teacher model to train a student.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This is the standard implementation of knowledge distillation.
/// It takes a large, accurate teacher model and uses it to train a smaller, faster student model.</para>
///
/// <para>The training process works as follows:
/// 1. For each input, get predictions from both teacher and student
/// 2. Compute distillation loss (how different are their predictions?)
/// 3. Update student parameters to minimize this loss
/// 4. Repeat until student learns to mimic teacher</para>
///
/// <para><b>Real-world Analogy:</b>
/// Think of this as an apprenticeship program. The master (teacher) demonstrates how to solve
/// problems, and the apprentice (student) learns by trying to replicate the master's approach.
/// The apprentice doesn't just learn the final answers, but also the reasoning process.</para>
///
/// <para><b>Benefits of Knowledge Distillation:</b>
/// - **Model Compression**: Deploy a 10x smaller model with &gt;90% of original accuracy
/// - **Faster Inference**: Smaller models run much faster on edge devices
/// - **Ensemble Distillation**: Combine knowledge from multiple teachers into one student
/// - **Transfer Learning**: Transfer knowledge across different architectures</para>
///
/// <para><b>Success Stories:</b>
/// - DistilBERT: 40% smaller than BERT, 97% of performance, 60% faster
/// - MobileNet: Distilled from ResNet, 10x fewer parameters, deployable on phones
/// - TinyBERT: 7.5x smaller than BERT, suitable for edge deployment</para>
/// </remarks>
public class KnowledgeDistillationTrainer<T> : KnowledgeDistillationTrainerBase<T, Vector<T>, Vector<T>>
{
    /// <summary>
    /// Initializes a new instance of the KnowledgeDistillationTrainer class.
    /// </summary>
    /// <param name="teacher">The teacher model to learn from.</param>
    /// <param name="distillationStrategy">The strategy for computing distillation loss.</param>
    /// <param name="checkpointConfig">Optional checkpoint configuration for automatic model saving during training.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Create a trainer by providing:
    /// 1. A trained teacher model (already performing well on your task)
    /// 2. A distillation strategy (defines how to transfer knowledge)
    /// 3. Optional checkpoint configuration (for automatic model saving)</para>
    ///
    /// <para>Example:
    /// <code>
    /// var teacher = new TeacherModelWrapper&lt;double&gt;(...);
    /// var distillationLoss = new DistillationLoss&lt;double&gt;(temperature: 3.0, alpha: 0.3);
    /// var trainer = new KnowledgeDistillationTrainer&lt;double&gt;(teacher, distillationLoss);
    /// </code>
    /// </para>
    ///
    /// <para>Example with automatic checkpointing:
    /// <code>
    /// var checkpointConfig = new DistillationCheckpointConfig
    /// {
    ///     SaveEveryEpochs = 5,
    ///     KeepBestN = 3
    /// };
    /// var trainer = new KnowledgeDistillationTrainer&lt;double&gt;(
    ///     teacher,
    ///     distillationLoss,
    ///     checkpointConfig: checkpointConfig
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public KnowledgeDistillationTrainer(
        ITeacherModel<Vector<T>, Vector<T>> teacher,
        IDistillationStrategy<T> distillationStrategy,
        DistillationCheckpointConfig? checkpointConfig = null,
        bool useEarlyStopping = false,
        double earlyStoppingMinDelta = 0.001,
        int earlyStoppingPatience = 5,
        int? seed = null)
        : base(teacher, distillationStrategy, checkpointConfig, useEarlyStopping, earlyStoppingMinDelta, earlyStoppingPatience, seed)
    {
    }

    /// <summary>
    /// Gets teacher predictions by calling the teacher model's GetLogits method.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="index">The index in the training batch (unused for standard distillation).</param>
    /// <returns>Teacher's logit predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> In standard distillation, we simply ask the teacher model
    /// for its predictions on each input. The teacher is frozen and doesn't change during training.</para>
    /// </remarks>
    protected override Vector<T> GetTeacherPredictions(Vector<T> input, int index)
    {
        return Teacher.GetLogits(input);
    }
}
