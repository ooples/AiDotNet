using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Config;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Base implementation for Reptile meta-learning trainers.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a first-order meta-learning algorithm (Nichol et al., 2018) that provides
/// a simple yet effective approach to few-shot learning. It works by repeatedly moving
/// model parameters toward task-adapted parameters, causing them to converge to an
/// initialization that enables rapid adaptation.
/// </para>
/// <para><b>Key Advantages:</b>
/// - Simpler than MAML (no second-order derivatives)
/// - Computationally efficient
/// - Works with any gradient-based model
/// - Strong empirical performance on few-shot tasks
/// </para>
/// <para><b>Algorithm Overview:</b>
/// <code>
/// Initialize θ (meta-parameters)
/// for each meta-iteration:
///     Sample batch of B tasks
///     for each task i in batch:
///         θ_i = θ (clone parameters)
///         for k = 1 to K (inner steps):
///             θ_i = θ_i - α∇L(θ_i, support_set_i)
///         Δθ_i = θ_i - θ
///     θ = θ + ε * Average(Δθ_i) (meta-update)
/// return θ
/// </code>
/// </para>
/// </remarks>
public abstract class ReptileTrainerBase<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the ReptileTrainerBase with a configuration object.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluation.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object with all hyperparameters. If null, uses default ReptileTrainerConfig.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Reptile meta-learning trainer with your model and settings.
    ///
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> The neural network or model you want to train for fast adaptation
    /// - <b>lossFunction:</b> How to measure prediction errors (e.g., MSE for regression, CrossEntropy for classification)
    /// - <b>dataLoader:</b> Provides different tasks for meta-training (N-way K-shot episodic sampling)
    /// - <b>config:</b> Learning rates and steps - can be left as default for standard meta-learning
    ///
    /// The configuration controls two learning processes:
    /// - Inner loop: How the model adapts to each individual task
    /// - Outer loop: How the meta-parameters improve across tasks
    /// </para>
    /// </remarks>
    protected ReptileTrainerBase(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        IMetaLearnerConfig<T>? config = null)
        : base(metaModel, lossFunction, dataLoader, config ?? new ReptileTrainerConfig<T>())
    {
    }
}
