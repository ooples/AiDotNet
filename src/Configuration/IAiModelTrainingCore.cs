using AiDotNet.Training.Memory;

namespace AiDotNet.Configuration;

/// <summary>
/// Component that owns the core training configuration for an AI model build: the model itself,
/// optimizer, regularization, fitness calculator, fit detector, training pipeline, training
/// monitor, checkpoint manager, and memory management. Extracted from <c>AiModelBuilder</c> as
/// slice 2 of the audit-2026-05 phase-2a DI refactor (see
/// <c>docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md</c>).
/// </summary>
/// <typeparam name="T">Element numeric type (e.g. <c>double</c> / <c>float</c>).</typeparam>
/// <typeparam name="TInput">Model input tensor type.</typeparam>
/// <typeparam name="TOutput">Model output tensor type.</typeparam>
/// <remarks>
/// <para>
/// This concern is the dependency root for slices 3 / 4 / 6 / 7 / 9 / 10 / 11 per the migration
/// plan: cross-validation, compliance evaluation, workflow orchestration (FL / distributed),
/// advanced learning, storage, observability, and agent / export each consume the trained model
/// or the optimizer in some way, so they all wait for this component to land before they can
/// migrate.
/// </para>
/// </remarks>
public interface IAiModelTrainingCore<T, TInput, TOutput>
{
    /// <summary>The configured model, or <c>null</c> if <c>ConfigureModel</c> hasn't been called.</summary>
    IFullModel<T, TInput, TOutput>? Model { get; }

    /// <summary>The configured optimizer, or <c>null</c> if <c>ConfigureOptimizer</c> hasn't been called.</summary>
    IOptimizer<T, TInput, TOutput>? Optimizer { get; }

    /// <summary>The configured regularization strategy, or <c>null</c> if not configured.</summary>
    IRegularization<T, TInput, TOutput>? Regularization { get; }

    /// <summary>The configured fitness calculator, or <c>null</c> if not configured.</summary>
    IFitnessCalculator<T, TInput, TOutput>? FitnessCalculator { get; }

    /// <summary>The configured fit detector (over- / under-fitting diagnostic), or <c>null</c> if not configured.</summary>
    IFitDetector<T, TInput, TOutput>? FitDetector { get; }

    /// <summary>The configured training-pipeline configuration (custom stage definitions), or <c>null</c> for the default linear training pipeline.</summary>
    TrainingPipelineConfiguration<T, TInput, TOutput>? TrainingPipelineConfiguration { get; }

    /// <summary>The configured checkpoint manager, or <c>null</c> if not configured (no checkpointing).</summary>
    ICheckpointManager<T, TInput, TOutput>? CheckpointManager { get; }

    /// <summary>The configured training-memory configuration (gradient checkpointing, activation pooling, sharding), or <c>null</c> for default settings.</summary>
    TrainingMemoryConfig? MemoryConfig { get; }

    /// <summary>The configured training monitor, or <c>null</c> if not configured (no monitoring callbacks).</summary>
    ITrainingMonitor<T>? TrainingMonitor { get; }

    /// <summary>Sets the model. <c>null</c> is allowed (clears the slot).</summary>
    void ConfigureModel(IFullModel<T, TInput, TOutput> model);

    /// <summary>Sets the optimizer.</summary>
    void ConfigureOptimizer(IOptimizer<T, TInput, TOutput> optimizationAlgorithm);

    /// <summary>Sets the regularization strategy.</summary>
    void ConfigureRegularization(IRegularization<T, TInput, TOutput> regularization);

    /// <summary>Sets the fitness calculator.</summary>
    void ConfigureFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> calculator);

    /// <summary>Sets the fit detector.</summary>
    void ConfigureFitDetector(IFitDetector<T, TInput, TOutput> detector);

    /// <summary>Sets the training-pipeline configuration. <c>null</c> uses the default linear training pipeline.</summary>
    void ConfigureTrainingPipeline(TrainingPipelineConfiguration<T, TInput, TOutput>? configuration);

    /// <summary>Sets the checkpoint manager.</summary>
    void ConfigureCheckpointManager(ICheckpointManager<T, TInput, TOutput> manager);

    /// <summary>Sets the training-memory configuration. <c>null</c> uses default settings.</summary>
    void ConfigureMemoryManagement(TrainingMemoryConfig? configuration);

    /// <summary>Sets the training monitor.</summary>
    void ConfigureTrainingMonitor(ITrainingMonitor<T> monitor);
}
