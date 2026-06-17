using AiDotNet.Training.Memory;

namespace AiDotNet.Configuration;

/// <summary>
/// Default implementation of <see cref="IAiModelTrainingCore{T,TInput,TOutput}"/>. Mirrors the
/// pre-refactor inline storage that <c>AiModelBuilder</c> used for its model / optimizer /
/// regularization / fitness / fit-detector / training-pipeline / checkpoint / memory / monitor
/// fields. Audit-2026-05 phase-2a slice 2 (see
/// <c>docs/internal/audit-2026-05-phase2a-aimodelbuilder-refactor.md</c>).
/// </summary>
/// <typeparam name="T">Element numeric type.</typeparam>
/// <typeparam name="TInput">Model input tensor type.</typeparam>
/// <typeparam name="TOutput">Model output tensor type.</typeparam>
public class AiModelTrainingCore<T, TInput, TOutput> : IAiModelTrainingCore<T, TInput, TOutput>
{
    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput>? Model { get; private set; }

    /// <inheritdoc/>
    public IOptimizer<T, TInput, TOutput>? Optimizer { get; private set; }

    /// <inheritdoc/>
    public IRegularization<T, TInput, TOutput>? Regularization { get; private set; }

    /// <inheritdoc/>
    public IFitnessCalculator<T, TInput, TOutput>? FitnessCalculator { get; private set; }

    /// <inheritdoc/>
    public IFitDetector<T, TInput, TOutput>? FitDetector { get; private set; }

    /// <inheritdoc/>
    public TrainingPipelineConfiguration<T, TInput, TOutput>? TrainingPipelineConfiguration { get; private set; }

    /// <inheritdoc/>
    public ICheckpointManager<T, TInput, TOutput>? CheckpointManager { get; private set; }

    /// <inheritdoc/>
    public TrainingMemoryConfig? MemoryConfig { get; private set; }

    /// <inheritdoc/>
    public ITrainingMonitor<T>? TrainingMonitor { get; private set; }

    /// <inheritdoc/>
    public void ConfigureModel(IFullModel<T, TInput, TOutput> model) => Model = model;

    /// <inheritdoc/>
    public void ConfigureOptimizer(IOptimizer<T, TInput, TOutput> optimizationAlgorithm)
        => Optimizer = optimizationAlgorithm;

    /// <inheritdoc/>
    public void ConfigureRegularization(IRegularization<T, TInput, TOutput> regularization)
        => Regularization = regularization;

    /// <inheritdoc/>
    public void ConfigureFitnessCalculator(IFitnessCalculator<T, TInput, TOutput> calculator)
        => FitnessCalculator = calculator;

    /// <inheritdoc/>
    public void ConfigureFitDetector(IFitDetector<T, TInput, TOutput> detector)
        => FitDetector = detector;

    /// <inheritdoc/>
    public void ConfigureTrainingPipeline(TrainingPipelineConfiguration<T, TInput, TOutput>? configuration)
        => TrainingPipelineConfiguration = configuration;

    /// <inheritdoc/>
    public void ConfigureCheckpointManager(ICheckpointManager<T, TInput, TOutput> manager)
        => CheckpointManager = manager;

    /// <inheritdoc/>
    public void ConfigureMemoryManagement(TrainingMemoryConfig? configuration)
        => MemoryConfig = configuration;

    /// <inheritdoc/>
    public void ConfigureTrainingMonitor(ITrainingMonitor<T> monitor)
        => TrainingMonitor = monitor;
}
