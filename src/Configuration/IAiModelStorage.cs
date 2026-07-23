using AiDotNet.Deployment.Configuration;

namespace AiDotNet.Configuration;

/// <summary>
/// Component that owns the storage / artifact-management configuration for an AI model build:
/// caching, versioning, A/B testing, experiment tracking, model registry, and data version
/// control. Extracted from <c>AiModelBuilder</c> as slice 9 of the audit-2026-05 phase-2a DI
/// refactor.
/// </summary>
internal interface IAiModelStorage<T, TInput, TOutput>
{
    CacheConfig? CacheConfig { get; }
    VersioningConfig? VersioningConfig { get; }
    ABTestingConfig? ABTestingConfig { get; }
    IExperimentTracker<T>? ExperimentTracker { get; }
    IModelRegistry<T, TInput, TOutput>? ModelRegistry { get; }
    IDataVersionControl<T>? DataVersionControl { get; }

    void ConfigureCaching(CacheConfig? config);
    void ConfigureVersioning(VersioningConfig? config);
    void ConfigureABTesting(ABTestingConfig? config);
    void ConfigureExperimentTracker(IExperimentTracker<T> tracker);
    void ConfigureModelRegistry(IModelRegistry<T, TInput, TOutput> registry);
    void ConfigureDataVersionControl(IDataVersionControl<T> dataVersionControl);
}
