using AiDotNet.Deployment.Configuration;

namespace AiDotNet.Configuration;

/// <summary>Default implementation of <see cref="IAiModelStorage{T,TInput,TOutput}"/>. Audit-2026-05 phase-2a slice 9.</summary>
public class AiModelStorage<T, TInput, TOutput> : IAiModelStorage<T, TInput, TOutput>
{
    public CacheConfig? CacheConfig { get; private set; }
    public VersioningConfig? VersioningConfig { get; private set; }
    public ABTestingConfig? ABTestingConfig { get; private set; }
    public IExperimentTracker<T>? ExperimentTracker { get; private set; }
    public IModelRegistry<T, TInput, TOutput>? ModelRegistry { get; private set; }
    public IDataVersionControl<T>? DataVersionControl { get; private set; }

    public void ConfigureCaching(CacheConfig? config) => CacheConfig = config;
    public void ConfigureVersioning(VersioningConfig? config) => VersioningConfig = config;
    public void ConfigureABTesting(ABTestingConfig? config) => ABTestingConfig = config;
    public void ConfigureExperimentTracker(IExperimentTracker<T> tracker) => ExperimentTracker = tracker;
    public void ConfigureModelRegistry(IModelRegistry<T, TInput, TOutput> registry) => ModelRegistry = registry;
    public void ConfigureDataVersionControl(IDataVersionControl<T> dataVersionControl) => DataVersionControl = dataVersionControl;
}
