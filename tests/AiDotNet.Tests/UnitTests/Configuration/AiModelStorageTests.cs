using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>Audit-2026-05 phase-2a slice 9 — storage component isolation tests.</summary>
public class AiModelStorageTests
{
    [Fact(Timeout = 30000)]
    public async Task InitialState_AllSlotsAreNull()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        Assert.Null(s.CacheConfig);
        Assert.Null(s.VersioningConfig);
        Assert.Null(s.ABTestingConfig);
        Assert.Null(s.ExperimentTracker);
        Assert.Null(s.ModelRegistry);
        Assert.Null(s.DataVersionControl);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureCaching_NullIsValid()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        s.ConfigureCaching(null);
        Assert.Null(s.CacheConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureCaching_ExplicitStored()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        var cfg = new CacheConfig();
        s.ConfigureCaching(cfg);
        Assert.Same(cfg, s.CacheConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureVersioning_ExplicitStored()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        var cfg = new VersioningConfig();
        s.ConfigureVersioning(cfg);
        Assert.Same(cfg, s.VersioningConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureABTesting_ExplicitStored()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        var cfg = new ABTestingConfig();
        s.ConfigureABTesting(cfg);
        Assert.Same(cfg, s.ABTestingConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureExperimentTracker_Stores()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        var tracker = Mock.Of<IExperimentTracker<double>>();
        s.ConfigureExperimentTracker(tracker);
        Assert.Same(tracker, s.ExperimentTracker);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureModelRegistry_Stores()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        var registry = Mock.Of<IModelRegistry<double, Matrix<double>, Vector<double>>>();
        s.ConfigureModelRegistry(registry);
        Assert.Same(registry, s.ModelRegistry);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureDataVersionControl_Stores()
    {
        await Task.Yield();
        var s = new AiModelStorage<double, Matrix<double>, Vector<double>>();
        var dvc = Mock.Of<IDataVersionControl<double>>();
        s.ConfigureDataVersionControl(dvc);
        Assert.Same(dvc, s.DataVersionControl);
    }
}
