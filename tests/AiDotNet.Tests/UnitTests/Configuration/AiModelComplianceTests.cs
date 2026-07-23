using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>Audit-2026-05 phase-2a slice 4 — compliance component isolation tests.</summary>
public class AiModelComplianceTests
{
    [Fact(Timeout = 30000)]
    public async Task InitialState_AllSlotsAreNull()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        Assert.Null(c.BiasDetector);
        Assert.Null(c.FairnessEvaluator);
        Assert.Null(c.InterpretabilityOptions);
        Assert.Null(c.AdversarialRobustnessConfiguration);
        Assert.Null(c.SafetyPipelineConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureBiasDetector_Stores()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        var detector = Mock.Of<IBiasDetector<double>>();
        c.ConfigureBiasDetector(detector);
        Assert.Same(detector, c.BiasDetector);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureFairnessEvaluator_Stores()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        var evaluator = Mock.Of<IFairnessEvaluator<double>>();
        c.ConfigureFairnessEvaluator(evaluator);
        Assert.Same(evaluator, c.FairnessEvaluator);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureInterpretability_NullAppliesDefault()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        c.ConfigureInterpretability(null);
        Assert.NotNull(c.InterpretabilityOptions);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureInterpretability_ExplicitOptions_Stored()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        var opts = new InterpretabilityOptions();
        c.ConfigureInterpretability(opts);
        Assert.Same(opts, c.InterpretabilityOptions);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureAdversarialRobustness_NullAppliesDefault()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        c.ConfigureAdversarialRobustness(null);
        Assert.NotNull(c.AdversarialRobustnessConfiguration);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureAdversarialRobustness_Explicit_Stored()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        var cfg = new AdversarialRobustnessConfiguration<double, Matrix<double>, Vector<double>>();
        c.ConfigureAdversarialRobustness(cfg);
        Assert.Same(cfg, c.AdversarialRobustnessConfiguration);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureSafety_NullActionLeavesEmptyConfig()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        c.ConfigureSafety(null);
        Assert.NotNull(c.SafetyPipelineConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureSafety_ActionInvokedOnNewConfig()
    {
        await Task.Yield();
        var c = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        SafetyConfig? captured = null;
        c.ConfigureSafety(sc => captured = sc);
        Assert.NotNull(captured);
        Assert.Same(c.SafetyPipelineConfig, captured);
    }

    [Fact(Timeout = 30000)]
    public async Task Interface_IsImplemented()
    {
        await Task.Yield();
        IAiModelCompliance<double, Matrix<double>, Vector<double>> c
            = new AiModelCompliance<double, Matrix<double>, Vector<double>>();
        var detector = Mock.Of<IBiasDetector<double>>();
        c.ConfigureBiasDetector(detector);
        Assert.Same(detector, c.BiasDetector);
    }
}
