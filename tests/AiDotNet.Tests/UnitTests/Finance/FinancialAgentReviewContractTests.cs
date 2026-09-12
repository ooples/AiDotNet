using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

[Trait("category", "unit")]
public sealed class FinancialAgentReviewContractTests
{
    public enum EpsilonEndpoint { Start, End }

    public FinancialAgentReviewContractTests() => TestModuleInitializer.EnsureInitialized();

    public static IEnumerable<object[]> InvalidProbabilities()
    {
        foreach (var endpoint in new[] { EpsilonEndpoint.Start, EpsilonEndpoint.End })
        foreach (double value in new[] { -0.001, 1.001, double.NaN, double.NegativeInfinity, double.PositiveInfinity })
            yield return new object[] { endpoint, value };
    }

    [Theory]
    [MemberData(nameof(InvalidProbabilities))]
    public void Options_reject_every_nonprobability(EpsilonEndpoint endpoint, double value)
    {
        var options = Options(Dqn, 4, 3, seed: 6);
        if (endpoint == EpsilonEndpoint.Start) options.EpsilonStart = value;
        else options.EpsilonEnd = value;
        string parameter = endpoint == EpsilonEndpoint.Start
            ? nameof(options.EpsilonStart) : nameof(options.EpsilonEnd);
        Assert.Equal(parameter, Assert.Throws<ArgumentException>(options.Validate).ParamName);
    }

    [Theory]
    [InlineData(0.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(1.0, 0.0)]
    [InlineData(0.75, 0.25)]
    public void Options_preserve_valid_probability_endpoints(double start, double end)
    {
        var options = Options(Dqn, 4, 3, seed: 6);
        options.EpsilonStart = start;
        options.EpsilonEnd = end;
        options.Validate();
        Assert.Equal(start, options.EpsilonStart);
        Assert.Equal(end, options.EpsilonEnd);
    }

    [Theory]
    [InlineData(Dqn)]
    [InlineData(A2C)]
    [InlineData(Ppo)]
    [InlineData(Sac)]
    [InlineData(MarketMaking)]
    public void Shared_constructor_rejects_invalid_options_before_mutating_architecture(FinancialAgentKind kind)
    {
        var options = Options(kind, 4, 3, seed: 6);
        options.EpsilonStart = double.NaN;
        var architecture = Arch(4, 3);
        Assert.Equal(nameof(options.EpsilonStart), Assert.Throws<ArgumentException>(
            () => { using var unexpected = Create(kind, options, architecture); }).ParamName);
        Assert.Empty(architecture.Layers);
        Assert.Null(architecture.RandomSeed);
    }

    [Theory]
    [InlineData(Dqn)]
    [InlineData(A2C)]
    [InlineData(Ppo)]
    [InlineData(Sac)]
    [InlineData(MarketMaking)]
    public void Shared_constructor_honors_the_options_validation_override(FinancialAgentKind kind)
    {
        var options = new RejectingOptions { StateSize = 4, ActionSize = 3, Seed = 6 };
        var architecture = Arch(4, 3);
        Assert.Throws<ValidationProbeException>(() => { using var unexpected = Create(kind, options, architecture); });
        Assert.Equal(1, options.Calls);
        Assert.Empty(architecture.Layers);
        Assert.Null(architecture.RandomSeed);
    }

    [Fact]
    public void Negative_legacy_hidden_count_is_rejected_without_mutating_architecture()
    {
        using var agent = new LegacyOverrideAgent();
        var architecture = Arch(4, 3);
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => agent.Build(architecture, -1, 5));
        Assert.Equal("hiddenLayerCount", error.ParamName);
        Assert.Empty(architecture.Layers);
        Assert.Null(architecture.RandomSeed);
    }

    [Theory]
    [InlineData(0, 5)]
    [InlineData(3, 5)]
    public void Valid_legacy_hidden_count_and_width_keep_their_meaning(int count, int width)
    {
        using var agent = new LegacyOverrideAgent();
        var architecture = Arch(4, 3);
        agent.Build(architecture, count, width);
        Assert.Equal(Enumerable.Repeat(width, count).Concat(new[] { 3 }),
            architecture.Layers.Select(layer => layer.GetOutputShape()[0]));
    }

    [Fact]
    public void Sac_critic_clones_keep_nondefault_topology_and_independent_parameter_storage()
    {
        var options = Options(Sac, 4, 3, seed: 6);
        options.HiddenLayers = new[] { 7, 5 };
        using var agent = Create(Sac, options);
        var critics = Networks(agent, Sac, 4, 3).Where(x => x.Role != FinancialNetworkRole.Policy).ToArray();
        Assert.Equal(4, critics.Length);
        foreach (var entry in critics)
        {
            Assert.Equal(new[] { 7, 5, 1 }, entry.Network.Layers.Select(x => x.GetOutputShape()[0]));
            entry.Network.Predict(Tensor<double>.FromVector(State(7, 1)));
        }
        var snapshots = critics.Select(x => x.Network.GetParameters().ToArray()).ToArray();
        Assert.All(snapshots, snapshot => Assert.NotEmpty(snapshot));
        var changed = new Vector<double>((double[])snapshots[0].Clone());
        changed[0] += 0.25;
        critics[0].Network.UpdateParameters(changed);
        Assert.Equal(changed.ToArray(), critics[0].Network.GetParameters().ToArray());
        for (int i = 1; i < critics.Length; i++)
            Assert.Equal(snapshots[i], critics[i].Network.GetParameters().ToArray());
    }

    private sealed class ValidationProbeException : Exception { }

    private sealed class RejectingOptions : MarketMakingOptions<double>
    {
        [SetsRequiredMembers]
        public RejectingOptions() { }
        public int Calls { get; private set; }
        public override void Validate()
        {
            Calls++;
            throw new ValidationProbeException();
        }
    }

    private sealed class LegacyOverrideAgent : FinancialDQNAgent<double>
    {
        public LegacyOverrideAgent() : base(Arch(4, 3), FinancialAgentTestKit.Options(Dqn, 4, 3, 6)) { }
        public void Build(NeuralNetworkArchitecture<double> architecture, int count, int width) =>
            EnsureDefaultLayers(architecture, 4, 3, count, width);
    }
}
