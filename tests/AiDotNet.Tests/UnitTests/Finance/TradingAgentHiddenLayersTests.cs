using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// The trading agents build the hidden layers their options declare (#2147).
/// </summary>
/// <remarks>
/// TradingAgentOptions.HiddenLayers was declared and never read: DQN, A2C and SAC always built 2 x 64 ReLU
/// and PPO 2 x 64 tanh. The options copy constructors also dropped settings - DQN's and PPO's copied only
/// their own fields, and SAC's missed its own temperature and target settings.
/// </remarks>
public class TradingAgentHiddenLayersTests
{
    private static NeuralNetworkArchitecture<double> Flat(int inputs, int outputs)
        => new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputs, outputSize: outputs);

    private static int[] Widths(NeuralNetworkArchitecture<double> architecture)
        => architecture.Layers.Select(layer =>
        {
            var shape = layer.GetOutputShape();
            return shape[shape.Length - 1];
        }).ToArray();

    private static T Configure<T>(T options) where T : TradingAgentOptions<double>
    {
        options.StateSize = 4;
        options.ActionSize = 3;
        options.HiddenLayers = new[] { 8, 5 };
        options.BatchSize = 1;
        options.ReplayBufferSize = 16;
        options.WarmupSteps = 0;
        options.Seed = 1;
        return options;
    }

    [Fact(Timeout = 60000)]
    public async Task DQN_BuildsTheDeclaredHiddenLayers()
    {
        await Task.Yield();
        var architecture = Flat(4, 3);

        _ = new FinancialDQNAgent<double>(architecture, Configure(new FinancialDQNAgentOptions<double>()));

        Assert.Equal(new[] { 8, 5, 3 }, Widths(architecture));
    }

    [Fact(Timeout = 60000)]
    public async Task A2C_BuildsTheDeclaredHiddenLayersForActorAndCritic()
    {
        await Task.Yield();
        var actor = Flat(4, 3);
        var critic = Flat(4, 1);

        _ = new FinancialA2CAgent<double>(actor, critic, Configure(new FinancialA2CAgentOptions<double>()));

        Assert.Equal(new[] { 8, 5, 3 }, Widths(actor));
        Assert.Equal(new[] { 8, 5, 1 }, Widths(critic));
    }

    [Fact(Timeout = 60000)]
    public async Task PPO_BuildsTheDeclaredWidthsWithThePapersTanh()
    {
        await Task.Yield();
        var actor = Flat(4, 3);
        var critic = Flat(4, 1);

        _ = new FinancialPPOAgent<double>(actor, critic, Configure(new FinancialPPOAgentOptions<double>()));

        Assert.Equal(new[] { 8, 5, 3 }, Widths(actor));
        var hidden = actor.Layers.Take(2).Cast<LayerBase<double>>();
        Assert.All(hidden, layer => Assert.IsType<TanhActivation<double>>(layer.ScalarActivation));
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_FollowEachPaper()
    {
        await Task.Yield();
        // Schulman et al. 2017: two hidden layers of 64. Haarnoja et al. 2018: two of 256. The generic trading
        // default stays the declared 256 / 128 / 64.
        Assert.Equal(new[] { 64, 64 }, new FinancialPPOAgentOptions<double>().HiddenLayers);
        Assert.Equal(new[] { 256, 256 }, new FinancialSACAgentOptions<double>().HiddenLayers);
        Assert.Equal(new[] { 256, 128, 64 }, new FinancialDQNAgentOptions<double>().HiddenLayers);
    }

    [Fact(Timeout = 60000)]
    public async Task CopyConstructors_KeepEverySetting()
    {
        await Task.Yield();
        var sac = Configure(new FinancialSACAgentOptions<double>());
        sac.SACAlpha = 0.7;
        sac.AutoTuneAlpha = false;
        sac.Tau = 0.02;
        sac.InitialLogAlpha = -1.5;

        var sacCopy = new FinancialSACAgentOptions<double>(sac);

        Assert.Equal(0.7, sacCopy.SACAlpha);
        Assert.False(sacCopy.AutoTuneAlpha);
        Assert.Equal(0.02, sacCopy.Tau);
        Assert.Equal(-1.5, sacCopy.InitialLogAlpha);
        Assert.Equal(4, sacCopy.StateSize);
        Assert.Equal(new[] { 8, 5 }, sacCopy.HiddenLayers);
        Assert.NotSame(sac.HiddenLayers, sacCopy.HiddenLayers);

        var dqn = Configure(new FinancialDQNAgentOptions<double>());
        dqn.UseDoubleDQN = false;
        var dqnCopy = new FinancialDQNAgentOptions<double>(dqn);
        Assert.False(dqnCopy.UseDoubleDQN);
        Assert.Equal(4, dqnCopy.StateSize);
        Assert.Equal(new[] { 8, 5 }, dqnCopy.HiddenLayers);

        var ppo = Configure(new FinancialPPOAgentOptions<double>());
        ppo.NumEpochs = 7;
        var ppoCopy = new FinancialPPOAgentOptions<double>(ppo);
        Assert.Equal(7, ppoCopy.NumEpochs);
        Assert.Equal(new[] { 8, 5 }, ppoCopy.HiddenLayers);
    }
}
