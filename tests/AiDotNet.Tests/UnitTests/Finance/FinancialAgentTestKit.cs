using System;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Shared construction helpers for the financial RL agent regression tests: one place that knows how each
/// agent is wired (actor/critic shapes, the SAC critic's state+action input, the market-making options type)
/// so every test builds agents exactly the way a downstream bake-off does.
/// </summary>
internal static class FinancialAgentTestKit
{
    internal const string Dqn = "DQN";
    internal const string A2C = "A2C";
    internal const string Ppo = "PPO";
    internal const string Sac = "SAC";
    internal const string MarketMaking = "MM";

    internal static NeuralNetworkArchitecture<double> Arch(int inputs, int outputs) =>
        new(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: inputs,
            outputSize: outputs);

    internal static TradingAgentOptions<double> Options(string kind, int stateSize, int actionSize, int? seed)
    {
        TradingAgentOptions<double> options = kind switch
        {
            Dqn => new FinancialDQNAgentOptions<double>(),
            A2C => new FinancialA2CAgentOptions<double>(),
            Ppo => new FinancialPPOAgentOptions<double>(),
            Sac => new FinancialSACAgentOptions<double>(),
            MarketMaking => new MarketMakingOptions<double>(),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown agent kind."),
        };

        options.StateSize = stateSize;
        options.ActionSize = actionSize;
        options.Seed = seed;
        options.ContinuousActions = kind is Sac or MarketMaking;
        return options;
    }

    internal static TradingAgentBase<double> Create(string kind, TradingAgentOptions<double> options) =>
        Create(kind, options, Arch(options.StateSize, options.ActionSize));

    internal static TradingAgentBase<double> Create(
        string kind,
        TradingAgentOptions<double> options,
        NeuralNetworkArchitecture<double> primary)
    {
        int s = options.StateSize;
        int a = options.ActionSize;
        return kind switch
        {
            Dqn => new FinancialDQNAgent<double>(primary, options),
            A2C => new FinancialA2CAgent<double>(primary, Arch(s, 1), options),
            Ppo => new FinancialPPOAgent<double>(primary, Arch(s, 1), options),
            Sac => new FinancialSACAgent<double>(primary, Arch(s + a, 1), options),
            MarketMaking => new MarketMakingAgent<double>(primary, (MarketMakingOptions<double>)options),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown agent kind."),
        };
    }

    /// <summary>A deterministic, bounded state vector (no RNG, so it is identical on every runtime).</summary>
    internal static Vector<double> State(int size, int salt)
    {
        var v = new Vector<double>(size);
        for (int i = 0; i < size; i++)
        {
            v[i] = Math.Sin(0.7 * (i + 1) + 1.3 * salt);
        }

        return v;
    }

    internal static Vector<double> OneHot(int size, int index)
    {
        var v = new Vector<double>(size);
        v[index] = 1.0;
        return v;
    }

    internal static int ArgMax(Vector<double> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary>
    /// A single-asset environment on a deterministic 8-bar zig-zag price path with a fixed start, so the
    /// environment itself contributes no randomness to a training run.
    /// </summary>
    internal static StockTradingEnvironment<double> ZigZagEnvironment(int bars = 80, int windowSize = 4)
    {
        var data = new Tensor<double>(new[] { bars, 1 });
        double price = 100.0;
        for (int i = 0; i < bars; i++)
        {
            data[i, 0] = price;
            price *= ((i / 4) % 2 == 0) ? 1.01 : 0.99;
        }

        return new StockTradingEnvironment<double>(
            data, windowSize, initialCapital: 1_000.0, tradeSize: 1.0,
            transactionCost: 0.0, allowShortSelling: true, randomStart: false, seed: 99);
    }
}
