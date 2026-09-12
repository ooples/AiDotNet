using System;
using AiDotNet.Finance.Trading.Agents;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Regression tests for the continuous-action exploration noise of <see cref="FinancialSACAgent{T}"/> and
/// <see cref="MarketMakingAgent{T}"/>. The noise used to be <c>U[0, scale)</c>: mean +scale/2 and never
/// negative, so every exploratory position was biased long (and, because both agents regress the actor onto
/// the actions they took, the bias compounded into the policy on every update). It was also drawn from an
/// unseeded RNG.
/// </summary>
public sealed class FinancialContinuousExplorationNoiseTests
{
    [Theory]
    [InlineData(Sac)]
    [InlineData(MarketMaking)]
    [Trait("category", "unit")]
    public void Training_noise_is_zero_mean_and_symmetric(FinancialAgentKind kind)
    {
        const int stateSize = 4;
        const int actionSize = 3;
        var options = Options(kind, stateSize, actionSize, seed: 5);
        using var agent = Create(kind, options);
        var state = State(stateSize, salt: 2);
        var greedy = agent.SelectAction(state, training: false);

        const int draws = 3000;
        double sum = 0;
        double sumSq = 0;
        int negatives = 0;
        int samples = 0;
        for (int i = 0; i < draws; i++)
        {
            var action = agent.SelectAction(state, training: true);
            for (int j = 0; j < actionSize; j++)
            {
                double noise = action[j] - greedy[j];
                sum += noise;
                sumSq += noise * noise;
                if (noise < 0)
                {
                    negatives++;
                }

                samples++;
            }
        }

        double mean = sum / samples;
        double std = Math.Sqrt(Math.Max(0.0, (sumSq / samples) - (mean * mean)));
        double negativeFraction = negatives / (double)samples;

        Assert.True(Math.Abs(mean) < 0.01, $"{kind} exploration noise mean {mean:F4} is biased (expected ~0).");
        Assert.True(negativeFraction > 0.45 && negativeFraction < 0.55,
            $"{kind} exploration noise is one-sided: {negativeFraction:P1} of draws were negative.");
        Assert.True(std > 0.0, $"{kind} exploration noise has zero spread.");
    }
}
