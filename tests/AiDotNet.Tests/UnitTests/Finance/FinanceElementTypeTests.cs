using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Finance.Tokenization;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Finance.Trading.Factors;
using AiDotNet.Finance.Volatility;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Element-type reference tests for the finance code converted to engine ops. Counts and indices must stay exact
/// past 2048 in Half, guards must stay positive in Half, and random draws and constant schedules must be formed in
/// double and rounded to the element type once, as master did.
/// </summary>
public class FinanceElementTypeTests
{
    // ---------------- policy agents: draws rounded once ----------------

    [Fact(Timeout = 60000)]
    public async Task FeedforwardPolicyAgent_CreateRandom_Float_RoundsEachScaledDrawOnce()
    {
        await Task.CompletedTask;
        double stddev = Math.Sqrt(2.0 / 8);
        var weights = FeedforwardPolicyAgent<float>.CreateRandom(new[] { 64, 8 }, stddev, 11);

        var rng = RandomHelper.CreateSeededRandom(11);
        for (int i = 0; i < weights.Length; i++)
        {
            float expected = (float)((rng.NextDouble() * 2 - 1) * stddev);
            Assert.Equal(expected, weights[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task RecurrentPolicyAgent_CreateRandom_Float_RoundsEachScaledDrawOnce()
    {
        await Task.CompletedTask;
        double stddev = Math.Sqrt(2.0 / 32);
        var weights = RecurrentPolicyAgent<float>.CreateRandom(new[] { 16, 32 }, stddev, 7);

        var rng = RandomHelper.CreateSeededRandom(7);
        for (int i = 0; i < weights.Length; i++)
        {
            float expected = (float)((rng.NextDouble() * 2 - 1) * stddev);
            Assert.Equal(expected, weights[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FeedforwardPolicyAgent_ExplorationNoise_Float_RoundsSigmaTimesZOnce()
    {
        await Task.CompletedTask;

        // A zero state gives tanh(meanW * tanh(W1 * 0 + 0) + 0) = 0, so the explored action is exactly the noise.
        const int actionDim = 128;
        const double sigma = 0.1;
        var agent = new FeedforwardPolicyAgent<float>(4, actionDim, hidden: 8, explorationSigma: sigma, seed: 3);

        var action = agent.SelectAction(new Vector<float>(4), explore: true);

        var rng = RandomHelper.CreateSeededRandom(3);
        for (int a = 0; a < actionDim; a++)
        {
            float expected = (float)(sigma * AiDotNet.Extensions.RandomExtensions.NextGaussian(rng));
            Assert.Equal(expected, action[a]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task TradingAgentBase_ExplorationNoise_Float_RoundsSigmaTimesZOnce()
    {
        await Task.CompletedTask;

        // Two agents built alike share the seeded stream: one supplies the standard-normal draws, the other adds
        // sigma * z to a zero action, which must be (float)(sigma * z).
        const int size = 64;
        const double sigma = 0.05;
        using var noisy = MarketMaking();
        using var reference = MarketMaking();
        var addNoise = typeof(TradingAgentBase<float>).GetMethod(
            "AddGaussianExplorationNoise", BindingFlags.Instance | BindingFlags.NonPublic, null,
            new[] { typeof(Vector<float>), typeof(double) }, null)
            ?? throw new InvalidOperationException("AddGaussianExplorationNoise(Vector<T>, double) not found.");
        var nextNormal = typeof(TradingAgentBase<float>).GetMethod(
            "NextStandardNormal", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("NextStandardNormal not found.");

        var result = (Vector<float>)(addNoise.Invoke(noisy, new object[] { new Vector<float>(size), sigma })
            ?? throw new InvalidOperationException("AddGaussianExplorationNoise returned null."));

        for (int i = 0; i < size; i++)
        {
            double z = (double)(nextNormal.Invoke(reference, Array.Empty<object>())
                ?? throw new InvalidOperationException("NextStandardNormal returned null."));
            Assert.Equal((float)(sigma * z), result[i]);
        }
    }

    private static MarketMakingAgent<float> MarketMaking() => new(new MarketMakingOptions<float>
    {
        StateSize = 4,
        ActionSize = 3,
        Seed = 7,
        ContinuousActions = true,
    });

    // ---------------- guards that must stay positive in Half ----------------

    [Fact(Timeout = 60000)]
    public async Task PolicyGradientLossWeights_Half_ConstantReturnsGiveZeroWeights()
    {
        await Task.CompletedTask;

        // Constant (zero) returns whiten to 0 / (0 + floor) = 0. In Half the 1e-8 floor rounded to zero and the
        // scale 0.5 / sigma^2 / std overflowed, so the weights came out NaN.
        var rewards = new[] { Half.Zero, Half.Zero, Half.Zero, Half.Zero };

        var weights = PolicyGradientRollout<Half>.LossWeights(rewards, (Half)0.99, 0.2);

        Assert.Equal(4, weights.Length);
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(0.0, (double)weights[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task RiskParity_Half_ZeroVolatilityAssetGetsAFiniteDominantWeight()
    {
        await Task.CompletedTask;

        // Asset 0 is flat (zero volatility), asset 1 moves. In Half the 1e-6 vol floor is subnormal and 1 / 1e-6
        // overflows, so the weights were inf / inf = NaN. With the floor raised to the smallest normal Half the flat
        // asset takes almost all the weight (16384 against about 2.5).
        var policy = BaselinePolicies.RiskParity<Half>(windowSize: 4, totalColumns: 2, tradableCount: 2);
        var state = new Vector<Half>(new[]
        {
            (Half)1, (Half)1,
            (Half)1, (Half)1.5,
            (Half)1, (Half)1,
            (Half)1, (Half)1.5,
        });

        var weights = policy(state);

        Assert.False(Half.IsNaN(weights[0]) || Half.IsNaN(weights[1]), "risk-parity weights are NaN");
        Assert.True((double)weights[0] > 0.99, $"flat asset weight {weights[0]}");
        Assert.InRange((double)weights[0] + (double)weights[1], 0.99, 1.01);
    }

    // ---------------- counts and indices past 2048 in Half ----------------

    [Fact(Timeout = 60000)]
    public async Task Momentum_Half_CountsWinnersPast2048()
    {
        await Task.CompletedTask;

        // 2049 winners each get 1 / 2049. A 0/1 sum in Half gives 2048, i.e. 2^-11, which is not Half(1 / 2049).
        const int assets = 2049;
        var policy = BaselinePolicies.Momentum<Half>(windowSize: 2, totalColumns: assets, tradableCount: assets);
        var state = new Vector<Half>(Enumerable.Range(0, 2 * assets).Select(i => i < assets ? (Half)1 : (Half)2).ToArray());

        var weights = policy(state);

        var expected = (Half)(1.0 / assets);
        Assert.NotEqual((Half)(1.0 / 2048), expected);
        for (int i = 0; i < assets; i++)
        {
            Assert.Equal(expected, weights[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task SharpeRatio_Half_CountsReturnsPast2048()
    {
        await Task.CompletedTask;

        // 3000 returns: +1 and -0.5 once each, zero otherwise. mean = 1/6000, sample variance
        // (1.25 - 3000 * mean^2) / 2999, Sharpe = mean / std * sqrt(252) = 0.12960. A count stuck at 2048 gives 0.157.
        var history = Enumerable.Repeat((Half)1, 3001).ToArray();
        history[1000] = (Half)2;

        double mean = 1.0 / 6000;
        double expected = mean / Math.Sqrt((1.25 - (3000 * mean * mean)) / 2999) * Math.Sqrt(252);

        var sharpe = TradingAgentBase<Half>.ComputeSharpeRatio(history, 0.0);

        Assert.Equal(expected, (double)sharpe, 0.004);
    }

    [Fact(Timeout = 60000)]
    public async Task MaskedMae_Half_CountsValidEntriesPast2048()
    {
        await Task.CompletedTask;

        // 3000 valid labels, one absolute error of 3: MAE = 3 / 3000 = 0.001 (3 / 2048 = 0.00146 with a Half count).
        const int n = 3000;
        var labels = new Vector<Half>(Enumerable.Repeat((Half)1, n).ToArray());
        var predictions = new Vector<Half>(Enumerable.Repeat((Half)1, n).ToArray());
        predictions[0] = (Half)4;

        var mae = StockformerMultiTaskLoss<Half>.MaskedMae(predictions, labels);

        Assert.Equal(0.001, (double)mae, 1e-5);
    }

    [Fact(Timeout = 60000)]
    public async Task CrossEntropy_Half_PicksTheTargetClassPast2048()
    {
        await Task.CompletedTask;

        // Target class 2048 has logit 0 and class 2049 logit 1. Class ids held in Half merge 2049 into 2048, so the
        // picked logit became 0 + 1. The loss is logsumexp(row) - 0.
        const int classes = 2050;
        var logits = new Half[classes];
        logits[2049] = (Half)1;
        var logitVector = new Vector<Half>(logits);

        var loss = StockformerMultiTaskLoss<Half>.CrossEntropy(logitVector, new Vector<Half>(new[] { (Half)2048 }), classes);

        var row = new Tensor<Half>(logits, new[] { 1, classes });
        var logSumExp = AiDotNetEngine.Current.TensorLogSumExp(row, 1, keepDims: false)[0];
        Assert.True(Math.Abs((double)loss - (double)logSumExp) < 0.01, $"loss {loss}, logsumexp {logSumExp}");
    }

    [Fact(Timeout = 60000)]
    public async Task DqnTemporalDifferenceTargets_Half_UseIntActionIndicesPast2048()
    {
        await Task.CompletedTask;

        // The greedy next action is 2049 (selection), worth 7 to the target network; action 2048 is worth 5.
        // Indices held in Half merge 2049 into 2048 and summed both values (12).
        const int actions = 2050;
        var shape = new[] { 1, actions };
        var currentQ = new Tensor<Half>(shape);
        var selection = new Tensor<Half>(shape);
        selection[0, 2049] = (Half)1;
        var evaluation = new Tensor<Half>(shape);
        evaluation[0, 2048] = (Half)5;
        evaluation[0, 2049] = (Half)7;
        var taken = new Tensor<Half>(shape);
        taken[0, 0] = (Half)1;

        var targets = FinancialDQNAgent<Half>.TemporalDifferenceTargets(
            currentQ, selection, evaluation, Enumerable.Repeat(true, actions).ToArray(),
            new Tensor<Half>(new[] { Half.Zero }, new[] { 1 }), new[] { false }, taken, (Half)1);

        Assert.Equal(7.0, (double)targets[0, 0]);
        for (int a = 1; a < actions; a++)
        {
            Assert.Equal(0.0, (double)targets[0, a]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task StockformerCausalWindow_Half_KeepsPositionsExactPast2048()
    {
        await Task.CompletedTask;

        // Kernel width 2: row t averages u = t - 1 and t. Positions held in Half put 2047, 2048 and 2049 at lag 0 or
        // 1 from row 2049 and averaged three entries.
        var op = StockformerDualEncoder<Half>.BuildCausalWindowOperator(2050, 2);

        Assert.Equal(1.0, (double)op[0, 0]);
        Assert.Equal(0.0, (double)op[2049, 2047]);
        Assert.Equal(0.5, (double)op[2049, 2048]);
        Assert.Equal(0.5, (double)op[2049, 2049]);
        Assert.Equal(0.0, (double)op[2048, 2049]);
    }

    // ---------------- constant schedules computed in double ----------------

    [Fact(Timeout = 60000)]
    public async Task StockformerTemporalEmbedding_Float_IsTheDoubleTableRoundedOnce()
    {
        await Task.CompletedTask;

        // angle = t / 10000^(2*(f/2) / width). Formed in float, the frequency's rounding grows with t to an
        // angle error near 1e-5 by t = 2000; the double table rounded once is within one float ulp.
        const int time = 2000;
        const int width = 16;
        var table = Stockformer<float>.TemporalEmbedding(time, width);

        for (int t = 0; t < time; t++)
        {
            for (int f = 0; f < width; f++)
            {
                double angle = t / Math.Pow(10000.0, (double)(f - (f % 2)) / width);
                double expected = f % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle);
                Assert.True(Math.Abs(expected - table[t, f]) <= 1.2e-7, $"table[{t}, {f}] = {table[t, f]}, expected {expected}.");
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task TokenizerBinEdges_Float_AreTheDoubleScheduleRoundedOnce()
    {
        await Task.CompletedTask;
        const int vocabulary = 1000;
        var tokenizer = new TimeSeriesTokenizer<float>(TimeSeriesTokenizationStrategy.Quantization, vocabularySize: vocabulary);
        var series = new Tensor<float>(new[] { 0.1f, 1.1f, 0.7f, 0.35f }, new[] { 4 });

        var (_, edges) = tokenizer.Quantize(series);

        double min = 0.1f;
        double range = (double)1.1f - min;
        Assert.Equal(vocabulary + 1, edges.Length);
        for (int i = 0; i <= vocabulary; i++)
        {
            Assert.Equal((float)(min + (range * i / vocabulary)), edges[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task TokenizerQuantize_Half_ConstantSeriesMapsToTokenZero()
    {
        await Task.CompletedTask;

        // master: range < 1e-10 becomes 1, so every value is in bin 0. In Half the 1e-10 guard rounded to zero and
        // the scaled value was 0 / 0.
        var tokenizer = new TimeSeriesTokenizer<Half>(TimeSeriesTokenizationStrategy.Quantization, vocabularySize: 64);
        var series = new Tensor<Half>(Enumerable.Repeat((Half)2, 5).ToArray(), new[] { 5 });

        var (tokens, _) = tokenizer.Quantize(series);

        Assert.Equal(new int[5], tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task GarchBackcastWeights_Float_AreTheDoubleScheduleRoundedOnce()
    {
        await Task.CompletedTask;

        // arch: w_i = 0.94^i / sum_j 0.94^j over the first 75 residuals.
        const int tau = 75;
        var weights = ClassicalVolatilityModelBase<float>.BackcastWeights(tau);

        double total = Enumerable.Range(0, tau).Sum(i => Math.Pow(0.94, i));
        Assert.Equal(tau, weights.Length);
        for (int i = 0; i < tau; i++)
        {
            double expected = Math.Pow(0.94, i) / total;
            Assert.True(Math.Abs(expected - weights[i]) <= 1.5e-7 * expected, $"w[{i}] = {weights[i]}, expected {expected}.");
        }
    }
}
