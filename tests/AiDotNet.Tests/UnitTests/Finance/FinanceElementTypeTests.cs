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

        Assert.Equal(64 * 8, weights.Length);
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

        Assert.Equal(16 * 32, weights.Length);
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
        var rewards = new[] { (Half)0, (Half)0, (Half)0, (Half)0 };

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
        Assert.Equal(assets, weights.Length);
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
            new Tensor<Half>(new[] { (Half)0 }, new[] { 1 }), new[] { false }, taken, (Half)1);

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

    // ---------------- review fixes: wide intermediates for Half and float ----------------

    [Fact(Timeout = 60000)]
    public async Task TokenizerQuantize_Half_RangePastHalfMaxKeepsTheMidpointInTheMiddleBin()
    {
        await Task.CompletedTask;

        // master (double): range 80000, so 0 maps to (0 + 40000) / 80000 * 64 = token 32. In Half the range and
        // value - min overflowed to infinity and every value but the maximum fell into bin 0.
        var tokenizer = new TimeSeriesTokenizer<Half>(TimeSeriesTokenizationStrategy.Quantization, vocabularySize: 64);
        var series = new Tensor<Half>(new[] { (Half)(-40000), (Half)0, (Half)40000, (Half)(-20000) }, new[] { 4 });

        var (tokens, _) = tokenizer.Quantize(series);

        Assert.Equal(new[] { 0, 32, 63, 16 }, tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task TokenizerQuantize_Half_DefaultVocabularyClampsTheMaximumToTheLastToken()
    {
        await Task.CompletedTask;

        // The maximum scales to 4096 and is clamped to 4095. In Half 4095 rounds to 4096, so a clamp in T
        // returned token 4096, outside the vocabulary.
        var tokenizer = new TimeSeriesTokenizer<Half>(TimeSeriesTokenizationStrategy.Quantization);
        var series = new Tensor<Half>(new[] { (Half)0, (Half)1 }, new[] { 2 });

        var (tokens, _) = tokenizer.Quantize(series);

        Assert.Equal(new[] { 0, 4095 }, tokens);
    }

    [Fact(Timeout = 60000)]
    public async Task TokenizerDequantize_Half_LargeConstantSeriesKeepsItsValue()
    {
        await Task.CompletedTask;

        // A constant series of 40000 gets edges 40000 + i / 64 (all rounded to 40000 in Half). The centre of bin 0
        // is 40000; adding two edges in Half overflowed to infinity first.
        var tokenizer = new TimeSeriesTokenizer<Half>(TimeSeriesTokenizationStrategy.Quantization, vocabularySize: 64);
        var series = new Tensor<Half>(Enumerable.Repeat((Half)40000, 3).ToArray(), new[] { 3 });

        var (tokens, edges) = tokenizer.Quantize(series);
        var values = tokenizer.Dequantize(tokens, edges);

        Assert.Equal(3, values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal((Half)40000, values[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task PolicyGradientLossWeights_Half_LargeReturnsWhitenInDouble()
    {
        await Task.CompletedTask;

        // gamma = 0: returns (1000, 0), centred (500, -500), std 500, weights +-1 * 0.5 / 0.04 = +-12.5. In Half the
        // sum of squares (500000) overflowed, std became infinite and both weights zero.
        var rewards = new[] { (Half)1000, (Half)0 };

        var weights = PolicyGradientRollout<Half>.LossWeights(rewards, (Half)0, 0.2);

        Assert.Equal(2, weights.Length);
        Assert.Equal(12.5, (double)weights[0], 1e-2);
        Assert.Equal(-12.5, (double)weights[1], 1e-2);
    }

    [Fact(Timeout = 60000)]
    public async Task PolicyGradientLossWeights_Half_SmallSigmaGivesZeroForAZeroReturn()
    {
        await Task.CompletedTask;

        // A one-step zero-reward episode whitens to 0. 0.5 / sigma^2 = 500000 is infinite in Half, and 0 * inf
        // made the weight NaN.
        var weights = PolicyGradientRollout<Half>.LossWeights(new[] { (Half)0 }, (Half)0.99, 0.001);

        Assert.Equal(1, weights.Length);
        Assert.Equal(0.0, (double)weights[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task GaussianLogDensity_Half_VerySmallLogStdIsFinite()
    {
        await Task.CompletedTask;

        // a = mu = 0, logStd = -20: log N = -0 - (-20) - 0.5 log(2 pi) = 19.0811. exp(-20) underflows to zero in
        // Half, and dividing by it gave NaN.
        var value = TradingAgentBase<Half>.GaussianLogDensity(
            new Vector<Half>(new[] { (Half)0 }), new Vector<Half>(new[] { (Half)0 }),
            new Vector<Half>(new[] { (Half)(-20) }));

        Assert.Equal(20.0 - (0.5 * Math.Log(2.0 * Math.PI)), (double)value, 2e-2);
    }

    [Fact(Timeout = 60000)]
    public async Task RiskParity_Half_FourFlatAssetsGetEqualWeights()
    {
        await Task.CompletedTask;

        // Four flat assets share the same inverse vol (1 / 2^-14 = 16384), so each weight is 0.25. Their raw sum,
        // 65536, is past Half's 65504, and the weights came out 16384 / inf = 0.
        var policy = BaselinePolicies.RiskParity<Half>(windowSize: 4, totalColumns: 4, tradableCount: 4);
        var state = new Vector<Half>(Enumerable.Repeat((Half)1, 16).ToArray());

        var weights = policy(state);

        Assert.Equal(4, weights.Length);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(0.25, (double)weights[i]);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task MaskedMae_Half_ErrorSumPastHalfMaxAveragesInDouble()
    {
        await Task.CompletedTask;

        // 128 errors of 1000: MAE 1000. The Half error sum (128000) overflowed before the division.
        const int n = 128;
        var labels = new Vector<Half>(Enumerable.Repeat((Half)1, n).ToArray());
        var predictions = new Vector<Half>(Enumerable.Repeat((Half)1001, n).ToArray());

        var mae = StockformerMultiTaskLoss<Half>.MaskedMae(predictions, labels);

        Assert.Equal(1000.0, (double)mae);
    }

    [Fact(Timeout = 60000)]
    public async Task CrossEntropy_Float_LargeCommonOffsetKeepsLogTwo()
    {
        await Task.CompletedTask;

        // Equal logits give -log(1/2) = log 2 whatever their common value. With logsumexp materialized unshifted
        // in float, 1e8 + log 2 rounded back to 1e8 and the loss came out 0.
        var offset = StockformerMultiTaskLoss<float>.CrossEntropy(
            new Vector<float>(new[] { 1e8f, 1e8f }), new Vector<float>(new[] { 0f }), 2);
        var plain = StockformerMultiTaskLoss<float>.CrossEntropy(
            new Vector<float>(new[] { 0f, 0f }), new Vector<float>(new[] { 0f }), 2);

        Assert.Equal(Math.Log(2.0), offset, 1e-6);
        Assert.Equal(plain, offset);
    }

    [Fact(Timeout = 120000)]
    public async Task GarchNegLogLikelihood_Half_FiniteTotalIsNotRejected()
    {
        await Task.CompletedTask;

        // 40000 returns of +-1 under (mu, omega, alpha, beta) = (0, 1, 0, 0): sigma^2 = 1, so the NLL is
        // 0.5 * (log(2 pi) * 40000 + 40000) = 56757.5, finite in Half. The Half constant term alone (73515)
        // overflowed and the likelihood was rejected with an infinite penalty.
        const int n = 40000;
        var returns = new Tensor<Half>(
            Enumerable.Range(0, n).Select(i => i % 2 == 0 ? (Half)1 : (Half)(-1)).ToArray(), new[] { n });
        var model = new Garch11Model<Half>();

        var nll = model.NegLogLikelihood(new Vector<Half>(new[] { (Half)0, (Half)1, (Half)0, (Half)0 }), returns);

        double expected = 0.5 * ((1.8378770664093453 * n) + n);
        Assert.Equal(expected, (double)nll, 32.0);
    }

    [Fact(Timeout = 60000)]
    public async Task GarchForecastVolatility_Float_FirstHorizonIsTheOneStepVariance()
    {
        await Task.CompletedTask;

        // alpha + beta = 0.99999994 makes the unconditional variance about 5.0e6 while the one-step variance is
        // about 0.9. uncond + (var1 - uncond) rounds var1 to float's 0.5 spacing at 5e6 and gave 1.0 at h = 1.
        var model = new Garch11Model<float>();
        model.SetParameters(new Vector<float>(new[] { 0f, 0.3f, 0f, 0.99999994f }));

        float var1 = model.ForecastNextVariance(new Vector<float>(new[] { 0f, 0f }));
        var forecast = model.ForecastVolatility(new Tensor<float>(new[] { 0f, 0f }, new[] { 2 }), 3);

        Assert.InRange(var1, 0.85f, 0.95f);
        Assert.Equal((float)Math.Sqrt(var1), forecast[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task GarchSerialize_Half_RoundTripsTheParameters()
    {
        await Task.CompletedTask;

        // SerializationHelper had no Half case, so a fitted Half model threw NotSupportedException on Serialize.
        var model = new Garch11Model<Half>();
        var parameters = new[] { (Half)0.5, (Half)0.25, (Half)0.125, (Half)0.75 };
        model.SetParameters(new Vector<Half>(parameters));

        var restored = new Garch11Model<Half>();
        restored.Deserialize(model.Serialize());

        Assert.Equal(parameters, restored.GetParameters().ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task GarchDeserialize_PreMeanThreeParameterCheckpoint_IsRejected()
    {
        await Task.CompletedTask;

        // master wrote GARCH(1,1) as an int count and [omega, alpha, beta] as doubles. Read in the current
        // (mu, omega, alpha, beta) layout, omega became mu and the unconditional variance indexed past the end.
        var bytes = new byte[sizeof(int) + (3 * sizeof(double))];
        BitConverter.GetBytes(3).CopyTo(bytes, 0);
        Buffer.BlockCopy(new[] { 2e-6, 0.08, 0.90 }, 0, bytes, sizeof(int), 3 * sizeof(double));

        var model = new Garch11Model<double>();

        Assert.Throws<InvalidOperationException>(() => model.Deserialize(bytes));
    }

    [Fact(Timeout = 60000)]
    public async Task EGarchVariancePath_Double_VeryNegativeLogVarianceFollowsTheFlooredRecursion()
    {
        await Task.CompletedTask;

        // An all-zero residual series backcasts ln(0) = -inf. NextVariance floors the previous variance at 1e-18
        // (ln = L), so ln s2_0 = w + b L and ln s2_t = w + a (0 - sqrt(2/pi)) + b L. Unfloored, exp(-ln s2 / 2)
        // overflowed and 0 * inf made the path NaN.
        const int n = 6;
        const double omega = -100, alpha = 0.1, gamma = 0.0, beta = 0.5;
        double floor = Math.Log(1e-18);
        var model = new EGarchModel<double>();
        var variancePath = typeof(EGarchModel<double>).GetMethod("VariancePath", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("VariancePath not found.");

        var path = (Tensor<double>)(variancePath.Invoke(model, new object[]
        {
            new Tensor<double>(new[] { n }), double.NegativeInfinity, new Vector<double>(new[] { omega, alpha, gamma, beta }),
        }) ?? throw new InvalidOperationException("VariancePath returned null."));

        Assert.Equal(n, path.Length);
        double first = omega + (beta * floor);
        Assert.Equal(Math.Exp(first), path[0], Math.Exp(first) * 1e-12);
        double rest = omega - (alpha * Math.Sqrt(2.0 / Math.PI)) + (beta * floor);
        for (int t = 1; t < n; t++)
        {
            Assert.Equal(Math.Exp(rest), path[t], Math.Exp(rest) * 1e-12);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task VolatilityStatistics_Half_SquaresAndScatterAccumulateInDouble()
    {
        await Task.CompletedTask;

        // Returns of +-100 over 8 samples: covariance 80000 / 7 = 11428.6, finite in Half, though the Half scatter
        // sum (80000) overflowed. A return of 300 squares to 90000, so the realized vol of 300 overflowed too.
        var alternating = new Tensor<Half>(
            Enumerable.Range(0, 8).Select(i => i % 2 == 0 ? (Half)100 : (Half)(-100)).ToArray(), new[] { 8, 1 });
        var large = new Tensor<Half>(Enumerable.Repeat((Half)300, 4).ToArray(), new[] { 4, 1 });

        var covariance = VolatilityStatistics<Half>.Covariance(alternating);
        var realized = VolatilityStatistics<Half>.RealizedVolatility(large);

        Assert.Equal((double)(Half)(80000.0 / 7), (double)covariance[0, 0]);
        Assert.Equal(300.0, (double)realized[0]);
    }
}
