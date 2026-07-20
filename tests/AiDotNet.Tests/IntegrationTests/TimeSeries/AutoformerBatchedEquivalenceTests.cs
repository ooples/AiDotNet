using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Equivalence cover for Autoformer's batched primitives.
///
/// Autoformer trained with a PER-SAMPLE forward while its siblings (Informer, TemporalFusionTransformer)
/// use a batched one, so it issued roughly batch-times more op dispatches for identical work — measured at
/// over 10 minutes for 200 synthetic samples on the GPU engine. The batched primitives collapse that.
///
/// The claim these tests defend is the one that actually matters for a perf change: batching must be a
/// DISPATCH optimisation only, never a numerical change. Each batched primitive runs the SAME ops one rank
/// wider, so per-sample arithmetic and summation ORDER are unchanged — results must therefore be exactly
/// equal, not merely close. The tolerance below is deliberately tight for that reason; if it ever needs
/// loosening, the batched path has changed the maths and the claim is void.
/// </summary>
public class AutoformerBatchedEquivalenceTests
{
    private const double Tolerance = 1e-12;

    private static Tensor<double> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int total = 1;
        foreach (var s in shape) total *= s;
        var data = new Vector<double>(total);
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble() * 2 - 1;
        return new Tensor<double>(shape, data);
    }

    private static AutoformerModel<double> Model(int embDim = 8) =>
        new(new AutoformerOptions<double>
        {
            LookbackWindow = 12,
            ForecastHorizon = 1,
            EmbeddingDim = embDim,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            Epochs = 1,
            UseEarlyStopping = false,
        });

    /// <summary>
    /// The batched correlation spectrum over [B, S, D] must equal the per-sample spectrum computed for each
    /// slice independently. This is the op-count hot spot (corrLen lags x ~5 ops, x4 call sites per forward),
    /// so it is where batching pays — and where a silent numerical drift would do the most damage.
    /// </summary>
    [Theory]
    [InlineData(1, 12, 8)]
    [InlineData(4, 12, 8)]
    [InlineData(3, 24, 16)]
    public void Batched_correlation_spectrum_equals_per_sample(int batch, int seq, int dim)
    {
        var model = Model(dim);
        var q = Rand([batch, seq, dim], seed: 3);
        var k = Rand([batch, seq, dim], seed: 7);

        var batched = model.CorrelationSpectrumBatched(q, k, seq, dim);
        Assert.Equal(batch, batched.Shape[0]);
        Assert.Equal(seq, batched.Shape[1]);

        // Reference: the same formula evaluated one sample at a time.
        // R[b, lag] = mean over (t < seq-lag, d) of q[b,t,d] * k[b,t+lag,d]
        for (int b = 0; b < batch; b++)
        {
            for (int lag = 0; lag < seq; lag++)
            {
                int valid = seq - lag;
                double sum = 0.0;
                for (int t = 0; t < valid; t++)
                    for (int d = 0; d < dim; d++)
                        sum += q[[b, t, d]] * k[[b, t + lag, d]];

                double expected = sum / (valid * dim);
                double actual = batched[[b, lag]];
                Assert.True(
                    Math.Abs(expected - actual) <= Tolerance,
                    $"spectrum mismatch at b={b} lag={lag}: expected {expected}, got {actual}");
            }
        }
    }

    /// <summary>
    /// The matmul spectrum must equal the original per-lag formula. This is the shipped hot-path change:
    /// R[lag] = mean over (t &lt; corrLen-lag, dim) of q[t,:]·k[t+lag,:] is the mean of the lag-th diagonal of
    /// Q·Kᵀ, and diagonal summation is linear in that product, so the whole spectrum is one constant operator
    /// applied to vec(Q·Kᵀ) — ~5 dispatches instead of corrLen x ~5.
    ///
    /// Tolerance is 1e-9, NOT the 1e-12 used for the batched primitives, and the difference is deliberate:
    /// this reformulation reassociates the summation (matmul reduction vs per-lag ReduceSum), so it is
    /// mathematically identical but not bit-identical. The reference below is computed independently in plain
    /// double arithmetic, so agreement means the FORMULA matches — not that two copies of the same code agree.
    /// </summary>
    [Theory]
    [InlineData(12, 8)]
    [InlineData(24, 16)]
    [InlineData(24, 64)]
    public void Matmul_spectrum_equals_per_lag_formula(int seq, int dim)
    {
        const double MatmulTolerance = 1e-9;
        var model = Model(dim);
        var q2 = Rand([seq, dim], seed: 5);
        var k2 = Rand([seq, dim], seed: 9);

        var spectrum = model.CorrelationSpectrumMatmul(q2, k2, seq, dim);
        Assert.Equal(seq, spectrum.Shape[0]);

        for (int lag = 0; lag < seq; lag++)
        {
            int valid = seq - lag;
            double sum = 0.0;
            for (int t = 0; t < valid; t++)
                for (int d = 0; d < dim; d++)
                    sum += q2[[t, d]] * k2[[t + lag, d]];

            double expected = sum / (valid * dim);
            double actual = spectrum[[lag]];
            Assert.True(
                Math.Abs(expected - actual) <= MatmulTolerance * Math.Max(1.0, Math.Abs(expected)),
                $"matmul spectrum mismatch at lag={lag}: expected {expected}, got {actual}");
        }
    }

    /// <summary>
    /// The batched moving average must equal the per-sample one, and specifically must NOT let one window's
    /// replication padding bleed into its neighbour. Flattening [B, S, D] to [B*S, D] and padding there would
    /// splice one sample's tail onto the next sample's head — this asserts each sample is padded from its own
    /// endpoints by giving neighbouring samples deliberately different levels.
    /// </summary>
    [Theory]
    [InlineData(1, 12, 4)]
    [InlineData(5, 12, 4)]
    [InlineData(3, 24, 8)]
    public void Batched_moving_average_equals_per_sample_and_does_not_cross_samples(int batch, int seq, int dim)
    {
        var model = Model(dim);

        // Give each sample a distinct offset so cross-sample contamination changes the result detectably.
        var x = Rand([batch, seq, dim], seed: 11);
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < seq; t++)
                for (int d = 0; d < dim; d++)
                    x[[b, t, d]] = x[[b, t, d]] + (b * 100.0);

        const int kernel = 5;
        var batched = model.MovingAverageBatched(x, kernel, seq);

        // Reference: replication-pad each sample from ITS OWN endpoints, then stride-1 windowed mean.
        for (int b = 0; b < batch; b++)
        {
            int leftPad = kernel / 2, rightPad = kernel - 1 - leftPad;
            for (int t = 0; t < seq; t++)
            {
                for (int d = 0; d < dim; d++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < kernel; j++)
                    {
                        int src = t + j - leftPad;
                        src = Math.Max(0, Math.Min(seq - 1, src)); // replication pad, per sample
                        sum += x[[b, src, d]];
                    }

                    double expected = sum / kernel;
                    double actual = batched[[b, t, d]];
                    Assert.True(
                        Math.Abs(expected - actual) <= Tolerance,
                        $"moving-average mismatch at b={b} t={t} d={d}: expected {expected}, got {actual}");
                }
            }
        }
    }
}
