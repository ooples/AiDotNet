using System;
using System.Linq;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Verifies the timestep-dependent low-rank adaptation behind <see cref="TLoRAModel{T}"/>, for
/// Soboleva et al., "T-LoRA: Single Image Diffusion Model Customization Without Overfitting"
/// (arXiv:2507.05964).
/// </summary>
/// <remarks>
/// <para>
/// The paper's two innovations are a rank schedule that tightens as the diffusion timestep rises and
/// an orthogonal (Ortho-LoRA) parametrization that makes that schedule meaningful. Both are asserted
/// here — a uniform-rank adapter, which is what plain LoRA is, would fail the first, and a correlated
/// initialization would silently defeat the second while still passing the first.
/// </para>
/// <para>
/// These assertions were TIGHTENED after checking the implementation against the paper line by line.
/// Three of them previously encoded a schedule the paper does not use: the floor is r_min = 50% of r,
/// not 1; the interpolation is <c>floor((r - r_min)(T - t)/T) + r_min</c>, not
/// <c>ceil(r(1 - t/T))</c>; and the identity-at-initialization property comes from subtracting the
/// frozen initial product, NOT from B = 0, which is the standard-LoRA convention this paper
/// specifically replaces in order to train a non-zero S.
/// </para>
/// </remarks>
public class TimestepDependentLoraTests
{
    private const int Rank = 8;
    private const int Width = 16;
    private const int Horizon = 1000;

    private static TimestepDependentLora<double> Adapter(int rank = Rank, int width = Width, int seed = 5)
        => new(rank, width, width, Horizon, new Random(seed));

    [Fact]
    public void RankIsFullAtTimestepZero()
    {
        // Low timesteps refine detail and are not the overfitting-prone regime, so the adapter keeps
        // all its capacity there.
        Assert.Equal(Rank, Adapter().EffectiveRank(0));
    }

    [Fact]
    public void MinimumRankIsHalfTheFullRank()
    {
        // "r_min is set to 50% of r." This is the paper's floor, and it is the assertion that
        // distinguishes the published schedule from an arbitrary decay to a single direction.
        var adapter = Adapter();
        Assert.Equal(Rank / 2, adapter.MinRank);
        Assert.Equal(adapter.MinRank, adapter.EffectiveRank(Horizon));
    }

    [Fact]
    public void ScheduleMatchesThePublishedFormula()
    {
        // r(t) = floor((r - r_min) * (T - t) / T) + r_min, checked pointwise across the horizon
        // rather than only at the endpoints, so a formula that happens to agree at t=0 and t=T but
        // interpolates differently in between cannot pass.
        var adapter = Adapter();
        int rMin = adapter.MinRank;

        for (int t = 0; t <= Horizon; t += 10)
        {
            int expected = (int)Math.Floor((double)(Rank - rMin) * (Horizon - t) / Horizon) + rMin;
            Assert.Equal(expected, adapter.EffectiveRank(t));
        }
    }

    [Fact]
    public void RankShrinksMonotonicallyAsTheTimestepRises()
    {
        // "Higher diffusion timesteps are more prone to overfitting than lower ones." The schedule
        // must therefore never increase with t — that direction is the entire contribution.
        var adapter = Adapter();
        int previous = adapter.EffectiveRank(0);

        for (int t = 0; t <= Horizon; t += 25)
        {
            int current = adapter.EffectiveRank(t);
            Assert.True(current <= previous,
                $"Effective rank rose from {previous} to {current} at timestep {t}.");
            previous = current;
        }

        Assert.True(adapter.EffectiveRank(Horizon) < adapter.EffectiveRank(0),
            "The schedule never tightened, so this is plain LoRA with extra steps.");
    }

    [Fact]
    public void RankNeverFallsBelowTheFloor()
    {
        // Beyond the horizon the schedule is clamped, never decayed further. A zero-rank adapter is
        // not "maximally constrained", it is DISCONNECTED.
        var adapter = Adapter();
        foreach (int t in new[] { Horizon, Horizon * 2, int.MaxValue })
        {
            Assert.Equal(adapter.MinRank, adapter.EffectiveRank(t));
        }
    }

    [Fact]
    public void TimestepsOutsideTheHorizonAreClamped()
    {
        var adapter = Adapter();
        Assert.Equal(adapter.EffectiveRank(0), adapter.EffectiveRank(-50));
        Assert.Equal(adapter.EffectiveRank(Horizon), adapter.EffectiveRank(Horizon + 500));
    }

    [Fact]
    public void DownProjectionRowsAreOrthonormal()
    {
        // The paper's second innovation: A_init = V_r^T, whose rows are orthonormal by construction
        // of the SVD. Without independence, masking the tail removes no capacity because the
        // surviving directions still span what was masked — the schedule would look right and do
        // nothing.
        var a = Adapter().DownProjection;

        for (int i = 0; i < Rank; i++)
        {
            double selfDot = 0.0;
            for (int c = 0; c < Width; c++) selfDot += a[i, c] * a[i, c];
            Assert.Equal(1.0, selfDot, 9);

            for (int j = i + 1; j < Rank; j++)
            {
                double dot = 0.0;
                for (int c = 0; c < Width; c++) dot += a[i, c] * a[j, c];
                Assert.Equal(0.0, dot, 9);
            }
        }
    }

    [Fact]
    public void UpProjectionColumnsAreOrthonormal()
    {
        // B_init = U_r, the matching half of the same SVD. Asserted separately because an
        // implementation could easily orthogonalize one factor and not the other.
        var b = Adapter().UpProjection;

        for (int i = 0; i < Rank; i++)
        {
            double selfDot = 0.0;
            for (int o = 0; o < b.Rows; o++) selfDot += b[o, i] * b[o, i];
            Assert.Equal(1.0, selfDot, 9);

            for (int j = i + 1; j < Rank; j++)
            {
                double dot = 0.0;
                for (int o = 0; o < b.Rows; o++) dot += b[o, i] * b[o, j];
                Assert.Equal(0.0, dot, 9);
            }
        }
    }

    [Fact]
    public void SingularValuesStartNonZero()
    {
        // The distinguishing feature of this paper's parametrization. Standard LoRA sets B = 0 to get
        // an identity adapter; T-LoRA keeps S = S_init non-zero and trainable, recovering the
        // identity by subtracting the frozen initial product instead. If S started at zero, the next
        // test would pass trivially and the reparametrization would be untested.
        var s = Adapter().SingularValues;
        Assert.Contains(true, Enumerable.Range(0, s.Length).Select(i => Math.Abs(s[i]) > 1e-9));
    }

    [Fact]
    public void AdapterStartsAsTheIdentity()
    {
        // Customization must not perturb the base model before it has learned anything. Here that
        // holds because B S M_t A and B_init S_init M_t A_init are the same product at init and
        // cancel — NOT because any factor is zero (see SingularValuesStartNonZero). Checked at both
        // ends of the schedule so the cancellation cannot depend on the mask width.
        var adapter = Adapter();
        var input = new Vector<double>(Width);
        for (int i = 0; i < Width; i++) input[i] = i + 1.0;

        foreach (int t in new[] { 0, Horizon / 2, Horizon })
        {
            var output = adapter.Apply(input, t);
            for (int i = 0; i < output.Length; i++) Assert.Equal(0.0, output[i], 12);
        }
    }

    [Fact]
    public void MaskingActuallyRemovesCapacity()
    {
        // With a trained (moved) B, a high timestep must produce a DIFFERENT update than a low one —
        // otherwise the mask is decorative.
        var adapter = Adapter();
        var rng = new Random(3);
        for (int o = 0; o < adapter.UpProjection.Rows; o++)
        {
            for (int r = 0; r < Rank; r++) adapter.UpProjection[o, r] = rng.NextDouble() - 0.5;
        }

        var input = new Vector<double>(Width);
        for (int i = 0; i < Width; i++) input[i] = Math.Sin(i);

        var low = adapter.Apply(input, timestep: 0);
        var high = adapter.Apply(input, timestep: Horizon);

        bool differ = Enumerable.Range(0, low.Length).Any(i => Math.Abs(low[i] - high[i]) > 1e-9);
        Assert.True(differ, "High and low timesteps produced the same update; the rank mask did nothing.");
    }

    [Fact]
    public void TrainingTheAdapterMovesItAwayFromTheIdentity()
    {
        // The flip side of AdapterStartsAsTheIdentity: the subtraction must not cancel FOREVER, or
        // the adapter could never learn. Moving S alone (the factor plain LoRA does not have) has to
        // be enough to produce a non-zero update.
        var adapter = Adapter();
        for (int r = 0; r < Rank; r++) adapter.SingularValues[r] += 0.5;

        var input = new Vector<double>(Width);
        for (int i = 0; i < Width; i++) input[i] = i + 1.0;

        var output = adapter.Apply(input, timestep: 0);
        bool moved = Enumerable.Range(0, output.Length).Any(i => Math.Abs(output[i]) > 1e-9);
        Assert.True(moved, "Training S produced no change; the reparametrization cancels unconditionally.");
    }

    [Fact]
    public void RankBeyondTheAmbientWidthIsRejected()
    {
        // Asking for more directions than the space has cannot produce independent ones, and the
        // paper's SVD initialization has no trailing triplet of that size to draw from. Rejected at
        // construction rather than silently yielding zero rows that merely look orthogonal.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new TimestepDependentLora<double>(rank: 6, inputDim: 3, outputDim: 3, Horizon, new Random(1)));
    }

    [Fact]
    public void ConstructorRejectsDegenerateConfiguration()
    {
        var rng = new Random(1);
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(0, Width, Width, Horizon, rng));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(Rank, 0, Width, Horizon, rng));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(Rank, Width, 0, Horizon, rng));
        // The horizon is the schedule's denominator.
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(Rank, Width, Width, 0, rng));
    }

    [Fact]
    public void SeededInitializationIsReproducible()
    {
        var a = Adapter(seed: 42).DownProjection;
        var b = Adapter(seed: 42).DownProjection;
        for (int r = 0; r < Rank; r++)
        {
            for (int c = 0; c < Width; c++) Assert.Equal(a[r, c], b[r, c], 12);
        }
    }
}
