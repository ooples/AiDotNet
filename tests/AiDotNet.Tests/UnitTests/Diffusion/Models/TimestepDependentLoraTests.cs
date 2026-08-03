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
/// The paper's two innovations are a rank schedule that tightens as the diffusion timestep rises and
/// an orthogonal parametrization that makes that schedule meaningful. Both are asserted here — a
/// uniform-rank adapter, which is what plain LoRA is, would fail the first, and a correlated
/// initialization would silently defeat the second while still passing the first.
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
    public void RankNeverReachesZero()
    {
        // A zero-rank adapter is not maximally constrained, it is DISCONNECTED: the update vanishes
        // and the most overfitting-prone timesteps would get no adaptation at all.
        var adapter = Adapter();
        foreach (int t in new[] { Horizon, Horizon * 2, int.MaxValue })
        {
            Assert.True(adapter.EffectiveRank(t) >= 1, $"Rank collapsed to zero at timestep {t}.");
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
        // The paper's second innovation. Without independence, masking the tail removes no capacity
        // because the surviving directions still span what was masked — the schedule would look
        // right and do nothing.
        var adapter = Adapter();
        var a = adapter.DownProjection;

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
    public void AdapterStartsAsTheIdentity()
    {
        // B is zero at initialization, the standard LoRA convention: customization must not perturb
        // the base model before it has learned anything.
        var adapter = Adapter();
        var input = new Vector<double>(Width);
        for (int i = 0; i < Width; i++) input[i] = i + 1.0;

        var output = adapter.Apply(input, timestep: 0);
        for (int i = 0; i < output.Length; i++) Assert.Equal(0.0, output[i], 12);
    }

    [Fact]
    public void MaskingActuallyRemovesCapacity()
    {
        // With a trained (non-zero) B, a high timestep must produce a DIFFERENT update than a low
        // one — otherwise the mask is decorative.
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
    public void RankBeyondTheAmbientWidthDoesNotFabricateDirections()
    {
        // Asking for more directions than the space has cannot produce independent ones. The extra
        // rows stay zero rather than being normalized noise that merely looks orthogonal.
        var adapter = new TimestepDependentLora<double>(rank: 6, inputDim: 3, outputDim: 3, Horizon, new Random(1));
        var a = adapter.DownProjection;

        for (int r = 3; r < 6; r++)
        {
            double norm = 0.0;
            for (int c = 0; c < 3; c++) norm += a[r, c] * a[r, c];
            Assert.Equal(0.0, norm, 9);
        }
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
