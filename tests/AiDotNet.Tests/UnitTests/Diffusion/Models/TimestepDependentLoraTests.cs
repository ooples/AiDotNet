using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.LinearAlgebra;
using Xunit;
using AiDotNet.Diffusion;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;

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
[Collection("EngineCurrentGlobalState")]
public class TimestepDependentLoraTests : IDisposable
{
    private readonly IEngine _previousEngine = AiDotNetEngine.Current;
    public TimestepDependentLoraTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _previousEngine;
    private const int Rank = 8;
    private const int Width = 16;
    private const int Horizon = 1000;

    private static TimestepDependentLora<double> Adapter(int rank = Rank, int width = Width, int seed = 5)
        => new(rank, width, width, Horizon, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));

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
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
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
            () => new TimestepDependentLora<double>(rank: 6, inputDim: 3, outputDim: 3, Horizon, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1)));
    }

    [Fact]
    public void ConstructorRejectsDegenerateConfiguration()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1);
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(0, Width, Width, Horizon, rng));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(Rank, 0, Width, Horizon, rng));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(Rank, Width, 0, Horizon, rng));
        // The horizon is the schedule's denominator.
        Assert.Throws<ArgumentOutOfRangeException>(() => new TimestepDependentLora<double>(Rank, Width, Width, 0, rng));
    }

    [Fact]
    public void CopyStateFromCarriesTheFrozenInitializationToo()
    {
        // The adapter's output is B S M_t A - B_init S_init M_t A_init, so "untrained" means the two
        // products CANCEL, not that anything is zero. Copying only A, B and S into an adapter whose
        // _init came from a different draw leaves them unequal, and the adapter then applies the
        // difference between two unrelated initializations. That is a large update masquerading as a
        // fresh adapter, and it made a cloned model's output differ from its source by 69.
        var source = Adapter(seed: 11);
        var target = Adapter(seed: 99);   // deliberately a DIFFERENT draw

        var input = new Vector<double>(Width);
        for (int i = 0; i < Width; i++) input[i] = i + 1.0;

        // Move the source off its initialization so the copy has something real to carry.
        for (int r = 0; r < Rank; r++) source.SingularValues[r] += 0.25;
        source.InvalidateCache();

        target.CopyStateFrom(source);

        var expected = source.Apply(input, timestep: Horizon / 3);
        var actual = target.Apply(input, timestep: Horizon / 3);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
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

    private static double[] ReferenceDelta(double[][] state, int rank, int input, int output, int effective)
    {
        var result = new double[input * output];
        for (int i = 0; i < input; i++)
            for (int o = 0; o < output; o++)
                for (int r = 0; r < effective; r++)
                    result[i * output + o] += state[1][o * rank + r] * state[2][r] * state[0][r * input + i]
                        - state[4][o * rank + r] * state[5][r] * state[3][r * input + i];
        return result;
    }

    private static double ReferenceLoss(double[][] state, int effective, double[] weights)
        => ReferenceDelta(state, 2, 3, 4, effective).Zip(weights, (value, weight) => value * weight).Sum();

    private static void AssertClose(double[] expected, double[] actual, double tolerance = 1e-10)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"index {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static void CheckFactorGradients(double[][] state, int effective, double[] weights, double[][] gradients)
    {
        for (int factor = 0; factor < 3; factor++)
            for (int i = 0; i < state[factor].Length; i++)
            {
                const double epsilon = 1e-6;
                double value = state[factor][i];
                state[factor][i] = value + epsilon;
                double plus = ReferenceLoss(state, effective, weights);
                state[factor][i] = value - epsilon;
                double minus = ReferenceLoss(state, effective, weights);
                state[factor][i] = value;
                Assert.True(Math.Abs(gradients[factor][i] - (plus - minus) / (2 * epsilon)) < 1e-7,
                    $"factor {factor}, index {i}: gradient {gradients[factor][i]:R}");
            }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(Horizon)]
    public void Warm_inference_cache_does_not_disconnect_eager_A_B_S_gradients(int timestep)
    {
        var engine = AiDotNetEngine.Current;
        var adapter = new TimestepDependentLora<double>(2, 3, 4, Horizon, new Random(17));
        var chunks = adapter.GetParameterStateChunks().ToArray();
        Assert.Equal(new[] { "A", "B", "S", "A_init", "B_init", "S_init" }, chunks.Select(chunk => chunk.StableId));
        Assert.All(chunks.Take(3), chunk => Assert.Equal(ParameterSlotRole.Trainable, chunk.Role));
        Assert.All(chunks.Skip(3), chunk => Assert.Equal(ParameterSlotRole.Frozen, chunk.Role));
        var warm = adapter.EffectiveDeltaTransposed(timestep);
        Assert.Same(warm, adapter.EffectiveDeltaTransposed(timestep));
        double[] weights = Enumerable.Range(0, 12).Select(i => Math.Cos(i * 0.7)).ToArray();
        var weightTensor = new Tensor<double>(weights, new[] { 3, 4 });
        using var tape = new GradientTape<double>();
        var delta = adapter.EffectiveDeltaTransposed(timestep);
        Assert.NotSame(warm, delta);
        var loss = engine.ReduceSum(engine.TensorMultiply(delta, weightTensor), null, false);
        var gradients = tape.ComputeGradients(loss, chunks.Select(chunk => chunk.Tensor).ToArray());
        Assert.All(chunks.Take(3), chunk => Assert.True(gradients.ContainsKey(chunk.Tensor)));
        Assert.All(chunks.Skip(3), chunk => Assert.False(gradients.ContainsKey(chunk.Tensor)));
        var state = chunks.Select(chunk => chunk.Tensor.ToArray()).ToArray();
        AssertClose(ReferenceDelta(state, 2, 3, 4, adapter.EffectiveRank(timestep)), delta.ToArray());
        CheckFactorGradients(state, adapter.EffectiveRank(timestep), weights,
            chunks.Take(3).Select(chunk => gradients[chunk.Tensor].ToArray()).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(Horizon)]
    public void Compiled_steps_and_probes_use_live_factors_without_caching_trace_outputs(int timestep)
    {
        var engine = AiDotNetEngine.Current;
        var adapter = new TimestepDependentLora<double>(2, 3, 4, Horizon, new Random(17));
        var chunks = adapter.GetParameterStateChunks().ToArray();
        var parameters = chunks.Take(3).Select(chunk => chunk.Tensor).ToArray();
        var initial = chunks.Skip(3).Select(chunk => chunk.Tensor.ToArray()).ToArray();
        double[] weights = Enumerable.Range(0, 12).Select(i => Math.Cos(i * 0.7)).ToArray();
        var weightTensor = new Tensor<double>(weights, new[] { 3, 4 });
        adapter.EffectiveDeltaTransposed(timestep).ToArray();
        Tensor<double> Objective() => engine.ReduceSum(engine.TensorMultiply(adapter.EffectiveDeltaTransposed(timestep), weightTensor), null, false);
        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            plan = scope.CompileTraining(parameters, Objective());
            Assert.Null(scope.HostReadOp);
        }
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            if (guard is null) throw new InvalidOperationException(reason);
            guard.AttachTo(plan);
            plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.01, weightDecay: 0);
            for (int step = 0; step < 3; step++)
            {
                var state = chunks.Select(chunk => chunk.Tensor.ToArray()).ToArray();
                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(Objective, parameters, out _));
                Assert.Equal(ReferenceLoss(state, adapter.EffectiveRank(timestep), weights), plan.Step().ToArray()[0], 10);
                CheckFactorGradients(state, adapter.EffectiveRank(timestep), weights, plan.Gradients.Select(gradient => gradient.ToArray()).ToArray());
                Assert.Equal(new long[] { step + 1, step + 1, step + 1 }, plan.ParameterAcceptedUpdates);
                for (int i = 0; i < 3; i++) Assert.Equal(initial[i], chunks[i + 3].Tensor.ToArray());
                var current = chunks.Select(chunk => chunk.Tensor.ToArray()).ToArray();
                AssertClose(ReferenceDelta(current, 2, 3, 4, adapter.EffectiveRank(timestep)), adapter.EffectiveDeltaTransposed(timestep).ToArray());
            }
        }
        var stateBeforeRefusal = adapter.GetParameters().ToArray();
        Assert.Throws<ArgumentException>(() => adapter.SetParameters(new Vector<double>(stateBeforeRefusal.Length - 1)));
        Assert.Equal(stateBeforeRefusal, adapter.GetParameters().ToArray());
        var copied = new TimestepDependentLora<double>(2, 3, 4, Horizon, new Random(91));
        copied.CopyStateFrom(adapter);
        Assert.Equal(stateBeforeRefusal, copied.GetParameters().ToArray());
        AssertClose(adapter.EffectiveDeltaTransposed(timestep).ToArray(), copied.EffectiveDeltaTransposed(timestep).ToArray());
    }

    private static TLoRAModel<float> SmallModel(int seed = 42)
        => new(predictor: new UNetNoisePredictor<float>(architecture: null, inputChannels: 4, outputChannels: 4,
                baseChannels: 32, channelMultipliers: new[] { 1, 2 }, numResBlocks: 1,
                attentionResolutions: new[] { 1 }, contextDim: 64, seed: seed),
            vae: new StandardVAE<float>(inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: seed), seed: seed, adapterRank: 2);

    [Fact]
    public void Model_steps_train_all_adapter_factors_and_preserve_frozen_state_across_clone_and_restore()
    {
        var model = SmallModel();
        _ = model.ParameterCount; // Latch the lazy registry before any warmup; adapters must already be present.
        var input = new Tensor<float>(Enumerable.Range(0, 256).Select(i => (float)Math.Sin(i * 0.1)).ToArray(), new[] { 1, 4, 8, 8 });
        var target = new Tensor<float>(Enumerable.Range(0, 256).Select(i => (float)Math.Cos(i * 0.2)).ToArray(), new[] { 1, 4, 8, 8 });
        model.PredictNoise(input, 0).ToArray(); // Warm every lazy base weight before taking the frozen snapshot.
        var baseWeights = model.NoisePredictor.GetParameters().ToArray();
        var vaeWeights = model.VAE.GetParameters().ToArray();
        var chunks = model.Adapters.Select(adapter => adapter.Adapter.GetParameterStateChunks().ToArray()).ToArray();
        Assert.NotEmpty(chunks);
        // Exercise the actual protected collector used by DiffusionModelBase.Train, rather than
        // treating the helper's declared role as proof that the model optimizer enrolls it.
        var collector = typeof(DiffusionModelBase<float>).GetMethod("CollectTrainableParameters", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("The diffusion training collector was not found.");
        var enrolled = Assert.IsType<Tensor<float>[]>(collector.Invoke(model, null));
        var expectedParameters = chunks.SelectMany(row => row.Take(3)).Select(chunk => chunk.Tensor).ToArray();
        Assert.Equal(expectedParameters.Length, enrolled.Length);
        foreach (var parameter in expectedParameters)
            Assert.Single(enrolled.Where(candidate => ReferenceEquals(candidate, parameter)));
        foreach (var frozen in chunks.SelectMany(row => row.Skip(3)))
            Assert.DoesNotContain(enrolled, candidate => ReferenceEquals(candidate, frozen.Tensor));
        var allState = model.GetParameterStateChunks().ToArray();
        foreach (var chunk in chunks.SelectMany(row => row))
            Assert.Single(allState.Where(candidate => ReferenceEquals(candidate.SourceTensor, chunk.Tensor)));
        Assert.Equal((long)baseWeights.Length + vaeWeights.Length + chunks.Sum(row => row.Sum(chunk => (long)chunk.Tensor.Length)), model.ParameterCount);
        var before = chunks.Select(row => row.Select(chunk => chunk.Tensor.ToArray()).ToArray()).ToArray();
        for (int step = 0; step < 3; step++) model.Train(input, target);
        for (int adapter = 0; adapter < chunks.Length; adapter++)
        {
            for (int factor = 0; factor < 3; factor++)
                Assert.False(before[adapter][factor].SequenceEqual(chunks[adapter][factor].Tensor.ToArray()), $"adapter {adapter}, factor {factor} did not train");
            for (int factor = 3; factor < 6; factor++) Assert.Equal(before[adapter][factor], chunks[adapter][factor].Tensor.ToArray());
        }
        Assert.Equal(baseWeights, model.NoisePredictor.GetParameters().ToArray());
        Assert.Equal(vaeWeights, model.VAE.GetParameters().ToArray());
        Assert.Equal(model.ParameterCount, model.GetParameters().Length);
        var expected = model.PredictNoise(input, 0).ToArray();
        Assert.All(expected, value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
        var clone = Assert.IsType<TLoRAModel<float>>(model.Clone());
        Assert.Equal(model.GetParameters().ToArray(), clone.GetParameters().ToArray());
        Assert.Equal(expected, clone.PredictNoise(input, 0).ToArray());
        var restored = SmallModel(91);
        restored.Deserialize(model.Serialize());
        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
        Assert.Equal(expected, restored.PredictNoise(input, 0).ToArray());
        // Restoring a clone must not replace or mutate the original model's canonical state.
        var sourceState = model.GetParameters().ToArray();
        clone.Adapters[0].Adapter.SingularValues[0] += 0.5f;
        Assert.Equal(sourceState, model.GetParameters().ToArray());
    }
}
