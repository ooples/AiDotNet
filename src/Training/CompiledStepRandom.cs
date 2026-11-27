using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// Random tensors a training step draws fresh on every step (reparameterization noise, a GAN's latent batch, a
/// diffusion step's noise), declared so that a compiled training plan redraws them on every replay.
/// </summary>
/// <remarks>
/// <para>
/// The fused training path traces the model's forward and loss once and then replays the compiled graph. A tensor the
/// forward builds on the host is captured by reference as a constant of that graph, and the closure that built it does
/// not run again on a replay. Noise drawn with <c>new Tensor</c> and a host loop therefore stays the first step's draw
/// for the whole run, and the model trains on one fixed sample: a VAE whose epsilon never changes, a generator that
/// only ever sees one latent batch. Nothing fails; the loss falls on the wrong objective.
/// </para>
/// <para>
/// Drawing through this type fixes that for every model at once. On the eager tape it is an ordinary draw. While a
/// plan is being traced, each draw is recorded with the plan, and before every later replay the plan's draws are
/// refilled in place from the same generator: the same mechanism dropout masks use, generalized so that a model does
/// not need a layer type to own its randomness.
/// </para>
/// <para><b>For Beginners:</b> if a training forward needs fresh random numbers each step, get them from here and the
/// fast compiled training path stays correct.</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
public static class CompiledStepRandom<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    internal sealed class RecordedDraw
    {
        private readonly Action<IEngine> _redraw;
        private RecordedDraw(Action<IEngine> redraw) => _redraw = redraw;

        // onDevice: fill writes the tensor through an engine op, so its device copy becomes the value; bumping the
        // version as a host fill needs would mark that copy stale and re-upload the old host values.
        internal static RecordedDraw Create<TData>(Tensor<TData> tensor, Action<Tensor<TData>> fill, bool onDevice = false)
            => new(engine =>
            {
                if (onDevice)
                {
                    using var suspension = AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording();
                    fill(tensor);
                    return;
                }
                fill(tensor);
                tensor.IncrementVersion();
                engine.InvalidatePersistentTensor(tensor);
            });

        internal void Redraw(IEngine engine) => _redraw(engine);
    }

    // The draws made while the current thread traces a plan; null when nothing is tracing.
    [ThreadStatic]
    private static List<RecordedDraw>? t_recording;

    // Depth of suppression scopes: a forward run only to resolve lazy shapes draws nothing, so the generator advances
    // exactly as it would on the eager tape.
    [ThreadStatic]
    private static int t_suppressDepth;

    // Depth of scopes that skip only the draws (a closure objective's per-step probe): the forward is the step's own,
    // so other per-step state still advances, but the plan's replay redraws, so the probe must not draw too.
    [ThreadStatic]
    private static int t_skipDrawsDepth;

    private static readonly ConditionalWeakTable<object, RecordedDraw[]> DrawsByPlan = new();

    /// <summary>A tensor of independent standard-normal samples, redrawn on every compiled replay.</summary>
    /// <param name="shape">The tensor shape.</param>
    /// <param name="random">The generator; it keeps advancing across replays, as it does eagerly.</param>
    public static Tensor<T> StandardNormal(int[] shape, Random random)
    {
        Guard.NotNull(random);
        return Draw(shape, t => FillStandardNormal(t, random));
    }

    /// <summary>A tensor of independent samples uniform on [<paramref name="min"/>, <paramref name="max"/>), redrawn on
    /// every compiled replay.</summary>
    public static Tensor<T> Uniform(int[] shape, Random random, double min = 0.0, double max = 1.0)
    {
        Guard.NotNull(shape);
        Guard.NotNull(random);
        // NaN in either bound is rejected too: "max <= min" alone is false for NaN and would let it through.
        if (double.IsNaN(min) || double.IsNaN(max) || max <= min)
            throw new ArgumentOutOfRangeException(nameof(max), "The upper bound must exceed the lower bound.");
        return Draw(shape, t => FillUniform(t, random, min, max));
    }

    /// <summary>
    /// A per-step random tensor of <paramref name="shape"/>, filled by <paramref name="fill"/>: on the eager tape an
    /// ordinary draw; while a plan is being traced, <paramref name="fill"/> is recorded and refills the tensor in place
    /// before every later replay.
    /// </summary>
    /// <param name="shape">The tensor shape.</param>
    /// <param name="fill">Writes one step's values into the tensor, in place, from the model's own generator.</param>
    /// <remarks>
    /// A forward that runs only to resolve lazily shaped layers draws nothing (the tensor is zeros and
    /// <paramref name="fill"/> is not called), so the model's generator advances exactly as on the eager tape.
    /// </remarks>
    public static Tensor<T> Draw(int[] shape, Action<Tensor<T>> fill) => DrawData<T>(shape, fill);

    /// <summary>Records a typed data draw in this model's recorder without narrowing it to the model's element type.</summary>
    /// <remarks>With <paramref name="onDevice"/>, <paramref name="fill"/> writes through an engine op (see
    /// <see cref="RecordedDraw.Create{TData}"/>) and runs outside any graph recording.</remarks>
    internal static Tensor<TData> DrawData<TData>(int[] shape, Action<Tensor<TData>> fill, bool onDevice = false)
    {
        Guard.NotNull(shape);
        Guard.NotNull(fill);
        var tensor = new Tensor<TData>(shape);
        if (t_suppressDepth > 0 || t_skipDrawsDepth > 0)
            return tensor;
        if (t_replay is { } replay && t_replayIndex < replay.Draws.Count
            && replay.Draws[t_replayIndex] is Tensor<TData> previous && SameShape(previous, shape))
        {
            previous.AsSpan().CopyTo(tensor.AsWritableSpan());
            t_replayIndex++;
        }
        else if (onDevice)
        {
            using var suspension = AiDotNet.Tensors.Engines.Compilation.GraphMode.SuspendRecording();
            fill(tensor);
        }
        else
            fill(tensor);
        t_captured?.Draws.Add(tensor);
        t_recording?.Add(RecordedDraw.Create(tensor, fill, onDevice));
        return tensor;
    }

    /// <summary>
    /// Registers an already-allocated per-step random tensor (such as a <see cref="NeuralNetworks.Layers.DropoutLayer{T}"/>
    /// mask) with the active plan recording so every compiled replay refills it in place, including when the same layer
    /// instance is invoked multiple times within one forward pass.
    /// </summary>
    /// <param name="tensor">The per-step tensor.</param>
    /// <param name="fill">Refills it in place.</param>
    /// <param name="onDevice">Whether <paramref name="fill"/> writes through an engine op (see
    /// <see cref="RecordedDraw.Create{TData}"/>).</param>
    internal static void RecordCustom(Tensor<T> tensor, Action<Tensor<T>> fill, bool onDevice = false)
    {
        if (t_recording is null)
            return;
        Guard.NotNull(tensor);
        Guard.NotNull(fill);
        t_recording.Add(RecordedDraw.Create(tensor, fill, onDevice));
    }

    /// <summary>
    /// An inverted-dropout mask of <paramref name="shape"/>, redrawn on every compiled replay: each entry is 0 with
    /// probability <paramref name="rate"/>, otherwise <c>1 / (1 - rate)</c>.
    /// </summary>
    /// <remarks>
    /// The engine draws the mask (<see cref="IEngine.TensorDropoutMask{T}"/>, on the device for a GPU engine) from one
    /// seed per draw taken from <paramref name="random"/>, and copies it into the step's tensor on the device: no host
    /// loop over the activation and no upload, as torch draws dropout masks. The masks are a deterministic function of
    /// <paramref name="random"/>'s stream, not of one <see cref="Random.NextDouble"/> per entry.
    /// </remarks>
    internal static Tensor<T> InvertedDropoutMask(int[] shape, Random random, double rate)
    {
        Guard.NotNull(random);
        T rateT = NumOps.FromDouble(rate);
        T keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        return DrawData<T>(shape, m =>
        {
            var engine = AiDotNetEngine.Current;
            int seed = random.Next();
            using var fresh = engine.TensorDropoutMask<T>(m._shape, rateT, keep, seed);
            engine.TensorCopy(fresh, m);
        }, onDevice: true);
    }

    /// <summary>
    /// A per-step 0/1 gate (shape <c>[1]</c>) that is 0 with probability <paramref name="dropProbability"/>, redrawn on
    /// every compiled replay: LayerDrop as <c>gate * layer(x) + (1 - gate) * x</c>, which is exactly <c>layer(x)</c> or
    /// <c>x</c>, keeps every layer in the traced graph instead of baking the traced step's choice of layers into it.
    /// </summary>
    internal static Tensor<T> KeepGate(Random random, double dropProbability)
    {
        Guard.NotNull(random);
        return Draw(new[] { 1 }, g => g.AsWritableSpan()[0] = random.NextDouble() < dropProbability ? NumOps.Zero : NumOps.One);
    }

    /// <summary>
    /// <paramref name="kept"/> where the <see cref="KeepGate"/> gate is 1 and <paramref name="skipped"/> where it is 0,
    /// as a selection that has the same shape as <paramref name="kept"/>.
    /// </summary>
    /// <remarks>
    /// A selection, not <c>gate * kept + (1 - gate) * skipped</c>. A dropped layer still runs so that it stays in the
    /// traced graph. If its output overflowed (inf or NaN, which Half reaches at 65504), <c>0 * inf</c> made the
    /// skipped step NaN, where LayerDrop's reference skips the layer and returns the input untouched.
    /// </remarks>
    internal static Tensor<T> Gated(IEngine engine, Tensor<T> gate, Tensor<T> kept, Tensor<T> skipped)
    {
        var shape = kept._shape;
        var ones = System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Repeat(1, shape.Length));
        var gateEverywhere = engine.TensorBroadcastTo(engine.Reshape(gate, ones), shape);
        return engine.TensorWhere(engine.TensorGreaterThan(gateEverywhere, NumOps.Zero), kept, skipped);
    }

    /// <summary>
    /// A per-step seed counter value for a layer that derives its randomness from one (dropout): normally
    /// <paramref name="advance"/>'s result. Inside <see cref="ReplayDraws"/> it is the value the probe of the same step
    /// took, in order, so a retrace does not advance the counter a second time.
    /// </summary>
    internal static ulong StepCounter(Func<ulong> advance)
    {
        Guard.NotNull(advance);
        if (t_replay is { } replay && t_replayCounterIndex < replay.Counters.Count)
            return replay.Counters[t_replayCounterIndex++];
        ulong value = advance();
        t_captured?.Counters.Add(value);
        return value;
    }

    /// <summary>
    /// Runs <paramref name="sample"/> for a layer that refills a persistent per-step sample in place, except inside
    /// <see cref="ReplayDraws"/> when the probe of the same step already ran it: the tensors then still hold that
    /// sample, and filling them again would consume the layer's generator twice in one step.
    /// </summary>
    internal static void StepSample(Action sample)
    {
        Guard.NotNull(sample);
        StepCounter(() =>
        {
            sample();
            return 0UL;
        });
    }

    // What a closure-objective probe drew this step, kept so that a retrace of the same step reuses it.
    [ThreadStatic]
    private static CaptureScope? t_captured;

    // What a retrace hands back in order instead of drawing again; the indices are the next entries.
    [ThreadStatic]
    private static CaptureScope? t_replay;

    [ThreadStatic]
    private static int t_replayIndex;

    [ThreadStatic]
    private static int t_replayCounterIndex;

    private static bool SameShape<TData>(Tensor<TData> tensor, int[] shape)
    {
        var own = tensor._shape;
        if (own.Length != shape.Length)
            return false;
        for (int i = 0; i < shape.Length; i++)
            if (own[i] != shape[i])
                return false;
        return true;
    }

    /// <summary>
    /// Keeps the draws made in this scope (a closure-objective probe): when the probe cannot replay a plan and the step
    /// traces the closure again, <see cref="ReplayDraws"/> hands the trace these values, so one step consumes the
    /// generator once, as it does on the eager tape.
    /// </summary>
    internal static CaptureScope CaptureDraws() => new CaptureScope();

    /// <summary>
    /// Makes the draws and step counters of this scope return what <paramref name="captured"/> took, in order (draws
    /// while the shapes agree), instead of consuming the generators; anything past the end draws normally.
    /// </summary>
    internal static ReplayScope ReplayDraws(CaptureScope captured) => new ReplayScope(captured);

    /// <summary>The draws and step counters of one probe; restores the enclosing capture when disposed.</summary>
    internal sealed class CaptureScope : IDisposable
    {
        private readonly CaptureScope? _previous;
        private bool _disposed;

        internal CaptureScope()
        {
            _previous = t_captured;
            t_captured = this;
        }

        internal List<object> Draws { get; } = new List<object>();

        internal List<ulong> Counters { get; } = new List<ulong>();

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            t_captured = _previous;
        }
    }

    /// <summary>A replay of captured draws; restores the enclosing replay when disposed.</summary>
    internal sealed class ReplayScope : IDisposable
    {
        private readonly CaptureScope? _previous;
        private readonly int _previousIndex;
        private readonly int _previousCounterIndex;
        private bool _disposed;

        internal ReplayScope(CaptureScope captured)
        {
            Guard.NotNull(captured);
            _previous = t_replay;
            _previousIndex = t_replayIndex;
            _previousCounterIndex = t_replayCounterIndex;
            t_replay = captured;
            t_replayIndex = 0;
            t_replayCounterIndex = 0;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            t_replay = _previous;
            t_replayIndex = _previousIndex;
            t_replayCounterIndex = _previousCounterIndex;
        }
    }

    /// <summary>Suppresses draws on this thread for a forward run only to resolve shapes; dispose to restore.</summary>
    internal static SuppressionScope SuppressDraws() => new SuppressionScope();

    /// <summary>
    /// Whether this thread is in a forward that is not a training step: one run only to resolve lazy shapes, or to
    /// check a plan against the eager tape. Per-step state other than random draws (a batch norm's running
    /// statistics) must not advance in it either, or the step that follows sees it advanced twice.
    /// </summary>
    internal static bool InNonStepForward => t_suppressDepth > 0;

    /// <summary>
    /// Whether this thread is tracing a compiled plan, capturing closure draws, or replaying captured draws.
    /// </summary>
    internal static bool IsCapturingOrReplaying => t_recording is not null || t_captured is not null || t_replay is not null;

    /// <summary>
    /// Whether per-step draws are suppressed or skipped on this thread (shape-resolution or non-step probe forwards).
    /// </summary>
    internal static bool AreDrawsSkipped => t_suppressDepth > 0 || t_skipDrawsDepth > 0;

    /// <summary>A scope in which <see cref="Draw"/> returns zeros without consuming the generator.</summary>
    internal sealed class SuppressionScope : IDisposable
    {
        private bool _disposed;

        internal SuppressionScope() => t_suppressDepth++;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            t_suppressDepth--;
        }
    }

    /// <summary>
    /// Skips draws on this thread (they return zeros without consuming the generator) while leaving
    /// <see cref="InNonStepForward"/> false: for a closure objective's per-step probe, whose step then replays a plan
    /// that redraws every recorded draw itself. Dispose to restore.
    /// </summary>
    internal static DrawSkipScope SkipDraws() => new DrawSkipScope();

    /// <summary>A scope in which <see cref="Draw"/> returns zeros without consuming the generator.</summary>
    internal sealed class DrawSkipScope : IDisposable
    {
        private bool _disposed;

        internal DrawSkipScope() => t_skipDrawsDepth++;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            t_skipDrawsDepth--;
        }
    }

    /// <summary>Starts recording the draws of a plan trace on this thread; dispose the scope when the trace ends.</summary>
    internal static RecordingScope BeginRecording() => new RecordingScope();

    /// <summary>Associates the draws a trace recorded with the plan it produced.</summary>
    internal static void Attach(object plan, RecordingScope recording)
    {
        Guard.NotNull(plan);
        var draws = recording.Draws;
        if (draws.Count == 0)
            return;
#if NET5_0_OR_GREATER
        DrawsByPlan.AddOrUpdate(plan, draws.ToArray());
#else
        DrawsByPlan.Remove(plan);
        DrawsByPlan.Add(plan, draws.ToArray());
#endif
    }

    /// <summary>The number of per-step draws recorded for <paramref name="plan"/>.</summary>
    internal static int DrawCount(object plan)
        => plan is not null && DrawsByPlan.TryGetValue(plan, out var draws) ? draws.Length : 0;

    /// <summary>Refills every draw recorded for <paramref name="plan"/> before it replays.</summary>
    internal static void RedrawFor(object plan)
    {
        if (plan is null || !DrawsByPlan.TryGetValue(plan, out var draws))
            return;
        var engine = AiDotNetEngine.Current;
        foreach (var draw in draws)
        {
            draw.Redraw(engine);
        }
    }

    /// <summary>The draws of one plan trace; restores the enclosing recording when disposed.</summary>
    internal sealed class RecordingScope : IDisposable
    {
        private readonly List<RecordedDraw>? _previous;
        private bool _disposed;

        internal RecordingScope()
        {
            _previous = t_recording;
            Draws = new List<RecordedDraw>();
            t_recording = Draws;
        }

        internal List<RecordedDraw> Draws { get; }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            t_recording = _previous;
        }
    }

    // Box-Muller, two samples per pair of uniforms, in the order an eager host loop would write them: the shared
    // definition in SeededNoiseTensors.
    private static void FillStandardNormal(Tensor<T> tensor, Random random) =>
        AiDotNet.Helpers.SeededNoiseTensors.WriteStandardNormalPairs(tensor.AsWritableSpan(), random);

    private static void FillUniform(Tensor<T> tensor, Random random, double min, double max)
    {
        var span = tensor.AsWritableSpan();
        double width = max - min;
        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble(min + width * random.NextDouble());
    }
}