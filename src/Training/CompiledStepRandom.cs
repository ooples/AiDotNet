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
        public RecordedDraw(Tensor<T> tensor, Action<Tensor<T>> redraw)
        {
            Tensor = tensor;
            Redraw = redraw;
        }

        public Tensor<T> Tensor { get; }
        public Action<Tensor<T>> Redraw { get; }
    }

    // The draws made while the current thread traces a plan; null when nothing is tracing.
    [ThreadStatic]
    private static List<RecordedDraw>? t_recording;

    // Depth of suppression scopes: a forward run only to resolve lazy shapes draws nothing, so the generator advances
    // exactly as it would on the eager tape.
    [ThreadStatic]
    private static int t_suppressDepth;

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
        if (!(max > min))
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
    public static Tensor<T> Draw(int[] shape, Action<Tensor<T>> fill)
    {
        Guard.NotNull(shape);
        Guard.NotNull(fill);
        var tensor = new Tensor<T>(shape);
        if (t_suppressDepth > 0)
            return tensor;
        fill(tensor);
        t_recording?.Add(new RecordedDraw(tensor, fill));
        return tensor;
    }

    /// <summary>Suppresses draws on this thread for a forward run only to resolve shapes; dispose to restore.</summary>
    internal static SuppressionScope SuppressDraws() => new SuppressionScope();

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
            draw.Redraw(draw.Tensor);
            draw.Tensor.IncrementVersion();
            engine.InvalidatePersistentTensor(draw.Tensor);
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

    // Box-Muller, two samples per pair of uniforms, in the order an eager host loop would write them.
    private static void FillStandardNormal(Tensor<T> tensor, Random random)
    {
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double angle = 2.0 * Math.PI * u2;
            span[i] = NumOps.FromDouble(radius * Math.Cos(angle));
            if (i + 1 < span.Length)
                span[i + 1] = NumOps.FromDouble(radius * Math.Sin(angle));
        }
    }

    private static void FillUniform(Tensor<T> tensor, Random random, double min, double max)
    {
        var span = tensor.AsWritableSpan();
        double width = max - min;
        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble(min + width * random.NextDouble());
    }
}