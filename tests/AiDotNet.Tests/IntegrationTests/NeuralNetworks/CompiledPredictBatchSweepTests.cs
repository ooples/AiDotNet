#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Every model-family network through the compiled Predict at several batch sizes: each batch's result matches the
/// eager forward. The compile cache is batch-dynamic, so the batches after the first two are built from the traces
/// (no re-trace) whenever the forward is batch-polymorphic; the output names the reason when it is not.
/// </summary>
[Collection(GpuEngineSwapCollection.Name)]
public class CompiledPredictBatchSweepTests
{
    // 1 is specialized; 2 is the base, 3 shows what scales; 5 and 4 are built; the second 2 is a cache hit.
    private static readonly int[] Batches = { 1, 2, 3, 5, 2, 4 };
    private const long SlowEagerMs = 5000;
    private readonly ITestOutputHelper _output;

    public CompiledPredictBatchSweepTests(ITestOutputHelper output) => _output = output;

    public static IEnumerable<object[]> ModelFamilies() => GpuPredictScopeSweepTests.ModelFamilies();

    [Theory(Timeout = 300000)]
    [MemberData(nameof(ModelFamilies))]
    public async Task CompiledPredict_MatchesEager_AtEveryBatch(string fixtureType)
    {
        await Task.Yield();
        var fixture = (IAsyncLifetime)Activator.CreateInstance(typeof(CompiledPredictBatchSweepTests).Assembly.GetType(fixtureType)!)!;
        await fixture.InitializeAsync();
        object? network = null;
        try
        {
            var type = fixture.GetType();
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            network = type.GetMethod("CreateNetwork", flags, null, Type.EmptyTypes, null)!.Invoke(fixture, null)!;
            var shape = (int[])type.GetProperty("EffectiveInputShape", flags)!.GetValue(fixture)!;
            if (shape.Length < 2 || shape[0] != 1)
            {
                _output.WriteLine($"{type.Name}: input [{string.Join(",", shape)}] has no batch dim");
                return;
            }
            var element = FindNetworkElementType(network.GetType());
            var run = typeof(CompiledPredictBatchSweepTests).GetMethod(nameof(RunBatches), BindingFlags.Static | BindingFlags.NonPublic)!
                .MakeGenericMethod(element);
            var makeConstant = type.GetMethod("CreateConstantTensor", flags, null, new[] { typeof(int[]), typeof(double) }, null)!;
            var constant = typeof(CompiledPredictBatchSweepTests).GetMethod(nameof(ConstantFactory), BindingFlags.Static | BindingFlags.NonPublic)!
                .MakeGenericMethod(element).Invoke(null, new object[] { fixture, makeConstant })!;
            var report = (string)run.Invoke(null, new[] { network, shape, constant })!;
            _output.WriteLine($"{type.Name}: {report}");
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }
        finally
        {
            (network as IDisposable)?.Dispose();
            await fixture.DisposeAsync();
        }
    }

    private static string RunBatches<T>(NeuralNetworkBase<T> network, int[] shape, Func<int[], Tensor<T>> constantInput)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var rng = new Random(17);
        var host = typeof(NeuralNetworkBase<T>).GetField("_compileHost", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(network)!;
        var cacheField = host.GetType().GetField("_cache", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var eagerOnlyField = host.GetType().GetField("_eagerOnlyShapeKeys", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var predictEager = typeof(NeuralNetworkBase<T>).GetMethod("PredictEager", BindingFlags.Instance | BindingFlags.NonPublic)!;

        // The reference is what PredictCompiled traces: PredictEager, in eval mode without a tape. Predict is not:
        // most models override PredictCore with their own forward.
        T[] Eager(Tensor<T> x)
        {
            using var noGrad = new NoGradScope<T>();
            try { return ((Tensor<T>)predictEager.Invoke(network, new object[] { x })!).ToArray(); }
            catch (TargetInvocationException ex) when (ex.InnerException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }
        network.SetTrainingMode(false);

        // Random inputs, unless the forward refuses them (token ids, masks): then the fixture's own constant input.
        bool random = true;
        Tensor<T> MakeInput(int[] s)
        {
            if (!random) return constantInput(s);
            var x = new Tensor<T>(s);
            var span = x.AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = ops.FromDouble(rng.NextDouble() * 2 - 1);
            return x;
        }
        var first = (int[])shape.Clone();
        first[0] = Batches[0];
        try { Eager(MakeInput(first)); }
        catch (Exception) { random = false; }
        var probe = MakeInput(first);
        T[] a1;
        try { a1 = Eager(probe); }
        catch (Exception ex)
        {
            // PredictEager (the layer chain PredictCompiled traces) is not this model's forward: its PredictCore
            // prepares the input first. Its compiled inference goes through Accelerate with its own forward instead.
            return $"PredictEager rejects the fixture input ({ex.GetType().Name}: {ex.Message}): not a layer-chain model";
        }
        var probeWatch = System.Diagnostics.Stopwatch.StartNew();
        var a2 = Eager(probe);
        // An expensive model (VGG16BNBackbone: ~10 s per sample on CPU) runs the shortest sequence that still covers each
        // plan source - 1 specialized, 2 traced, 3 built from the template - so the test fits its timeout.
        var batches = probeWatch.ElapsedMilliseconds > SlowEagerMs ? new[] { 1, 2, 3 } : Batches;
        if (!a1.SequenceEqual(a2)) return $"eager forward is not deterministic ({(random ? "random" : "constant")} input): nothing to compare";

        var how = new List<string>();
        foreach (var batch in batches)
        {
            var s = (int[])shape.Clone();
            s[0] = batch;
            var input = MakeInput(s);

            T[] expected;
            var watch = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                expected = Eager(input);
            }
            catch (Exception ex) when (batch != 1)
            {
                // Dim 0 of this fixture's input is not a batch the eager forward accepts: nothing to compare.
                return $"[{string.Join(" ", how)}] eager forward rejects batch {batch} ({ex.GetType().Name}: {ex.Message})";
            }
            long eagerMs = watch.ElapsedMilliseconds;
            watch.Restart();
            int builtBefore = ((CompiledModelCache<T>?)cacheField.GetValue(host))?.InstantiatedPlanCount ?? 0;
            T[] actual;
            using (var refusals = new RefusalListener())
            {
                actual = network.PredictCompiled(input).ToArray();
                if (refusals.HostBoundary is { } refusal) how.Add($"{batch}:eager({refusal})");
            }
            var cache = (CompiledModelCache<T>?)cacheField.GetValue(host);
            bool eagerOnly = (eagerOnlyField.GetValue(host) as System.Collections.ICollection)?.Count > 0;
            string source = cache is null ? "eager (no cache)"
                : cache.InstantiatedPlanCount > builtBefore ? "built"
                : eagerOnly ? "traced or eager-only" : "traced";
            how.Add($"{batch}:{source} {eagerMs}/{watch.ElapsedMilliseconds}ms");
            string? mismatch = expected.Length != actual.Length
                ? $"eager returned {expected.Length} values, compiled {actual.Length}"
                : FirstMismatch(ops, expected, actual);
            if (mismatch is not null)
                Assert.Fail($"batch {batch} ({source}): {mismatch}; first diverging layer: {BisectLayers(network, input)}");
        }

        var final = (CompiledModelCache<T>?)cacheField.GetValue(host);
        if (final is null) return "no compile cache (compilation disabled)";
        return $"{(batches.Length < Batches.Length ? "slow eager forward, short batch sequence: " : "")}[{string.Join(" ", how)}] plans {final.InferencePlanCount}, built {final.InstantiatedPlanCount}, "
               + $"polymorphism: {final.GetBatchPolymorphismFailure(SymbolicShape.BatchDynamic(shape)) ?? "ok"}";
    }

    private static string? FirstMismatch<T>(AiDotNet.Tensors.Interfaces.INumericOperations<T> ops, T[] expected, T[] actual)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            double e = ops.ToDouble(expected[i]), a = ops.ToDouble(actual[i]);
            if (!(Math.Abs(e - a) <= 1e-4 + 1e-3 * Math.Abs(e) || double.IsNaN(e) && double.IsNaN(a)))
                return $"[{i}] eager {e} compiled {a}";
        }
        return null;
    }

    /// <summary>
    /// Compiles each layer's forward on its own, fed the eager output of the layers before it, and names the first
    /// whose compiled output differs from its eager one.
    /// </summary>
    private static string BisectLayers<T>(NeuralNetworkBase<T> network, Tensor<T> input)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        using var noGrad = new NoGradScope<T>();
        var x = input;
        var layers = network.LayersReadOnly;
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            var layerInput = x;
            Tensor<T> eager;
            try { eager = layer.Forward(layerInput); }
            catch (Exception ex) { return $"#{i} {layer.GetType().Name}: eager layer forward threw {ex.GetType().Name} (the model's forward is not a plain layer chain)"; }
            try
            {
                using var cache = new CompiledModelCache<T>();
                // The layer's input tensor names the mutable slot: a shape alone is ambiguous when a weight shares it.
                var plan = cache.GetOrCompileInference(layerInput, () => layer.Forward(layerInput));
                plan.SetInputs(new[] { layerInput });
                var compiled = plan.Execute().ToArray();
                var reference = layer.Forward(layerInput).ToArray();
                var m = compiled.Length != reference.Length
                    ? $"eager {reference.Length} values, compiled {compiled.Length}"
                    : FirstMismatch(ops, reference, compiled);
                if (m is not null) return $"#{i} {layer.GetType().Name} [{string.Join(",", layerInput._shape)}]: {m}";
            }
            catch (GraphCaptureNotSupportedException ex) when (ex.Limitation == GraphCaptureLimitation.NoCompilableOperations)
            {
                // Records nothing (Dropout in eval): it passes its input through, which is not a divergence.
            }
            catch (GraphCaptureNotSupportedException ex) when (ex.Limitation == GraphCaptureLimitation.HostBoundary)
            {
                return $"#{i} {layer.GetType().Name} [{string.Join(",", layerInput._shape)}]: refuses to compile, its forward reads a traced value on the host";
            }
            catch (Exception ex)
            {
                return $"#{i} {layer.GetType().Name} [{string.Join(",", layerInput._shape)}]: compiling it alone threw {ex.GetType().Name}: {ex.Message}";
            }
            x = eager;
        }
        return "none alone (the divergence needs layers together, or the model's PredictEager is not the layer chain)";
    }

    /// <summary>Catches the compile host's warning when a trace refuses to compile at a host boundary.</summary>
    private sealed class RefusalListener : System.Diagnostics.TraceListener
    {
        internal string? HostBoundary;

        internal RefusalListener() => System.Diagnostics.Trace.Listeners.Add(this);

        public override void Write(string? message) => Note(message);
        public override void WriteLine(string? message) => Note(message);

        private void Note(string? message)
        {
            if (HostBoundary is null && message is not null && message.Contains("on the host while tracing"))
                HostBoundary = message.Length > 160 ? message.Substring(0, 160) : message;
        }

        protected override void Dispose(bool disposing)
        {
            System.Diagnostics.Trace.Listeners.Remove(this);
            base.Dispose(disposing);
        }
    }

    private static Func<int[], Tensor<T>> ConstantFactory<T>(object fixture, MethodInfo createConstant)
        => s => (Tensor<T>)createConstant.Invoke(fixture, new object[] { s, 0.1 })!;

    private static Type FindNetworkElementType(Type type)
    {
        for (var t = type; t is not null; t = t.BaseType)
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return t.GetGenericArguments()[0];
        throw new InvalidOperationException($"{type} is not a NeuralNetworkBase<T>");
    }
}
#endif
