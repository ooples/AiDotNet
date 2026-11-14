using System.Collections.Concurrent;
using AiDotNet.Configuration;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Export;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference;

/// <summary>
/// Compiled inference for a neural network, used by the model facade for every prediction when JIT compilation is
/// on (the default).
/// </summary>
/// <remarks>
/// <para>Three tiers, best first:</para>
/// <list type="number">
/// <item>A portable <see cref="TensorProgram"/> of the network's forward, traced once with the network's own
/// parameter tensors bound, and run through a <see cref="ProgramPackage"/>: a compiled plan per device and input
/// shape, replayed with no re-trace. Tracing verifies the program against the forward on the example and on a
/// value-reversed copy of it, so a network that reads its input on the host (token ids used as indices) is detected
/// up front and never served from a stale plan. The package is what a saved model embeds, so a loaded model predicts
/// from a compiled plan on its first call, and recompiles itself from the program on new hardware.</item>
/// <item>The graph-capture plan cache, for a network the program tracer cannot record (an op with no portable form),
/// with its value-stability guard: until a differing input shows the plan responds to its input, answers come from
/// the eager forward.</item>
/// <item>The eager forward.</item>
/// </list>
/// </remarks>
internal sealed class CompiledNeuralPredictor<T>
{
    private readonly NeuralNetworkBase<T> _model;
    private readonly JitCompilationConfig _config;
    private readonly object _programGate = new();
    private ProgramPackage? _package;
    private bool _programDisabled;
    private Tensor<T>? _example;

    private readonly CompiledModelCache<T> _cache = new();
    private readonly ConcurrentDictionary<TensorShape, JitValueStability<T>> _stabilityByShape = new();

    /// <summary>How far a replayed graph-capture plan may sit from the eager forward before it is not value-stable.</summary>
    internal const double ValueStabilityTolerance = 1e-3;

    internal CompiledNeuralPredictor(NeuralNetworkBase<T> model, JitCompilationConfig config, ProgramPackage? package = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        if (package is not null && TryBind(package)) _package = package;
    }

    /// <summary>The network this predictor serves.</summary>
    internal NeuralNetworkBase<T> Model => _model;

    /// <summary>True when a saved package's compiled plans gained a target for this machine (re-save to keep it).</summary>
    internal bool HasNewCompiledTargets => _package?.IsDirty == true;

    /// <summary>Predicts through the best available tier.</summary>
    internal Tensor<T>[] Predict(Tensor<T>[] inputs)
    {
        if (inputs is null || inputs.Length == 0)
            throw new ArgumentException("A compiled prediction requires at least one input tensor.", nameof(inputs));
        var input = inputs[0];
        // Request-pool workers don't inherit the thread-static codec options set on the builder thread.
        _config.ApplyToTensorCodec();

        if (inputs.Length == 1 && TryProgram(input, out var programOutput)) return programOutput;
        return GraphCapturePredict(input);
    }

    /// <summary>
    /// The package to embed in a saved model: traced from an input seen so far (or <paramref name="example"/>) and
    /// compiled for this device, so the loaded model starts warm. Null when the network cannot be traced.
    /// </summary>
    internal ProgramPackage? Export(Tensor<T>? example)
    {
        lock (_programGate)
        {
            var sample = _example ?? example;
            if (_package is null && sample is not null) TryTrace(sample);
            if (_package is null || sample is null) return _package;
            try
            {
                _package.Precompile(AiDotNetEngine.Current, new[] { new[] { sample } });
            }
            catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or ArgumentException)
            {
                System.Diagnostics.Trace.TraceWarning($"Compiled plan not embedded for {_model.GetType().Name}: {ex.Message}");
            }
            return _package;
        }
    }

    private bool TryProgram(Tensor<T> input, out Tensor<T>[] output)
    {
        output = Array.Empty<Tensor<T>>();
        lock (_programGate)
        {
            if (_programDisabled) return false;
            _example ??= input.Clone();
            if (_package is null && !TryTrace(input)) return false;
            var package = _package!;
            if (package.Program.InputDimensions[0].Count != input.Rank) return false;
            try
            {
                using var noGrad = new NoGradScope<T>();
                output = package.Execute(AiDotNetEngine.Current, new[] { input });
                return true;
            }
            catch (ArgumentException)
            {
                // An input shape the program fixed (a non-batch axis): this shape takes the next tier.
                return false;
            }
            catch (Exception ex) when (!_config.ThrowOnFailure
                                       && ex is InvalidOperationException or NotSupportedException or ProgramFormatException)
            {
                // A compiled or replayed program that fails is never allowed to fail the prediction: stop using the
                // program tier for this model and answer from the next one.
                _programDisabled = true;
                System.Diagnostics.Trace.TraceWarning(
                    $"Program tier disabled for {_model.GetType().FullName}: {ex.GetType().Name}: {ex.Message}");
                return false;
            }
        }
    }

    private bool TryTrace(Tensor<T> example)
    {
        var chunks = _model.GetParameterStateChunks().ToList();
        var names = new Dictionary<object, string>(ReferenceComparer.Instance);
        foreach (var chunk in chunks) names[chunk.Tensor] = chunk.StableId;
        try
        {
            var program = TensorProgram.Trace(x => CaptureCompatiblePredict(_model, x), example.Clone(),
                new AiDotNet.Tensors.Engines.Export.TraceOptions { Parameters = chunks.Select(c => (object)c.Tensor), Names = names });
            _package = new ProgramPackage(program);
            return true;
        }
        catch (Exception ex) when (ex is TraceException or NotSupportedException or InvalidOperationException or ArgumentException)
        {
            _programDisabled = true;
            System.Diagnostics.Trace.TraceWarning(
                $"Portable program unavailable for {_model.GetType().FullName}: {ex.Message} Using the graph-capture plan cache.");
            return false;
        }
    }

    // A loaded package's parameters were saved by stable id and shape only; attach the model's live tensors.
    private bool TryBind(ProgramPackage package)
    {
        var byId = _model.GetParameterStateChunks().ToDictionary(c => c.StableId, c => c.Tensor, StringComparer.Ordinal);
        var bound = new List<Tensor<T>>();
        foreach (var id in package.Program.ParameterIds)
        {
            var name = package.Program.Values[id].Name;
            if (name is null || !byId.TryGetValue(name, out var tensor))
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"The saved compiled program does not match {_model.GetType().Name}'s parameters ('{name}'); it will be re-traced.");
                return false;
            }
            bound.Add(tensor);
        }
        try
        {
            package.Program.BindParameters(bound);
            return true;
        }
        catch (ArgumentException ex)
        {
            System.Diagnostics.Trace.TraceWarning($"The saved compiled program does not fit {_model.GetType().Name}: {ex.Message} It will be re-traced.");
            return false;
        }
    }

    // ── Graph-capture tier (with value-stability guard) ──────────────────────────────────────────────────

    private Tensor<T>[] GraphCapturePredict(Tensor<T> input)
    {
        var stability = _stabilityByShape.GetOrAdd(input.Shape, static _ => new JitValueStability<T>());
        try
        {
            Func<Tensor<T>> traceForward = () => CaptureCompatiblePredict(_model, input);
            if (stability.IsPermanentlyDisabled) return new[] { EagerPredict(_model, input) };

            var plan = _cache.GetOrCompileInference(input, traceForward);
            if (stability.IsVerified) return new[] { plan.Execute() };

            var reference = CaptureCompatiblePredict(_model, input);
            // At the traced input a broken plan and a correct one agree, so wait for an input that differs.
            if (!stability.RecordAndCheckDiffers(input)) return new[] { reference };

            var replayed = plan.Execute();
            double discrepancy = MaxAbsoluteDifference(replayed, reference);
            if (discrepancy <= ValueStabilityTolerance)
            {
                stability.MarkVerified();
                return new[] { replayed };
            }
            if (stability.TryMarkPermanentlyDisabled())
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"JIT disabled for {_model.GetType().FullName}: the compiled plan is not value-stable at shape " +
                    $"[{string.Join(", ", input.Shape)}] (off by {discrepancy:E3}); the trace captured input values as " +
                    "constants. Falling back to eager inference for this shape.");
            }
            return new[] { reference };
        }
        catch (GraphCaptureNotSupportedException ex) when (!_config.ThrowOnFailure)
        {
            if (stability.TryMarkPermanentlyDisabled())
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"JIT disabled for {_model.GetType().FullName}: graph capture is not supported for input shape " +
                    $"[{string.Join(", ", input.Shape)}] ({ex.Limitation}). {ex.Message}");
            }
            return new[] { EagerPredict(_model, input) };
        }
        catch (Exception ex) when (!_config.ThrowOnFailure)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"JIT fallback for {_model.GetType().FullName} with input shape [{string.Join(", ", input.Shape)}]: " +
                $"{ex.GetType().Name}: {ex.Message}");
            return new[] { EagerPredict(_model, input) };
        }
    }

    private static Tensor<T> EagerPredict(NeuralNetworkBase<T> model, Tensor<T> input)
    {
        using var noGrad = new NoGradScope<T>();
        return model.Predict(input);
    }

    /// <summary>The canonical forward used for program tracing, graph capture and its value-stability check.</summary>
    internal static Tensor<T> CaptureCompatiblePredict(NeuralNetworkBase<T> model, Tensor<T> input)
    {
        // Guard against nested GraphMode: a model's own compiled path would open a second scope and drop the outer
        // trace's operations. The compatibility scope also selects the same graph-safe fast paths outside GraphMode.
        var savedOptions = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        var captureOptions = GraphCaptureCompatibility.CreateOptionsWithoutCompilation(savedOptions);
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(captureOptions);
        try
        {
            using var compatibility = GraphCaptureCompatibility.Enter();
            using var noGrad = new NoGradScope<T>();
            return model.Predict(input);
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(savedOptions);
        }
    }

    private static double MaxAbsoluteDifference(Tensor<T> left, Tensor<T> right)
    {
        if (left.Length != right.Length) return double.PositiveInfinity;
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.ToDouble(engine.TensorMaxValue(engine.TensorAbs(engine.TensorSubtract(left, right))));
    }

    private sealed class ReferenceComparer : IEqualityComparer<object>
    {
        internal static readonly ReferenceComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }
}
