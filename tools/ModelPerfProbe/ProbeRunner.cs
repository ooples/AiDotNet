using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.ModelPerfProbe;

/// <summary>
/// Drives one probe per model: construct → warm-up forward → warm-up train →
/// measured train loop → metrics. Generic over the model type via reflection
/// so the same runner covers every <c>IFullModel&lt;float, Tensor&lt;float&gt;,
/// Tensor&lt;float&gt;&gt;</c> in the registry.
/// </summary>
internal static class ProbeRunner
{
    /// <summary>
    /// Constructs the model with a default <c>NeuralNetworkArchitecture&lt;float&gt;</c>
    /// (mirroring what the model-family test scaffold emits), then runs the workload
    /// and captures metrics.
    /// </summary>
    public static ProbeResult Run(
        Type closedModelType,
        int steps,
        int seqLen,
        double slowStepMs,
        double slowAllocMb)
    {
        var result = new ProbeResult { Model = closedModelType.Name, StepCount = steps };

        IFullModel<float, Tensor<float>, Tensor<float>>? model = null;
        try
        {
            var (ctor, args) = PickCtor(closedModelType, seqLen);
            var swCtor = Stopwatch.StartNew();
            model = (IFullModel<float, Tensor<float>, Tensor<float>>)ctor.Invoke(args)!;
            swCtor.Stop();
            result.ConstructMs = swCtor.Elapsed.TotalMilliseconds;
        }
        catch (TargetInvocationException tie)
        {
            var inner = tie.InnerException ?? tie;
            result.Status = "construct-failed";
            result.Error = $"{inner.GetType().Name}: {inner.Message}";
            return result;
        }
        catch (Exception ex)
        {
            result.Status = "construct-failed";
            result.Error = $"{ex.GetType().Name}: {ex.Message}";
            return result;
        }

        try
        {
            var input = BuildInput(seqLen);
            var output = model.Predict(input);
            var outShape = new int[output.Shape.Length];
            for (int s = 0; s < output.Shape.Length; s++) outShape[s] = output.Shape[s];
            var target = BuildTarget(outShape);

            // Warm-up forward: many models initialize their lazy layers on the first call;
            // separate this from per-step measurement so the steady-state metrics aren't
            // contaminated by the one-time init cost.
            var swWarmFwd = Stopwatch.StartNew();
            _ = model.Predict(input);
            swWarmFwd.Stop();
            result.WarmupForwardMs = swWarmFwd.Elapsed.TotalMilliseconds;

            var swWarmTrain = Stopwatch.StartNew();
            model.Train(input, target);
            swWarmTrain.Stop();
            result.WarmupTrainMs = swWarmTrain.Elapsed.TotalMilliseconds;

            long gen0Before = GC.CollectionCount(0);
            long gen1Before = GC.CollectionCount(1);
            long gen2Before = GC.CollectionCount(2);
            long allocBefore = GC.GetTotalAllocatedBytes(precise: false);

            var swAll = Stopwatch.StartNew();
            for (int i = 0; i < steps; i++)
                model.Train(input, target);
            swAll.Stop();

            result.TotalMs = swAll.Elapsed.TotalMilliseconds;
            result.AvgStepMs = result.TotalMs / Math.Max(1, steps);
            result.AllocBytes = GC.GetTotalAllocatedBytes(precise: false) - allocBefore;
            result.AllocMbPerStep = result.AllocBytes / (1024.0 * 1024.0) / Math.Max(1, steps);
            result.Gen0 = (int)(GC.CollectionCount(0) - gen0Before);
            result.Gen1 = (int)(GC.CollectionCount(1) - gen1Before);
            result.Gen2 = (int)(GC.CollectionCount(2) - gen2Before);
            // Model-family TrainingIterations*3 = 30 default; project to that horizon.
            result.Projected30IterS = result.AvgStepMs * 30 / 1000.0;

            // Slow-budget tagging — flag and explain so the manifest is actionable.
            if (result.AvgStepMs > slowStepMs)
            {
                result.Flagged = true;
                result.FlagReason = $"avgStepMs={result.AvgStepMs:F1} > slowStepMs={slowStepMs:F1}";
            }
            else if (result.AllocMbPerStep > slowAllocMb)
            {
                result.Flagged = true;
                result.FlagReason = $"allocMbPerStep={result.AllocMbPerStep:F1} > slowAllocMb={slowAllocMb:F1}";
            }
        }
        catch (Exception ex)
        {
            result.Status = "probe-failed";
            result.Error = $"{ex.GetType().Name}: {ex.Message}";
        }
        finally
        {
            if (model is IDisposable d) d.Dispose();
        }

        return result;
    }

    /// <summary>
    /// Picks the parameter signature most likely to succeed for the model. The
    /// canonical ctor across the registry is <c>(NeuralNetworkArchitecture&lt;float&gt;
    /// architecture, ...optional defaults)</c>; satisfied with a default
    /// architecture, the rest filled with each parameter's compile-time default.
    /// Falls back to a parameterless ctor when no arch-first variant exists.
    /// </summary>
    private static (ConstructorInfo Ctor, object?[] Args) PickCtor(Type closedModelType, int seqLen)
    {
        var archType = typeof(NeuralNetworkArchitecture<float>);

        // Prefer an (arch, ...all-optional) ctor.
        foreach (var c in closedModelType.GetConstructors())
        {
            var pars = c.GetParameters();
            if (pars.Length >= 1 && pars[0].ParameterType == archType
                && pars.Skip(1).All(p => p.HasDefaultValue))
            {
                var args = new object?[pars.Length];
                args[0] = BuildArchitecture(seqLen);
                for (int i = 1; i < pars.Length; i++)
                    args[i] = pars[i].DefaultValue;
                return (c, args);
            }
        }

        // Parameterless fallback.
        var nullary = closedModelType.GetConstructor(Type.EmptyTypes);
        if (nullary is not null)
            return (nullary, Array.Empty<object?>());

        throw new InvalidOperationException(
            $"{closedModelType.Name}: no probeable constructor (need either parameterless or " +
            $"(NeuralNetworkArchitecture<float>, ...optional)).");
    }

    /// <summary>
    /// Probe-default architecture: 1-D input with the requested sequence length,
    /// regression-style 7-class output. Matches the model-family scaffold's
    /// default input shape for the "language or multimodal" branch, so the probe's
    /// workload tracks what the existing tests run.
    /// </summary>
    private static NeuralNetworkArchitecture<float> BuildArchitecture(int seqLen)
        => new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: seqLen,
            outputSize: 7);

    private static Tensor<float> BuildInput(int seqLen)
    {
        var rng = new Random(42);
        var input = new Tensor<float>(new[] { seqLen });
        for (int i = 0; i < seqLen; i++) input[i] = rng.Next(0, 1000);
        return input;
    }

    /// <summary>One-hot target shaped to whatever the model's forward emitted.</summary>
    private static Tensor<float> BuildTarget(int[] outputShape)
    {
        var rng = new Random(42);
        var target = new Tensor<float>(outputShape);
        int total = target.Length;
        for (int i = 0; i < total; i++) target[i] = (float)rng.NextDouble();
        return target;
    }
}
