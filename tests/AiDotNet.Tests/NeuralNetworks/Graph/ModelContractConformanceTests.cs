using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Every model that DECLARES a shape contract must have that contract predict what Predict returns.
/// </summary>
/// <remarks>
/// <para>
/// This is the model-side counterpart of the layer conformance sweep, and it is what makes a declared
/// contract worth anything. A declaration nothing checks is a comment: it drifts the moment a decoder
/// changes its stride, and drifts silently, because nothing downstream can tell a stale contract from
/// a correct one.
/// </para>
/// <para>
/// WHAT THIS BUYS OVER THE INDUSTRY STANDARD. PyTorch cannot answer "what shape does this model
/// return" without RUNNING the model - meta tensors make that cheap, but it is still execution, it
/// yields one concrete shape rather than a relation, and nothing verifies it because nothing is
/// declared. Here a caller gets a symbolic relation statically, and this test is the machinery that
/// keeps that relation honest against the real forward pass.
/// </para>
/// <para>
/// It asserts ZERO disagreements. A contract that declines (returns null) is not a disagreement -
/// declining is the honest answer for a rank or configuration nobody measured. Only a contract that
/// CLAIMS a shape and gets it wrong fails.
/// </para>
/// </remarks>
public class ModelContractConformanceTests
{
    private readonly ITestOutputHelper _out;
    public ModelContractConformanceTests(ITestOutputHelper output) => _out = output;

    private const int Extent = 64;
    private const int Classes = 7;

    [Trait("Category", "Sweep")]
    [Fact(Timeout = 1800000)]
    public async Task EveryDeclaredModelContractPredictsWhatPredictReturned()
    {
        await Task.Yield();

        // MODELS only. Layers implement IShapeContract too - 317 of them - and none has an
        // architecture constructor, so they all landed in "skipped" and buried the real skips under
        // noise. Their conformance is already covered by the layer sweep; this one is about models.
        var models = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && t.GetInterfaces().Any(i => i.Name == "IShapeContract")
                        && !DerivesFromLayerBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        // Optional window over the candidate list, for running this in passes. The DEFAULTS are the
        // real configuration and are what CI runs; this only lets a developer split one long pass into
        // several short ones on a machine that cannot hold a ten-minute run open. It never reduces
        // what a default run covers.
        // Optional namespace filter, for checking ONE family's freshly-declared law without paying for
        // the whole inventory. Same rule as the window below: empty is the real configuration.
        string? nsFilter = Environment.GetEnvironmentVariable("ADNSHAPE_CONF_NAMESPACE");
        if (!string.IsNullOrWhiteSpace(nsFilter))
        {
            models = models
                .Where(t => t.Namespace is not null
                            && t.Namespace.Contains(nsFilter, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        int offset = EnvInt("ADNSHAPE_CONF_OFFSET", 0, 0);
        int budget = EnvInt("ADNSHAPE_CONF_BUDGET", models.Count, 1);
        if (offset > 0) models = models.Skip(offset).ToList();
        if (budget < models.Count) models = models.Take(budget).ToList();
        _out.WriteLine($"window: offset={offset} budget={budget} -> {models.Count} candidates");

        int declared = 0, agreed = 0, declined = 0;
        var disagreed = new List<string>();
        var skipped = new List<string>();

        foreach (var open in models)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            // TRY EACH INPUT TYPE the families actually use, and keep the first that produces a real
            // comparison. Building every model as a 3-D image made the entire audio family report
            // "declined": its contract declares rank 2, the harness fed rank 4, and the contract
            // correctly declined on RANK - which looks identical to declining for want of a width.
            // A harness that cannot tell those apart reports a family as unverified when it is simply
            // being asked the wrong question.
            object? model = null;
            IShapeContract? contract = null;
            int[]? shape = null;
            string? lastNote = null;

            // TwoDimensional is here because its per-sample shape is [Height, Width] - rank 3 once
            // batched - and NOTHING else in this list produces rank 3. Without it the whole forecasting
            // family reported 71 declared / 0 agreed / 71 DECLINED, which reads as "no model conforms"
            // when the truth was that the harness never asked them a rank-3 question. A sweep that can
            // only pose two of the three ranks the library uses cannot tell a wrong contract from an
            // unasked one.
            foreach (var inputType in new[]
                     { InputType.ThreeDimensional, InputType.TwoDimensional, InputType.OneDimensional })
            {
                object? candidate = null;
                try { candidate = Construct(closed, inputType); }
                catch (Exception ex) { lastNote ??= $"{Unwrap(ex).GetType().Name} constructing"; continue; }
                if (candidate is null) { lastNote ??= "no usable constructor"; continue; }

                if (candidate is not IShapeContract c) { (candidate as IDisposable)?.Dispose(); lastNote ??= "not IShapeContract"; continue; }

                int[]? per = TryArchitectureInputShape(candidate);
                if (per is null || per.Length == 0 || per.Any(d => d <= 0))
                {
                    (candidate as IDisposable)?.Dispose();
                    lastNote ??= "no concrete declared input shape";
                    continue;
                }

                var candidateShape = new int[per.Length + 1];
                candidateShape[0] = 1;
                for (int i = 0; i < per.Length; i++) candidateShape[i + 1] = Math.Min(per[i], Extent);

                // Prefer an input type whose rank the contract actually answers for.
                bool answers = ShapeInference.InferOutputShape(c, candidateShape) is not null;
                if (model is null || answers)
                {
                    (model as IDisposable)?.Dispose();
                    model = candidate; contract = c; shape = candidateShape;
                    if (answers) break;
                }
                else { (candidate as IDisposable)?.Dispose(); }
            }

            if (model is null || contract is null || shape is null)
            {
                skipped.Add($"{open.Name}: {lastNote ?? "no usable input type"}");
                continue;
            }

            try
            {
                declared++;

                // What the CONTRACT says, without running the model.
                int[]? predictedShape = ShapeInference.InferOutputShape(contract, shape);
                if (predictedShape is null)
                {
                    declined++;
                    continue;
                }

                // What the model ACTUALLY does.
                var (actual, failure) = TryPredict(model, shape);

                // NOTE on a skip this harness cannot fix, seen with the vocoders. Every axis is
                // clamped to Extent to keep a probe cheap, and for HiFiGAN that handed 64 mel
                // channels to kernels built for 80: "Input channels (64) must match kernel
                // in_channels (80)". Retrying UNCLAMPED does not help, because the architecture the
                // harness constructed also says 64 - the 80 comes from the model's own options and is
                // never reflected back into the architecture it was handed. That is a model-side
                // inconsistency, not a clamp that can be widened, so those models are verified by
                // VocoderShapeContractTests instead, which builds them at their real mel width.
                if (actual is null) { skipped.Add($"{open.Name}: {failure}"); continue; }

                if (predictedShape.SequenceEqual(actual)) { agreed++; continue; }

                disagreed.Add($"{open.Name}: in [{string.Join(",", shape)}] "
                    + $"contract says [{string.Join(",", predictedShape)}] "
                    + $"but Predict returned [{string.Join(",", actual)}]");
            }
            finally { (model as IDisposable)?.Dispose(); }
        }

        _out.WriteLine($"models declaring a contract : {declared}");
        _out.WriteLine($"  contract agreed with Predict : {agreed}");
        _out.WriteLine($"  contract declined (null)     : {declined}");
        _out.WriteLine($"  DISAGREED                    : {disagreed.Count}");
        _out.WriteLine($"  skipped                      : {skipped.Count}");
        foreach (var s in skipped.Take(15)) _out.WriteLine($"    skipped: {s}");
        foreach (var d in disagreed) _out.WriteLine($"    DISAGREED: {d}");

        // Assert the EXERCISED count too. Without it, a run where every model failed to construct
        // would pass while verifying nothing - the vacuous-sweep failure that hid 13 dead layer
        // contracts until the layer sweep printed its own counts.
        Assert.True(agreed > 0,
            "no model contract was verified against a real forward pass, so this proved nothing");

        Assert.True(disagreed.Count == 0,
            $"{disagreed.Count} model contract(s) claim a shape their own Predict does not produce."
            + Environment.NewLine + string.Join(Environment.NewLine, disagreed));
    }

    private static object? Construct(Type closed, InputType inputType)
    {
        var ctor = closed.GetConstructors().FirstOrDefault(c =>
        {
            var ps = c.GetParameters();
            return ps.Length > 0
                && ps[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && ps.Skip(1).All(p => p.HasDefaultValue);
        });

        if (ctor is null)
        {
            return closed.GetConstructor(Type.EmptyTypes) is not null
                ? Activator.CreateInstance(closed) : null;
        }

        var pars = ctor.GetParameters();
        var args = new object?[pars.Length];
        args[0] = inputType switch
        {
            InputType.OneDimensional => new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: Extent, outputSize: Classes),

            // [Height, Width] per sample, so [1, Extent, Extent] batched. For a sequence family that
            // reads as [Batch, SequenceLength, NumFeatures].
            InputType.TwoDimensional => new NeuralNetworkArchitecture<double>(
                InputType.TwoDimensional, NeuralNetworkTaskType.Regression,
                inputHeight: Extent, inputWidth: Extent, outputSize: Classes),

            _ => new NeuralNetworkArchitecture<double>(
                InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
                inputDepth: 3, inputHeight: Extent, inputWidth: Extent, outputSize: Classes),
        };

        for (int i = 1; i < pars.Length; i++)
        {
            var p = pars[i];
            bool isClassCount = p.ParameterType == typeof(int)
                && (p.Name?.IndexOf("numClasses", StringComparison.OrdinalIgnoreCase) >= 0
                    || p.Name?.IndexOf("classCount", StringComparison.OrdinalIgnoreCase) >= 0);
            args[i] = isClassCount ? Classes : p.DefaultValue;
        }

        return ctor.Invoke(args);
    }

    private static int[]? TryArchitectureInputShape(object model)
    {
        try
        {
            dynamic arch = ((dynamic)model).GetArchitecture();
            int[] shape = arch.GetInputShape();
            return shape;
        }
        catch { return null; }
    }

    private static (int[]? Shape, string? Failure) TryPredict(object model, int[] shape)
    {
        try
        {
            var probe = new Tensor<double>(shape);
            for (int i = 0; i < probe.Length; i++) probe[i] = (i * 7) % 13;
            var result = ((dynamic)model).Predict(probe);
            return result is null ? (null, "Predict returned null") : ((int[])result._shape, null);
        }
        catch (Exception ex)
        {
            var root = Unwrap(ex);
            var msg = root.Message.Split('\n')[0].Trim();
            return (null, $"{root.GetType().Name}: {(msg.Length > 80 ? msg.Substring(0, 80) + "..." : msg)}");
        }
    }

    private static int EnvInt(string name, int fallback, int minimum) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v >= minimum ? v : fallback;

    private static bool DerivesFromLayerBase(Type type)
    {
        for (var a = type.BaseType; a is not null; a = a.BaseType)
        {
            var def = a.IsGenericType ? a.GetGenericTypeDefinition() : a;
            if (def.Name == "LayerBase`1") return true;
        }
        return false;
    }

    private static Exception Unwrap(Exception ex) =>
        ex is TargetInvocationException { InnerException: not null } tie ? tie.InnerException : ex;
}
