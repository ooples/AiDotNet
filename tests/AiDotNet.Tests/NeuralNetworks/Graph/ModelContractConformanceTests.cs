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

        var models = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && t.GetInterfaces().Any(i => i.Name == "IShapeContract"))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        int declared = 0, agreed = 0, declined = 0;
        var disagreed = new List<string>();
        var skipped = new List<string>();

        foreach (var open in models)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            object? model = null;
            try { model = Construct(closed); }
            catch (Exception ex) { skipped.Add($"{open.Name}: {Unwrap(ex).GetType().Name} constructing"); continue; }
            if (model is null) { skipped.Add($"{open.Name}: no usable constructor"); continue; }

            try
            {
                if (model is not IShapeContract contract) { skipped.Add($"{open.Name}: not IShapeContract"); continue; }

                int[]? perSample = TryArchitectureInputShape(model);
                if (perSample is null || perSample.Length == 0 || perSample.Any(d => d <= 0))
                {
                    skipped.Add($"{open.Name}: no concrete declared input shape");
                    continue;
                }

                var shape = new int[perSample.Length + 1];
                shape[0] = 1;
                for (int i = 0; i < perSample.Length; i++) shape[i + 1] = Math.Min(perSample[i], Extent);

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

    private static object? Construct(Type closed)
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
        args[0] = new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            inputDepth: 3, inputHeight: Extent, inputWidth: Extent, outputSize: Classes);

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

    private static Exception Unwrap(Exception ex) =>
        ex is TargetInvocationException { InnerException: not null } tie ? tie.InnerException : ex;
}
