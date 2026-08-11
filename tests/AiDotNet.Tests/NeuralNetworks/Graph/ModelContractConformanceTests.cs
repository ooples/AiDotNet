using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
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

        int declared = 0, concrete = 0, unavailable = 0, agreed = 0, declined = 0;
        var disagreed = new List<string>();
        var unverified = new List<string>();

        foreach (var open in models)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            declared++;

            // Capability is inspected before construction. VisionLanguageModelBase explicitly marks
            // its honest family-wide null as unavailable, so 170 paper-scale descendants no longer
            // have to allocate billions of parameters merely to return that null. An override resolves
            // to a different interface target and is therefore a real contract that must be probed.
            if (!ShapeInference.HasDeclaredOutputShapeContract(closed))
            {
                unavailable++;
                declined++;
                continue;
            }

            concrete++;
            var result = await ModelShapeConformanceProcess.ProbeAsync(
                closed, Extent, Classes, TimeSpan.FromMinutes(3));

            if (result.Status == "agreed")
            {
                agreed++;
                continue;
            }

            if (result.Status == "disagreed")
            {
                disagreed.Add($"{open.Name}: in [{Join(result.InputShape)}] "
                    + $"contract says [{Join(result.PredictedShape)}] "
                    + $"but Predict returned [{Join(result.ActualShape)}]");
                continue;
            }

            if (result.Status == "declined") declined++;
            unverified.Add($"{open.Name}: {result.Status}"
                + (string.IsNullOrWhiteSpace(result.Error) ? string.Empty : $" - {result.Error}"));
        }

        _out.WriteLine($"models declaring a contract : {declared}");
        _out.WriteLine($"  concrete contracts           : {concrete}");
        _out.WriteLine($"  explicitly unavailable       : {unavailable}");
        _out.WriteLine($"  contract agreed with Predict : {agreed}");
        _out.WriteLine($"  contract declined (null)     : {declined}");
        _out.WriteLine($"  DISAGREED                    : {disagreed.Count}");
        _out.WriteLine($"  unverified                   : {unverified.Count}");
        foreach (var s in unverified.Take(15)) _out.WriteLine($"    unverified: {s}");
        foreach (var d in disagreed) _out.WriteLine($"    DISAGREED: {d}");

        // A window made exclusively of EXPLICIT unavailability declarations is a valid inventory
        // result, not a failed conformance run. The old agreed > 0 assertion made 34 arbitrary
        // windows fail even though none contained a concrete contract. Non-vacuity is now exact:
        // every concrete contract in this window must complete a real comparison, while every
        // unavailable contract is accounted for without construction.
        Assert.True(declared > 0, "the conformance window selected no model contracts");
        Assert.Equal(declared, concrete + unavailable);
        Assert.True(unverified.Count == 0,
            $"{unverified.Count} concrete model contract(s) could not be verified."
            + Environment.NewLine + string.Join(Environment.NewLine, unverified));
        Assert.Equal(concrete, agreed + disagreed.Count);

        Assert.True(disagreed.Count == 0,
            $"{disagreed.Count} model contract(s) claim a shape their own Predict does not produce."
            + Environment.NewLine + string.Join(Environment.NewLine, disagreed));
    }

    private static string Join(int[]? shape) => shape is null ? "?" : string.Join(",", shape);

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
}
