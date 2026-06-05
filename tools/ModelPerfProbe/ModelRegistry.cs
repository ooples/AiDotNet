using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.ModelPerfProbe;

/// <summary>
/// Discovers probeable models by reflecting over the AiDotNet assembly's
/// <see cref="ModelDomainAttribute"/>-tagged types. Mirrors the discovery the
/// existing TestScaffoldGenerator does for test scaffolds, so the perf-probe
/// scope tracks the test scope automatically.
/// </summary>
internal static class ModelRegistry
{
    /// <summary>
    /// Returns every probeable model type — i.e., every <c>[ModelDomain]</c>-tagged
    /// generic class that closes over <c>float</c> for the standard tape-trained
    /// <c>IFullModel&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;</c>
    /// surface the probe drives.
    /// </summary>
    public static List<Type> Discover()
    {
        var assembly = typeof(AiDotNet.NeuralNetworks.NeuralNetworkBase<>).Assembly;
        var openModelInterface = typeof(IFullModel<,,>);

        var hits = new List<Type>();
        foreach (var t in assembly.GetTypes())
        {
            if (!t.IsClass || t.IsAbstract) continue;

            // Require a [ModelDomain] / [Model*] tag so we don't probe helper types.
            var hasModelTag = t.GetCustomAttributes(inherit: false)
                .Any(a => a.GetType().Name.StartsWith("Model", StringComparison.Ordinal));
            if (!hasModelTag) continue;

            // Only generic-of-T models — the probe closes T=float.
            if (!t.IsGenericTypeDefinition) continue;
            if (t.GetGenericArguments().Length != 1) continue;

            Type closed;
            try
            {
                closed = t.MakeGenericType(typeof(float));
            }
            catch
            {
                continue;
            }

            // Skip models whose closed form doesn't implement the tape-trainable surface.
            // (Some models are <T, TInput, TOutput> with three generic params; skipped here.)
            bool implementsFullModel = closed.GetInterfaces().Any(i =>
                i.IsGenericType && i.GetGenericTypeDefinition() == openModelInterface
                && i.GetGenericArguments()[0] == typeof(float)
                && i.GetGenericArguments()[1] == typeof(Tensor<float>)
                && i.GetGenericArguments()[2] == typeof(Tensor<float>));
            if (!implementsFullModel) continue;

            hits.Add(closed);
        }

        hits.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
        return hits;
    }

    /// <summary>Closes <typeparamref name="float"/> over the named open generic class.</summary>
    public static Type? ResolveByName(string name)
    {
        var matches = Discover().Where(t =>
            string.Equals(t.Name, name, StringComparison.OrdinalIgnoreCase)
            || string.Equals(StripBacktick(t.Name), name, StringComparison.OrdinalIgnoreCase)).ToList();
        return matches.Count == 1 ? matches[0] : null;
    }

    private static string StripBacktick(string typeName)
    {
        int tick = typeName.IndexOf('`');
        return tick >= 0 ? typeName.Substring(0, tick) : typeName;
    }
}
