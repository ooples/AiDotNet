using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Guards the facade's public surface: every FLUENT builder method on the concrete
/// <see cref="AiModelBuilder{T,TInput,TOutput}"/> — i.e. one that returns
/// <see cref="IAiModelBuilder{T,TInput,TOutput}"/> for chaining — MUST also be declared on the
/// <see cref="IAiModelBuilder{T,TInput,TOutput}"/> interface. Otherwise a caller holding the interface (the
/// normal facade surface, e.g. after the first <c>Configure*</c> call in a chain) cannot invoke it — the exact
/// gap that hid <c>ConfigureMemoryManagement</c>. Methods that deliberately return the CONCRETE builder
/// (e.g. ONNX-export helpers) are exempt, since they are intentionally not part of the interface contract.
/// </summary>
public sealed class AiModelBuilderInterfaceCompletenessTests
{
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Every_fluent_concrete_builder_method_is_declared_on_the_interface()
    {
        var concrete = typeof(AiModelBuilder<,,>);
        var iface = typeof(IAiModelBuilder<,,>);

        // A fluent method counts as "reachable" if it is declared on IAiModelBuilder OR on one of its documented
        // derived capability interfaces (e.g. IWeightStreamingCapableBuilder : IAiModelBuilder) — those are
        // opt-in-via-cast surfaces the facade deliberately splits out to keep IAiModelBuilder's binary compat.
        var builderInterfaces = new List<Type> { iface };
        builderInterfaces.AddRange(iface.Assembly.GetTypes().Where(t =>
            t.IsInterface && t.IsGenericTypeDefinition && t != iface &&
            t.GetInterfaces().Any(bi => bi.IsGenericType && bi.GetGenericTypeDefinition() == typeof(IAiModelBuilder<,,>))));

        var interfaceSignatures = builderInterfaces
            .SelectMany(i => i.GetMethods(BindingFlags.Public | BindingFlags.Instance))
            .Select(Signature)
            .ToHashSet(StringComparer.Ordinal);

        var missing = new List<string>();
        foreach (var method in concrete.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (method.IsSpecialName) continue;                 // property/event accessors
            if (!ReturnsBuilderInterface(method)) continue;     // only fluent methods that return IAiModelBuilder<,,>

            var sig = Signature(method);
            if (!interfaceSignatures.Contains(sig))
            {
                missing.Add(sig);
            }
        }

        Assert.True(
            missing.Count == 0,
            "These fluent AiModelBuilder methods return IAiModelBuilder for chaining but are NOT declared on " +
            "IAiModelBuilder, so interface-typed callers can't use them. Add them to IAiModelBuilder " +
            "(or make them return the concrete builder if they are intentionally concrete-only):\n  " +
            string.Join("\n  ", missing.OrderBy(s => s, StringComparer.Ordinal)));
    }

    private static bool ReturnsBuilderInterface(MethodInfo method)
    {
        var rt = method.ReturnType;
        return rt.IsGenericType && rt.GetGenericTypeDefinition() == typeof(IAiModelBuilder<,,>);
    }

    // Name + ordered parameter types. Uses the (open) generic parameter names / type names, which are identical
    // between the concrete and interface open generic definitions, so a match means a real signature match.
    private static string Signature(MethodInfo method)
    {
        var parameters = string.Join(", ", method.GetParameters().Select(p => p.ParameterType.ToString()));
        return $"{method.Name}({parameters})";
    }
}
