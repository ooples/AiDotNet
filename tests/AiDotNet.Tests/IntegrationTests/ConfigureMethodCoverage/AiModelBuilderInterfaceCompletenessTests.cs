using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Guards the facade's public surface: EVERY fluent builder method on the concrete
/// <see cref="AiModelBuilder{T,TInput,TOutput}"/> — one that returns <see cref="IAiModelBuilder{T,TInput,TOutput}"/>
/// for chaining — MUST also be declared on the <see cref="IAiModelBuilder{T,TInput,TOutput}"/> interface. There are
/// no concrete-only shortcuts and no allowlist: adding a Configure method to the concrete builder without adding
/// it to the interface makes it unreachable to interface-typed callers (the normal facade surface once a chain
/// returns the interface), which is the gap this test forbids. Adding a member to the interface is a breaking
/// change for external implementers — that trade-off is accepted; the facade surface must stay complete and
/// uniform.
/// </summary>
public sealed class AiModelBuilderInterfaceCompletenessTests
{
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Every_fluent_concrete_builder_method_is_declared_on_the_interface()
    {
        var concrete = typeof(AiModelBuilder<,,>);
        var iface = typeof(IAiModelBuilder<,,>);

        // The interface's own declared methods plus any inherited from base interfaces it extends.
        var interfaceSignatures = new[] { iface }
            .Concat(iface.GetInterfaces())
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
            "Every fluent AiModelBuilder method (one that returns IAiModelBuilder for chaining) MUST be declared " +
            "on IAiModelBuilder so interface-typed callers can invoke it — no concrete-only methods. Declare " +
            "these on IAiModelBuilder in the same style as the rest (a breaking change is acceptable):\n  " +
            string.Join("\n  ", missing.OrderBy(s => s, StringComparer.Ordinal)));
    }

    /// <summary>
    /// Terminal builder methods — those that end a chain rather than continue it — must also be on the interface.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The fluent test above cannot see these, because it selects on the RETURN type being
    /// <see cref="IAiModelBuilder{T, TInput, TOutput}"/> and a terminal call returns a result instead. That blind
    /// spot is not theoretical: <c>Build(features, labels)</c> was first added to the concrete builder only, and
    /// was unreachable from every documented example, because each example begins with a <c>Configure*</c> call
    /// and is therefore interface-typed by the time it reaches the terminal call. The build compiled and the
    /// fluent completeness test stayed green.
    /// </para>
    /// <para>
    /// "Terminal" is identified by name rather than by return type, because the results are heterogeneous
    /// (<c>AiModelResult</c>, <c>Task&lt;AiModelResult&gt;</c>) and a name-based rule states the intent directly:
    /// anything a caller invokes to finish a chain belongs on the interface.
    /// </para>
    /// </remarks>
    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Every_terminal_builder_method_is_declared_on_the_interface()
    {
        var concrete = typeof(AiModelBuilder<,,>);
        var iface = typeof(IAiModelBuilder<,,>);

        var interfaceSignatures = new[] { iface }
            .Concat(iface.GetInterfaces())
            .SelectMany(i => i.GetMethods(BindingFlags.Public | BindingFlags.Instance))
            .Select(Signature)
            .ToHashSet(StringComparer.Ordinal);

        var missing = new List<string>();
        foreach (var method in concrete.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (method.IsSpecialName) continue;
            if (!IsTerminalBuilderMethod(method)) continue;

            var sig = Signature(method);
            if (!interfaceSignatures.Contains(sig))
            {
                missing.Add(sig);
            }
        }

        Assert.True(
            missing.Count == 0,
            "Every terminal AiModelBuilder method (Build/BuildAsync and friends) MUST be declared on " +
            "IAiModelBuilder. A chain becomes interface-typed at its first Configure* call, so a terminal method " +
            "declared only on the concrete builder is unreachable exactly where callers need it:\n  " +
            string.Join("\n  ", missing.OrderBy(s => s, StringComparer.Ordinal)));
    }

    private static bool ReturnsBuilderInterface(MethodInfo method)
    {
        var rt = method.ReturnType;
        return rt.IsGenericType && rt.GetGenericTypeDefinition() == typeof(IAiModelBuilder<,,>);
    }

    private static bool IsTerminalBuilderMethod(MethodInfo method) =>
        !ReturnsBuilderInterface(method) &&
        (method.Name == "Build" || method.Name == "BuildAsync");

    // Name + ordered parameter types. Uses the (open) generic parameter names / type names, which are identical
    // between the concrete and interface open generic definitions, so a match means a real signature match.
    private static string Signature(MethodInfo method)
    {
        var parameters = string.Join(", ", method.GetParameters().Select(p => p.ParameterType.ToString()));
        return $"{method.Name}({parameters})";
    }
}
