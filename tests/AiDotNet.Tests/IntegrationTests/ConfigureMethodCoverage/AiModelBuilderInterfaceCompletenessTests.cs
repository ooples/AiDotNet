using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Guards the facade's public surface. Every FLUENT builder method on the concrete
/// <see cref="AiModelBuilder{T,TInput,TOutput}"/> — one that returns <see cref="IAiModelBuilder{T,TInput,TOutput}"/>
/// for chaining — must be a CONSCIOUS surface decision: reachable through <see cref="IAiModelBuilder{T,TInput,TOutput}"/>
/// or a derived capability interface the builder implements (e.g. <c>IWeightStreamingCapableBuilder</c>), OR
/// explicitly pinned in <see cref="IntentionallyConcreteOnly"/>. This catches the accidental gap that hid
/// <c>ConfigureMemoryManagement</c> from callers, while respecting the documented policy that adding abstract
/// members to <see cref="IAiModelBuilder{T,TInput,TOutput}"/> is a breaking change — so advanced methods stay
/// concrete-only (the concrete type is also what the generated YAML applier dispatches against).
/// </summary>
public sealed class AiModelBuilderInterfaceCompletenessTests
{
    /// <summary>
    /// Fluent methods intentionally NOT on IAiModelBuilder for binary compatibility. Adding a NEW fluent method
    /// forces a deliberate choice: declare it on the interface / a derived capability interface (a documented
    /// breaking change with a version bump), or add it here with the reason. Keyed by name + parameter types.
    /// </summary>
    private static readonly HashSet<string> IntentionallyConcreteOnly = new(StringComparer.Ordinal)
    {
        "ConfigureMemoryManagement(AiDotNet.Training.Memory.TrainingMemoryConfig)",
        "ConfigureProfiling(AiDotNet.Deployment.Configuration.ProfilingConfig)",
        "ConfigureInterpretability(AiDotNet.Models.Options.InterpretabilityOptions)",
        "ConfigurePreprocessing(AiDotNet.Preprocessing.PreprocessingPipeline`3[T,TInput,TInput])",
        "ConfigureAugmentation(AiDotNet.Augmentation.AugmentationConfig`2[T,TInput])",
        "ConfigureAutoML(AiDotNet.Interfaces.IAutoMLModel`3[T,TInput,TOutput])",
        "ConfigureDataLoader(AiDotNet.Interfaces.IDataLoader`2[AiDotNet.NeuralRadianceFields.Data.ImageView`1[T],AiDotNet.NeuralRadianceFields.Data.PixelBatch`1[T]])",
    };

    [Fact]
    [Trait("category", "integration-configure-method")]
    public void Every_fluent_concrete_builder_method_is_a_conscious_surface_decision()
    {
        var concrete = typeof(AiModelBuilder<,,>);
        var iface = typeof(IAiModelBuilder<,,>);

        // Derived capability interfaces count as reachable ONLY if the concrete builder actually implements them
        // (so a caller can cast to them). Collect the open generic definitions the concrete implements.
        var implementedInterfaceDefs = concrete.GetInterfaces()
            .Where(i => i.IsGenericType)
            .Select(i => i.GetGenericTypeDefinition())
            .ToHashSet();

        var reachableInterfaces = new List<Type> { iface };
        reachableInterfaces.AddRange(iface.Assembly.GetTypes().Where(t =>
            t.IsInterface && t.IsGenericTypeDefinition && t != iface &&
            implementedInterfaceDefs.Contains(t) &&
            t.GetInterfaces().Any(bi => bi.IsGenericType && bi.GetGenericTypeDefinition() == typeof(IAiModelBuilder<,,>))));

        var reachableSignatures = reachableInterfaces
            .SelectMany(i => i.GetMethods(BindingFlags.Public | BindingFlags.Instance))
            .Select(Signature)
            .ToHashSet(StringComparer.Ordinal);

        var missing = new List<string>();
        foreach (var method in concrete.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            if (method.IsSpecialName) continue;                 // property/event accessors
            if (!ReturnsBuilderInterface(method)) continue;     // only fluent methods that return IAiModelBuilder<,,>

            var sig = Signature(method);
            if (!reachableSignatures.Contains(sig) && !IntentionallyConcreteOnly.Contains(sig))
            {
                missing.Add(sig);
            }
        }

        Assert.True(
            missing.Count == 0,
            "These fluent AiModelBuilder methods return IAiModelBuilder for chaining but are NEITHER reachable " +
            "through IAiModelBuilder / a derived capability interface the builder implements, NOR pinned as " +
            "intentionally concrete-only. Make a deliberate choice for each: declare it on the interface (a " +
            "documented breaking change) or add it to IntentionallyConcreteOnly with a reason:\n  " +
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
