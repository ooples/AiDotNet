using System.Reflection;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serialization;

/// <summary>
/// The clone registry registers a generated plan only for the type it is asked about, looked up by the runtime
/// full name the generator recorded. A name the generator spells differently from <c>Type.FullName</c> would send
/// that type to the reflection fallback, silently replacing its compile-time plan.
/// </summary>
public sealed class CloneLazyRegistrationTests
{
    public CloneLazyRegistrationTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void EveryGeneratedPlan_IsFoundByItsRuntimeName()
    {
        var registrations = typeof(CloneRegistry).Assembly.GetType("AiDotNet.Generated.CloneRegistrations");
        Assert.NotNull(registrations);
        var tryRegister = registrations!.GetMethod("TryRegister", BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(tryRegister);

        var types = CloneRegistry.VerifiedTypes().ToList();
        Assert.NotEmpty(types);
        var missed = types
            .Where(type => !(bool)tryRegister!.Invoke(null, new object[] { type })!)
            .Select(type => type.FullName)
            .ToList();

        Assert.True(missed.Count == 0,
            $"{missed.Count} of {types.Count} generated plans are not found by name: {string.Join(", ", missed.Take(10))}");
    }

    [Fact]
    public void ClosedGeneric_ResolvesThroughItsOpenDefinition()
    {
        var registrations = typeof(CloneRegistry).Assembly.GetType("AiDotNet.Generated.CloneRegistrations")!;
        var tryRegister = registrations.GetMethod("TryRegister", BindingFlags.Static | BindingFlags.NonPublic)!;

        Assert.True((bool)tryRegister.Invoke(null, new object[] { typeof(AiDotNet.NeuralNetworks.Layers.DenseLayer<double>) })!);
        Assert.True(CloneRegistry.IsVerified(typeof(AiDotNet.NeuralNetworks.Layers.DenseLayer<double>)));
    }
}
