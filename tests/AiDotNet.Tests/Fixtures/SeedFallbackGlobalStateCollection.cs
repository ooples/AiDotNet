using Xunit;

namespace AiDotNet.Tests.Fixtures;

/// <summary>
/// Test classes that change <c>NeuralNetworkArchitecture&lt;T&gt;.DefaultRandomSeedOverride</c>, the process-wide
/// fallback seed, and so must not run beside the model-family tests that pin it.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public class SeedFallbackGlobalStateCollection
{
    /// <summary>Name used in <c>[Collection(...)]</c> attributes on classes that change the fallback seed.</summary>
    public const string Name = "SeedFallbackGlobalState";
}