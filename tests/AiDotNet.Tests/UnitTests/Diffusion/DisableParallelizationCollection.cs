using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Marker collection for diffusion tests that mutate PROCESS-GLOBAL static state (for example
/// <see cref="AiDotNet.Diffusion.NoisePredictors.NoisePredictorBase{T}.CheckpointingThresholdOverride"/>).
/// </summary>
/// <remarks>
/// <c>DisableParallelization = true</c> stops member classes from running concurrently with any other
/// test collection in the assembly, so a global a test temporarily changes cannot leak into predictors
/// constructed on parallel threads. Member tests must still restore the global with try/finally.
/// </remarks>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class DisableParallelizationCollection
{
    public const string Name = "DiffusionGlobalState-Serial";
}
