using AiDotNet.Models;
using AiDotNet.VisionLanguage.Editing;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class MgieCloneConfigurationReviewTests
{
    private readonly ITestOutputHelper _output;

    public MgieCloneConfigurationReviewTests(ITestOutputHelper output)
    {
        TestModuleInitializer.EnsureInitialized();
        _output = output;
    }

    [Fact]
    public void GeneratedClonePlan_PreservesAllInjectedComponents()
    {
        var modelType = typeof(MGIE<float>);
        var plan = CloneRegistry.GetPlan(modelType);
        _output.WriteLine("Verified generated plan: " + CloneRegistry.IsVerified(modelType));
        foreach (var candidate in plan.ConstructorCandidates)
            _output.WriteLine("Candidate: " + string.Join(", ", candidate));
        Assert.True(CloneRegistry.IsVerified(modelType));
        Assert.Contains(plan.ConstructorCandidates, candidate =>
            candidate.Contains("_instructionEncoder") && candidate.Contains("_unet") &&
            candidate.Contains("_vae") && candidate.Contains("_options"));
    }
}
