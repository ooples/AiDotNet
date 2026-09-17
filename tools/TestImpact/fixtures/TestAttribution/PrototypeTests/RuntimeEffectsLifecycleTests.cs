using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class RuntimeEffectsLifecycleTests
{
    [Fact]
    public void AWorkloadKeepsSharedRootsButNotUnrelatedOwnerRoots()
    {
        var unrelated = Enumerable.Range(0, 20_000).Select(index => new SourceTestLifecycle("other-" + index, ["expensive-" + index], true));
        var map = new SourceLifecycleMap(1, [new("selected", ["entry", "hook"], true), .. unrelated], ["discovery", "startup"]);
        Assert.Equal(["discovery", "entry", "hook", "startup"], RuntimeEffectsExperiment.LifecycleRoots(map, ["selected"]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MissingOrDuplicateOwnerBindingsAreRejected(bool duplicate)
    {
        var map = new SourceLifecycleMap(1, duplicate
            ? [new("selected", ["entry"], true), new("selected", ["other"], true)] : [], ["startup"]);
        Assert.Throws<InvalidDataException>(() => RuntimeEffectsExperiment.LifecycleRoots(map, ["selected"]));
    }

    [Fact]
    public void EmptyOrDuplicateInventoriesAreRejected()
    {
        var map = new SourceLifecycleMap(1, [new("selected", ["entry"], true)], []);
        Assert.Throws<InvalidDataException>(() => RuntimeEffectsExperiment.LifecycleRoots(map, []));
        Assert.Throws<InvalidDataException>(() => RuntimeEffectsExperiment.LifecycleRoots(map, ["selected", "selected"]));
    }
}
