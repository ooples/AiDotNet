using AiDotNet.TestImpact;
using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class RuntimeEffectsLifecycleTests
{
    [Fact]
    public void ObservedOwnerSliceMatchesItsFullLifecycleWithoutPublishingAPartialGroupGraph()
    {
        using var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(RuntimeEffectsLifecycleTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(RuntimeEffectsLifecycleTests).Assembly.Location,
            new ReaderParameters { AssemblyResolver = resolver });
        string owner = "PrototypeTests:PrototypeTests.TrialHookTests.HookedFact";
        SourceLifecycleMap full = XunitLifecycleReader.Read(assembly).Map;
        SourceTestLifecycle expected = Assert.Single(full.Tests, test => test.Owner == owner);
        SourceTestLifecycle actual = Assert.Single(XunitLifecycleReader.ReadObservedOwners(assembly, [owner]));
        Assert.Equal(expected.Owner, actual.Owner);
        Assert.Equal(expected.Complete, actual.Complete);
        Assert.Equal(expected.Roots, actual.Roots);
        Assert.True(full.Tests.Length > 1);
        Assert.Empty(XunitLifecycleReader.ReadObservedOwners(assembly, [owner + "Missing"]));
    }

    [Fact]
    public void MalformedObservedOwnerSlicesAreRejected()
    {
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(RuntimeEffectsLifecycleTests).Assembly.Location);
        Assert.Throws<InvalidDataException>(() => XunitLifecycleReader.ReadObservedOwners(assembly, []));
        Assert.Throws<InvalidDataException>(() => XunitLifecycleReader.ReadObservedOwners(assembly, ["malformed"]));
        Assert.Throws<InvalidDataException>(() => XunitLifecycleReader.ReadObservedOwners(assembly, ["A.B", "A.B"]));
    }

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
