using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class ManagedSourceTests
{
    private static readonly TestCaseIdentity[] Cases = [new("a", "A"), new("b", "B")];
    private static SourceManagedDependencies Managed(char hash = 'd', DependencyBoundary boundary = DependencyBoundary.Closed) => new(1,
        [new("Library.dll", new(hash, 64), ManagedBinaryOrigin.Bundle)],
        [new(new("Library:Helper", [], [], boundary), "Library:Helper", new('e', 64), [], false)], false);
    private static SourceSnapshot Snapshot(char revision) => new(1, new(revision, 40), "Tests.dll", new('c', 64), new('d', 64),
        new('e', 64), SourceMapStatus.Verified,
        [new(new("a", ["Library:Helper"], [], DependencyBoundary.Closed), "A", new('f', 64), [new("Tests.cs", 10, 10)], false),
         new(new("b", [], [], DependencyBoundary.Closed), "B", new('f', 64), [new("Tests.cs", 20, 20)], false)], ManagedDependencies: Managed());
    private static SourceDelta Delta(int line = 10) => new(new('a', 40), new('b', 40), [new("Tests.cs", line, 1)], [new("Tests.cs", line, 1)], false);
    private static SourceBundleSnapshot Bundle(char revision, SourceManagedDependencies other) => new(1, new(revision, 40), "Tests.dll",
        [Snapshot(revision), new(1, new(revision, 40), "Other.dll", new('c', 64), new('d', 64), new('e', 64),
            SourceMapStatus.Verified, [], ManagedDependencies: other)]);

    [Fact]
    public void ClosedManagedDependencyDoesNotForceUnrelatedTestExecution()
        => Assert.Equal("A", Assert.Single(SourceImpact.Select(Snapshot('a'), Snapshot('b'), Cases, Cases, Delta()).Methods).MethodId);

    [Fact]
    public void NativeManagedEndpointStillForcesItsCallerToExecute()
    {
        SourceSnapshot before = Snapshot('a') with { ManagedDependencies = Managed(boundary: DependencyBoundary.Native) };
        SourceSnapshot after = Snapshot('b') with { ManagedDependencies = Managed(boundary: DependencyBoundary.Native) };
        Assert.Equal(2, SourceImpact.Select(before, after, Cases, Cases, Delta(20)).Methods.Length);
    }

    [Fact]
    public void ChangedDependencyBinaryInvalidatesFullWorkload()
    {
        SourceSelection selection = SourceImpact.Select(Snapshot('a'), Snapshot('b') with { ManagedDependencies = Managed('f') }, Cases, Cases, Delta());
        Assert.True(selection.FullFallback);
        Assert.Equal(2, selection.Methods.Length);
    }

    [Fact]
    public void SharedManagedDefinitionsMergeWithoutLosingOpenBoundaries()
    {
        SourceManagedDependencies open = Managed(boundary: DependencyBoundary.Unresolved);
        Assert.Equal(2, SourceImpact.Select(Bundle('a', open), Bundle('b', open), Cases, Cases, Delta(20)).Methods.Length);
        Assert.Equal("A", Assert.Single(SourceImpact.Select(Bundle('a', Managed()), Bundle('b', Managed()), Cases, Cases, Delta()).Methods).MethodId);
    }

    [Fact]
    public void UnboundOrConflictingManagedDefinitionsAreRejected()
    {
        SourceManagedDependencies missing = Managed() with { Files = [] };
        Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() =>
            SourceImpact.Select(Snapshot('a'), Snapshot('b') with { ManagedDependencies = missing }, Cases, Cases, Delta())).Reason);
        SourceManagedDependencies conflicting = Managed() with { Methods = Managed().Methods.Select(method => method with { BodyHash = new('f', 64) }).ToArray() };
        Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() =>
            SourceImpact.Select(Bundle('a', Managed()), Bundle('b', conflicting), Cases, Cases, Delta())).Reason);
    }
}
