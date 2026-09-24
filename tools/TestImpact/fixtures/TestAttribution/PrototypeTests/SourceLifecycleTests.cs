using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class SourceLifecycleTests
{
    private static readonly TestCaseIdentity[] Cases = [new("a", "A"), new("b", "B")];
    private static SourceSnapshot Snapshot(char revision) => new(1, new(revision, 40), "tests.dll", new('c', 64), new('d', 64),
        new('e', 64), SourceMapStatus.Verified,
        [Node("a", "A", 10), Node("b", "B", 20), Node("ctor-a", "A..ctor", 30), Node("ctor-b", "B..ctor", 40),
         Node("unrelated", "Unrelated.Helper", 50, DependencyBoundary.Reflection)],
        new(1, [new("A", ["a", "ctor-a"], true), new("B", ["b", "ctor-b"], true)], []));
    private static SourceMethod Node(string id, string owner, int line, DependencyBoundary boundary = DependencyBoundary.Closed) =>
        new(new(id, [], [], boundary), owner, new('f', 64), [new("Tests.cs", line, line)], false);
    private static SourceSelection Select(SourceSnapshot before, SourceSnapshot after, int line = 10) =>
        SourceImpact.Select(before, after, Cases, Cases, new(before.SourceTree, after.SourceTree,
            [new("Tests.cs", line, 1)], [new("Tests.cs", line, 1)], false));

    [Fact]
    public void UnrelatedOpenHelperDoesNotBecomeAnAssemblyFixture()
        => Assert.Equal("A", Assert.Single(Select(Snapshot('a'), Snapshot('b')).Methods).MethodId);

    [Fact]
    public void ConstructorAndInheritedMethodRootsFollowTheirActualOwners()
    {
        Assert.Equal("A", Assert.Single(Select(Snapshot('a'), Snapshot('b'), 30).Methods).MethodId);
        SourceSnapshot before = Snapshot('a');
        SourceSnapshot after = Snapshot('b');
        before = before with { Methods = before.Methods.Select(method => method.Dependency.Id == "a" ? method with { Owner = "Base.Inherited" } : method).ToArray() };
        after = after with { Methods = after.Methods.Select(method => method.Dependency.Id == "a" ? method with { Owner = "Base.Inherited" } : method).ToArray() };
        Assert.Equal("A", Assert.Single(Select(before, after).Methods).MethodId);
    }

    [Fact]
    public void SharedFixtureStateCouplesTestsEvenWithoutStaticFields()
    {
        SourceSnapshot WithFixture(SourceSnapshot value) => value with
        {
            Methods = [.. value.Methods, new(new("fixture", [], ["collection:shared"], DependencyBoundary.Closed), "Fixture", new('f', 64), [], false)],
            LifecycleMap = new(1, [new("A", ["a", "fixture"], true), new("B", ["b", "fixture"], true)], [])
        };
        Assert.Equal(2, Select(WithFixture(Snapshot('a')), WithFixture(Snapshot('b'))).Methods.Length);
    }

    [Fact]
    public void UnknownDiscoveryCallbackStillInvalidatesTheWholeGroup()
    {
        SourceSnapshot Group(SourceSnapshot value) => value with { LifecycleMap = new(1,
            [new("A", ["a"], true), new("B", ["b"], true)], ["unrelated"]) };
        Assert.Equal(2, Select(Group(Snapshot('a')), Group(Snapshot('b'))).Methods.Length);
    }

    [Fact]
    public void UnsupportedOrMissingLifecycleOwnerCannotBeSkipped()
    {
        foreach (SourceLifecycleMap map in new[] {
            new SourceLifecycleMap(1, [new("A", ["a"], true), new("B", ["b"], false)], []),
            new SourceLifecycleMap(1, [new("A", ["a"], true)], []) })
            Assert.Equal(2, Select(Snapshot('a'), Snapshot('b') with { LifecycleMap = map }).Methods.Length);
    }

    [Fact]
    public void DuplicateLifecycleOwnersAreTypedFormatFailures()
    {
        var bad = Snapshot('b') with { LifecycleMap = new(1, [new("A", ["a"], true), new("A", ["b"], true)], []) };
        Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() => Select(Snapshot('a'), bad)).Reason);
    }
}
