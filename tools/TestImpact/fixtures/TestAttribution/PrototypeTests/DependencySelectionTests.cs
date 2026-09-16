using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "DependencySelection")]
public sealed class DependencySelectionTests
{
    private static DependencyNode Node(string id, string[]? calls = null, string[]? state = null,
        DependencyBoundary boundary = DependencyBoundary.Closed) => new(id, calls ?? [], state ?? [], boundary);
    private static DependencySnapshot Snapshot(DependencyNode[] nodes, TestDependencyRoots[]? tests = null,
        string[]? group = null) => new(nodes, tests ?? [new("Left", ["left"]), new("Right", ["right"])], group ?? []);
    private static string[] Selected(DependencySnapshot before, DependencySnapshot after, params string[] changed) =>
        DependencySelection.Select(before, after, changed, false).Select(item => item.MethodId).ToArray();

    [Fact]
    public void UntakenStaticBranchStillSelectsCallerOnly()
    {
        var graph = Snapshot([Node("left", ["branch"]), Node("branch"), Node("right")]);
        Assert.Equal(new[] { "Left" }, Selected(graph, graph, "branch"));
    }

    [Fact]
    public void RemovedCallRetainsOldCallerDependency()
    {
        var old = Snapshot([Node("left", ["deleted"]), Node("deleted"), Node("right")]);
        var next = Snapshot([Node("left"), Node("right")]);
        Assert.Equal(new[] { "Left" }, Selected(old, next, "deleted"));
    }

    [Fact]
    public void AddedBranchUsesNewGraph()
    {
        var old = Snapshot([Node("left"), Node("right")]);
        var next = Snapshot([Node("left", ["new"]), Node("new"), Node("right")]);
        Assert.Equal(new[] { "Left" }, Selected(old, next, "new"));
    }

    [Fact]
    public void MissingCallTargetCannotAuthorizeSkipping()
    {
        var graph = Snapshot([Node("left", ["missing"]), Node("right")]);
        Assert.Equal(new[] { "Left", "Right" }, Selected(graph, graph, "right"));
    }

    [Theory]
    [InlineData(DependencyBoundary.Unresolved)]
    [InlineData(DependencyBoundary.VirtualDispatch)]
    [InlineData(DependencyBoundary.Reflection)]
    [InlineData(DependencyBoundary.External)]
    [InlineData(DependencyBoundary.Native)]
    public void OpenBoundaryCannotAuthorizeSkipping(DependencyBoundary boundary)
    {
        var graph = Snapshot([Node("left", boundary: boundary), Node("right")]);
        Assert.Equal(new[] { "Left", "Right" }, Selected(graph, graph, "right"));
    }

    [Fact]
    public void GroupDependencySelectsEntireWorkload()
    {
        var graph = Snapshot([Node("left"), Node("right"), Node("fixture", ["dependency"]), Node("dependency")], group: ["fixture"]);
        Assert.Equal(new[] { "Left", "Right" }, Selected(graph, graph, "dependency"));
    }

    [Fact]
    public void SharedStateClosesTransitivelyAcrossTests()
    {
        var graph = Snapshot([Node("left", state: ["one"]), Node("right", state: ["one", "two"]), Node("third", state: ["two"])],
            [new("Left", ["left"]), new("Right", ["right"]), new("Third", ["third"])]);
        Assert.Equal(new[] { "Left", "Right", "Third" }, Selected(graph, graph, "left"));
    }

    [Fact]
    public void DeletedTestMayHaveInitializedSharedState()
    {
        var old = Snapshot([Node("left", state: ["one"]), Node("right", state: ["one"])]);
        var next = Snapshot([Node("right", state: ["one"])], [new("Right", ["right"])]);
        Assert.Equal(new[] { "Right" }, Selected(old, next, "left"));
    }

    [Fact]
    public void ChangedTestSharingStateWithFixtureSelectsWholeWorkload()
    {
        var graph = Snapshot([Node("left", state: ["shared"]), Node("right"), Node("fixture", state: ["shared"])], group: ["fixture"]);
        Assert.Equal(new[] { "Left", "Right" }, Selected(graph, graph, "left"));
    }

    [Fact]
    public void UnknownChangedInputExpandsRatherThanDisappears()
    {
        var graph = Snapshot([Node("left"), Node("right")]);
        Assert.Equal(new[] { "Left", "Right" }, Selected(graph, graph, "unmapped"));
        Assert.All(DependencySelection.Select(graph, graph, ["unmapped"], false), item => Assert.Equal(SelectionReason.UnmappedChange, item.Reason));
        Assert.Equal(2, DependencySelection.Select(graph, graph, [], true).Count);
    }

    [Fact]
    public void NewTestRunsWithoutOldCoverage()
    {
        var old = Snapshot([Node("right")], [new("Right", ["right"])]);
        var next = Snapshot([Node("left"), Node("right")]);
        Assert.Equal(new[] { "Left" }, Selected(old, next));
    }

    [Fact]
    public void CyclesTerminateAndDoNotSelectDisconnectedTest()
    {
        var graph = Snapshot([Node("left", ["cycle"]), Node("cycle", ["left"]), Node("right")]);
        Assert.Equal(new[] { "Left" }, Selected(graph, graph, "cycle"));
        Assert.Empty(Selected(graph, graph));
    }

    [Fact]
    public void InvalidGraphIdentitiesAndMissingRootsAreRejected()
    {
        var valid = Snapshot([Node("left"), Node("right")]);
        Assert.Throws<InvalidDataException>(() => Selected(valid, Snapshot([Node("left"), Node("left")])));
        Assert.Throws<InvalidDataException>(() => Selected(valid, Snapshot([Node("left")], [new("Left", [])])));
        Assert.Throws<InvalidDataException>(() => Selected(valid, valid, "left", "left"));
        Assert.Throws<InvalidDataException>(() => Selected(valid, Snapshot([Node("left", boundary: (DependencyBoundary)99)])));
    }
}
