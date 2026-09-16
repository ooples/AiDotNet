using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class SourceImpactTests
{
    public enum MalformedMember { NullCalls, NullState, NullCallEntry, NullStateEntry, DuplicateCalls, DuplicateState, Boundary, NullSpan }

    [Theory]
    [InlineData(MalformedMember.NullCalls)]
    [InlineData(MalformedMember.NullState)]
    [InlineData(MalformedMember.NullCallEntry)]
    [InlineData(MalformedMember.NullStateEntry)]
    [InlineData(MalformedMember.DuplicateCalls)]
    [InlineData(MalformedMember.DuplicateState)]
    [InlineData(MalformedMember.Boundary)]
    [InlineData(MalformedMember.NullSpan)]
    public void MalformedNestedMembersHaveTypedFormatFailures(MalformedMember mutation)
    {
        var json = System.Text.Json.Nodes.JsonNode.Parse(System.Text.Json.JsonSerializer.Serialize(Snapshot('a')))
            ?? throw new InvalidDataException();
        var method = json[nameof(SourceSnapshot.Methods)]?[0] ?? throw new InvalidDataException();
        var dependency = method[nameof(SourceMethod.Dependency)] ?? throw new InvalidDataException();
        switch (mutation)
        {
            case MalformedMember.NullCalls: dependency[nameof(DependencyNode.Calls)] = null; break;
            case MalformedMember.NullState: dependency[nameof(DependencyNode.SharedState)] = null; break;
            case MalformedMember.NullCallEntry: dependency[nameof(DependencyNode.Calls)] = System.Text.Json.Nodes.JsonNode.Parse("[null]"); break;
            case MalformedMember.NullStateEntry: dependency[nameof(DependencyNode.SharedState)] = System.Text.Json.Nodes.JsonNode.Parse("[null]"); break;
            case MalformedMember.DuplicateCalls: dependency[nameof(DependencyNode.Calls)] = System.Text.Json.Nodes.JsonNode.Parse("[\"a\",\"a\"]"); break;
            case MalformedMember.DuplicateState: dependency[nameof(DependencyNode.SharedState)] = System.Text.Json.Nodes.JsonNode.Parse("[\"a\",\"a\"]"); break;
            case MalformedMember.Boundary: dependency[nameof(DependencyNode.Boundary)] = 999; break;
            case MalformedMember.NullSpan: method[nameof(SourceMethod.Spans)] = System.Text.Json.Nodes.JsonNode.Parse("[null]"); break;
            default: throw new ArgumentOutOfRangeException(nameof(mutation));
        }
        SourceSnapshot malformed = System.Text.Json.JsonSerializer.Deserialize<SourceSnapshot>(json.ToJsonString()) ?? throw new InvalidDataException();
        Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() => Select(before: malformed)).Reason);
    }
    private static TestCaseIdentity[] Cases => [new("a1", "A"), new("a2", "A"), new("b", "B")];
    private static SourceSnapshot Snapshot(char revision) => new(1, new(revision, 40), "tests.dll", new('c', 64), new('c', 64), new('d', 64), SourceMapStatus.Verified,
        [new(new("a", [], [], DependencyBoundary.Closed), "A", new('e', 64), [new("Tests.cs", 10, 14)], false),
         new(new("b", [], [], DependencyBoundary.Closed), "B", new('e', 64), [new("Tests.cs", 20, 24)], false),
         new(new("helper", [], [], DependencyBoundary.Closed), "Helper", new('e', 64), [new("Tests.cs", 30, 34)], false)]);
    private static SourceDelta Delta(ChangedLines[] before, ChangedLines[] after, bool unmapped = false) => new(new('a', 40), new('b', 40), before, after, unmapped);
    private static SourceSelection Select(SourceSnapshot? before = null, SourceSnapshot? after = null, SourceDelta? delta = null, TestCaseIdentity[]? inventory = null) =>
        SourceImpact.Select(before ?? Snapshot('a'), after ?? Snapshot('b'), Cases, inventory ?? Cases,
            delta ?? Delta([new("Tests.cs", 11, 1)], [new("Tests.cs", 11, 1)]));

    [Fact]
    public void BodyEditSelectsItsOwnerRatherThanTheWholeWorkload()
    {
        SourceSelection selected = Select();
        Assert.False(selected.FullFallback);
        Assert.Equal("A", Assert.Single(selected.Methods).MethodId);
        Assert.Equal("a", Assert.Single(selected.ChangedNodes));
    }

    [Fact]
    public void UnmappedLineBroadensRatherThanSkipping()
    {
        SourceSelection selected = Select(delta: Delta([], [new("Tests.cs", 9, 1)]));
        Assert.True(selected.FullFallback);
        Assert.Equal(2, selected.Methods.Length);
    }

    [Fact]
    public void HoleBetweenSourceSpansCannotBeMistakenForCoverage()
    {
        SourceSelection selected = Select(delta: Delta([], [new("Tests.cs", 12, 10)]));
        Assert.True(selected.FullFallback);
        Assert.Equal(2, selected.Methods.Length);
    }

    [Fact]
    public void WrongRevisionIsRejected()
    {
        Assert.Throws<EvidenceException>(() => Select(before: Snapshot('f')));
        Assert.Throws<EvidenceException>(() => Select(after: Snapshot('f')));
    }

    [Fact]
    public void DuplicateAndEmptyMethodIdsHaveTypedFormatFailures()
    {
        SourceSnapshot before = Snapshot('a');
        before.Methods[1] = before.Methods[1] with { Dependency = before.Methods[1].Dependency with { Id = "a" } };
        Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() => Select(before: before)).Reason);
        before.Methods[1] = before.Methods[1] with { Dependency = before.Methods[1].Dependency with { Id = "" } };
        Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() => Select(before: before)).Reason);
    }

    [Fact]
    public void UnverifiableMapCannotNarrowExecution()
    {
        Assert.True(Select(before: Snapshot('a') with { Status = SourceMapStatus.Unverifiable }).FullFallback);
        Assert.True(Select(after: Snapshot('b') with { Status = SourceMapStatus.Unverifiable }).FullFallback);
    }

    [Fact]
    public void BinaryBodyChangeIsIncludedEvenWithoutAnObservedSourceEdit()
    {
        SourceSnapshot after = Snapshot('b');
        after.Methods[0] = after.Methods[0] with { BodyHash = new('f', 64) };
        Assert.Equal("A", Assert.Single(Select(after: after, delta: Delta([], [])).Methods).MethodId);
    }

    [Fact]
    public void MetadataAndUnknownConfigurationChangesForceFullExecution()
    {
        Assert.True(Select(after: Snapshot('b') with { ConfigurationHash = new('f', 64) }).FullFallback);
        Assert.True(Select(delta: Delta([], [], true)).FullFallback);
    }

    [Fact]
    public void RemovedSourceLineStillUsesTheOldDependencyMap()
    {
        SourceSelection selected = Select(delta: Delta([new("Tests.cs", 11, 1)], [new("Tests.cs", 10, 0)]));
        Assert.Equal("A", Assert.Single(selected.Methods).MethodId);
    }

    [Fact]
    public void MissingTestRootCannotSilentlyDisappear()
    {
        SourceSelection selected = Select(inventory: [.. Cases, new("c", "C")]);
        Assert.True(selected.FullFallback);
        Assert.Equal(3, selected.Methods.Length);
    }

    [Fact]
    public void ChangedTheoryInventorySelectsItsCompleteMethod()
    {
        SourceSelection selected = Select(delta: Delta([], []), inventory: [.. Cases, new("a3", "A")]);
        Assert.Equal("A", Assert.Single(selected.Methods).MethodId);
    }

    [Fact]
    public void OpenDependencyIsNotTreatedAsUnchangedAndSafe()
    {
        SourceSnapshot after = Snapshot('b');
        after.Methods[1] = after.Methods[1] with { Dependency = after.Methods[1].Dependency with { Boundary = DependencyBoundary.External } };
        Assert.Equal(2, Select(after: after).Methods.Length);
    }

    [Fact]
    public void HelperOrFixtureChangeAppliesToTheWholeExecutionGroup()
    {
        SourceSelection selected = Select(delta: Delta([new("Tests.cs", 31, 1)], [new("Tests.cs", 31, 1)]));
        Assert.Equal(2, selected.Methods.Length);
        Assert.All(selected.Methods, method => Assert.Equal(SelectionReason.GroupDependency, method.Reason));
    }
}
