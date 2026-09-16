using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "ReuseProtocol")]
public sealed class ReuseTests
{
    private static ExecutionContextIdentity Old => new(new('a', 40), new('b', 64), new('c', 64));
    private static ExecutionContextIdentity Current => new(new('d', 40), new('e', 64), new('c', 64));
    private static TestCaseIdentity[] Cases => [new("a1", "A"), new("a2", "A"), new("b", "B")];
    private static DependencySnapshot Graph => new(
        [new("a", [], [], DependencyBoundary.Closed), new("b", [], [], DependencyBoundary.Closed)],
        [new("A", ["a"]), new("B", ["b"])], []);
    private static VerifiedExecution Pass(ExecutionPlan plan, TestCaseIdentity[] inventory)
    {
        var origin = new RunIdentity("owner/repository", 1, 1);
        return ExecutionEvidence.Verify(plan, inventory, new(1, plan.Workload, plan.Scope, plan.Context,
            plan.InventoryHash, plan.PlanHash, origin, plan.RequiredCases.Select(test => new TestCaseResult(test.CaseId, CaseOutcome.Passed)).ToArray()), origin);
    }
    private static VerifiedExecution Baseline => Pass(ExecutionEvidence.CreatePlan("work", Cases, [], ValidationScope.FullWorkload, Old), Cases);
    private static ReusePartition Partition(string[] changes, TestCaseIdentity[]? cases = null,
        DependencySnapshot? before = null, DependencySnapshot? after = null, ExecutionContextIdentity? context = null,
        bool unmapped = false) => ExecutionReuse.Prepare(Baseline, "work", cases ?? Cases, context ?? Current,
            new(Old, before ?? Graph), new(context ?? Current, after ?? Graph), changes, unmapped);

    [Fact]
    public void ChangedMethodExecutesEveryRowAndRetainsUnaffectedProvenance()
    {
        ReusePartition partition = Partition(["a"]);
        ExecutionPlan run = Assert.IsType<ExecutionPlan>(partition.Execution);
        Assert.Equal(new[] { "a1", "a2" }, run.RequiredCases.Select(test => test.CaseId));
        Assert.Equal("b", Assert.Single(partition.ReusedCases).CaseId);
        VerifiedReusePartition complete = ExecutionReuse.Complete(partition, Pass(run, Cases));
        Assert.False(complete.CanReplaceFullBaseline);
        Assert.Equal(Old, complete.Partition.Baseline.Context);
        Assert.Equal(Current, complete.Partition.Context);
    }

    [Fact]
    public void ReuseOnlyIsNotAZeroTestExecutionOrFreshBaseline()
    {
        ReusePartition partition = Partition([]);
        Assert.Null(partition.Execution);
        Assert.Equal(3, partition.ReusedCases.Count);
        Assert.False(ExecutionReuse.Complete(partition, null).CanReplaceFullBaseline);
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Complete(partition, Baseline));
    }

    [Fact]
    public void PartialBaselineCannotSeedReuse()
    {
        VerifiedExecution partial = Pass(ExecutionEvidence.CreatePlan("work", Cases, ["A"], ValidationScope.SelectedMethods, Old), Cases);
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Prepare(partial, "work", Cases, Current, new(Old, Graph), new(Current, Graph), [], false));
    }

    [Fact]
    public void WrongWorkloadAndGraphRevisionAreRejected()
    {
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Prepare(Baseline, "other", Cases, Current, new(Old, Graph), new(Current, Graph), [], false));
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Prepare(Baseline, "work", Cases, Current, new(Current, Graph), new(Current, Graph), [], false));
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Prepare(Baseline, "work", Cases, Current, new(Old, Graph), new(Old, Graph), [], false));
    }

    [Fact]
    public void MissingExtraAndDuplicateGraphMethodsCannotHideTests()
    {
        foreach (DependencySnapshot bad in new[] { Graph with { Tests = [new("A", ["a"])] },
            Graph with { Tests = [.. Graph.Tests, new("C", ["a"])] },
            Graph with { Tests = [.. Graph.Tests, new("A", ["a"])] } })
        {
            Assert.Throws<EvidenceException>(() => Partition([], before: bad));
            Assert.Throws<EvidenceException>(() => Partition([], after: bad));
        }
    }

    [Fact]
    public void AddedAndRemovedTheoryRowsExecuteTheWholeCurrentMethod()
    {
        foreach (TestCaseIdentity[] inventory in new[] { Cases.Append(new TestCaseIdentity("a3", "A")).ToArray(),
            Cases.Where(test => test.CaseId != "a2").ToArray() })
        {
            ReusePartition partition = Partition([], cases: inventory);
            Assert.Equal(inventory.Where(test => test.MethodId == "A"), Assert.IsType<ExecutionPlan>(partition.Execution).RequiredCases);
            Assert.Single(partition.ReusedCases);
        }
    }

    [Fact]
    public void ProfileChangeForcesFreshFullExecution()
    {
        ReusePartition partition = Partition([], context: Current with { ProfileFingerprint = new('f', 64) });
        ExecutionPlan run = Assert.IsType<ExecutionPlan>(partition.Execution);
        Assert.Equal(ValidationScope.FullWorkload, run.Scope);
        Assert.Empty(partition.ReusedCases);
        Assert.True(ExecutionReuse.Complete(partition, Pass(run, Cases)).CanReplaceFullBaseline);
    }

    [Fact]
    public void UnmappedAndUnknownChangesCannotReuseAnything()
    {
        foreach (ReusePartition partition in new[] { Partition([], unmapped: true), Partition(["unknown"]) })
        {
            Assert.Empty(partition.ReusedCases);
            Assert.Equal(ValidationScope.FullWorkload, Assert.IsType<ExecutionPlan>(partition.Execution).Scope);
        }
    }

    [Fact]
    public void MissingStaleAndWrongScopeExecutionsCannotCompletePartition()
    {
        ReusePartition partition = Partition(["a"]);
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Complete(partition, null));
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Complete(partition, Baseline));
        VerifiedExecution full = Pass(ExecutionEvidence.CreatePlan("work", Cases, [], ValidationScope.FullWorkload, Current), Cases);
        Assert.Throws<EvidenceException>(() => ExecutionReuse.Complete(partition, full));
    }

    [Fact]
    public void CallerCannotMutateTheStoredRequiredCaseSet()
    {
        ReusePartition partition = Partition(["a"]);
        ExecutionPlan exposed = Assert.IsType<ExecutionPlan>(partition.Execution);
        exposed.RequiredCases[0] = new("b", "B");
        Assert.Equal("a1", Assert.IsType<ExecutionPlan>(partition.Execution).RequiredCases[0].CaseId);
        Assert.Throws<EvidenceException>(() => Pass(exposed, Cases));
    }

    [Fact]
    public void RemovedEdgesAndOpenDependenciesStillInvalidateReuse()
    {
        DependencySnapshot prior = Graph with { Nodes = [new("a", ["b"], [], DependencyBoundary.Closed), Graph.Nodes[1]] };
        Assert.Empty(Partition(["b"], before: prior).ReusedCases);
        DependencySnapshot open = Graph with { Nodes = [Graph.Nodes[0], Graph.Nodes[1] with { Boundary = DependencyBoundary.Native }] };
        ReusePartition partition = Partition([], after: open);
        Assert.Equal("B", Assert.Single(Assert.IsType<ExecutionPlan>(partition.Execution).RequiredCases).MethodId);
    }

    [Fact]
    public void SharedStateCouplesThePartitionAcrossMethods()
    {
        DependencySnapshot state = Graph with { Nodes = Graph.Nodes.Select(node => node with { SharedState = ["shared"] }).ToArray() };
        Assert.Empty(Partition(["a"], after: state).ReusedCases);
    }
}
