using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "WorkflowProtocol")]
public sealed class WorkflowEvidenceTests
{
    private static readonly string Head = new('a', 40);
    private static readonly string Base = new('b', 40);
    private static readonly string Merge = new('c', 40);
    private static readonly WorkflowImportPolicy Policy = new("owner/repo", 12, 2, Head,
        ".github/workflows/ci.yml", "validation", "method-proof-12-2", WorkflowCompletionScope.WholeRun, ValidationScope.FullWorkload);
    private static readonly WorkflowRunEvidence Run = new("owner/repo", "owner/repo", 12, 2, Head,
        ".github/workflows/ci.yml", WorkflowEvent.PullRequest, WorkflowStatus.Completed, WorkflowConclusion.Success, [new(Base, Head, "main")]);
    private static readonly WorkflowJobEvidence Job = new(12, 2, "validation", WorkflowStatus.Completed, WorkflowConclusion.Success);
    private static readonly WorkflowArtifactEvidence Artifact = new(51, 12, "method-proof-12-2", false);
    private static void Validate(WorkflowImportPolicy? policy = null, WorkflowRunEvidence? run = null,
        WorkflowJobEvidence[]? jobs = null, WorkflowArtifactEvidence? artifact = null, WorkflowSourceEvidence? source = null) =>
        WorkflowEvidence.Validate(policy ?? Policy, run ?? Run, jobs ?? [Job], artifact ?? Artifact, source ?? new(Head, []));

    [Fact]
    public void ExactHeadAndExactMergeSourcesAreAccepted()
    {
        Validate();
        Validate(source: new(Merge, [Base, Head]));
    }

    [Fact]
    public void ForeignOrChangedBaseMergeIsRejected()
    {
        Assert.Throws<EvidenceException>(() => Validate(source: new(Merge, [new('d', 40), Head])));
        Assert.Throws<EvidenceException>(() => Validate(source: new(Merge, [Base, new('d', 40)])));
        Assert.Throws<EvidenceException>(() => Validate(source: new(Merge, [Base, Head, new('d', 40)])));
        Assert.Throws<EvidenceException>(() => Validate(run: Run with { Event = WorkflowEvent.Push }, source: new(Merge, [Base, Head])));
    }

    [Fact]
    public void AdvancedBaseRequiresBothAuthenticatedAncestryChecks()
    {
        string advanced = new('d', 40);
        var binding = new WorkflowBaseBinding(Base, advanced, "main", new('e', 40), true, true);
        var source = new WorkflowSourceEvidence(Merge, [advanced, Head], binding);
        Validate(source: source);
        Assert.Throws<EvidenceException>(() => Validate(source: source with { BaseBinding = binding with { RecordedBaseIsAncestor = false } }));
        Assert.Throws<EvidenceException>(() => Validate(source: source with { BaseBinding = binding with { TestedBaseIsAncestor = false } }));
        Assert.Throws<EvidenceException>(() => Validate(source: source with { BaseBinding = binding with { BaseBranch = "other" } }));
        Assert.Throws<EvidenceException>(() => Validate(source: source with { BaseBinding = binding with { TestedBaseSha = Base } }));
    }

    [Fact]
    public void ForeignRepositoryWorkflowHeadAndRunAreRejected()
    {
        foreach (WorkflowRunEvidence run in new[] { Run with { Repository = "foreign/repo" }, Run with { HeadRepository = "fork/repo" },
            Run with { WorkflowPath = ".github/workflows/other.yml" }, Run with { HeadSha = new('d', 40) }, Run with { RunId = 13 } })
            Assert.Throws<EvidenceException>(() => Validate(run: run));
    }

    [Fact]
    public void RerunAttemptAndOldAttemptArtifactAreRejected()
    {
        Assert.Throws<EvidenceException>(() => Validate(run: Run with { Attempt = 3 }));
        Assert.Throws<EvidenceException>(() => Validate(jobs: [Job with { Attempt = 1 }]));
        Assert.Throws<EvidenceException>(() => Validate(artifact: Artifact with { Name = "method-proof-12-1" }));
        Assert.Throws<EvidenceException>(() => Validate(policy: Policy with { ArtifactName = "method-proof-12-1" }, artifact: Artifact with { Name = "method-proof-12-1" }));
    }

    [Fact]
    public void UnfinishedCancelledAndSkippedWorkflowsCannotAuthorizeReuse()
    {
        Assert.Throws<EvidenceException>(() => Validate(run: Run with { Status = WorkflowStatus.Queued }));
        Assert.Throws<EvidenceException>(() => Validate(run: Run with { Status = WorkflowStatus.InProgress }));
        foreach (WorkflowConclusion conclusion in new[] { WorkflowConclusion.Cancelled, WorkflowConclusion.Skipped,
            WorkflowConclusion.TimedOut, WorkflowConclusion.None, WorkflowConclusion.ActionRequired, WorkflowConclusion.Neutral, WorkflowConclusion.Stale })
            Assert.Throws<EvidenceException>(() => Validate(run: Run with { Conclusion = conclusion }));
    }

    [Fact]
    public void MissingDuplicateAndUnsuccessfulValidationJobsAreRejected()
    {
        Assert.Throws<EvidenceException>(() => Validate(jobs: []));
        Assert.Throws<EvidenceException>(() => Validate(jobs: [Job, Job]));
        Assert.Throws<EvidenceException>(() => Validate(jobs: [Job with { Conclusion = WorkflowConclusion.Skipped }]));
        Assert.Throws<EvidenceException>(() => Validate(jobs: [Job with { Conclusion = WorkflowConclusion.Failure }]));
        Assert.Throws<EvidenceException>(() => Validate(jobs: [Job with { Status = WorkflowStatus.InProgress }]));
        Assert.Throws<EvidenceException>(() => Validate(jobs: [Job with { RunId = 13 }]));
    }

    [Fact]
    public void ExpiredMissingAndForeignArtifactsAreRejected()
    {
        foreach (WorkflowArtifactEvidence artifact in new[] { Artifact with { Expired = true }, Artifact with { Id = 0 },
            Artifact with { RunId = 13 }, Artifact with { Name = "another-12-2" } })
            Assert.Throws<EvidenceException>(() => Validate(artifact: artifact));
    }

    [Fact]
    public void FailedQualityStageRequiresExplicitValidationOnlyPolicy()
    {
        Assert.Throws<EvidenceException>(() => Validate(run: Run with { Conclusion = WorkflowConclusion.Failure }));
        Validate(policy: Policy with { CompletionScope = WorkflowCompletionScope.ValidationJob }, run: Run with { Conclusion = WorkflowConclusion.Failure });
        Assert.Throws<EvidenceException>(() => Validate(policy: Policy with { CompletionScope = WorkflowCompletionScope.ValidationJob },
            run: Run with { Conclusion = WorkflowConclusion.Cancelled }));
    }

    [Fact]
    public void InvalidPolicyEnumsAndCommitIdsAreRejected()
    {
        Assert.Throws<EvidenceException>(() => Validate(policy: Policy with { CompletionScope = (WorkflowCompletionScope)42 }));
        Assert.Throws<EvidenceException>(() => Validate(policy: Policy with { RequiredScope = (ValidationScope)42 }));
        Assert.Throws<EvidenceException>(() => Validate(source: new("not-a-commit", [])));
        Assert.Throws<EvidenceException>(() => Validate(policy: Policy with { Attempt = 0 }));
    }
}
