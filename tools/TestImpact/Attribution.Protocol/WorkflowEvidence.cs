namespace AiDotNet.TestImpact;

public enum WorkflowEvent { PullRequest, Push, MergeGroup, WorkflowDispatch }
public enum WorkflowStatus { Queued, InProgress, Completed }
public enum WorkflowConclusion { None, Success, Failure, Cancelled, Skipped, TimedOut, ActionRequired, Neutral, Stale }
public enum WorkflowCompletionScope { WholeRun, ValidationJob }
public sealed record WorkflowImportPolicy(string Repository, long RunId, int Attempt, string HeadSha,
    string WorkflowPath, string JobName, string ArtifactName, WorkflowCompletionScope CompletionScope, ValidationScope RequiredScope);
public sealed record WorkflowPullIdentity(string BaseSha, string HeadSha, string BaseBranch);
public sealed record WorkflowRunEvidence(string Repository, string HeadRepository, long RunId, int Attempt, string HeadSha,
    string WorkflowPath, WorkflowEvent Event, WorkflowStatus Status, WorkflowConclusion Conclusion, WorkflowPullIdentity[] Pulls);
public sealed record WorkflowJobEvidence(long RunId, int Attempt, string Name, WorkflowStatus Status, WorkflowConclusion Conclusion);
public sealed record WorkflowArtifactEvidence(long Id, long RunId, string Name, bool Expired);
public sealed record WorkflowBaseBinding(string RecordedBaseSha, string TestedBaseSha, string BaseBranch, string CurrentBaseTip,
    bool RecordedBaseIsAncestor, bool TestedBaseIsAncestor);
public sealed record WorkflowSourceEvidence(string Sha, string[] Parents, WorkflowBaseBinding? BaseBinding = null);

// These pure checks do not authenticate caller-supplied JSON. The CLI transport
// obtains metadata from GitHub independently, pins the artifact ID, and repeats
// the run/attempt checks after download before passing evidence to this policy.
public static class WorkflowEvidence
{
    public static void Validate(WorkflowImportPolicy policy, WorkflowRunEvidence run, WorkflowJobEvidence[] jobs,
        WorkflowArtifactEvidence artifact, WorkflowSourceEvidence source)
    {
        static bool Sha(string value) => value is not null && value.Length == 40 &&
            value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');
        if (policy.RunId < 1 || policy.Attempt < 1 || !Sha(policy.HeadSha) || !Sha(source.Sha) ||
            !Enum.IsDefined(policy.CompletionScope) || !Enum.IsDefined(policy.RequiredScope) ||
            !Enum.IsDefined(run.Event) || !Enum.IsDefined(run.Conclusion) || source.Parents is null || run.Pulls is null ||
            string.IsNullOrWhiteSpace(policy.Repository) || string.IsNullOrWhiteSpace(policy.JobName) ||
            string.IsNullOrWhiteSpace(policy.ArtifactName) ||
            !policy.ArtifactName.EndsWith($"-{policy.RunId}-{policy.Attempt}", StringComparison.Ordinal) ||
            !policy.WorkflowPath.StartsWith(".github/workflows/", StringComparison.Ordinal))
            throw new EvidenceException(EvidenceFailure.Provenance, "Incomplete workflow import policy.");
        if (run.Repository != policy.Repository || run.HeadRepository != policy.Repository || run.RunId != policy.RunId ||
            run.Attempt != policy.Attempt || run.HeadSha != policy.HeadSha || run.WorkflowPath != policy.WorkflowPath ||
            run.Status != WorkflowStatus.Completed || run.Conclusion is not (WorkflowConclusion.Success or WorkflowConclusion.Failure) ||
            (policy.CompletionScope == WorkflowCompletionScope.WholeRun && run.Conclusion != WorkflowConclusion.Success))
            throw new EvidenceException(EvidenceFailure.Provenance, "Workflow origin, attempt or completion does not match policy.");
        WorkflowJobEvidence[] required = jobs.Where(job => job.Name == policy.JobName).ToArray();
        if (required.Length != 1 || required[0].RunId != policy.RunId || required[0].Attempt != policy.Attempt ||
            required[0].Status != WorkflowStatus.Completed || required[0].Conclusion != WorkflowConclusion.Success)
            throw new EvidenceException(EvidenceFailure.Provenance, "Required validation job did not complete successfully in this attempt.");
        if (artifact.Id < 1 || artifact.RunId != policy.RunId || artifact.Name != policy.ArtifactName || artifact.Expired)
            throw new EvidenceException(EvidenceFailure.Provenance, "Artifact does not belong to this completed workflow.");
        if (source.Sha == run.HeadSha) return;
        bool MatchesBase(WorkflowPullIdentity pull) => source.Parents[0] == pull.BaseSha ||
            (source.BaseBinding is WorkflowBaseBinding binding && binding.RecordedBaseSha == pull.BaseSha &&
             binding.TestedBaseSha == source.Parents[0] && binding.BaseBranch == pull.BaseBranch && Sha(binding.CurrentBaseTip) &&
             binding.RecordedBaseIsAncestor && binding.TestedBaseIsAncestor);
        if (run.Event != WorkflowEvent.PullRequest || source.Parents.Length != 2 ||
            !run.Pulls.Any(pull => pull.HeadSha == run.HeadSha && Sha(pull.BaseSha) &&
                MatchesBase(pull) && source.Parents[1] == pull.HeadSha))
            throw new EvidenceException(EvidenceFailure.Context, "Tested source is neither the workflow head nor its exact PR merge.");
    }
}
