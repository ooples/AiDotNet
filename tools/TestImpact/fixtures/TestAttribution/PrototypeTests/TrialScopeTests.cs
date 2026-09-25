using AiDotNet.TestImpact;
using AttributionRuntime;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class TrialScopeTests
{
    public enum LedgerMutation { MissingEnd, WrongActive, WrongRestore, WrongPath, ChangedRoot, FileBefore, FileAfter,
        Unavailable, RepeatedStart, RepeatedEnd, EndWithoutStart, Unowned, DuplicatePath, AliasPrevious }
    public enum FileCase { Absent, MainFile, Tombstone, Directory, ForeignRoot, MalformedName, Relative }
    public enum MissingProof { Legacy, Open, Rejected, InvalidLedger, CustomCase, MultipleRows }
    public enum BadReport { DuplicateOwner, UnknownOwner, BadHash, UnknownState, Collision, AliasPrevious }
    private static string Hash(char value) => new(value, 64);
    private static TrialPathSample Sample => new(Hash('a'), Hash('b'), TrialPathState.Absent);

    [Fact]
    public void ACompletedScopeIsAnIndependentSnapshot()
    {
        var ledger = new TrialPathScopes();
        ledger.Begin("Owner", Sample, Hash('c'), Sample.PathHash);
        TrialScopeReport before = ledger.Snapshot();
        ledger.End("Owner", Sample, Hash('c'));
        Assert.Equal(TrialScopeState.Open, Assert.Single(before.Scopes).State);
        TrialScopeReport after = ledger.Snapshot();
        Assert.Equal(TrialScopeState.Complete, Assert.Single(after.Scopes).State);
        after.Scopes[0] = after.Scopes[0] with { State = TrialScopeState.Rejected };
        Assert.Equal(TrialScopeState.Complete, Assert.Single(ledger.Snapshot().Scopes).State);
    }

    [Theory]
    [InlineData(LedgerMutation.MissingEnd)]
    [InlineData(LedgerMutation.WrongActive)]
    [InlineData(LedgerMutation.WrongRestore)]
    [InlineData(LedgerMutation.WrongPath)]
    [InlineData(LedgerMutation.ChangedRoot)]
    [InlineData(LedgerMutation.FileBefore)]
    [InlineData(LedgerMutation.FileAfter)]
    [InlineData(LedgerMutation.Unavailable)]
    [InlineData(LedgerMutation.RepeatedStart)]
    [InlineData(LedgerMutation.RepeatedEnd)]
    [InlineData(LedgerMutation.EndWithoutStart)]
    [InlineData(LedgerMutation.Unowned)]
    [InlineData(LedgerMutation.DuplicatePath)]
    [InlineData(LedgerMutation.AliasPrevious)]
    public void IncompleteOrConflictingLifetimesCannotProduceProof(LedgerMutation mutation)
    {
        var ledger = new TrialPathScopes();
        TrialPathSample start = mutation switch
        {
            LedgerMutation.FileBefore => Sample with { State = TrialPathState.Present },
            LedgerMutation.Unavailable => Sample with { State = TrialPathState.Unavailable }, _ => Sample
        };
        if (mutation != LedgerMutation.EndWithoutStart)
            ledger.Begin("Owner", start, mutation == LedgerMutation.AliasPrevious ? Sample.PathHash : Hash('c'),
                mutation == LedgerMutation.WrongActive ? Hash('d') : Sample.PathHash);
        if (mutation == LedgerMutation.RepeatedStart) ledger.Begin("Owner", Sample, Hash('c'), Sample.PathHash);
        if (mutation == LedgerMutation.DuplicatePath) ledger.Begin("Other", Sample, Hash('c'), Sample.PathHash);
        if (mutation == LedgerMutation.Unowned) ledger.Invalidate();
        TrialPathSample end = mutation switch
        {
            LedgerMutation.WrongPath => Sample with { PathHash = Hash('d') },
            LedgerMutation.ChangedRoot => Sample with { RootHash = Hash('d') },
            LedgerMutation.FileAfter => Sample with { State = TrialPathState.Present }, _ => Sample
        };
        if (mutation != LedgerMutation.MissingEnd)
            ledger.End("Owner", end, mutation == LedgerMutation.WrongRestore ? Hash('d') : Hash('c'));
        if (mutation == LedgerMutation.RepeatedEnd) ledger.End("Owner", Sample, Hash('c'));
        TrialScopeReport report = ledger.Snapshot();
        Assert.True(report.State == TrialScopeLedgerState.Invalid || report.Scopes.All(scope => scope.State != TrialScopeState.Complete));
    }

    [Theory]
    [InlineData(FileCase.Absent)]
    [InlineData(FileCase.MainFile)]
    [InlineData(FileCase.Tombstone)]
    [InlineData(FileCase.Directory)]
    [InlineData(FileCase.ForeignRoot)]
    [InlineData(FileCase.MalformedName)]
    [InlineData(FileCase.Relative)]
    public void FileInspectionIsBoundedToAnAbsentPrivateTrialPath(FileCase kind)
    {
        string temporary = System.IO.Directory.CreateTempSubdirectory("trial-inspect-").FullName;
        try
        {
            string root = System.IO.Directory.CreateDirectory(Path.Combine(temporary, "aidotnet-trial-tests")).FullName;
            string path = Path.Combine(root, Guid.NewGuid().ToString("N") + ".json");
            switch (kind)
            {
                case FileCase.MainFile: File.WriteAllText(path, "trial"); break;
                case FileCase.Tombstone: File.WriteAllText(path + ".tombstone", "trial"); break;
                case FileCase.Directory: System.IO.Directory.CreateDirectory(path); break;
                case FileCase.ForeignRoot: path = Path.Combine(temporary, Path.GetFileName(path)); break;
                case FileCase.MalformedName: path = Path.Combine(root, "not-a-guid.json"); break;
                case FileCase.Relative: path = "relative.json"; break;
            }
            TrialPathState expected = kind switch
            {
                FileCase.Absent => TrialPathState.Absent,
                FileCase.MainFile or FileCase.Tombstone or FileCase.Directory => TrialPathState.Present,
                _ => TrialPathState.Unavailable
            };
            Assert.Equal(expected, TrialPathScopes.Inspect(path, temporary).State);
        }
        finally { System.IO.Directory.Delete(temporary, recursive: true); }
    }

    [Fact]
    public void LexicalAliasesShareOnePathIdentity()
    {
        string root = Path.GetFullPath(Path.GetTempPath());
        string path = Path.Combine(root, "aidotnet-trial-tests", "01234567890123456789012345678901.json");
        string alias = Path.Combine(root, "unused", "..", "aidotnet-trial-tests", Path.GetFileName(path));
        Assert.Equal(TrialPathScopes.PathHash(path), TrialPathScopes.PathHash(alias));
        if (OperatingSystem.IsWindows()) Assert.Equal(TrialPathScopes.PathHash(path), TrialPathScopes.PathHash(path.ToUpperInvariant()));
    }

    [Fact]
    public void RestoreRequiresTheExactValueNotJustAnEquivalentPath()
    {
        string previous = Path.Combine(Path.GetTempPath(), "previous.json");
        string changed = previous.ToUpperInvariant();
        Assert.NotEqual(previous, changed);
        var ledger = new TrialPathScopes();
        ledger.Begin("Owner", Sample, TrialPathScopes.ValueHash(previous), Sample.PathHash, TrialPathScopes.PathHash(previous));
        ledger.End("Owner", Sample, TrialPathScopes.ValueHash(changed));
        Assert.Equal(TrialScopeState.Rejected, Assert.Single(ledger.Snapshot().Scopes).State);
    }

    [Fact]
    public void ReportFieldsDoNotContainRawPaths()
    {
        var ledger = new TrialPathScopes();
        string path = Path.Combine(Path.GetTempPath(), "aidotnet-trial-tests", Guid.NewGuid().ToString("N") + ".json");
        TrialPathSample sample = TrialPathScopes.Inspect(path);
        ledger.Begin("Owner", sample, TrialPathScopes.PathHash(null), sample.PathHash);
        ledger.End("Owner", sample, TrialPathScopes.PathHash(null));
        Assert.DoesNotContain(path, System.Text.Json.JsonSerializer.Serialize(ledger.Snapshot()), StringComparison.Ordinal);
        Assert.NotEqual(TrialPathScopes.PathHash(null), TrialPathScopes.PathHash(""));
    }

    [Fact]
    public void VerifiedObservationRetainsOnlyCheckedImmutableScopeRecords()
    {
        using var fixture = new ObservedOwnerTests.Evidence([new("one", "Owner")]);
        TrialScopeReport report = Report();
        fixture.Report = fixture.Report with { TrialScopes = report };
        VerifiedObservedExecution observed = fixture.Verify();
        Assert.Equal(TrialScopeState.Complete, Assert.Single(observed.TrialScopes).State);
        report.Scopes[0] = report.Scopes[0] with { State = TrialScopeState.Rejected };
        Assert.Equal(TrialScopeState.Complete, Assert.Single(observed.TrialScopes).State);
        Assert.Empty(fixture.Verify().TrialScopes);
    }

    [Theory]
    [InlineData(MissingProof.Legacy)]
    [InlineData(MissingProof.Open)]
    [InlineData(MissingProof.Rejected)]
    [InlineData(MissingProof.InvalidLedger)]
    [InlineData(MissingProof.CustomCase)]
    [InlineData(MissingProof.MultipleRows)]
    public void PassingExecutionAloneDoesNotProveScopeIsolation(MissingProof missing)
    {
        using var fixture = new ObservedOwnerTests.Evidence(missing == MissingProof.MultipleRows
            ? [new("one", "Owner"), new("two", "Owner")] : [new("one", "Owner")]);
        TrialScopeReport report = Report();
        if (missing == MissingProof.Open) report.Scopes[0] = report.Scopes[0] with { State = TrialScopeState.Open };
        if (missing == MissingProof.Rejected) report.Scopes[0] = report.Scopes[0] with { State = TrialScopeState.Rejected };
        if (missing == MissingProof.InvalidLedger) report = report with { State = TrialScopeLedgerState.Invalid };
        if (missing == MissingProof.CustomCase) fixture.Report.Cases[0] = fixture.Report.Cases[0] with
        { Case = fixture.Report.Cases[0].Case with { Kind = DiscoveredCaseKind.DeferredOrCustom } };
        fixture.Report = fixture.Report with { TrialScopes = missing == MissingProof.Legacy ? null : report };
        Assert.Empty(fixture.Verify().TrialScopes);
    }

    [Theory]
    [InlineData(BadReport.DuplicateOwner)]
    [InlineData(BadReport.UnknownOwner)]
    [InlineData(BadReport.BadHash)]
    [InlineData(BadReport.UnknownState)]
    [InlineData(BadReport.Collision)]
    [InlineData(BadReport.AliasPrevious)]
    public void MalformedScopeEvidenceCannotBeImported(BadReport mutation)
    {
        using var fixture = new ObservedOwnerTests.Evidence([new("one", "Owner"), new("two", "Other")]);
        TrialScopeReport report = Report();
        switch (mutation)
        {
            case BadReport.DuplicateOwner: report = report with { Scopes = [report.Scopes[0], report.Scopes[0]] }; break;
            case BadReport.UnknownOwner: report.Scopes[0] = report.Scopes[0] with { Owner = "Unexecuted" }; break;
            case BadReport.BadHash: report.Scopes[0] = report.Scopes[0] with { PathHash = "forged" }; break;
            case BadReport.UnknownState: report.Scopes[0] = report.Scopes[0] with { State = (TrialScopeState)99 }; break;
            case BadReport.Collision: report = report with { Scopes = [report.Scopes[0], report.Scopes[0] with { Owner = "Other" }] }; break;
            case BadReport.AliasPrevious: report.Scopes[0] = report.Scopes[0] with { PreviousPathHash = report.Scopes[0].PathHash }; break;
        }
        fixture.Report = fixture.Report with { TrialScopes = report };
        Assert.Throws<EvidenceException>(() => fixture.Verify());
    }

    private static TrialScopeReport Report() => new(TrialScopeLedgerState.Recorded,
        [new("Owner", Hash('a'), Hash('b'), Hash('c'), Hash('c'), TrialScopeState.Complete)]);
}
