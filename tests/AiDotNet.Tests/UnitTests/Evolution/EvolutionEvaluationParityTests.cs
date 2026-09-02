using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionEvaluationParityTests
{
    [Fact]
    public async Task DefaultOptionsRunTheDirectEvaluatorWithNoArtifactsOrEarlyStopping()
    {
        var task = new StagedEvolutionTask();
        var observer = new EvaluationRecordingObserver();
        EvolutionEngineOptions options = Options(maxAttempts: 10, batchSize: 4);
        options.MaxProposals = 4;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(task, new IncrementVariation(),
            _ => TestArchive(), options, observer: observer).RunAsync(Seeds(4));

        Assert.Equal(4, task.DirectCalls);
        Assert.Equal(0, task.StageCalls(0));
        Assert.Equal(4, result.Counters.EvaluationAttempts);
        Assert.Equal(0, result.Counters.AbandonedEvaluations);
        Assert.Empty(result.PendingArtifacts);
        Assert.All(observer.Evaluations, evaluation =>
        {
            Assert.Empty(evaluation.Artifacts);
            Assert.Empty(evaluation.Cost.StageCostUnits);
            Assert.Null(evaluation.Cost.RejectedStage);
        });
        Assert.Equal(EvolutionStopReason.ProposalBudgetReached, result.StopReason);
    }

    [Fact]
    public async Task CascadeRejectionDoesNotConsumeTheEvaluationBudget()
    {
        var task = new StagedEvolutionTask();
        EvolutionEngineOptions options = CascadeOptions(maxAttempts: 4, batchSize: 4, threshold: 3);
        options.MaxProposals = 4;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(task, new IncrementVariation(),
            _ => TestArchive(), options).RunAsync(Seeds(4));

        Assert.Equal(0, task.DirectCalls);
        Assert.Equal(4, task.StageCalls(0));
        Assert.Equal(2, task.StageCalls(1));
        Assert.Equal(4, result.Counters.Proposals);
        Assert.Equal(2, result.Counters.EvaluationAttempts);
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Skipped]);
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Completed]);
    }

    [Fact]
    public async Task CascadeReportsTheRejectingStageAndItsPerStageCost()
    {
        var task = new StagedEvolutionTask();
        var observer = new EvaluationRecordingObserver();
        EvolutionEngineOptions options = CascadeOptions(maxAttempts: 4, batchSize: 4, threshold: 3);
        options.MaxProposals = 4;
        await new EvolutionEngine<TestGenome>(task, new IncrementVariation(), _ => TestArchive(), options,
            observer: observer).RunAsync(Seeds(4));

        EvolutionEvaluation rejected = Assert.Single(observer.Evaluations,
            item => item.Status == EvolutionEvaluationStatus.Skipped && item.GenomeId == "1");
        Assert.Equal(0, rejected.Cost.RejectedStage);
        Assert.Equal(new[] { 1.0 }, rejected.Cost.StageCostUnits);
        EvolutionDiagnostic diagnostic = Assert.Single(rejected.Diagnostics,
            item => item.Code == "cascade_stage_rejected");
        Assert.Equal("0", diagnostic.Data["stage"]);
        Assert.Equal("3", diagnostic.Data["threshold"]);
        Assert.Equal("1", diagnostic.Data["quality"]);

        EvolutionEvaluation completed = Assert.Single(observer.Evaluations,
            item => item.Status == EvolutionEvaluationStatus.Completed && item.GenomeId == "4");
        Assert.Null(completed.Cost.RejectedStage);
        Assert.Equal(new[] { 1.0, 2.0 }, completed.Cost.StageCostUnits);
        Assert.Equal(3.0, completed.Cost.CostUnits);
        Assert.Equal(4.0, completed.Metrics["stage0"]);
        Assert.Equal(4.0, completed.Metrics["stage1"]);
    }

    [Fact]
    public async Task CascadeChargesRejectionsWhenTheOptionIsSet()
    {
        var task = new StagedEvolutionTask();
        EvolutionEngineOptions options = CascadeOptions(maxAttempts: 4, batchSize: 4, threshold: 3);
        options.MaxProposals = 4;
        options.Cascade.ChargeRejectedStagesToBudget = true;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(task, new IncrementVariation(),
            _ => TestArchive(), options).RunAsync(Seeds(4));

        Assert.Equal(4, result.Counters.EvaluationAttempts);
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Skipped]);
    }

    [Fact]
    public async Task CascadeThresholdRespectsMinimization()
    {
        var task = new StagedEvolutionTask(direction: EvolutionOptimizationDirection.Minimize);
        EvolutionEngineOptions options = CascadeOptions(maxAttempts: 4, batchSize: 4, threshold: 2);
        options.MaxProposals = 4;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(task, new IncrementVariation(),
            _ => MinimizingArchive(), options).RunAsync(Seeds(4));

        Assert.Equal(4, task.StageCalls(0));
        Assert.Equal(2, task.StageCalls(1));
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Skipped]);
        Assert.Equal(2, result.Counters.EvaluationAttempts);
    }

    [Fact]
    public void CascadeConfigurationIsValidatedAtConstructionTime()
    {
        EvolutionEngineOptions withoutCascadeTask = CascadeOptions(2, 2, 1);
        Assert.Throws<ArgumentException>(() => new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(),
            new IncrementVariation(), _ => TestArchive(), withoutCascadeTask));

        EvolutionEngineOptions tooFewThresholds = CascadeOptions(2, 2, 1);
        tooFewThresholds.Cascade.Thresholds = Array.Empty<double>();
        Assert.Throws<ArgumentException>(() => new EvolutionEngine<TestGenome>(new StagedEvolutionTask(3),
            new IncrementVariation(), _ => TestArchive(), tooFewThresholds));

        EvolutionEngineOptions nonMonotone = CascadeOptions(2, 2, 1);
        nonMonotone.Cascade.Thresholds = new[] { 5.0, 1.0 };
        Assert.Throws<ArgumentException>(() => new EvolutionEngine<TestGenome>(new StagedEvolutionTask(3),
            new IncrementVariation(), _ => TestArchive(), nonMonotone));

        EvolutionEngineOptions wrongTimeoutCount = CascadeOptions(2, 2, 1);
        wrongTimeoutCount.Cascade.StageTimeouts = new[] { TimeSpan.FromSeconds(1) };
        Assert.Throws<ArgumentException>(() => new EvolutionEngine<TestGenome>(new StagedEvolutionTask(),
            new IncrementVariation(), _ => TestArchive(), wrongTimeoutCount));

        EvolutionEngineOptions negativeTimeout = CascadeOptions(2, 2, 1);
        negativeTimeout.Cascade.StageTimeouts = new[] { TimeSpan.Zero, TimeSpan.FromSeconds(1) };
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionEngine<TestGenome>(new StagedEvolutionTask(),
            new IncrementVariation(), _ => TestArchive(), negativeTimeout));
    }

    [Fact]
    public async Task TargetQualityStopsTheRunAndWritesAFinalCheckpoint()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions options = Options(maxAttempts: 20, batchSize: 1);
        options.TargetQuality = 3;
        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), options, store)
            .RunAsync(Seeds(1));

        Assert.Equal(EvolutionStopReason.TargetReached, result.StopReason);
        Assert.Equal(3.0, result.Best?.Evaluation.Quality);
        EvolutionCheckpoint checkpoint = Assert.IsType<EvolutionCheckpoint>(await store.LoadLatestAsync(options.RunId));
        Assert.True(checkpoint.Sequence > 0);
    }

    [Fact]
    public async Task TargetQualityUsesTheArchiveDirectionWhenMinimizing()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 20, batchSize: 1);
        options.TargetQuality = 3;
        var engine = new EvolutionEngine<TestGenome>(
            new StagedEvolutionTask(direction: EvolutionOptimizationDirection.Minimize), new IncrementVariation(),
            _ => MinimizingArchive(), options);

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(Seeds(1));

        Assert.Equal(EvolutionStopReason.TargetReached, result.StopReason);
        Assert.Equal(1, result.Counters.EvaluationAttempts);
    }

    [Fact]
    public async Task EarlyStoppingHaltsAPlateauAtADeterministicEvaluationCount()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions options = Options(maxAttempts: 40, batchSize: 2);
        options.EarlyStopping.PatienceEvaluations = 4;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(new PlateauEvolutionTask(),
            new SequentialVariation(), _ => TestArchive(), options, checkpointStore: store,
            genomeCodec: new TestGenomeCodec()).RunAsync(Seeds(1));

        Assert.Equal(EvolutionStopReason.EarlyStopped, result.StopReason);
        Assert.Equal(5, result.Counters.Proposals);
        Assert.NotNull(await store.LoadLatestAsync(options.RunId));
    }

    [Fact]
    public async Task EarlyStoppingIsSuppressedWhileTheMetricKeepsImproving()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 6, batchSize: 2);
        options.EarlyStopping.PatienceEvaluations = 2;
        EvolutionRunResult<TestGenome> result = await Engine(new SyntheticEvolutionTask(), options).RunAsync(Seeds(1));

        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, result.StopReason);
        Assert.Equal(6, result.Counters.EvaluationAttempts);
    }

    [Fact]
    public async Task CoverageEarlyStoppingSeesProgressThatBestQualityWouldMiss()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 8, batchSize: 2);
        options.EarlyStopping.PatienceEvaluations = 2;
        options.EarlyStopping.Metric = EvolutionEarlyStoppingMetric.Coverage;
        options.EarlyStopping.MinimumImprovement = 1e-9;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(new PlateauEvolutionTask(),
            new SequentialVariation(), _ => TestArchive(), options).RunAsync(Seeds(1));

        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, result.StopReason);
        Assert.Equal(8, result.Counters.EvaluationAttempts);
    }

    [Fact]
    public async Task ResumeAfterEarlyStoppingContinuesFromTheCheckpointedPlateau()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions options = ResumableEarlyStoppingOptions();
        EvolutionRunResult<TestGenome> first = await new EvolutionEngine<TestGenome>(new PlateauEvolutionTask(),
            new SequentialVariation(), _ => TestArchive(), options, checkpointStore: store,
            genomeCodec: new TestGenomeCodec()).RunAsync(Seeds(1));
        Assert.Equal(EvolutionStopReason.EarlyStopped, first.StopReason);

        EvolutionEngineOptions resumedOptions = ResumableEarlyStoppingOptions();
        resumedOptions.Resume = true;
        EvolutionRunResult<TestGenome> resumed = await new EvolutionEngine<TestGenome>(new PlateauEvolutionTask(),
            new SequentialVariation(), _ => TestArchive(), resumedOptions, checkpointStore: store,
            genomeCodec: new TestGenomeCodec()).RunAsync(Seeds(1));

        Assert.Equal(EvolutionStopReason.EarlyStopped, resumed.StopReason);
        Assert.Equal(first.Counters.Proposals + resumedOptions.ProposalBatchSize, resumed.Counters.Proposals);
    }

    [Fact]
    public async Task ArtifactsAreRetainedSanitizedAndDeliveredToTheNextProposalExactlyOnce()
    {
        var variation = new ArtifactRecordingVariation();
        var observer = new EvaluationRecordingObserver();
        EvolutionEngineOptions options = Options(maxAttempts: 3, batchSize: 1);
        options.Artifacts.Enabled = true;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(
            new ArtifactEvolutionTask("boom token=hunter2"), variation, _ => TestArchive(), options,
            observer: observer).RunAsync(Seeds(1));

        EvolutionEvaluation first = observer.Evaluations[0];
        Assert.Equal(2, first.Artifacts.Count);
        Assert.Equal("stderr", first.Artifacts[0].Key);
        Assert.Equal("boom token=<REDACTED>", first.Artifacts[0].Text);
        Assert.True(first.Artifacts[0].IsRedacted);
        Assert.Contains("stderr=boom token=<REDACTED>", variation.Received);
        Assert.Equal(2, variation.Received.Count(item => item.StartsWith("stderr=", StringComparison.Ordinal)));
        Assert.Equal(3, result.Counters.EvaluationAttempts);
        Assert.Single(result.PendingArtifacts);
    }

    [Fact]
    public async Task AFailedChildLeavesItsArtifactsForTheNextProposalOfTheSameLineage()
    {
        var variation = new ArtifactRecordingVariation();
        EvolutionEngineOptions options = Options(maxAttempts: 3, batchSize: 1);
        options.Artifacts.Enabled = true;
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(
            new FailingChildArtifactTask(), variation, _ => TestArchive(), options).RunAsync(Seeds(1));

        Assert.Equal(3, result.Counters.EvaluationAttempts);
        Assert.Equal(2, result.Counters.StatusCounts[EvolutionEvaluationStatus.Failed]);
        Assert.Contains("stderr=child 2 failed", variation.Received);
        Assert.Equal(new[] { "1" }, result.PendingArtifacts.Keys);
    }

    [Fact]
    public async Task ArtifactsAreDisabledByDefaultAndStoreNoBytes()
    {
        var variation = new ArtifactRecordingVariation();
        var observer = new EvaluationRecordingObserver();
        EvolutionEngineOptions options = Options(maxAttempts: 3, batchSize: 1);
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(
            new ArtifactEvolutionTask("boom"), variation, _ => TestArchive(), options,
            observer: observer).RunAsync(Seeds(1));

        Assert.All(observer.Evaluations, evaluation => Assert.Empty(evaluation.Artifacts));
        Assert.Empty(variation.Received);
        Assert.Empty(result.PendingArtifacts);
    }

    [Fact]
    public async Task ArtifactsAreTruncatedAndCappedByTheConfiguredBudgets()
    {
        var observer = new EvaluationRecordingObserver();
        EvolutionEngineOptions options = Options(maxAttempts: 1, batchSize: 1);
        options.Artifacts.Enabled = true;
        options.Artifacts.MaxArtifactBytes = 4;
        options.Artifacts.MaxArtifactsPerEvaluation = 1;
        await new EvolutionEngine<TestGenome>(
            new ArtifactEvolutionTask(string.Concat(Enumerable.Repeat("zz ", 50))),
            new IncrementVariation(), _ => TestArchive(), options, observer: observer).RunAsync(Seeds(1));

        EvolutionArtifact artifact = Assert.Single(observer.Evaluations[0].Artifacts);
        Assert.Equal("zz z", artifact.Text);
        Assert.Equal(4, artifact.SizeBytes);
        Assert.True(artifact.IsTruncated);
    }

    [Fact]
    public async Task ALongUnbrokenTokenIsRedactedBeforeItReachesAStoredArtifact()
    {
        var observer = new EvaluationRecordingObserver();
        EvolutionEngineOptions options = Options(maxAttempts: 1, batchSize: 1);
        options.Artifacts.Enabled = true;
        await new EvolutionEngine<TestGenome>(new ArtifactEvolutionTask(new string('z', 100)),
            new IncrementVariation(), _ => TestArchive(), options, observer: observer).RunAsync(Seeds(1));

        EvolutionArtifact artifact = observer.Evaluations[0].Artifacts[0];
        Assert.Equal("<REDACTED_TOKEN>", artifact.Text);
        Assert.True(artifact.IsRedacted);
        Assert.False(artifact.IsTruncated);
    }

    [Fact]
    public async Task ArtifactsSurviveACheckpointRoundTrip()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions options = ResumableArtifactOptions();
        EvolutionRunResult<TestGenome> uninterrupted = await new EvolutionEngine<TestGenome>(
            new ArtifactEvolutionTask("compiler said no"), new IncrementVariation(), _ => TestArchive(),
            options, checkpointStore: store, genomeCodec: new TestGenomeCodec()).RunAsync(Seeds(1));

        EvolutionCheckpoint checkpoint = Assert.IsType<EvolutionCheckpoint>(
            await store.LoadLatestAsync(options.RunId));
        Assert.Contains("compiler said no", checkpoint.Payload, StringComparison.Ordinal);
        Assert.NotEmpty(uninterrupted.PendingArtifacts);
        Assert.All(uninterrupted.Islands.SelectMany(island => island.Entries),
            entry => Assert.Equal(2, entry.Evaluation.Artifacts.Count));
    }

    [Fact]
    public async Task ResumeAfterCancellationPreservesArtifactsAndMatchesTheUninterruptedStateHash()
    {
        TestGenome[] seeds = Seeds(8);
        var uninterruptedStore = new InMemoryEvolutionCheckpointStore();
        EvolutionRunResult<TestGenome> uninterrupted = await new EvolutionEngine<TestGenome>(
            new ArtifactEvolutionTask("note"), new IncrementVariation(), _ => TestArchive(),
            ResumableArtifactOptions(), checkpointStore: uninterruptedStore,
            genomeCodec: new TestGenomeCodec()).RunAsync(seeds);

        var sharedStore = new InMemoryEvolutionCheckpointStore();
        using var cancellation = new CancellationTokenSource();
        EvolutionEngine<TestGenome> interrupted = new(new ArtifactEvolutionTask("note", cancellation),
            new IncrementVariation(), _ => TestArchive(), ResumableArtifactOptions(),
            checkpointStore: sharedStore, genomeCodec: new TestGenomeCodec());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => interrupted.RunAsync(seeds, cancellation.Token));

        EvolutionEngineOptions resumedOptions = ResumableArtifactOptions();
        resumedOptions.Resume = true;
        EvolutionRunResult<TestGenome> resumed = await new EvolutionEngine<TestGenome>(
            new ArtifactEvolutionTask("note"), new IncrementVariation(), _ => TestArchive(),
            resumedOptions, checkpointStore: sharedStore, genomeCodec: new TestGenomeCodec()).RunAsync(seeds);

        Assert.Equal(uninterrupted.StateHash, resumed.StateHash);
        Assert.Equal(uninterrupted.PendingArtifacts.Count, resumed.PendingArtifacts.Count);
        Assert.NotEmpty(resumed.PendingArtifacts);
    }

    [Fact]
    public async Task RetryOnExcludesTimeoutsWhenConfigured()
    {
        var retried = new TimeoutOnceEvolutionTask();
        EvolutionEngineOptions retryOptions = Options(maxAttempts: 2, batchSize: 1);
        retryOptions.MaxProposals = 1;
        retryOptions.MaxRetries = 1;
        EvolutionRunResult<TestGenome> withRetry = await new EvolutionEngine<TestGenome>(retried,
            new IncrementVariation(), _ => TestArchive(), retryOptions).RunAsync(Seeds(1));
        Assert.Equal(2, retried.Calls);
        Assert.Equal(1, withRetry.Counters.StatusCounts[EvolutionEvaluationStatus.Completed]);

        var notRetried = new TimeoutOnceEvolutionTask();
        EvolutionEngineOptions noTimeoutRetry = Options(maxAttempts: 2, batchSize: 1);
        noTimeoutRetry.MaxProposals = 1;
        noTimeoutRetry.MaxRetries = 1;
        noTimeoutRetry.RetryOn = EvolutionRetryStatuses.Failed | EvolutionRetryStatuses.Canceled;
        EvolutionRunResult<TestGenome> withoutRetry = await new EvolutionEngine<TestGenome>(notRetried,
            new IncrementVariation(), _ => TestArchive(), noTimeoutRetry).RunAsync(Seeds(1));

        Assert.Equal(1, notRetried.Calls);
        Assert.Equal(1, withoutRetry.Counters.StatusCounts[EvolutionEvaluationStatus.TimedOut]);
    }

    [Fact]
    public async Task NonCooperativeEvaluatorIsAbandonedInsteadOfBlockingTheRun()
    {
        var task = new NonCooperativeEvolutionTask(blockMilliseconds: 60_000);
        EvolutionEngineOptions options = Options(maxAttempts: 1, batchSize: 1);
        options.EvaluationTimeout = TimeSpan.FromMilliseconds(20);
        options.EvaluationGracePeriod = TimeSpan.FromMilliseconds(20);
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(task, new IncrementVariation(),
            _ => TestArchive(), options).RunAsync(Seeds(1));

        Assert.False(task.HasFinished);
        Assert.Equal(1, task.Calls);
        Assert.Equal(1, result.Counters.AbandonedEvaluations);
        Assert.Equal(1, result.Counters.StatusCounts[EvolutionEvaluationStatus.TimedOut]);
    }

    [Fact]
    public async Task CooperativeTimeoutCarriesStructuredContextAndIsNeverAbandoned()
    {
        var task = new CooperativeBlockingEvolutionTask();
        var observer = new EvaluationRecordingObserver();
        EvolutionEngineOptions options = Options(maxAttempts: 1, batchSize: 1);
        options.EvaluationTimeout = TimeSpan.FromMilliseconds(20);
        EvolutionRunResult<TestGenome> result = await new EvolutionEngine<TestGenome>(task, new IncrementVariation(),
            _ => TestArchive(), options, observer: observer).RunAsync(Seeds(1));

        Assert.Equal(0, result.Counters.AbandonedEvaluations);
        EvolutionDiagnostic diagnostic = Assert.Single(observer.Evaluations[0].Diagnostics,
            item => item.Code == "evaluation_timeout");
        Assert.Equal("false", diagnostic.Data["abandoned"]);
        Assert.Equal("1", diagnostic.Data["attempt"]);
        Assert.Equal(TimeSpan.FromMilliseconds(20).Ticks.ToString(System.Globalization.CultureInfo.InvariantCulture),
            diagnostic.Data["timeout_ticks"]);
    }

    [Fact]
    public void GracePeriodWithoutATimeoutIsRejected()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 1, batchSize: 1);
        options.EvaluationGracePeriod = TimeSpan.FromSeconds(1);
        Assert.Throws<ArgumentException>(() => new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(),
            new IncrementVariation(), _ => TestArchive(), options));
    }

    [Fact]
    public void SanitizerRedactsCredentialsWithoutDestroyingContentDigests()
    {
        const string digest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
        string sanitized = EvolutionArtifactSanitizer.Sanitize(
            "\u001b[31mfail\u001b[0m sk-" + new string('a', 48) + " AKIAIOSFODNN7EXAMPLE " +
            "Authorization: Bearer abcdefghijklmnop password=hunter2 sha=" + digest);

        Assert.DoesNotContain("\u001b", sanitized, StringComparison.Ordinal);
        Assert.Contains("<REDACTED_API_KEY>", sanitized, StringComparison.Ordinal);
        Assert.Contains("<REDACTED_ACCESS_KEY>", sanitized, StringComparison.Ordinal);
        Assert.Contains("Bearer <REDACTED>", sanitized, StringComparison.Ordinal);
        Assert.Contains("password=<REDACTED>", sanitized, StringComparison.Ordinal);
        Assert.Contains(digest, sanitized, StringComparison.Ordinal);
        Assert.DoesNotContain("hunter2", sanitized, StringComparison.Ordinal);
        Assert.True(EvolutionArtifactSanitizer.WouldRedact(sanitized + " token=abc"));
        Assert.False(EvolutionArtifactSanitizer.WouldRedact("all clear, digest " + digest));
    }

    [Fact]
    public void SanitizerIsIdempotentSoRepeatedBoundingCannotDrift()
    {
        string once = EvolutionArtifactSanitizer.Sanitize("token=abc123 sk-" + new string('b', 48));
        Assert.Equal(once, EvolutionArtifactSanitizer.Sanitize(once));
    }

    [Fact]
    public void ArtifactAndDiagnosticBoundsAreEnforcedAtConstruction()
    {
        Assert.Throws<ArgumentException>(() => new EvolutionArtifact(" ", "text"));
        Assert.Throws<ArgumentException>(() => new EvolutionArtifact(new string('k', 129), "text"));
        Assert.Throws<ArgumentException>(() => new EvolutionDiagnostic("code", "message", false,
            new Dictionary<string, string> { [new string('k', 65)] = "value" }));
        Assert.Throws<ArgumentException>(() => new EvolutionDiagnostic("code", "message", false,
            Enumerable.Range(0, 17).ToDictionary(index => index.ToString(
                System.Globalization.CultureInfo.InvariantCulture), _ => "v")));
        Assert.Throws<ArgumentException>(() => new EvolutionTaskResult(EvolutionEvaluationStatus.Failed,
            artifacts: Enumerable.Range(0, 65).Select(index => new EvolutionArtifact(
                "k" + index.ToString(System.Globalization.CultureInfo.InvariantCulture), "v"))));
        Assert.Equal(4, new EvolutionArtifact("k", "\U0001F600").SizeBytes);
    }

    [Fact]
    public async Task RequestStopReturnsPartialArchivesInsteadOfThrowing()
    {
        var store = new InMemoryEvolutionCheckpointStore();
        EvolutionEngineOptions options = Options(maxAttempts: 20, batchSize: 2);
        EvolutionEngine<TestGenome>? engine = null;
        var observer = new StopRequestingObserver(() => engine?.RequestStop(), afterEvaluations: 2);
        engine = new EvolutionEngine<TestGenome>(new SyntheticEvolutionTask(), new IncrementVariation(),
            _ => TestArchive(), options, observer: observer, checkpointStore: store,
            genomeCodec: new TestGenomeCodec());

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(Seeds(4));

        Assert.Equal(EvolutionStopReason.Canceled, result.StopReason);
        Assert.Equal(2, result.Counters.EvaluationAttempts);
        Assert.NotNull(result.Best);
        Assert.NotNull(await store.LoadLatestAsync(options.RunId));
    }

    [Fact]
    public async Task RequestStopBeforeTheRunStartsReturnsAnEmptyResult()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 20, batchSize: 2);
        EvolutionEngine<TestGenome> engine = Engine(new SyntheticEvolutionTask(), options);
        engine.RequestStop();

        EvolutionRunResult<TestGenome> result = await engine.RunAsync(Seeds(4));

        Assert.Equal(EvolutionStopReason.Canceled, result.StopReason);
        Assert.Equal(0, result.Counters.Proposals);
        Assert.Null(result.Best);
    }

    [Fact]
    public void ArchiveEntriesAreServedFromACacheUntilTheArchiveChanges()
    {
        MapElitesArchive<TestGenome> archive = TestArchive();
        IReadOnlyList<EvolutionArchiveEntry<TestGenome>> empty = archive.Entries;
        Assert.Same(empty, archive.Entries);

        MapElitesArchiveTests.Add(archive, 1, "1", quality: 1, descriptor: 5);
        IReadOnlyList<EvolutionArchiveEntry<TestGenome>> afterInsert = archive.Entries;
        Assert.NotSame(empty, afterInsert);
        Assert.Same(afterInsert, archive.Entries);
        Assert.Single(afterInsert);

        MapElitesArchiveTests.Add(archive, 2, "2", quality: 0, descriptor: 5);
        Assert.Same(afterInsert, archive.Entries);
    }

    [Fact]
    public void ArtifactOptionsRejectABoundLargerThanTheResultCap()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 1, batchSize: 1);
        options.Artifacts.MaxArtifactsPerEvaluation = EvolutionTaskResult.MaximumArtifacts + 1;
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionEngine<TestGenome>(
            new SyntheticEvolutionTask(), new IncrementVariation(), _ => TestArchive(), options));
    }

    private static EvolutionEngineOptions CascadeOptions(int maxAttempts, int batchSize, double threshold)
    {
        EvolutionEngineOptions options = Options(maxAttempts, batchSize);
        options.Cascade.Enabled = true;
        options.Cascade.Thresholds = new[] { threshold };
        return options;
    }

    private static EvolutionEngineOptions ResumableEarlyStoppingOptions()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 40, batchSize: 2);
        options.CheckpointInterval = 1;
        options.EarlyStopping.PatienceEvaluations = 4;
        return options;
    }

    private static EvolutionEngineOptions ResumableArtifactOptions()
    {
        EvolutionEngineOptions options = Options(maxAttempts: 8, batchSize: 2);
        options.CheckpointInterval = 2;
        options.Artifacts.Enabled = true;
        return options;
    }

    private static EvolutionEngineOptions Options(int maxAttempts, int batchSize) => new()
    {
        RunId = "evaluation-parity",
        Seed = 91,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = batchSize,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1,
        CheckpointInterval = 0
    };

    private static TestGenome[] Seeds(int count) =>
        Enumerable.Range(1, count).Select(value => new TestGenome(value)).ToArray();

    private static EvolutionEngine<TestGenome> Engine(IEvolutionTask<TestGenome> task, EvolutionEngineOptions options,
        IEvolutionCheckpointStore? checkpointStore = null) => new(
        task, new IncrementVariation(), _ => TestArchive(), options,
        checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());

    private static MapElitesArchive<TestGenome> TestArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    });

    private static MapElitesArchive<TestGenome> MinimizingArchive() => new(new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    }, EvolutionOptimizationDirection.Minimize);
}
