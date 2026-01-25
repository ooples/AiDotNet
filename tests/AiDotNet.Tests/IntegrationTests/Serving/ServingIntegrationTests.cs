using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using AiDotNet.Serving.ContinuousBatching;

namespace AiDotNet.Tests.IntegrationTests.Serving;

/// <summary>
/// Comprehensive integration tests for the Serving module.
/// Tests cover batch scheduling, sequence state management, generation requests/results,
/// and continuous batching infrastructure.
/// </summary>
public class ServingIntegrationTests
{
    #region ContinuousBatcherConfig Tests

    [Fact]
    public void ContinuousBatcherConfig_DefaultValues_AreCorrect()
    {
        var config = new ContinuousBatcherConfig();

        Assert.NotNull(config.SchedulerConfig);
        Assert.Equal(2, config.EosTokenId);
        Assert.Equal(10, config.IdleSleepMs);
        Assert.True(config.AutoStart);
        Assert.Equal(4096, config.MaxContextLength);
        Assert.False(config.EnableSpeculativeDecoding);
        Assert.Equal(AiDotNet.Configuration.SpeculationPolicy.Auto, config.SpeculationPolicy);
        Assert.Equal(4, config.SpeculationDepth);
        Assert.False(config.UseTreeSpeculation);
    }

    [Fact]
    public void ContinuousBatcherConfig_CanBeConfigured()
    {
        var config = new ContinuousBatcherConfig
        {
            EosTokenId = 50256,
            IdleSleepMs = 50,
            AutoStart = false,
            MaxContextLength = 8192,
            EnableSpeculativeDecoding = true,
            SpeculationPolicy = AiDotNet.Configuration.SpeculationPolicy.ForceOn,
            SpeculationDepth = 8,
            UseTreeSpeculation = true
        };

        Assert.Equal(50256, config.EosTokenId);
        Assert.Equal(50, config.IdleSleepMs);
        Assert.False(config.AutoStart);
        Assert.Equal(8192, config.MaxContextLength);
        Assert.True(config.EnableSpeculativeDecoding);
        Assert.Equal(AiDotNet.Configuration.SpeculationPolicy.ForceOn, config.SpeculationPolicy);
        Assert.Equal(8, config.SpeculationDepth);
        Assert.True(config.UseTreeSpeculation);
    }

    [Fact]
    public void ContinuousBatcherConfig_ForModel_LLama7B()
    {
        var config = ContinuousBatcherConfig.ForModel("llama-7b", maxBatchSize: 16);

        Assert.Equal(16, config.SchedulerConfig.MaxBatchSize);
        Assert.Equal(4096, config.MaxContextLength);
    }

    [Fact]
    public void ContinuousBatcherConfig_ForModel_LLama70B()
    {
        var config = ContinuousBatcherConfig.ForModel("llama-70b");

        Assert.Equal(4096, config.MaxContextLength);
    }

    [Fact]
    public void ContinuousBatcherConfig_ForModel_GPT2()
    {
        var config = ContinuousBatcherConfig.ForModel("gpt2");

        Assert.Equal(1024, config.MaxContextLength);
    }

    [Fact]
    public void ContinuousBatcherConfig_ForModel_Unknown()
    {
        var config = ContinuousBatcherConfig.ForModel("unknown-model");

        Assert.Equal(2048, config.MaxContextLength);
    }

    #endregion

    #region BatchSchedulerConfig Tests

    [Fact]
    public void BatchSchedulerConfig_DefaultValues_AreCorrect()
    {
        var config = new BatchSchedulerConfig();

        Assert.Equal(8, config.MaxBatchSize);
        Assert.Equal(256, config.MaxCacheSlots);
        Assert.Equal(8L * 1024 * 1024 * 1024, config.MaxMemoryBytes); // 8GB
        Assert.True(config.AllowPreemption);
        Assert.Equal(SchedulingPolicy.Priority, config.Policy);
        Assert.Equal(32, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(32, config.NumLayers);
    }

    [Fact]
    public void BatchSchedulerConfig_CanBeConfigured()
    {
        var config = new BatchSchedulerConfig
        {
            MaxBatchSize = 16,
            MaxCacheSlots = 512,
            MaxMemoryBytes = 16L * 1024 * 1024 * 1024,
            AllowPreemption = false,
            Policy = SchedulingPolicy.FCFS,
            NumHeads = 64,
            HeadDimension = 256,
            NumLayers = 80
        };

        Assert.Equal(16, config.MaxBatchSize);
        Assert.Equal(512, config.MaxCacheSlots);
        Assert.Equal(16L * 1024 * 1024 * 1024, config.MaxMemoryBytes);
        Assert.False(config.AllowPreemption);
        Assert.Equal(SchedulingPolicy.FCFS, config.Policy);
        Assert.Equal(64, config.NumHeads);
        Assert.Equal(256, config.HeadDimension);
        Assert.Equal(80, config.NumLayers);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_LLama7B()
    {
        var config = BatchSchedulerConfig.ForModel("llama-7b", maxBatchSize: 12);

        Assert.Equal(12, config.MaxBatchSize);
        Assert.Equal(32, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(32, config.NumLayers);
        Assert.Equal(4L * 1024 * 1024 * 1024, config.MaxMemoryBytes);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_LLama13B()
    {
        var config = BatchSchedulerConfig.ForModel("llama-13b");

        Assert.Equal(40, config.NumHeads);
        Assert.Equal(40, config.NumLayers);
        Assert.Equal(8L * 1024 * 1024 * 1024, config.MaxMemoryBytes);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_LLama70B()
    {
        var config = BatchSchedulerConfig.ForModel("llama-70b", maxBatchSize: 8);

        Assert.True(config.MaxBatchSize <= 4); // Constrained for 70B
        Assert.Equal(64, config.NumHeads);
        Assert.Equal(80, config.NumLayers);
        Assert.Equal(16L * 1024 * 1024 * 1024, config.MaxMemoryBytes);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_Unknown()
    {
        var config = BatchSchedulerConfig.ForModel("custom-model", maxBatchSize: 10);

        Assert.Equal(10, config.MaxBatchSize);
    }

    #endregion

    #region BatchScheduler Tests

    [Fact]
    public void BatchScheduler_Creation_WithDefaultConfig()
    {
        var scheduler = new BatchScheduler<double>();

        Assert.Equal(0, scheduler.WaitingCount);
        Assert.Equal(0, scheduler.RunningCount);
        Assert.Equal(0, scheduler.PreemptedCount);
    }

    [Fact]
    public void BatchScheduler_Creation_WithCustomConfig()
    {
        var config = new BatchSchedulerConfig { MaxBatchSize = 16 };
        var scheduler = new BatchScheduler<double>(config);

        Assert.NotNull(scheduler.Config);
        Assert.Equal(16, scheduler.Config.MaxBatchSize);
    }

    [Fact]
    public void BatchScheduler_Creation_ThrowsOnNullConfig()
    {
        Assert.Throws<ArgumentNullException>(() => new BatchScheduler<double>(null!));
    }

    [Fact]
    public void BatchScheduler_AddSequence_IncrementsWaitingCount()
    {
        var scheduler = new BatchScheduler<double>();
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        scheduler.AddSequence(sequence);

        Assert.Equal(1, scheduler.WaitingCount);
    }

    [Fact]
    public void BatchScheduler_AddSequence_ThrowsOnNull()
    {
        var scheduler = new BatchScheduler<double>();

        Assert.Throws<ArgumentNullException>(() => scheduler.AddSequence(null!));
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_MovesSequencesToRunning()
    {
        var scheduler = new BatchScheduler<double>();
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        scheduler.AddSequence(sequence);

        var batch = scheduler.ScheduleNextBatch();

        Assert.Single(batch);
        Assert.Equal(0, scheduler.WaitingCount);
        Assert.Equal(1, scheduler.RunningCount);
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_RespectsMaxBatchSize()
    {
        var config = new BatchSchedulerConfig { MaxBatchSize = 2 };
        var scheduler = new BatchScheduler<double>(config);

        // Add 5 sequences
        for (int i = 0; i < 5; i++)
        {
            var request = CreateGenerationRequest();
            var sequence = new SequenceState<double>(request);
            scheduler.AddSequence(sequence);
        }

        var batch = scheduler.ScheduleNextBatch();

        Assert.Equal(2, batch.Count);
        Assert.Equal(3, scheduler.WaitingCount);
        Assert.Equal(2, scheduler.RunningCount);
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_IncludesRunningSequences()
    {
        var scheduler = new BatchScheduler<double>();
        var request1 = CreateGenerationRequest();
        var sequence1 = new SequenceState<double>(request1);
        scheduler.AddSequence(sequence1);

        // First schedule - moves to running
        var batch1 = scheduler.ScheduleNextBatch();
        batch1[0].Status = SequenceStatus.Generating;

        // Add another sequence
        var request2 = CreateGenerationRequest();
        var sequence2 = new SequenceState<double>(request2);
        scheduler.AddSequence(sequence2);

        // Second schedule - should include both
        var batch2 = scheduler.ScheduleNextBatch();

        Assert.Equal(2, batch2.Count);
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_AssignsBatchIndices()
    {
        var scheduler = new BatchScheduler<double>();

        for (int i = 0; i < 3; i++)
        {
            var request = CreateGenerationRequest();
            var sequence = new SequenceState<double>(request);
            scheduler.AddSequence(sequence);
        }

        var batch = scheduler.ScheduleNextBatch();

        for (int i = 0; i < batch.Count; i++)
        {
            Assert.Equal(i, batch[i].BatchIndex);
        }
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_EmptyWhenNoSequences()
    {
        var scheduler = new BatchScheduler<double>();

        var batch = scheduler.ScheduleNextBatch();

        Assert.Empty(batch);
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_PrioritizesHigherPriority()
    {
        var scheduler = new BatchScheduler<double>();

        var lowPriorityRequest = CreateGenerationRequest(priority: 1);
        var highPriorityRequest = CreateGenerationRequest(priority: 10);

        scheduler.AddSequence(new SequenceState<double>(lowPriorityRequest));
        scheduler.AddSequence(new SequenceState<double>(highPriorityRequest));

        var batch = scheduler.ScheduleNextBatch();

        // Higher priority should come first
        Assert.Equal(10, batch[0].Priority);
    }

    [Fact]
    public void BatchScheduler_CompleteSequence_RemovesFromRunning()
    {
        var scheduler = new BatchScheduler<double>();
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        scheduler.AddSequence(sequence);
        scheduler.ScheduleNextBatch();

        Assert.Equal(1, scheduler.RunningCount);

        scheduler.CompleteSequence(sequence);

        Assert.Equal(0, scheduler.RunningCount);
    }

    [Fact]
    public void BatchScheduler_CompleteSequence_HandlesNull()
    {
        var scheduler = new BatchScheduler<double>();

        // Should not throw
        scheduler.CompleteSequence(null!);
    }

    [Fact]
    public void BatchScheduler_PreemptSequence_MovesToPreempted()
    {
        var scheduler = new BatchScheduler<double>();
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        scheduler.AddSequence(sequence);
        scheduler.ScheduleNextBatch();

        Assert.Equal(1, scheduler.RunningCount);
        Assert.Equal(0, scheduler.PreemptedCount);

        scheduler.PreemptSequence(sequence);

        Assert.Equal(0, scheduler.RunningCount);
        Assert.Equal(1, scheduler.PreemptedCount);
        Assert.Equal(SequenceStatus.Paused, sequence.Status);
    }

    [Fact]
    public void BatchScheduler_CancelSequence_RemovesRunningSequence()
    {
        var scheduler = new BatchScheduler<double>();
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        scheduler.AddSequence(sequence);
        scheduler.ScheduleNextBatch();

        var cancelled = scheduler.CancelSequence(sequence.SequenceId);

        Assert.True(cancelled);
        Assert.Equal(0, scheduler.RunningCount);
        Assert.Equal(SequenceStatus.Cancelled, sequence.Status);
    }

    [Fact]
    public void BatchScheduler_CancelSequence_ReturnsFalseForUnknown()
    {
        var scheduler = new BatchScheduler<double>();

        var cancelled = scheduler.CancelSequence(99999);

        Assert.False(cancelled);
    }

    [Fact]
    public void BatchScheduler_GetRunningSequences_ReturnsCorrectList()
    {
        var scheduler = new BatchScheduler<double>();
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        scheduler.AddSequence(sequence);
        scheduler.ScheduleNextBatch();

        var running = scheduler.GetRunningSequences();

        Assert.Single(running);
        Assert.Equal(sequence.SequenceId, running[0].SequenceId);
    }

    [Fact]
    public void BatchScheduler_GetStatistics_ReturnsCorrectStats()
    {
        var scheduler = new BatchScheduler<double>();
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        scheduler.AddSequence(sequence);
        scheduler.ScheduleNextBatch();

        var stats = scheduler.GetStatistics();

        Assert.Equal(0, stats.WaitingSequences);
        Assert.Equal(1, stats.RunningSequences);
        Assert.Equal(0, stats.PreemptedSequences);
        Assert.Equal(1, stats.TotalSequences);
    }

    [Fact]
    public void BatchScheduler_ReorderByPriority_SortsByPriority()
    {
        var config = new BatchSchedulerConfig { MaxBatchSize = 10 };
        var scheduler = new BatchScheduler<double>(config);

        // Add sequences with different priorities
        for (int i = 0; i < 5; i++)
        {
            var request = CreateGenerationRequest(priority: i);
            scheduler.AddSequence(new SequenceState<double>(request));
        }

        scheduler.ScheduleNextBatch();
        scheduler.ReorderByPriority();

        var running = scheduler.GetRunningSequences();

        // Should be sorted by descending priority
        for (int i = 0; i < running.Count - 1; i++)
        {
            Assert.True(running[i].Priority >= running[i + 1].Priority);
        }
    }

    #endregion

    #region SequenceState Tests

    [Fact]
    public void SequenceState_Creation_InitializesCorrectly()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        Assert.True(sequence.SequenceId > 0);
        Assert.Equal(request, sequence.Request);
        Assert.Equal(SequenceStatus.Pending, sequence.Status);
        Assert.Equal(request.PromptTokenIds.Count, sequence.PromptLength);
        Assert.Equal(0, sequence.GeneratedLength);
        Assert.Equal(request.MaxNewTokens, sequence.MaxNewTokens);
        Assert.Equal(-1, sequence.BatchIndex);
        Assert.Equal(-1, sequence.CacheSlot);
        Assert.False(sequence.PrefillComplete);
        Assert.Null(sequence.FinishReason);
    }

    [Fact]
    public void SequenceState_Creation_ThrowsOnNullRequest()
    {
        Assert.Throws<ArgumentNullException>(() => new SequenceState<double>(null!));
    }

    [Fact]
    public void SequenceState_UniqueIds_AreIncremental()
    {
        var request1 = CreateGenerationRequest();
        var request2 = CreateGenerationRequest();
        var sequence1 = new SequenceState<double>(request1);
        var sequence2 = new SequenceState<double>(request2);

        Assert.True(sequence2.SequenceId > sequence1.SequenceId);
    }

    [Fact]
    public void SequenceState_AppendToken_IncreasesGeneratedLength()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        sequence.AppendToken(100);
        sequence.AppendToken(200);
        sequence.AppendToken(300);

        Assert.Equal(3, sequence.GeneratedLength);
        Assert.Equal(request.PromptTokenIds.Count + 3, sequence.TokenIds.Count);
    }

    [Fact]
    public void SequenceState_AppendToken_AccumulatesLogProb()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        sequence.AppendToken(100, -0.5);
        sequence.AppendToken(200, -0.3);

        Assert.Equal(-0.8, sequence.CumulativeLogProb, precision: 5);
    }

    [Fact]
    public void SequenceState_ShouldStop_MaxLength()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 3
        };
        var sequence = new SequenceState<double>(request);

        sequence.AppendToken(10);
        sequence.AppendToken(20);
        Assert.False(sequence.ShouldStop(eosTokenId: 0));

        sequence.AppendToken(30);
        Assert.True(sequence.ShouldStop(eosTokenId: 0));
        Assert.Equal(StopReason.MaxLength, sequence.FinishReason);
    }

    [Fact]
    public void SequenceState_ShouldStop_EosToken()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        sequence.AppendToken(100);
        Assert.False(sequence.ShouldStop(eosTokenId: 2));

        sequence.AppendToken(2); // EOS token
        Assert.True(sequence.ShouldStop(eosTokenId: 2));
        Assert.Equal(StopReason.EndOfSequence, sequence.FinishReason);
    }

    [Fact]
    public void SequenceState_ShouldStop_StopTokens()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        var stopTokens = new List<int> { 999, 888 };

        sequence.AppendToken(100);
        Assert.False(sequence.ShouldStop(eosTokenId: 2, stopTokenIds: stopTokens));

        sequence.AppendToken(888);
        Assert.True(sequence.ShouldStop(eosTokenId: 2, stopTokenIds: stopTokens));
        Assert.Equal(StopReason.StopToken, sequence.FinishReason);
    }

    [Fact]
    public void SequenceState_Complete_SetsStatusAndTimestamp()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        sequence.GenerationStartedAt = DateTime.UtcNow.AddSeconds(-1);

        sequence.Complete(StopReason.EndOfSequence);

        Assert.Equal(SequenceStatus.Completed, sequence.Status);
        Assert.Equal(StopReason.EndOfSequence, sequence.FinishReason);
        Assert.NotNull(sequence.CompletedAt);
    }

    [Fact]
    public void SequenceState_Cancel_SetsStatusCorrectly()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        sequence.Cancel();

        Assert.Equal(SequenceStatus.Cancelled, sequence.Status);
        Assert.Equal(StopReason.Cancelled, sequence.FinishReason);
        Assert.NotNull(sequence.CompletedAt);
    }

    [Fact]
    public void SequenceState_Fail_SetsStatusCorrectly()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        sequence.Fail("Test error");

        Assert.Equal(SequenceStatus.Failed, sequence.Status);
        Assert.Equal(StopReason.Error, sequence.FinishReason);
        Assert.NotNull(sequence.CompletedAt);
    }

    [Fact]
    public void SequenceState_QueueTime_CalculatedCorrectly()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        // Set GenerationStartedAt to a known offset from CreatedAt (deterministic, no sleeping)
        sequence.GenerationStartedAt = sequence.CreatedAt.AddMilliseconds(50);

        // Assert the exact expected duration
        Assert.Equal(50, sequence.QueueTime.TotalMilliseconds);
    }

    [Fact]
    public void SequenceState_GenerationTime_CalculatedCorrectly()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        sequence.GenerationStartedAt = DateTime.UtcNow.AddSeconds(-2);
        sequence.CompletedAt = DateTime.UtcNow;

        Assert.True(sequence.GenerationTime.HasValue);
        Assert.True(sequence.GenerationTime.Value.TotalSeconds >= 1.9);
    }

    [Fact]
    public void SequenceState_TokensPerSecond_CalculatedCorrectly()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);
        sequence.GenerationStartedAt = DateTime.UtcNow.AddSeconds(-1);

        sequence.AppendToken(10);
        sequence.AppendToken(20);
        sequence.AppendToken(30);
        sequence.AppendToken(40);
        sequence.AppendToken(50);

        sequence.CompletedAt = DateTime.UtcNow;

        Assert.True(sequence.TokensPerSecond.HasValue);
        Assert.True(sequence.TokensPerSecond.Value > 0);
    }

    [Fact]
    public void SequenceState_Priority_InheritedFromRequest()
    {
        var request = CreateGenerationRequest(priority: 42);
        var sequence = new SequenceState<double>(request);

        Assert.Equal(42, sequence.Priority);
    }

    [Fact]
    public void SequenceState_UserContext_InheritedFromRequest()
    {
        var request = CreateGenerationRequest();
        request.UserContext = "test context";
        var sequence = new SequenceState<double>(request);

        Assert.Equal("test context", sequence.UserContext);
    }

    #endregion

    #region GenerationRequest Tests

    [Fact]
    public void GenerationRequest_DefaultValues_AreCorrect()
    {
        var request = new GenerationRequest<double>();

        Assert.Empty(request.PromptTokenIds);
        Assert.Equal(100, request.MaxNewTokens);
        Assert.Equal(1.0f, request.Temperature);
        Assert.Equal(1.0f, request.TopP);
        Assert.Equal(0, request.TopK);
        Assert.Equal(1.0f, request.RepetitionPenalty);
        Assert.False(request.UseBeamSearch);
        Assert.Equal(1, request.NumBeams);
        Assert.Equal(0, request.Priority);
        Assert.Null(request.UserContext);
        Assert.Null(request.OnTokenGenerated);
        Assert.Null(request.StopTokenIds);
    }

    [Fact]
    public void GenerationRequest_CanBeFullyConfigured()
    {
        var tokenCallback = (int token) => { };
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1, 2, 3, 4, 5 },
            MaxNewTokens = 200,
            Temperature = 0.7f,
            TopP = 0.9f,
            TopK = 50,
            RepetitionPenalty = 1.2f,
            UseBeamSearch = true,
            NumBeams = 4,
            Priority = 10,
            UserContext = "custom context",
            OnTokenGenerated = tokenCallback,
            StopTokenIds = new List<int> { 50256 }
        };

        Assert.Equal(5, request.PromptTokenIds.Count);
        Assert.Equal(200, request.MaxNewTokens);
        Assert.Equal(0.7f, request.Temperature);
        Assert.Equal(0.9f, request.TopP);
        Assert.Equal(50, request.TopK);
        Assert.Equal(1.2f, request.RepetitionPenalty);
        Assert.True(request.UseBeamSearch);
        Assert.Equal(4, request.NumBeams);
        Assert.Equal(10, request.Priority);
        Assert.Equal("custom context", request.UserContext);
        Assert.NotNull(request.OnTokenGenerated);
        Assert.Single(request.StopTokenIds!);
    }

    #endregion

    #region GenerationResult Tests

    [Fact]
    public void GenerationResult_DefaultValues_AreCorrect()
    {
        var result = new GenerationResult<double>();

        Assert.Equal(0, result.SequenceId);
        Assert.Empty(result.TokenIds);
        Assert.Empty(result.GeneratedTokens);
        Assert.Equal(StopReason.MaxLength, result.FinishReason);
        Assert.Equal(0, result.GeneratedLength);
        Assert.Equal(default, result.QueueTime);
        Assert.Null(result.GenerationTime);
        Assert.Null(result.TokensPerSecond);
    }

    [Fact]
    public void GenerationResult_CanBeFullyPopulated()
    {
        var result = new GenerationResult<double>
        {
            SequenceId = 12345,
            TokenIds = new List<int> { 1, 2, 3, 100, 200, 300 },
            GeneratedTokens = new List<int> { 100, 200, 300 },
            FinishReason = StopReason.EndOfSequence,
            GeneratedLength = 3,
            QueueTime = TimeSpan.FromMilliseconds(100),
            GenerationTime = TimeSpan.FromSeconds(2),
            TokensPerSecond = 1.5
        };

        Assert.Equal(12345, result.SequenceId);
        Assert.Equal(6, result.TokenIds.Count);
        Assert.Equal(3, result.GeneratedTokens.Count);
        Assert.Equal(StopReason.EndOfSequence, result.FinishReason);
        Assert.Equal(3, result.GeneratedLength);
        Assert.Equal(100, result.QueueTime.TotalMilliseconds);
        Assert.Equal(2, result.GenerationTime.Value.TotalSeconds);
        Assert.Equal(1.5, result.TokensPerSecond);
    }

    #endregion

    #region BatcherStatistics Tests

    [Fact]
    public void BatcherStatistics_DefaultValues_AreZero()
    {
        var stats = new BatcherStatistics();

        Assert.Equal(0, stats.TotalTokensGenerated);
        Assert.Equal(0, stats.TotalRequestsProcessed);
        Assert.Equal(0, stats.TotalIterations);
        Assert.Equal(0, stats.TokensPerSecond);
        Assert.Equal(0, stats.RequestsPerSecond);
        Assert.Equal(0, stats.AverageBatchSize);
        Assert.Equal(0, stats.WaitingRequests);
        Assert.Equal(0, stats.RunningRequests);
        Assert.Equal(0, stats.MemoryUtilization);
        Assert.Equal(0, stats.RuntimeSeconds);
    }

    [Fact]
    public void BatcherStatistics_CanBePopulated()
    {
        var stats = new BatcherStatistics
        {
            TotalTokensGenerated = 1000,
            TotalRequestsProcessed = 50,
            TotalIterations = 200,
            TokensPerSecond = 100.0,
            RequestsPerSecond = 5.0,
            AverageBatchSize = 5.0,
            WaitingRequests = 10,
            RunningRequests = 3,
            MemoryUtilization = 0.75,
            RuntimeSeconds = 10.0
        };

        Assert.Equal(1000, stats.TotalTokensGenerated);
        Assert.Equal(50, stats.TotalRequestsProcessed);
        Assert.Equal(200, stats.TotalIterations);
        Assert.Equal(100.0, stats.TokensPerSecond);
        Assert.Equal(5.0, stats.RequestsPerSecond);
        Assert.Equal(5.0, stats.AverageBatchSize);
        Assert.Equal(10, stats.WaitingRequests);
        Assert.Equal(3, stats.RunningRequests);
        Assert.Equal(0.75, stats.MemoryUtilization);
        Assert.Equal(10.0, stats.RuntimeSeconds);
    }

    #endregion

    #region SchedulerStatistics Tests

    [Fact]
    public void SchedulerStatistics_DefaultValues_AreZero()
    {
        var stats = new SchedulerStatistics();

        Assert.Equal(0, stats.WaitingSequences);
        Assert.Equal(0, stats.RunningSequences);
        Assert.Equal(0, stats.PreemptedSequences);
        Assert.Equal(0, stats.UsedCacheSlots);
        Assert.Equal(0, stats.MaxCacheSlots);
        Assert.Equal(0, stats.UsedMemoryBytes);
        Assert.Equal(0, stats.MaxMemoryBytes);
        Assert.Equal(0, stats.MemoryUtilization);
        Assert.Equal(0, stats.TotalSequences);
        Assert.Equal(0, stats.SlotUtilization);
    }

    [Fact]
    public void SchedulerStatistics_TotalSequences_CalculatedCorrectly()
    {
        var stats = new SchedulerStatistics
        {
            WaitingSequences = 5,
            RunningSequences = 3,
            PreemptedSequences = 2
        };

        Assert.Equal(10, stats.TotalSequences);
    }

    [Fact]
    public void SchedulerStatistics_SlotUtilization_CalculatedCorrectly()
    {
        var stats = new SchedulerStatistics
        {
            UsedCacheSlots = 64,
            MaxCacheSlots = 256
        };

        Assert.Equal(0.25, stats.SlotUtilization);
    }

    [Fact]
    public void SchedulerStatistics_SlotUtilization_HandlesZeroMax()
    {
        var stats = new SchedulerStatistics
        {
            UsedCacheSlots = 10,
            MaxCacheSlots = 0
        };

        Assert.Equal(0, stats.SlotUtilization);
    }

    #endregion

    #region Enum Tests

    [Fact]
    public void SequenceStatus_ContainsExpectedValues()
    {
        var values = Enum.GetValues<SequenceStatus>();

        Assert.Contains(SequenceStatus.Pending, values);
        Assert.Contains(SequenceStatus.Prefilling, values);
        Assert.Contains(SequenceStatus.Generating, values);
        Assert.Contains(SequenceStatus.Completed, values);
        Assert.Contains(SequenceStatus.Cancelled, values);
        Assert.Contains(SequenceStatus.Failed, values);
        Assert.Contains(SequenceStatus.Paused, values);
    }

    [Fact]
    public void StopReason_ContainsExpectedValues()
    {
        var values = Enum.GetValues<StopReason>();

        Assert.Contains(StopReason.MaxLength, values);
        Assert.Contains(StopReason.EndOfSequence, values);
        Assert.Contains(StopReason.StopToken, values);
        Assert.Contains(StopReason.Cancelled, values);
        Assert.Contains(StopReason.Error, values);
    }

    [Fact]
    public void SchedulingPolicy_ContainsExpectedValues()
    {
        var values = Enum.GetValues<SchedulingPolicy>();

        Assert.Contains(SchedulingPolicy.FCFS, values);
        Assert.Contains(SchedulingPolicy.Priority, values);
        Assert.Contains(SchedulingPolicy.ShortestFirst, values);
        Assert.Contains(SchedulingPolicy.Fair, values);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Integration_FullSchedulingWorkflow()
    {
        var scheduler = new BatchScheduler<double>();

        // Add multiple sequences with different priorities
        var requests = new[]
        {
            CreateGenerationRequest(priority: 5),
            CreateGenerationRequest(priority: 10),
            CreateGenerationRequest(priority: 1)
        };

        foreach (var request in requests)
        {
            scheduler.AddSequence(new SequenceState<double>(request));
        }

        Assert.Equal(3, scheduler.WaitingCount);

        // Schedule batch
        var batch = scheduler.ScheduleNextBatch();

        Assert.Equal(3, batch.Count);
        Assert.Equal(0, scheduler.WaitingCount);
        Assert.Equal(3, scheduler.RunningCount);

        // Highest priority should be first
        Assert.Equal(10, batch[0].Priority);

        // Mark first sequence as generating
        batch[0].Status = SequenceStatus.Generating;

        // Complete first sequence
        batch[0].Complete(StopReason.EndOfSequence);
        scheduler.CompleteSequence(batch[0]);

        Assert.Equal(2, scheduler.RunningCount);

        // Preempt another
        scheduler.PreemptSequence(batch[1]);

        Assert.Equal(1, scheduler.RunningCount);
        Assert.Equal(1, scheduler.PreemptedCount);

        // Check statistics
        var stats = scheduler.GetStatistics();
        Assert.Equal(0, stats.WaitingSequences);
        Assert.Equal(1, stats.RunningSequences);
        Assert.Equal(1, stats.PreemptedSequences);
    }

    [Fact]
    public void Integration_SequenceLifecycle()
    {
        var request = CreateGenerationRequest();
        var sequence = new SequenceState<double>(request);

        // Initial state
        Assert.Equal(SequenceStatus.Pending, sequence.Status);
        Assert.Equal(0, sequence.GeneratedLength);

        // Prefilling
        sequence.Status = SequenceStatus.Prefilling;
        sequence.GenerationStartedAt = DateTime.UtcNow;

        // Generate some tokens
        sequence.Status = SequenceStatus.Generating;
        sequence.PrefillComplete = true;

        for (int i = 0; i < 5; i++)
        {
            sequence.AppendToken(100 + i);
        }

        Assert.Equal(5, sequence.GeneratedLength);

        // Check for stop (not reached yet)
        Assert.False(sequence.ShouldStop(eosTokenId: 2));

        // Generate EOS
        sequence.AppendToken(2);
        Assert.True(sequence.ShouldStop(eosTokenId: 2));

        // Complete
        sequence.Complete(StopReason.EndOfSequence);
        Assert.Equal(SequenceStatus.Completed, sequence.Status);
        Assert.NotNull(sequence.GenerationTime);
    }

    #endregion

    #region Helper Methods

    private static GenerationRequest<double> CreateGenerationRequest(int priority = 0)
    {
        return new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1, 2, 3, 4, 5 },
            MaxNewTokens = 100,
            Temperature = 1.0f,
            TopP = 1.0f,
            Priority = priority
        };
    }

    #endregion
}
