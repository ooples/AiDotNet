using AiDotNet.Serving.ContinuousBatching;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Serving;

/// <summary>
/// Deep integration tests for Serving:
/// BatchSchedulerConfig (defaults, factory methods for LLM models),
/// ContinuousBatcherConfig (defaults, model-specific configs),
/// BatcherStatistics/SchedulerStatistics (data models, computed properties),
/// BatchScheduler (scheduling, queuing, preemption, statistics),
/// SchedulingPolicy enum.
/// </summary>
public class ServingDeepMathIntegrationTests
{
    // ============================
    // BatchSchedulerConfig: Defaults
    // ============================

    [Fact]
    public void BatchSchedulerConfig_Defaults_MaxBatchSize()
    {
        var config = new BatchSchedulerConfig();
        Assert.Equal(8, config.MaxBatchSize);
    }

    [Fact]
    public void BatchSchedulerConfig_Defaults_MaxCacheSlots()
    {
        var config = new BatchSchedulerConfig();
        Assert.Equal(256, config.MaxCacheSlots);
    }

    [Fact]
    public void BatchSchedulerConfig_Defaults_MaxMemoryBytes()
    {
        var config = new BatchSchedulerConfig();
        Assert.Equal(8L * 1024 * 1024 * 1024, config.MaxMemoryBytes); // 8GB
    }

    [Fact]
    public void BatchSchedulerConfig_Defaults_AllowPreemption()
    {
        var config = new BatchSchedulerConfig();
        Assert.True(config.AllowPreemption);
    }

    [Fact]
    public void BatchSchedulerConfig_Defaults_PolicyPriority()
    {
        var config = new BatchSchedulerConfig();
        Assert.Equal(SchedulingPolicy.Priority, config.Policy);
    }

    [Fact]
    public void BatchSchedulerConfig_Defaults_NumHeads()
    {
        var config = new BatchSchedulerConfig();
        Assert.Equal(32, config.NumHeads);
    }

    [Fact]
    public void BatchSchedulerConfig_Defaults_HeadDimension()
    {
        var config = new BatchSchedulerConfig();
        Assert.Equal(128, config.HeadDimension);
    }

    [Fact]
    public void BatchSchedulerConfig_Defaults_NumLayers()
    {
        var config = new BatchSchedulerConfig();
        Assert.Equal(32, config.NumLayers);
    }

    // ============================
    // BatchSchedulerConfig: ForModel Factory
    // ============================

    [Fact]
    public void BatchSchedulerConfig_ForModel_Llama7B()
    {
        var config = BatchSchedulerConfig.ForModel("llama-7b");
        Assert.Equal(8, config.MaxBatchSize);
        Assert.Equal(32, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(32, config.NumLayers);
        Assert.Equal(4L * 1024 * 1024 * 1024, config.MaxMemoryBytes);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_Llama13B()
    {
        var config = BatchSchedulerConfig.ForModel("llama-13b");
        Assert.Equal(8, config.MaxBatchSize);
        Assert.Equal(40, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(40, config.NumLayers);
        Assert.Equal(8L * 1024 * 1024 * 1024, config.MaxMemoryBytes);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_Llama70B()
    {
        var config = BatchSchedulerConfig.ForModel("llama-70b");
        Assert.Equal(4, config.MaxBatchSize); // Capped to min(8, 4) = 4
        Assert.Equal(64, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(80, config.NumLayers);
        Assert.Equal(16L * 1024 * 1024 * 1024, config.MaxMemoryBytes);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_Llama70B_CustomBatch()
    {
        var config = BatchSchedulerConfig.ForModel("llama-70b", maxBatchSize: 2);
        Assert.Equal(2, config.MaxBatchSize); // min(2, 4) = 2
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_Unknown_UsesDefaults()
    {
        var config = BatchSchedulerConfig.ForModel("unknown-model");
        Assert.Equal(8, config.MaxBatchSize);
    }

    [Fact]
    public void BatchSchedulerConfig_ForModel_CaseInsensitive()
    {
        var config = BatchSchedulerConfig.ForModel("LLAMA-7B");
        Assert.Equal(32, config.NumHeads);
    }

    // ============================
    // ContinuousBatcherConfig: Defaults
    // ============================

    [Fact]
    public void ContinuousBatcherConfig_Defaults_EosTokenId()
    {
        var config = new ContinuousBatcherConfig();
        Assert.Equal(2, config.EosTokenId);
    }

    [Fact]
    public void ContinuousBatcherConfig_Defaults_IdleSleepMs()
    {
        var config = new ContinuousBatcherConfig();
        Assert.Equal(10, config.IdleSleepMs);
    }

    [Fact]
    public void ContinuousBatcherConfig_Defaults_AutoStart()
    {
        var config = new ContinuousBatcherConfig();
        Assert.True(config.AutoStart);
    }

    [Fact]
    public void ContinuousBatcherConfig_Defaults_MaxContextLength()
    {
        var config = new ContinuousBatcherConfig();
        Assert.Equal(4096, config.MaxContextLength);
    }

    [Fact]
    public void ContinuousBatcherConfig_Defaults_SpeculativeDecoding()
    {
        var config = new ContinuousBatcherConfig();
        Assert.False(config.EnableSpeculativeDecoding);
        Assert.Equal(4, config.SpeculationDepth);
        Assert.False(config.UseTreeSpeculation);
    }

    // ============================
    // ContinuousBatcherConfig: ForModel Factory
    // ============================

    [Fact]
    public void ContinuousBatcherConfig_ForModel_Llama7B_ContextLength()
    {
        var config = ContinuousBatcherConfig.ForModel("llama-7b");
        Assert.Equal(4096, config.MaxContextLength);
    }

    [Fact]
    public void ContinuousBatcherConfig_ForModel_GPT2_ContextLength()
    {
        var config = ContinuousBatcherConfig.ForModel("gpt2");
        Assert.Equal(1024, config.MaxContextLength);
    }

    [Fact]
    public void ContinuousBatcherConfig_ForModel_Unknown_DefaultContext()
    {
        var config = ContinuousBatcherConfig.ForModel("custom-model");
        Assert.Equal(2048, config.MaxContextLength);
    }

    // ============================
    // BatcherStatistics: Defaults
    // ============================

    [Fact]
    public void BatcherStatistics_Defaults_AllZero()
    {
        var stats = new BatcherStatistics();
        Assert.Equal(0, stats.TotalTokensGenerated);
        Assert.Equal(0, stats.TotalRequestsProcessed);
        Assert.Equal(0, stats.TotalIterations);
        Assert.Equal(0.0, stats.TokensPerSecond);
        Assert.Equal(0.0, stats.RequestsPerSecond);
        Assert.Equal(0.0, stats.AverageBatchSize);
        Assert.Equal(0, stats.WaitingRequests);
        Assert.Equal(0, stats.RunningRequests);
        Assert.Equal(0.0, stats.MemoryUtilization);
        Assert.Equal(0.0, stats.RuntimeSeconds);
    }

    [Fact]
    public void BatcherStatistics_SetProperties()
    {
        var stats = new BatcherStatistics
        {
            TotalTokensGenerated = 10000,
            TotalRequestsProcessed = 50,
            TotalIterations = 1000,
            TokensPerSecond = 500.0,
            RequestsPerSecond = 2.5,
            AverageBatchSize = 4.2,
            WaitingRequests = 3,
            RunningRequests = 5,
            MemoryUtilization = 0.75,
            RuntimeSeconds = 20.0
        };

        Assert.Equal(10000, stats.TotalTokensGenerated);
        Assert.Equal(50, stats.TotalRequestsProcessed);
        Assert.Equal(1000, stats.TotalIterations);
        Assert.Equal(500.0, stats.TokensPerSecond);
        Assert.Equal(2.5, stats.RequestsPerSecond);
        Assert.Equal(4.2, stats.AverageBatchSize, 1e-10);
        Assert.Equal(3, stats.WaitingRequests);
        Assert.Equal(5, stats.RunningRequests);
        Assert.Equal(0.75, stats.MemoryUtilization, 1e-10);
        Assert.Equal(20.0, stats.RuntimeSeconds);
    }

    // ============================
    // SchedulerStatistics: Defaults and Computed
    // ============================

    [Fact]
    public void SchedulerStatistics_Defaults_AllZero()
    {
        var stats = new SchedulerStatistics();
        Assert.Equal(0, stats.WaitingSequences);
        Assert.Equal(0, stats.RunningSequences);
        Assert.Equal(0, stats.PreemptedSequences);
        Assert.Equal(0, stats.UsedCacheSlots);
        Assert.Equal(0, stats.MaxCacheSlots);
        Assert.Equal(0, stats.UsedMemoryBytes);
        Assert.Equal(0, stats.MaxMemoryBytes);
        Assert.Equal(0.0, stats.MemoryUtilization);
    }

    [Fact]
    public void SchedulerStatistics_TotalSequences_Computed()
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
    public void SchedulerStatistics_SlotUtilization_Computed()
    {
        var stats = new SchedulerStatistics
        {
            UsedCacheSlots = 64,
            MaxCacheSlots = 256
        };
        Assert.Equal(0.25, stats.SlotUtilization, 1e-10);
    }

    [Fact]
    public void SchedulerStatistics_SlotUtilization_MaxZero_ReturnsZero()
    {
        var stats = new SchedulerStatistics
        {
            UsedCacheSlots = 10,
            MaxCacheSlots = 0
        };
        Assert.Equal(0.0, stats.SlotUtilization);
    }

    [Fact]
    public void SchedulerStatistics_SlotUtilization_Full()
    {
        var stats = new SchedulerStatistics
        {
            UsedCacheSlots = 256,
            MaxCacheSlots = 256
        };
        Assert.Equal(1.0, stats.SlotUtilization, 1e-10);
    }

    // ============================
    // SchedulingPolicy Enum
    // ============================

    [Fact]
    public void SchedulingPolicy_HasFourValues()
    {
        var values = (((SchedulingPolicy[])Enum.GetValues(typeof(SchedulingPolicy))));
        Assert.Equal(4, values.Length);
    }

    [Theory]
    [InlineData(SchedulingPolicy.FCFS)]
    [InlineData(SchedulingPolicy.Priority)]
    [InlineData(SchedulingPolicy.ShortestFirst)]
    [InlineData(SchedulingPolicy.Fair)]
    public void SchedulingPolicy_AllValuesValid(SchedulingPolicy policy)
    {
        Assert.True(Enum.IsDefined(typeof(SchedulingPolicy), policy));
    }

    // ============================
    // BatchScheduler: Construction and Initial State
    // ============================

    [Fact]
    public void BatchScheduler_DefaultConstruction_EmptyQueues()
    {
        var scheduler = new BatchScheduler<double>();
        Assert.Equal(0, scheduler.WaitingCount);
        Assert.Equal(0, scheduler.RunningCount);
        Assert.Equal(0, scheduler.PreemptedCount);
    }

    [Fact]
    public void BatchScheduler_CustomConfig_StoresConfig()
    {
        var config = new BatchSchedulerConfig { MaxBatchSize = 16, NumHeads = 64 };
        var scheduler = new BatchScheduler<double>(config);
        Assert.Equal(16, scheduler.Config.MaxBatchSize);
        Assert.Equal(64, scheduler.Config.NumHeads);
    }

    [Fact]
    public void BatchScheduler_InitialStatistics_AllZero()
    {
        var scheduler = new BatchScheduler<double>();
        var stats = scheduler.GetStatistics();
        Assert.Equal(0, stats.WaitingSequences);
        Assert.Equal(0, stats.RunningSequences);
        Assert.Equal(0, stats.PreemptedSequences);
        Assert.Equal(0, stats.UsedCacheSlots);
        Assert.Equal(0, stats.TotalSequences);
    }

    [Fact]
    public void BatchScheduler_ScheduleEmpty_ReturnsEmptyBatch()
    {
        var scheduler = new BatchScheduler<double>();
        var batch = scheduler.ScheduleNextBatch();
        Assert.Empty(batch);
    }

    [Fact]
    public void BatchScheduler_GetRunningSequences_InitiallyEmpty()
    {
        var scheduler = new BatchScheduler<double>();
        var running = scheduler.GetRunningSequences();
        Assert.Empty(running);
    }

    [Fact]
    public void BatchScheduler_CancelNonExistent_ReturnsFalse()
    {
        var scheduler = new BatchScheduler<double>();
        bool cancelled = scheduler.CancelSequence(999);
        Assert.False(cancelled);
    }

    // ============================
    // BatchScheduler: AddSequence and Scheduling
    // ============================

    [Fact]
    public void BatchScheduler_AddSequence_IncreasesWaitingCount()
    {
        var scheduler = new BatchScheduler<double>();
        var request = new GenerationRequest<double> { PromptTokenIds = new List<int> { 1, 2, 3 }, MaxNewTokens = 10 };
        var seq = new SequenceState<double>(request);
        scheduler.AddSequence(seq);
        Assert.Equal(1, scheduler.WaitingCount);
    }

    [Fact]
    public void BatchScheduler_AddMultipleSequences_CountsCorrectly()
    {
        var scheduler = new BatchScheduler<double>();
        for (int i = 0; i < 5; i++)
        {
            var request = new GenerationRequest<double> { PromptTokenIds = new List<int> { 1, 2 }, MaxNewTokens = 10 };
            var seq = new SequenceState<double>(request);
            scheduler.AddSequence(seq);
        }
        Assert.Equal(5, scheduler.WaitingCount);
    }

    [Fact]
    public void BatchScheduler_AddNull_Throws()
    {
        var scheduler = new BatchScheduler<double>();
        Assert.Throws<ArgumentNullException>(() => scheduler.AddSequence(null!));
    }

    [Fact]
    public void BatchScheduler_ScheduleOneBatch_MovesToRunning()
    {
        var config = new BatchSchedulerConfig { MaxBatchSize = 4 };
        var scheduler = new BatchScheduler<double>(config);

        for (int i = 0; i < 3; i++)
        {
            var request = new GenerationRequest<double> { PromptTokenIds = new List<int> { 1, 2 }, MaxNewTokens = 10 };
            var seq = new SequenceState<double>(request);
            scheduler.AddSequence(seq);
        }

        var batch = scheduler.ScheduleNextBatch();
        Assert.True(batch.Count <= 4);
        Assert.True(scheduler.RunningCount <= 4);
    }

    [Fact]
    public void BatchScheduler_ReorderByPriority_NoExceptions()
    {
        var scheduler = new BatchScheduler<double>();
        // Should not throw even with empty running list
        scheduler.ReorderByPriority();
    }

    // ============================
    // SequenceState: Construction and Properties
    // ============================

    [Fact]
    public void SequenceState_Construction_InitializesCorrectly()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 10, 20, 30 },
            MaxNewTokens = 50,
            Priority = 5
        };
        var seq = new SequenceState<double>(request);

        Assert.True(seq.SequenceId > 0);
        Assert.Equal(SequenceStatus.Pending, seq.Status);
        Assert.Equal(3, seq.PromptLength);
        Assert.Equal(0, seq.GeneratedLength);
        Assert.Equal(50, seq.MaxNewTokens);
        Assert.Equal(5, seq.Priority);
        Assert.Equal(-1, seq.BatchIndex);
        Assert.Equal(-1, seq.CacheSlot);
        Assert.False(seq.PrefillComplete);
        Assert.Null(seq.FinishReason);
        Assert.Equal(0.0, seq.CumulativeLogProb);
    }

    [Fact]
    public void SequenceState_AppendToken_IncreasesGeneratedLength()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1, 2 },
            MaxNewTokens = 10
        };
        var seq = new SequenceState<double>(request);

        Assert.Equal(0, seq.GeneratedLength);
        seq.AppendToken(100, logProb: -0.5);
        Assert.Equal(1, seq.GeneratedLength);
        Assert.Equal(3, seq.TokenIds.Count); // 2 prompt + 1 generated
        Assert.Equal(-0.5, seq.CumulativeLogProb, 1e-10);

        seq.AppendToken(200, logProb: -0.3);
        Assert.Equal(2, seq.GeneratedLength);
        Assert.Equal(-0.8, seq.CumulativeLogProb, 1e-10);
    }

    [Fact]
    public void SequenceState_ShouldStop_MaxLength()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 2
        };
        var seq = new SequenceState<double>(request);

        seq.AppendToken(10);
        Assert.False(seq.ShouldStop(99));

        seq.AppendToken(20);
        Assert.True(seq.ShouldStop(99));
        Assert.Equal(StopReason.MaxLength, seq.FinishReason);
    }

    [Fact]
    public void SequenceState_ShouldStop_EosToken()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 100
        };
        var seq = new SequenceState<double>(request);

        seq.AppendToken(2); // EOS token
        Assert.True(seq.ShouldStop(eosTokenId: 2));
        Assert.Equal(StopReason.EndOfSequence, seq.FinishReason);
    }

    [Fact]
    public void SequenceState_ShouldStop_StopToken()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 100
        };
        var seq = new SequenceState<double>(request);

        seq.AppendToken(50);
        Assert.True(seq.ShouldStop(eosTokenId: 2, stopTokenIds: new List<int> { 50 }));
        Assert.Equal(StopReason.StopToken, seq.FinishReason);
    }

    [Fact]
    public void SequenceState_Complete_SetsStatusAndReason()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 10
        };
        var seq = new SequenceState<double>(request);

        seq.Complete(StopReason.EndOfSequence);
        Assert.Equal(SequenceStatus.Completed, seq.Status);
        Assert.Equal(StopReason.EndOfSequence, seq.FinishReason);
        Assert.NotNull(seq.CompletedAt);
    }

    [Fact]
    public void SequenceState_Cancel_SetsStatusAndReason()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 10
        };
        var seq = new SequenceState<double>(request);

        seq.Cancel();
        Assert.Equal(SequenceStatus.Cancelled, seq.Status);
        Assert.Equal(StopReason.Cancelled, seq.FinishReason);
    }

    [Fact]
    public void SequenceState_Fail_SetsStatusAndReason()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 10
        };
        var seq = new SequenceState<double>(request);

        seq.Fail("test error");
        Assert.Equal(SequenceStatus.Failed, seq.Status);
        Assert.Equal(StopReason.Error, seq.FinishReason);
    }

    // ============================
    // SequenceState: Unique IDs
    // ============================

    [Fact]
    public void SequenceState_UniqueIds()
    {
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 10
        };

        var seq1 = new SequenceState<double>(request);
        var seq2 = new SequenceState<double>(request);
        Assert.NotEqual(seq1.SequenceId, seq2.SequenceId);
    }

    // ============================
    // GenerationRequest: Defaults
    // ============================

    [Fact]
    public void GenerationRequest_Defaults()
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

    // ============================
    // SequenceStatus/StopReason Enums
    // ============================

    [Fact]
    public void SequenceStatus_HasSevenValues()
    {
        var values = (((SequenceStatus[])Enum.GetValues(typeof(SequenceStatus))));
        Assert.Equal(7, values.Length);
    }

    [Fact]
    public void StopReason_HasFiveValues()
    {
        var values = (((StopReason[])Enum.GetValues(typeof(StopReason))));
        Assert.Equal(5, values.Length);
    }
}
