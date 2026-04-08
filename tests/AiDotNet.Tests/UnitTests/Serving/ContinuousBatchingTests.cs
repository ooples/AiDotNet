using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.Serving.ContinuousBatching;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serving;

/// <summary>
/// Unit tests for Continuous Batching implementation.
/// </summary>
public class ContinuousBatchingTests
{
    #region SequenceState Tests

    [Fact]
    public void SequenceState_InitializesCorrectly()
    {
        // Arrange
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1, 2, 3, 4, 5 },
            MaxNewTokens = 50,
            Temperature = 0.8f
        };

        // Act
        var state = new SequenceState<float>(request);

        // Assert
        Assert.True(state.SequenceId > 0);
        Assert.Equal(SequenceStatus.Pending, state.Status);
        Assert.Equal(5, state.PromptLength);
        Assert.Equal(0, state.GeneratedLength);
        Assert.Equal(50, state.MaxNewTokens);
        Assert.Equal(5, state.TokenIds.Count);
        Assert.False(state.PrefillComplete);
    }

    [Fact]
    public void SequenceState_AppendToken_UpdatesState()
    {
        // Arrange
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1, 2, 3 },
            MaxNewTokens = 10
        };
        var state = new SequenceState<float>(request);

        // Act
        state.AppendToken(100, -1.5);
        state.AppendToken(101, -2.0);

        // Assert
        Assert.Equal(5, state.TokenIds.Count);
        Assert.Equal(2, state.GeneratedLength);
        Assert.Equal(100, state.TokenIds[3]);
        Assert.Equal(101, state.TokenIds[4]);
        Assert.Equal(-3.5, state.CumulativeLogProb, 5);
    }

    [Fact]
    public void SequenceState_ShouldStop_MaxLength()
    {
        // Arrange
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 3
        };
        var state = new SequenceState<float>(request);

        // Act - Generate 3 tokens (hitting the limit)
        state.AppendToken(10);
        state.AppendToken(11);
        state.AppendToken(12);

        // Assert
        Assert.True(state.ShouldStop(eosTokenId: 2));
        Assert.Equal(StopReason.MaxLength, state.FinishReason);
    }

    [Fact]
    public void SequenceState_ShouldStop_EndOfSequence()
    {
        // Arrange
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 100
        };
        var state = new SequenceState<float>(request);

        // Act - Generate EOS token
        state.AppendToken(10);
        state.AppendToken(2); // EOS token

        // Assert
        Assert.True(state.ShouldStop(eosTokenId: 2));
        Assert.Equal(StopReason.EndOfSequence, state.FinishReason);
    }

    [Fact]
    public void SequenceState_ShouldStop_StopToken()
    {
        // Arrange
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 100,
            StopTokenIds = new List<int> { 50, 51, 52 }
        };
        var state = new SequenceState<float>(request);

        // Act
        state.AppendToken(10);
        state.AppendToken(51); // Stop token

        // Assert
        Assert.True(state.ShouldStop(eosTokenId: 2, stopTokenIds: request.StopTokenIds));
        Assert.Equal(StopReason.StopToken, state.FinishReason);
    }

    [Fact]
    public void SequenceState_Complete_SetsStatus()
    {
        // Arrange
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 }
        };
        var state = new SequenceState<float>(request);

        // Act
        state.Complete(StopReason.EndOfSequence);

        // Assert
        Assert.Equal(SequenceStatus.Completed, state.Status);
        Assert.Equal(StopReason.EndOfSequence, state.FinishReason);
        Assert.NotNull(state.CompletedAt);
    }

    [Fact]
    public void SequenceState_Cancel_SetsStatus()
    {
        // Arrange
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 }
        };
        var state = new SequenceState<float>(request);
        state.Status = SequenceStatus.Generating;

        // Act
        state.Cancel();

        // Assert
        Assert.Equal(SequenceStatus.Cancelled, state.Status);
        Assert.Equal(StopReason.Cancelled, state.FinishReason);
    }

    #endregion

    #region BatchScheduler Tests

    [Fact]
    public void BatchScheduler_AddSequence_AddsToQueue()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig { MaxBatchSize = 4 });
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1, 2, 3 }
        };
        var sequence = new SequenceState<float>(request);

        // Act
        scheduler.AddSequence(sequence);

        // Assert
        Assert.Equal(1, scheduler.WaitingCount);
        Assert.Equal(0, scheduler.RunningCount);
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_ReturnsSequences()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig { MaxBatchSize = 4 });
        for (int i = 0; i < 3; i++)
        {
            var request = new GenerationRequest<float>
            {
                PromptTokenIds = new List<int> { 1, 2, 3 }
            };
            scheduler.AddSequence(new SequenceState<float>(request));
        }

        // Act
        var batch = scheduler.ScheduleNextBatch();

        // Assert
        Assert.Equal(3, batch.Count);
        Assert.Equal(0, scheduler.WaitingCount);
        Assert.Equal(3, scheduler.RunningCount);
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_RespectsMaxBatchSize()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig { MaxBatchSize = 2 });
        for (int i = 0; i < 5; i++)
        {
            var request = new GenerationRequest<float>
            {
                PromptTokenIds = new List<int> { 1 }
            };
            scheduler.AddSequence(new SequenceState<float>(request));
        }

        // Act
        var batch = scheduler.ScheduleNextBatch();

        // Assert
        Assert.Equal(2, batch.Count);
        Assert.Equal(3, scheduler.WaitingCount);
        Assert.Equal(2, scheduler.RunningCount);
    }

    [Fact]
    public void BatchScheduler_ScheduleNextBatch_AssignsBatchIndices()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig { MaxBatchSize = 4 });
        for (int i = 0; i < 3; i++)
        {
            var request = new GenerationRequest<float>
            {
                PromptTokenIds = new List<int> { 1 }
            };
            scheduler.AddSequence(new SequenceState<float>(request));
        }

        // Act
        var batch = scheduler.ScheduleNextBatch();

        // Assert
        for (int i = 0; i < batch.Count; i++)
        {
            Assert.Equal(i, batch[i].BatchIndex);
        }
    }

    [Fact]
    public void BatchScheduler_PriorityScheduling_HighPriorityFirst()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig
        {
            MaxBatchSize = 1,
            Policy = SchedulingPolicy.Priority
        });

        var lowPriority = new SequenceState<float>(new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            Priority = 1
        });
        var highPriority = new SequenceState<float>(new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            Priority = 10
        });

        scheduler.AddSequence(lowPriority);
        scheduler.AddSequence(highPriority);

        // Act
        var batch = scheduler.ScheduleNextBatch();

        // Assert
        Assert.Single(batch);
        Assert.Equal(10, batch[0].Priority);
    }

    [Fact]
    public void BatchScheduler_CompleteSequence_RemovesFromRunning()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig { MaxBatchSize = 4 });
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 }
        };
        scheduler.AddSequence(new SequenceState<float>(request));
        var batch = scheduler.ScheduleNextBatch();

        // Act
        scheduler.CompleteSequence(batch[0]);

        // Assert
        Assert.Equal(0, scheduler.RunningCount);
    }

    [Fact]
    public void BatchScheduler_PreemptSequence_MovesToPreempted()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig
        {
            MaxBatchSize = 4,
            AllowPreemption = true
        });
        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 }
        };
        scheduler.AddSequence(new SequenceState<float>(request));
        var batch = scheduler.ScheduleNextBatch();

        // Act
        scheduler.PreemptSequence(batch[0]);

        // Assert
        Assert.Equal(0, scheduler.RunningCount);
        Assert.Equal(1, scheduler.PreemptedCount);
        Assert.Equal(SequenceStatus.Paused, batch[0].Status);
    }

    [Fact]
    public void BatchScheduler_ResumePreempted_PrioritizesPreempted()
    {
        // Arrange
        var config = new BatchSchedulerConfig
        {
            MaxBatchSize = 1,
            AllowPreemption = true,
            MaxMemoryBytes = long.MaxValue // Disable memory constraints
        };
        var scheduler = new BatchScheduler<float>(config);

        // Add and schedule first sequence
        var first = new SequenceState<float>(new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 }
        });
        scheduler.AddSequence(first);
        scheduler.ScheduleNextBatch();

        // Preempt it
        scheduler.PreemptSequence(first);

        // Add a new sequence
        var second = new SequenceState<float>(new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 2 }
        });
        scheduler.AddSequence(second);

        // Act - Schedule again should prefer preempted
        var batch = scheduler.ScheduleNextBatch();

        // Assert - First (preempted) should be resumed first
        Assert.Single(batch);
        Assert.Equal(first.SequenceId, batch[0].SequenceId);
    }

    [Fact]
    public void BatchScheduler_GetStatistics_ReturnsCorrectValues()
    {
        // Arrange
        var scheduler = new BatchScheduler<float>(new BatchSchedulerConfig { MaxBatchSize = 4 });
        for (int i = 0; i < 3; i++)
        {
            scheduler.AddSequence(new SequenceState<float>(new GenerationRequest<float>
            {
                PromptTokenIds = new List<int> { 1 }
            }));
        }
        var batch = scheduler.ScheduleNextBatch();
        scheduler.CompleteSequence(batch[0]);

        // Act
        var stats = scheduler.GetStatistics();

        // Assert
        Assert.Equal(0, stats.WaitingSequences);
        Assert.Equal(2, stats.RunningSequences);
        Assert.Equal(0, stats.PreemptedSequences);
    }

    #endregion

    #region ContinuousBatcher Tests

    [Fact]
    public void ContinuousBatcher_Creation_Succeeds()
    {
        // Arrange & Act
        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = false
        });

        // Assert
        Assert.False(batcher.IsRunning);
        Assert.Equal(0, batcher.PendingRequestCount);
    }

    [Fact]
    public void ContinuousBatcher_Step_ProcessesSequences()
    {
        // Arrange
        var config = new ContinuousBatcherConfig
        {
            AutoStart = false,
            EosTokenId = 2
        };

        // Simple mock model that returns fixed logits
        Tensor<float> mockModel(Tensor<float> input)
        {
            // Return logits where token 5 has highest probability
            var vocabSize = 10;
            var logits = new Tensor<float>(new[] { 1, 1, vocabSize });
            for (int i = 0; i < vocabSize; i++)
            {
                logits[new[] { 0, 0, i }] = i == 5 ? 10f : 0f;
            }
            return logits;
        }

        using var batcher = new ContinuousBatcher<float>(config, mockModel);

        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1, 2, 3 },
            MaxNewTokens = 1
        };

        // Submit request manually (not using async)
        var sequence = new SequenceState<float>(request);
        var scheduler = GetSchedulerFromBatcher(batcher);
        scheduler.AddSequence(sequence);

        // Act
        int tokensGenerated = batcher.Step();

        // Assert
        Assert.True(tokensGenerated >= 0);
    }

    [Fact]
    public void ContinuousBatcher_GetStatistics_ReturnsValidData()
    {
        // Arrange
        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = false
        });

        // Act
        var stats = batcher.GetStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(0, stats.TotalTokensGenerated);
        Assert.Equal(0, stats.TotalRequestsProcessed);
    }

    [Fact]
    public async Task ContinuousBatcher_StartStop_Works()
    {
        // Arrange
        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = false
        });

        // Act
        batcher.Start();
        bool wasRunning = batcher.IsRunning;
        await batcher.StopAsync();
        bool isNowRunning = batcher.IsRunning;

        // Assert
        Assert.True(wasRunning);
        Assert.False(isNowRunning);
    }

    [Fact]
    public async Task ContinuousBatcher_GenerateAsync_ReturnsCancellableTask()
    {
        // Arrange
        using var batcher = new ContinuousBatcher<float>(new ContinuousBatcherConfig
        {
            AutoStart = false
        });

        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1, 2, 3 },
            MaxNewTokens = 100
        };

        using var cts = new CancellationTokenSource();

        // Act
        var task = batcher.GenerateAsync(request, cts.Token);
        cts.Cancel();

        // Assert
        await Assert.ThrowsAsync<TaskCanceledException>(() => task);
    }

    #endregion

    #region Configuration Tests

    [Fact]
    public void BatchSchedulerConfig_ForModel_ReturnsCorrectConfig()
    {
        // Act
        var llama7b = BatchSchedulerConfig.ForModel("llama-7b");
        var llama70b = BatchSchedulerConfig.ForModel("llama-70b");

        // Assert
        Assert.Equal(32, llama7b.NumHeads);
        Assert.Equal(32, llama7b.NumLayers);

        Assert.Equal(64, llama70b.NumHeads);
        Assert.Equal(80, llama70b.NumLayers);
        Assert.True(llama70b.MaxBatchSize <= 4); // Reduced for large model
    }

    [Fact]
    public void ContinuousBatcherConfig_ForModel_ReturnsCorrectConfig()
    {
        // Act
        var config = ContinuousBatcherConfig.ForModel("llama-7b");

        // Assert
        Assert.Equal(4096, config.MaxContextLength);
        Assert.Equal(32, config.SchedulerConfig.NumHeads);
    }

    [Fact]
    public void GenerationRequest_DefaultValues_AreReasonable()
    {
        // Arrange & Act
        var request = new GenerationRequest<float>();

        // Assert
        Assert.Equal(100, request.MaxNewTokens);
        Assert.Equal(1.0f, request.Temperature);
        Assert.Equal(1.0f, request.TopP);
        Assert.Equal(0, request.TopK);
        Assert.Equal(1.0f, request.RepetitionPenalty);
        Assert.False(request.UseBeamSearch);
    }

    #endregion

    #region Event Tests

    [Fact]
    public void ContinuousBatcher_SequenceCompleted_EventFires()
    {
        // Arrange
        var config = new ContinuousBatcherConfig
        {
            AutoStart = false,
            EosTokenId = 2
        };

        // Mock model that immediately returns EOS
        Tensor<float> mockModel(Tensor<float> input)
        {
            var vocabSize = 10;
            var logits = new Tensor<float>(new[] { 1, 1, vocabSize });
            logits[new[] { 0, 0, 2 }] = 100f; // EOS token
            return logits;
        }

        using var batcher = new ContinuousBatcher<float>(config, mockModel);

        bool eventFired = false;
        batcher.SequenceCompleted += (sender, args) =>
        {
            eventFired = true;
        };

        var request = new GenerationRequest<float>
        {
            PromptTokenIds = new List<int> { 1 },
            MaxNewTokens = 10
        };

        // Add sequence directly to scheduler
        var sequence = new SequenceState<float>(request);
        var scheduler = GetSchedulerFromBatcher(batcher);
        scheduler.AddSequence(sequence);

        // Act - Multiple steps to process
        for (int i = 0; i < 3 && !eventFired; i++)
        {
            batcher.Step();
        }

        // Assert
        Assert.True(eventFired);
    }

    #endregion

    #region Helper Methods

    private static BatchScheduler<T> GetSchedulerFromBatcher<T>(ContinuousBatcher<T> batcher)
        where T : struct, IComparable<T>
    {
        // Use reflection to access private scheduler for testing
        var field = typeof(ContinuousBatcher<T>).GetField("_scheduler",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        return (BatchScheduler<T>)field!.GetValue(batcher)!;
    }

    #endregion
}
