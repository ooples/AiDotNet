using AiDotNet.DistributedTraining;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.DistributedTraining;

/// <summary>
/// Deep integration tests for DistributedTraining:
/// ActivationCheckpointConfig (defaults, validation, strategies),
/// InMemoryCommunicationBackend (construction, rank/worldSize, initialization, collective ops),
/// RecomputeStrategy enum.
/// </summary>
public class DistributedTrainingDeepMathIntegrationTests
{
    // ============================
    // ActivationCheckpointConfig: Defaults
    // ============================

    [Fact]
    public void ActivationCheckpointConfig_Defaults_Disabled()
    {
        var config = new ActivationCheckpointConfig();
        Assert.False(config.Enabled);
    }

    [Fact]
    public void ActivationCheckpointConfig_Defaults_CheckpointEveryTenLayers()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Equal(10, config.CheckpointEveryNLayers);
    }

    [Fact]
    public void ActivationCheckpointConfig_Defaults_RecomputeStrategyNone()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Equal(RecomputeStrategy.None, config.RecomputeStrategy);
    }

    [Fact]
    public void ActivationCheckpointConfig_Defaults_MaxActivationsZero()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Equal(0, config.MaxActivationsInMemory);
    }

    [Fact]
    public void ActivationCheckpointConfig_Defaults_CheckpointFirstLayerTrue()
    {
        var config = new ActivationCheckpointConfig();
        Assert.True(config.CheckpointFirstLayer);
    }

    // ============================
    // ActivationCheckpointConfig: Property Setting
    // ============================

    [Fact]
    public void ActivationCheckpointConfig_SetAll()
    {
        var config = new ActivationCheckpointConfig
        {
            Enabled = true,
            CheckpointEveryNLayers = 5,
            RecomputeStrategy = RecomputeStrategy.Selective,
            MaxActivationsInMemory = 20,
            CheckpointFirstLayer = false
        };

        Assert.True(config.Enabled);
        Assert.Equal(5, config.CheckpointEveryNLayers);
        Assert.Equal(RecomputeStrategy.Selective, config.RecomputeStrategy);
        Assert.Equal(20, config.MaxActivationsInMemory);
        Assert.False(config.CheckpointFirstLayer);
    }

    // ============================
    // ActivationCheckpointConfig: Validation
    // ============================

    [Fact]
    public void ActivationCheckpointConfig_CheckpointEveryNLayers_ZeroThrows()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Throws<ArgumentOutOfRangeException>(() => config.CheckpointEveryNLayers = 0);
    }

    [Fact]
    public void ActivationCheckpointConfig_CheckpointEveryNLayers_NegativeThrows()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Throws<ArgumentOutOfRangeException>(() => config.CheckpointEveryNLayers = -1);
    }

    [Fact]
    public void ActivationCheckpointConfig_CheckpointEveryNLayers_OneIsValid()
    {
        var config = new ActivationCheckpointConfig();
        config.CheckpointEveryNLayers = 1;
        Assert.Equal(1, config.CheckpointEveryNLayers);
    }

    [Fact]
    public void ActivationCheckpointConfig_MaxActivationsInMemory_NegativeThrows()
    {
        var config = new ActivationCheckpointConfig();
        Assert.Throws<ArgumentOutOfRangeException>(() => config.MaxActivationsInMemory = -1);
    }

    [Fact]
    public void ActivationCheckpointConfig_MaxActivationsInMemory_ZeroIsValid()
    {
        var config = new ActivationCheckpointConfig();
        config.MaxActivationsInMemory = 0;
        Assert.Equal(0, config.MaxActivationsInMemory);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(10)]
    [InlineData(50)]
    [InlineData(100)]
    public void ActivationCheckpointConfig_CheckpointEveryNLayers_ValidValues(int value)
    {
        var config = new ActivationCheckpointConfig { CheckpointEveryNLayers = value };
        Assert.Equal(value, config.CheckpointEveryNLayers);
    }

    // ============================
    // RecomputeStrategy Enum
    // ============================

    [Fact]
    public void RecomputeStrategy_HasThreeValues()
    {
        var values = Enum.GetValues<RecomputeStrategy>();
        Assert.Equal(3, values.Length);
    }

    [Theory]
    [InlineData(RecomputeStrategy.Selective)]
    [InlineData(RecomputeStrategy.Full)]
    [InlineData(RecomputeStrategy.None)]
    public void RecomputeStrategy_AllValuesValid(RecomputeStrategy strategy)
    {
        Assert.True(Enum.IsDefined(strategy));
    }

    // ============================
    // InMemoryCommunicationBackend: Construction
    // ============================

    [Fact]
    public void InMemoryBackend_Construction_RankAndWorldSize()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);
        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(0, 4)]
    [InlineData(3, 4)]
    [InlineData(0, 8)]
    [InlineData(7, 8)]
    public void InMemoryBackend_ValidRankWorldSize(int rank, int worldSize)
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: rank, worldSize: worldSize);
        Assert.Equal(rank, backend.Rank);
        Assert.Equal(worldSize, backend.WorldSize);
    }

    [Fact]
    public void InMemoryBackend_Initialize_Succeeds()
    {
        var envId = Guid.NewGuid().ToString("N");
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();
        // Initialize should not throw for a single-process backend
    }

    [Fact]
    public void InMemoryBackend_FloatType_Constructs()
    {
        var backend = new InMemoryCommunicationBackend<float>(rank: 0, worldSize: 2);
        Assert.Equal(0, backend.Rank);
        Assert.Equal(2, backend.WorldSize);
    }

    // ============================
    // InMemoryCommunicationBackend: Broadcast (single rank)
    // ============================

    [Fact]
    public void InMemoryBackend_Broadcast_SingleProcess_PreservesData()
    {
        var envId = Guid.NewGuid().ToString("N");
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1, environmentId: envId);
        backend.Initialize();

        var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        backend.Broadcast(data, 0);

        // With worldSize=1, broadcast should be a no-op
        Assert.Equal(1.0, data[0]);
        Assert.Equal(2.0, data[1]);
        Assert.Equal(3.0, data[2]);
    }

    // ============================
    // ActivationCheckpointConfig: Optimal Checkpoint Frequency
    // ============================

    [Theory]
    [InlineData(100, 10)]  // sqrt(100) = 10
    [InlineData(16, 4)]    // sqrt(16) = 4
    [InlineData(25, 5)]    // sqrt(25) = 5
    [InlineData(64, 8)]    // sqrt(64) = 8
    public void ActivationCheckpoint_OptimalFrequency_IsSqrtLayers(int totalLayers, int expectedCheckpointInterval)
    {
        // Optimal checkpoint frequency is approximately sqrt(L) where L = total layers
        // This gives O(sqrt(L)) memory instead of O(L)
        int optimalInterval = (int)Math.Round(Math.Sqrt(totalLayers));
        Assert.Equal(expectedCheckpointInterval, optimalInterval);

        // Verify we can set this value
        var config = new ActivationCheckpointConfig { CheckpointEveryNLayers = optimalInterval };
        Assert.Equal(expectedCheckpointInterval, config.CheckpointEveryNLayers);
    }

    [Theory]
    [InlineData(100, 10)]  // 100/10 = 10 checkpoints
    [InlineData(100, 1)]   // 100/1 = 100 checkpoints (every layer = no recomputation)
    [InlineData(100, 100)] // 100/100 = 1 checkpoint (maximum recomputation)
    public void ActivationCheckpoint_NumberOfCheckpoints(int totalLayers, int interval)
    {
        // Number of checkpoints = ceil(totalLayers / interval)
        int numCheckpoints = (int)Math.Ceiling((double)totalLayers / interval);

        var config = new ActivationCheckpointConfig { CheckpointEveryNLayers = interval };
        Assert.True(numCheckpoints >= 1);
        Assert.True(numCheckpoints <= totalLayers);
    }

    // ============================
    // Memory savings calculation
    // ============================

    [Theory]
    [InlineData(100)]
    [InlineData(200)]
    [InlineData(400)]
    public void ActivationCheckpoint_MemorySavings_SublinearVsLinear(int totalLayers)
    {
        // Without checkpointing: memory = O(L) = totalLayers activations
        int withoutCheckpointing = totalLayers;

        // With optimal checkpointing: memory = O(sqrt(L))
        int optimalInterval = (int)Math.Round(Math.Sqrt(totalLayers));
        int numCheckpoints = (int)Math.Ceiling((double)totalLayers / optimalInterval);
        // Memory = checkpoints stored + max segment between checkpoints
        int withCheckpointing = numCheckpoints + optimalInterval;

        // Checkpointing should use significantly less memory than no checkpointing
        Assert.True(withCheckpointing < withoutCheckpointing,
            $"Checkpointing ({withCheckpointing}) should use less memory than no checkpointing ({withoutCheckpointing}) for {totalLayers} layers");

        // Savings ratio should be > 2x for large models
        if (totalLayers >= 100)
        {
            double ratio = (double)withoutCheckpointing / withCheckpointing;
            Assert.True(ratio > 2.0,
                $"Memory savings ratio should be > 2x for {totalLayers} layers, got {ratio:F2}x");
        }
    }
}
