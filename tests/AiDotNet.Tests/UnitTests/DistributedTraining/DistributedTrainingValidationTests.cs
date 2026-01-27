using AiDotNet.DistributedTraining;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.DistributedTraining;

/// <summary>
/// Unit tests for parameter validation in DistributedTraining classes.
/// These tests verify that constructors properly validate their inputs to prevent
/// runtime errors from invalid configurations.
/// </summary>
public class DistributedTrainingValidationTests
{
    #region PR #754 Bug Fix Tests - Parameter Validation

    #region ShardingConfiguration Validation Tests

    [Fact]
    public void ShardingConfiguration_Constructor_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ShardingConfiguration<double>(null!));
    }

    [Fact]
    public void ShardingConfiguration_Constructor_ThrowsOnZeroLearningRate()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShardingConfiguration<double>(backend, learningRate: 0));
    }

    [Fact]
    public void ShardingConfiguration_Constructor_ThrowsOnNegativeLearningRate()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ShardingConfiguration<double>(backend, learningRate: -0.01));
    }

    [Fact]
    public void ShardingConfiguration_Constructor_AcceptsValidLearningRate()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);

        var config = new ShardingConfiguration<double>(backend, learningRate: 0.001);

        Assert.NotNull(config);
        Assert.Same(backend, config.CommunicationBackend);
    }

    [Fact]
    public void ShardingConfiguration_CreateDefault_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateDefault(null!));
    }

    [Fact]
    public void ShardingConfiguration_CreateForHighBandwidth_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateForHighBandwidth(null!));
    }

    [Fact]
    public void ShardingConfiguration_CreateForLowBandwidth_ThrowsOnNullBackend()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ShardingConfiguration<double>.CreateForLowBandwidth(null!));
    }

    #endregion

    #region PipelineParallelModel Validation Tests

    [Fact]
    public void PipelineParallelModel_Constructor_ThrowsOnZeroMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PipelineParallelModel<double, Matrix<double>, Vector<double>>(model, config, microBatchSize: 0));

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_Constructor_ThrowsOnNegativeMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PipelineParallelModel<double, Matrix<double>, Vector<double>>(model, config, microBatchSize: -1));

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_Constructor_AcceptsMinimumMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var pipelineModel = new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
            model, config, microBatchSize: 1);

        Assert.NotNull(pipelineModel);

        backend.Shutdown();
    }

    [Fact]
    public void PipelineParallelModel_Constructor_AcceptsLargeMicroBatchSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 2);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var pipelineModel = new PipelineParallelModel<double, Matrix<double>, Vector<double>>(
            model, config, microBatchSize: 64);

        Assert.NotNull(pipelineModel);

        backend.Shutdown();
    }

    #endregion

    #region HybridShardedModel Validation Tests

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnZeroPipelineParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, pipelineParallelSize: 0));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnNegativePipelineParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, pipelineParallelSize: -1));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnZeroTensorParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, tensorParallelSize: 0));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsOnNegativeTensorParallelSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, tensorParallelSize: -1));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_AcceptsMinimumValidSizes()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        var hybridModel = new HybridShardedModel<double, Matrix<double>, Vector<double>>(
            model, config, pipelineParallelSize: 1, tensorParallelSize: 1, dataParallelSize: 1);

        Assert.NotNull(hybridModel);

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_ThrowsWhenSizesDontMatchWorldSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 8);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        // 2 x 2 x 3 = 12 != 8
        Assert.Throws<ArgumentException>(() =>
            new HybridShardedModel<double, Matrix<double>, Vector<double>>(
                model, config, pipelineParallelSize: 2, tensorParallelSize: 2, dataParallelSize: 3));

        backend.Shutdown();
    }

    [Fact]
    public void HybridShardedModel_Constructor_AcceptsMatchingWorldSize()
    {
        var model = CreateMockModel();
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 8);
        backend.Initialize();
        var config = new ShardingConfiguration<double>(backend);

        // 2 x 2 x 2 = 8 = worldSize
        var hybridModel = new HybridShardedModel<double, Matrix<double>, Vector<double>>(
            model, config, pipelineParallelSize: 2, tensorParallelSize: 2, dataParallelSize: 2);

        Assert.NotNull(hybridModel);

        backend.Shutdown();
    }

    #endregion

    #region InMemoryCommunicationBackend Validation Tests

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnNegativeRank()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: -1, worldSize: 4));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnRankExceedsWorldSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 4, worldSize: 4));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnZeroWorldSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 0));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnNegativeWorldSize()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: -1));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnEmptyEnvironmentId()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: ""));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_ThrowsOnWhitespaceEnvironmentId()
    {
        Assert.Throws<ArgumentException>(() =>
            new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: "   "));
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_AcceptsValidParameters()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4);

        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
    }

    [Fact]
    public void InMemoryCommunicationBackend_Constructor_AcceptsCustomEnvironmentId()
    {
        var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 4, environmentId: "test-env-1");

        Assert.Equal(0, backend.Rank);
        Assert.Equal(4, backend.WorldSize);
    }

    #endregion

    #endregion

    /// <summary>
    /// Creates a simple mock model for testing purposes.
    /// Uses VectorModel which implements IFullModel.
    /// </summary>
    private static IFullModel<double, Matrix<double>, Vector<double>> CreateMockModel()
    {
        var coefficients = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
        return new VectorModel<double>(coefficients);
    }
}
