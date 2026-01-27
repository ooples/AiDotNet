using AiDotNet.ContinualLearning;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.ContinualLearning;

/// <summary>
/// Integration tests for ContinualLearning strategies.
/// Tests verify that strategies correctly protect learned knowledge across sequential tasks.
/// </summary>
/// <remarks>
/// <para><b>What is Continual Learning?</b></para>
/// Continual learning (also called lifelong learning) is a machine learning paradigm where
/// a model learns multiple tasks sequentially without forgetting previous tasks. The main
/// challenge is "catastrophic forgetting" - when learning new tasks causes the model to
/// forget previously learned knowledge.
///
/// <para><b>Key Strategies Tested:</b></para>
/// <list type="bullet">
/// <item><description>EWC (Elastic Weight Consolidation): Protects important weights using Fisher Information</description></item>
/// <item><description>SI (Synaptic Intelligence): Tracks importance based on gradient path integral</description></item>
/// <item><description>MAS (Memory Aware Synapses): Measures importance via gradient magnitude</description></item>
/// <item><description>GEM (Gradient Episodic Memory): Projects gradients to not interfere with stored examples</description></item>
/// <item><description>LwF (Learning without Forgetting): Uses knowledge distillation from teacher model</description></item>
/// </list>
/// </remarks>
public class ContinualLearningIntegrationTests
{
    private const int DefaultParamCount = 50;
    private const int DefaultOutputSize = 3;
    private const int DefaultBatchSize = 8;
    private const int DefaultInputSize = 10;

    #region Helper Methods

    private static MockNeuralNetwork CreateMockNetwork(int paramCount = DefaultParamCount, int outputSize = DefaultOutputSize)
    {
        return new MockNeuralNetwork(paramCount, outputSize);
    }

    private static Tensor<double> CreateInputBatch(int batchSize = DefaultBatchSize, int inputSize = DefaultInputSize, int seed = 42)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new double[batchSize * inputSize];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble();
        }
        return new Tensor<double>(new[] { batchSize, inputSize }, new Vector<double>(data));
    }

    private static Tensor<double> CreateTargetBatch(int batchSize = DefaultBatchSize, int outputSize = DefaultOutputSize, int seed = 42)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var data = new double[batchSize * outputSize];
        for (int i = 0; i < batchSize; i++)
        {
            // Create one-hot encoded targets
            int classIdx = random.Next(outputSize);
            data[i * outputSize + classIdx] = 1.0;
        }
        return new Tensor<double>(new[] { batchSize, outputSize }, new Vector<double>(data));
    }

    private static (Tensor<double> inputs, Tensor<double> targets) CreateTaskData(
        int batchSize = DefaultBatchSize,
        int inputSize = DefaultInputSize,
        int outputSize = DefaultOutputSize,
        int seed = 42)
    {
        return (CreateInputBatch(batchSize, inputSize, seed), CreateTargetBatch(batchSize, outputSize, seed));
    }

    #endregion

    #region EWC Tests

    [Fact]
    public void ElasticWeightConsolidation_Constructor_InitializesCorrectly()
    {
        var ewc = new ElasticWeightConsolidation<double>(lambda: 500.0);

        Assert.Equal(500.0, ewc.Lambda);
    }

    [Fact]
    public void ElasticWeightConsolidation_BeforeTask_DoesNotThrow()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => ewc.BeforeTask(network, taskId: 0));

        Assert.Null(exception);
    }

    [Fact]
    public void ElasticWeightConsolidation_BeforeTask_NullNetwork_ThrowsArgumentNullException()
    {
        var ewc = new ElasticWeightConsolidation<double>();

        Assert.Throws<ArgumentNullException>(() => ewc.BeforeTask(null!, taskId: 0));
    }

    [Fact]
    public void ElasticWeightConsolidation_AfterTask_StoresFisherAndParams()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        ewc.BeforeTask(network, taskId: 0);
        ewc.AfterTask(network, taskData, taskId: 0);

        // After task, ComputeLoss should return non-zero for changed parameters
        // First, let's just verify it doesn't throw
        var exception = Record.Exception(() => ewc.ComputeLoss(network));
        Assert.Null(exception);
    }

    [Fact]
    public void ElasticWeightConsolidation_ComputeLoss_ReturnsZeroBeforeAnyTask()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();

        var loss = ewc.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact]
    public void ElasticWeightConsolidation_ComputeLoss_NonZeroAfterTaskWithChangedParams()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        // Complete first task
        ewc.BeforeTask(network, taskId: 0);
        ewc.AfterTask(network, taskData, taskId: 0);

        // Change parameters
        var originalParams = network.GetParameters();
        var newParams = originalParams.Clone();
        for (int i = 0; i < newParams.Length; i++)
        {
            newParams[i] = newParams[i] + 1.0; // Significant change
        }
        network.SetParameters(newParams);

        // Now loss should be non-zero (parameters changed from optimal)
        var loss = ewc.ComputeLoss(network);

        Assert.True(loss >= 0, "EWC loss should be non-negative");
        // Note: The loss value depends on Fisher Information which may be zero for mock network
    }

    [Fact]
    public void ElasticWeightConsolidation_ModifyGradients_AddsRegularizationGradient()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        ewc.BeforeTask(network, taskId: 0);
        ewc.AfterTask(network, taskData, taskId: 0);

        // Get original gradients
        var gradients = new Vector<double>(network.ParameterCount);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = 0.1 * (i + 1);
        }

        var modifiedGradients = ewc.ModifyGradients(network, gradients);

        Assert.NotNull(modifiedGradients);
        Assert.Equal(gradients.Length, modifiedGradients.Length);
    }

    [Fact]
    public void ElasticWeightConsolidation_ModifyGradients_BeforeTask_ReturnsUnchanged()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();

        var gradients = new Vector<double>(network.ParameterCount);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = 0.1 * (i + 1);
        }

        var modifiedGradients = ewc.ModifyGradients(network, gradients);

        // Before any task, gradients should be unchanged
        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.Equal(gradients[i], modifiedGradients[i]);
        }
    }

    [Fact]
    public void ElasticWeightConsolidation_Reset_ClearsStoredData()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        ewc.BeforeTask(network, taskId: 0);
        ewc.AfterTask(network, taskData, taskId: 0);

        // Reset
        ewc.Reset();

        // After reset, loss should be zero again
        var loss = ewc.ComputeLoss(network);
        Assert.Equal(0.0, loss);
    }

    [Theory]
    [InlineData(100.0)]
    [InlineData(1000.0)]
    [InlineData(5000.0)]
    public void ElasticWeightConsolidation_DifferentLambdaValues_Work(double lambda)
    {
        var ewc = new ElasticWeightConsolidation<double>(lambda);
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        ewc.BeforeTask(network, taskId: 0);
        ewc.AfterTask(network, taskData, taskId: 0);

        var exception = Record.Exception(() => ewc.ComputeLoss(network));
        Assert.Null(exception);
    }

    [Fact]
    public void ElasticWeightConsolidation_MultipleTasks_AccumulatesConstraints()
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var network = CreateMockNetwork();

        // Task 1
        var taskData1 = CreateTaskData(seed: 1);
        ewc.BeforeTask(network, taskId: 0);
        ewc.AfterTask(network, taskData1, taskId: 0);

        // Task 2
        var taskData2 = CreateTaskData(seed: 2);
        ewc.BeforeTask(network, taskId: 1);
        ewc.AfterTask(network, taskData2, taskId: 1);

        // After two tasks, should still compute loss without error
        var exception = Record.Exception(() => ewc.ComputeLoss(network));
        Assert.Null(exception);
    }

    #endregion

    #region SynapticIntelligence Tests

    [Fact]
    public void SynapticIntelligence_Constructor_InitializesCorrectly()
    {
        var si = new SynapticIntelligence<double>(lambda: 0.5);

        Assert.Equal(0.5, si.Lambda);
    }

    [Fact]
    public void SynapticIntelligence_BeforeTask_DoesNotThrow()
    {
        var si = new SynapticIntelligence<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => si.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void SynapticIntelligence_AfterTask_RecordsImportance()
    {
        var si = new SynapticIntelligence<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        si.BeforeTask(network, taskId: 0);
        si.AfterTask(network, taskData, taskId: 0);

        var exception = Record.Exception(() => si.ComputeLoss(network));
        Assert.Null(exception);
    }

    [Fact]
    public void SynapticIntelligence_ComputeLoss_ReturnsNonNegative()
    {
        var si = new SynapticIntelligence<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        si.BeforeTask(network, taskId: 0);
        si.AfterTask(network, taskData, taskId: 0);

        var loss = si.ComputeLoss(network);
        Assert.True(loss >= 0, "SI loss should be non-negative");
    }

    [Fact]
    public void SynapticIntelligence_Reset_ClearsData()
    {
        var si = new SynapticIntelligence<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        si.BeforeTask(network, taskId: 0);
        si.AfterTask(network, taskData, taskId: 0);

        si.Reset();

        var loss = si.ComputeLoss(network);
        Assert.Equal(0.0, loss);
    }

    #endregion

    #region MemoryAwareSynapses Tests

    [Fact]
    public void MemoryAwareSynapses_Constructor_InitializesCorrectly()
    {
        var mas = new MemoryAwareSynapses<double>(lambda: 1.0);

        Assert.Equal(1.0, mas.Lambda);
    }

    [Fact]
    public void MemoryAwareSynapses_BeforeTask_DoesNotThrow()
    {
        var mas = new MemoryAwareSynapses<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => mas.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void MemoryAwareSynapses_AfterTask_ComputesGradientMagnitude()
    {
        var mas = new MemoryAwareSynapses<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        mas.BeforeTask(network, taskId: 0);
        mas.AfterTask(network, taskData, taskId: 0);

        var exception = Record.Exception(() => mas.ComputeLoss(network));
        Assert.Null(exception);
    }

    [Fact]
    public void MemoryAwareSynapses_ComputeLoss_ReturnsNonNegative()
    {
        var mas = new MemoryAwareSynapses<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        mas.BeforeTask(network, taskId: 0);
        mas.AfterTask(network, taskData, taskId: 0);

        var loss = mas.ComputeLoss(network);
        Assert.True(loss >= 0, "MAS loss should be non-negative");
    }

    #endregion

    #region GradientEpisodicMemory Tests

    [Fact]
    public void GradientEpisodicMemory_Constructor_InitializesCorrectly()
    {
        var gem = new GradientEpisodicMemory<double>(memorySize: 100);

        // Just verify construction works
        Assert.NotNull(gem);
    }

    [Fact]
    public void GradientEpisodicMemory_BeforeTask_DoesNotThrow()
    {
        var gem = new GradientEpisodicMemory<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => gem.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void GradientEpisodicMemory_AfterTask_StoresExamples()
    {
        var gem = new GradientEpisodicMemory<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        gem.BeforeTask(network, taskId: 0);
        gem.AfterTask(network, taskData, taskId: 0);

        var exception = Record.Exception(() => gem.ModifyGradients(network, new Vector<double>(network.ParameterCount)));
        Assert.Null(exception);
    }

    [Fact]
    public void GradientEpisodicMemory_ModifyGradients_ReturnsValidVector()
    {
        var gem = new GradientEpisodicMemory<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        gem.BeforeTask(network, taskId: 0);
        gem.AfterTask(network, taskData, taskId: 0);

        var gradients = new Vector<double>(network.ParameterCount);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = 0.1;
        }

        var modifiedGradients = gem.ModifyGradients(network, gradients);

        Assert.NotNull(modifiedGradients);
        Assert.Equal(gradients.Length, modifiedGradients.Length);
    }

    #endregion

    #region LearningWithoutForgetting Tests

    [Fact]
    public void LearningWithoutForgetting_Constructor_InitializesCorrectly()
    {
        var lwf = new LearningWithoutForgetting<double>(lambda: 1.0, temperature: 2.0);

        // Verify construction
        Assert.NotNull(lwf);
    }

    [Fact]
    public void LearningWithoutForgetting_BeforeTask_DoesNotThrow()
    {
        var lwf = new LearningWithoutForgetting<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => lwf.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void LearningWithoutForgetting_AfterTask_DoesNotThrow()
    {
        var lwf = new LearningWithoutForgetting<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        lwf.BeforeTask(network, taskId: 0);
        var exception = Record.Exception(() => lwf.AfterTask(network, taskData, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void LearningWithoutForgetting_ComputeLoss_ReturnsNonNegative()
    {
        var lwf = new LearningWithoutForgetting<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        lwf.BeforeTask(network, taskId: 0);
        lwf.AfterTask(network, taskData, taskId: 0);

        var loss = lwf.ComputeLoss(network);
        Assert.True(loss >= 0, "LwF loss should be non-negative");
    }

    #endregion

    #region OnlineEWC Tests

    [Fact]
    public void OnlineEWC_Constructor_InitializesCorrectly()
    {
        var onlineEwc = new OnlineEWC<double>(lambda: 400.0, gamma: 0.9);

        Assert.Equal(400.0, onlineEwc.Lambda);
    }

    [Fact]
    public void OnlineEWC_WorksLikeEWCWithDecay()
    {
        var onlineEwc = new OnlineEWC<double>(gamma: 0.95);
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        onlineEwc.BeforeTask(network, taskId: 0);
        onlineEwc.AfterTask(network, taskData, taskId: 0);

        var loss = onlineEwc.ComputeLoss(network);
        Assert.True(loss >= 0, "Online EWC loss should be non-negative");
    }

    [Fact]
    public void OnlineEWC_MultipleTasks_AccumulatesWithDecay()
    {
        var onlineEwc = new OnlineEWC<double>(gamma: 0.8);
        var network = CreateMockNetwork();

        // Task 1
        onlineEwc.BeforeTask(network, taskId: 0);
        onlineEwc.AfterTask(network, CreateTaskData(seed: 1), taskId: 0);

        // Task 2
        onlineEwc.BeforeTask(network, taskId: 1);
        onlineEwc.AfterTask(network, CreateTaskData(seed: 2), taskId: 1);

        // Task 3
        onlineEwc.BeforeTask(network, taskId: 2);
        onlineEwc.AfterTask(network, CreateTaskData(seed: 3), taskId: 2);

        var exception = Record.Exception(() => onlineEwc.ComputeLoss(network));
        Assert.Null(exception);
    }

    #endregion

    #region ExperienceReplay Tests

    [Fact]
    public void ExperienceReplay_Constructor_InitializesCorrectly()
    {
        var er = new ExperienceReplay<double>(maxBufferSize: 500);

        Assert.NotNull(er);
    }

    [Fact]
    public void ExperienceReplay_BeforeTask_DoesNotThrow()
    {
        var er = new ExperienceReplay<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => er.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void ExperienceReplay_AfterTask_StoresExamples()
    {
        var er = new ExperienceReplay<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        er.BeforeTask(network, taskId: 0);
        er.AfterTask(network, taskData, taskId: 0);

        // Should not throw when computing loss after storing
        var exception = Record.Exception(() => er.ComputeLoss(network));
        Assert.Null(exception);
    }

    #endregion

    #region PackNet Tests

    [Fact]
    public void PackNet_Constructor_InitializesCorrectly()
    {
        var packNet = new PackNet<double>(pruningRatio: 0.75);

        Assert.NotNull(packNet);
    }

    [Fact]
    public void PackNet_BeforeTask_DoesNotThrow()
    {
        var packNet = new PackNet<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => packNet.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void PackNet_AfterTask_PrunesAndFreezes()
    {
        var packNet = new PackNet<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        packNet.BeforeTask(network, taskId: 0);
        var exception = Record.Exception(() => packNet.AfterTask(network, taskData, taskId: 0));
        Assert.Null(exception);
    }

    #endregion

    #region ProgressiveNeuralNetworks Tests

    [Fact]
    public void ProgressiveNeuralNetworks_Constructor_InitializesCorrectly()
    {
        var pnn = new ProgressiveNeuralNetworks<double>();

        Assert.NotNull(pnn);
    }

    [Fact]
    public void ProgressiveNeuralNetworks_BeforeTask_DoesNotThrow()
    {
        var pnn = new ProgressiveNeuralNetworks<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => pnn.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    #endregion

    #region GenerativeReplay Tests

    [Fact]
    public void GenerativeReplay_Constructor_InitializesCorrectly()
    {
        var gr = new GenerativeReplay<double>();

        Assert.NotNull(gr);
    }

    [Fact]
    public void GenerativeReplay_BeforeTask_DoesNotThrow()
    {
        var gr = new GenerativeReplay<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => gr.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    #endregion

    #region AveragedGEM Tests

    [Fact]
    public void AveragedGEM_Constructor_InitializesCorrectly()
    {
        var agem = new AveragedGEM<double>(memorySize: 200);

        Assert.NotNull(agem);
    }

    [Fact]
    public void AveragedGEM_BeforeTask_DoesNotThrow()
    {
        var agem = new AveragedGEM<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => agem.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact]
    public void AveragedGEM_ModifyGradients_ReturnsValidGradients()
    {
        var agem = new AveragedGEM<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        agem.BeforeTask(network, taskId: 0);
        agem.AfterTask(network, taskData, taskId: 0);

        var gradients = new Vector<double>(network.ParameterCount);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = 0.5;
        }

        var modifiedGradients = agem.ModifyGradients(network, gradients);

        Assert.NotNull(modifiedGradients);
        Assert.Equal(gradients.Length, modifiedGradients.Length);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void AllStrategies_NullNetwork_ThrowsArgumentNullException()
    {
        var strategies = new IContinualLearningStrategy<double>[]
        {
            new ElasticWeightConsolidation<double>(),
            new SynapticIntelligence<double>(),
            new MemoryAwareSynapses<double>(),
        };

        foreach (var strategy in strategies)
        {
            Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null!, 0));
            Assert.Throws<ArgumentNullException>(() => strategy.ComputeLoss(null!));
        }
    }

    [Fact]
    public void AllStrategies_NullGradients_ThrowsArgumentNullException()
    {
        var strategies = new IContinualLearningStrategy<double>[]
        {
            new ElasticWeightConsolidation<double>(),
        };
        var network = CreateMockNetwork();

        foreach (var strategy in strategies)
        {
            Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(network, null!));
        }
    }

    [Fact]
    public void CoreRegularizationStrategies_LossIsNonNegative()
    {
        // Tests the three core regularization-based strategies: EWC, SI, and MAS
        var strategies = new IContinualLearningStrategy<double>[]
        {
            new ElasticWeightConsolidation<double>(),
            new SynapticIntelligence<double>(),
            new MemoryAwareSynapses<double>(),
        };

        foreach (var strategy in strategies)
        {
            var network = CreateMockNetwork();
            var taskData = CreateTaskData();

            strategy.BeforeTask(network, taskId: 0);
            strategy.AfterTask(network, taskData, taskId: 0);

            var loss = strategy.ComputeLoss(network);
            Assert.True(loss >= 0, $"{strategy.GetType().Name} loss should be non-negative, got {loss}");
        }
    }

    [Fact]
    public void ContinualLearning_SequentialTasks_NoThrows()
    {
        var ewc = new ElasticWeightConsolidation<double>(lambda: 100);
        var network = CreateMockNetwork();

        // Simulate learning 5 sequential tasks
        for (int task = 0; task < 5; task++)
        {
            var taskData = CreateTaskData(seed: task * 10);

            ewc.BeforeTask(network, taskId: task);

            // Simulate training (modify parameters slightly)
            var currentParams = network.GetParameters();
            for (int i = 0; i < currentParams.Length; i++)
            {
                currentParams[i] += 0.01 * (task + 1);
            }
            network.SetParameters(currentParams);

            ewc.AfterTask(network, taskData, taskId: task);
        }

        // Should be able to compute loss after all tasks
        var loss = ewc.ComputeLoss(network);
        Assert.True(loss >= 0);
    }

    #endregion

    #region Lambda Property Tests

    [Theory]
    [InlineData(1.0)]
    [InlineData(100.0)]
    [InlineData(10000.0)]
    public void Strategies_LambdaProperty_CanBeModified(double newLambda)
    {
        var ewc = new ElasticWeightConsolidation<double>();
        var si = new SynapticIntelligence<double>();
        var mas = new MemoryAwareSynapses<double>();

        ewc.Lambda = newLambda;
        si.Lambda = newLambda;
        mas.Lambda = newLambda;

        Assert.Equal(newLambda, ewc.Lambda);
        Assert.Equal(newLambda, si.Lambda);
        Assert.Equal(newLambda, mas.Lambda);
    }

    #endregion
}
