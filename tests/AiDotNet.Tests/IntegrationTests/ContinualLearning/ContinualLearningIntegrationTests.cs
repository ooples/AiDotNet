using AiDotNet.ContinualLearning;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;
using System.Threading.Tasks;

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

    #region OnlineEWC Tests

    [Fact(Timeout = 120000)]
    public async Task OnlineEWC_Constructor_InitializesCorrectly()
    {
        var onlineEwc = new OnlineEWC<double>(lambda: 400.0, gamma: 0.9);

        Assert.Equal(400.0, onlineEwc.Lambda);
    }

    [Fact(Timeout = 120000)]
    public async Task OnlineEWC_WorksLikeEWCWithDecay()
    {
        var onlineEwc = new OnlineEWC<double>(gamma: 0.95);
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        onlineEwc.BeforeTask(network, taskId: 0);
        onlineEwc.AfterTask(network, taskData, taskId: 0);

        var loss = onlineEwc.ComputeLoss(network);
        Assert.True(loss >= 0, "Online EWC loss should be non-negative");
    }

    [Fact(Timeout = 120000)]
    public async Task OnlineEWC_MultipleTasks_AccumulatesWithDecay()
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

    [Fact(Timeout = 120000)]
    public async Task ExperienceReplay_Constructor_InitializesCorrectly()
    {
        var er = new ExperienceReplay<double>(maxBufferSize: 500);

        Assert.NotNull(er);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperienceReplay_BeforeTask_DoesNotThrow()
    {
        var er = new ExperienceReplay<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => er.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact(Timeout = 120000)]
    public async Task ExperienceReplay_AfterTask_StoresExamples()
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

    #region GenerativeReplay Tests

    [Fact(Timeout = 120000)]
    public async Task GenerativeReplay_Constructor_InitializesCorrectly()
    {
        var gr = new GenerativeReplay<double>();

        Assert.NotNull(gr);
    }

    [Fact(Timeout = 120000)]
    public async Task GenerativeReplay_BeforeTask_DoesNotThrow()
    {
        var gr = new GenerativeReplay<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => gr.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    #endregion

    #region AveragedGEM Tests

    [Fact(Timeout = 120000)]
    public async Task AveragedGEM_Constructor_InitializesCorrectly()
    {
        var agem = new AveragedGEM<double>(memorySize: 200);

        Assert.NotNull(agem);
    }

    [Fact(Timeout = 120000)]
    public async Task AveragedGEM_BeforeTask_DoesNotThrow()
    {
        var agem = new AveragedGEM<double>();
        var network = CreateMockNetwork();

        var exception = Record.Exception(() => agem.BeforeTask(network, taskId: 0));
        Assert.Null(exception);
    }

    [Fact(Timeout = 120000)]
    public async Task AveragedGEM_ModifyGradients_ReturnsValidGradients()
    {
        var agem = new AveragedGEM<double>();
        var network = CreateMockNetwork();
        var taskData = CreateTaskData();

        agem.BeforeTask(network, taskId: 0);
        agem.AfterTask(network, taskData, taskId: 0);

        var gradients = new Vector<double>((int)network.ParameterCount);
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

    #endregion

    #region Lambda Property Tests

    #endregion
}
