using AiDotNet.ContinualLearning;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;
using System.Threading.Tasks;

#nullable disable

namespace AiDotNet.Tests.ContinualLearning;

/// <summary>
/// Comprehensive tests for all Continual Learning strategies.
/// </summary>
public class ContinualLearningStrategyTests
{
    #region OnlineEWC Tests

    [Fact(Timeout = 60000)]
    public async Task OnlineEWC_Constructor_DefaultParameters()
    {
        var strategy = new OnlineEWC<double>();

        Assert.Equal(400.0, strategy.Lambda);
        Assert.Equal(1.0, strategy.Gamma);
        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineEWC_Constructor_CustomParameters()
    {
        var strategy = new OnlineEWC<double>(lambda: 800.0, gamma: 0.9);

        Assert.Equal(800.0, strategy.Lambda);
        Assert.Equal(0.9, strategy.Gamma);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineEWC_TaskCount_IncreasesAfterTask()
    {
        var strategy = new OnlineEWC<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        Assert.Equal(0, strategy.TaskCount);
        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        Assert.Equal(1, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineEWC_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new OnlineEWC<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineEWC_AfterTask_ThrowsOnNullInputs()
    {
        var strategy = new OnlineEWC<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var (_, targets) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(network, (null, targets), 0));
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineEWC_ComputeLoss_ReturnsZeroBeforeAnyTasks()
    {
        var strategy = new OnlineEWC<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineEWC_Reset_ClearsStateAndTaskCount()
    {
        var strategy = new OnlineEWC<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.TaskCount);
    }

    #endregion

    #region AveragedGEM Tests

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_Constructor_DefaultParameters()
    {
        var strategy = new AveragedGEM<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.Equal(0, strategy.TaskCount);
        Assert.Equal(0, strategy.TotalMemorySize);
    }

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_Constructor_CustomParameters()
    {
        var strategy = new AveragedGEM<double>(memorySize: 512, sampleSize: 128, lambda: 2.0);

        Assert.Equal(2.0, strategy.Lambda);
    }

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_TaskCount_IncreasesAfterTask()
    {
        var strategy = new AveragedGEM<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        Assert.Equal(1, strategy.TaskCount);
        Assert.True(strategy.TotalMemorySize > 0);
    }

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new AveragedGEM<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_AfterTask_ThrowsOnNullInputs()
    {
        var strategy = new AveragedGEM<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var (_, targets) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(network, (null, targets), 0));
    }

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_ComputeLoss_ReturnsZeroAlways()
    {
        var strategy = new AveragedGEM<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_ModifyGradients_ReturnsSameVectorWhenNoTasks()
    {
        var strategy = new AveragedGEM<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        var result = strategy.ModifyGradients(network, gradients);

        Assert.Same(gradients, result);
    }

    [Fact(Timeout = 60000)]
    public async Task AveragedGEM_Reset_ClearsMemory()
    {
        var strategy = new AveragedGEM<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.TaskCount);
        Assert.Equal(0, strategy.TotalMemorySize);
    }

    #endregion

    #region ExperienceReplay Tests

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_Constructor_DefaultParameters()
    {
        var strategy = new ExperienceReplay<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.Equal(0.5, strategy.ReplayRatio);
        Assert.Equal(ExperienceReplay<double>.BufferStrategy.Reservoir, strategy.Strategy);
        Assert.Equal(0, strategy.BufferSize);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_Constructor_CustomParameters()
    {
        var strategy = new ExperienceReplay<double>(
            maxBufferSize: 500,
            replayRatio: 0.3,
            strategy: ExperienceReplay<double>.BufferStrategy.Ring,
            lambda: 2.0);

        Assert.Equal(2.0, strategy.Lambda);
        Assert.Equal(0.3, strategy.ReplayRatio);
        Assert.Equal(ExperienceReplay<double>.BufferStrategy.Ring, strategy.Strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_BufferStrategy_ClassBalanced()
    {
        var strategy = new ExperienceReplay<double>(strategy: ExperienceReplay<double>.BufferStrategy.ClassBalanced);

        Assert.Equal(ExperienceReplay<double>.BufferStrategy.ClassBalanced, strategy.Strategy);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_BufferSize_IncreasesAfterTask()
    {
        var strategy = new ExperienceReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        Assert.True(strategy.BufferSize > 0);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new ExperienceReplay<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_AfterTask_ThrowsOnNullInputs()
    {
        var strategy = new ExperienceReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var (_, targets) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(network, (null, targets), 0));
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_ComputeLoss_ReturnsZeroWhenBufferEmpty()
    {
        var strategy = new ExperienceReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_Reset_ClearsBuffer()
    {
        var strategy = new ExperienceReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.BufferSize);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_SampleReplayBatch_ThrowsWhenBufferEmpty()
    {
        var strategy = new ExperienceReplay<double>();

        Assert.Throws<InvalidOperationException>(() => strategy.SampleReplayBatch());
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_SampleReplayBatch_ReturnsDataWhenBufferPopulated()
    {
        var strategy = new ExperienceReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        var (inputs, targets) = strategy.SampleReplayBatch();

        Assert.NotNull(inputs);
        Assert.NotNull(targets);
        Assert.True(inputs.Shape[0] > 0);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_ExhaustiveSample_PreservesInsertionOrder()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var strategy = new ExperienceReplay<double>(seed: 42);
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        var (inputs, targets) = strategy.SampleReplayBatch();

        Assert.Equal(taskData.inputs.Shape, inputs.Shape);
        Assert.Equal(taskData.targets.Shape, targets.Shape);
        for (int i = 0; i < inputs.Length; i++)
            Assert.Equal(taskData.inputs[i], inputs[i]);
        for (int i = 0; i < targets.Length; i++)
            Assert.Equal(taskData.targets[i], targets[i]);
    }

    [Fact(Timeout = 60000)]
    public async Task ExperienceReplay_ReplayObjectiveAveragesAccumulatedMemories()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var taskData = ContinualLearningTestHelper.CreateTaskData();
        int samplesPerTask = taskData.inputs.Shape[0];
        var strategy = new ExperienceReplay<double>(
            maxBufferSize: checked(samplesPerTask * 20),
            replayRatio: 1.0,
            seed: 42);
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        Assert.False(strategy.AccumulatesAcrossTasks);

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        int firstBufferSize = strategy.BufferSize;
        double oneCopyLoss = strategy.ComputeLoss(network);

        strategy.BeforeTask(network, 1);
        strategy.AfterTask(network, taskData, 1);
        double twoCopyLoss = strategy.ComputeLoss(network);

        Assert.Equal(checked(firstBufferSize * 2), strategy.BufferSize);
        double relativeTolerance = Math.Max(1.0, Math.Abs(oneCopyLoss)) * 1e-10;
        Assert.InRange(Math.Abs(twoCopyLoss - oneCopyLoss), 0.0, relativeTolerance);
    }


    #endregion

    #region GenerativeReplay Tests

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_Constructor_DefaultParameters()
    {
        var strategy = new GenerativeReplay<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.Equal(32, strategy.ReplayBatchSize);
        Assert.Equal(0.5, strategy.ReplayRatio);
        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_Constructor_CustomParameters()
    {
        var strategy = new GenerativeReplay<double>(
            replayBatchSize: 64,
            replayRatio: 0.3,
            lambda: 2.0);

        Assert.Equal(2.0, strategy.Lambda);
        Assert.Equal(64, strategy.ReplayBatchSize);
        Assert.Equal(0.3, strategy.ReplayRatio);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_TaskCount_IncreasesAfterTask()
    {
        var strategy = new GenerativeReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        Assert.Equal(1, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new GenerativeReplay<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_AfterTask_ThrowsOnNullInputs()
    {
        var strategy = new GenerativeReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var (_, targets) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(network, (null, targets), 0));
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_SetGenerator_ThrowsOnNull()
    {
        var strategy = new GenerativeReplay<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.SetGenerator(null));
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_ComputeLoss_ReturnsZeroWithoutGenerator()
    {
        var strategy = new GenerativeReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_GenerateReplaySamples_ReturnsNullWithoutGenerator()
    {
        var strategy = new GenerativeReplay<double>();

        var (inputs, targets) = strategy.GenerateReplaySamples();

        Assert.Null(inputs);
        Assert.Null(targets);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_Reset_ClearsTaskCount()
    {
        var strategy = new GenerativeReplay<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_CreateMixedBatch_ThrowsOnNullInputs()
    {
        var strategy = new GenerativeReplay<double>();
        var (_, targets) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.CreateMixedBatch(null, targets, 32));
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_CreateMixedBatch_ThrowsOnNullTargets()
    {
        var strategy = new GenerativeReplay<double>();
        var (inputs, _) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.CreateMixedBatch(inputs, null, 32));
    }

    [Fact(Timeout = 60000)]
    public async Task GenerativeReplay_CreateMixedBatch_ReturnsOriginalWhenNoGenerator()
    {
        var strategy = new GenerativeReplay<double>();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        var (resultInputs, resultTargets) = strategy.CreateMixedBatch(taskData.inputs, taskData.targets, 32);

        Assert.Same(taskData.inputs, resultInputs);
        Assert.Same(taskData.targets, resultTargets);
    }

    #endregion

    #region VariationalContinualLearning Tests

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_Constructor_DefaultParameters()
    {
        var strategy = new VariationalContinualLearning<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.Equal(-3.0, strategy.InitialLogVar);
        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_Constructor_CustomParameters()
    {
        var strategy = new VariationalContinualLearning<double>(lambda: 2.0, initialLogVar: -2.0);

        Assert.Equal(2.0, strategy.Lambda);
        Assert.Equal(-2.0, strategy.InitialLogVar);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_TaskCount_IncreasesAfterTask()
    {
        var strategy = new VariationalContinualLearning<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        Assert.Equal(1, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new VariationalContinualLearning<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_ComputeLoss_ReturnsZeroBeforeAnyTasks()
    {
        var strategy = new VariationalContinualLearning<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_ModifyGradients_ThrowsOnNullNetwork()
    {
        var strategy = new VariationalContinualLearning<double>();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(null, gradients));
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_ModifyGradients_ThrowsOnNullGradients()
    {
        var strategy = new VariationalContinualLearning<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(network, null));
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_ModifyGradients_ReturnsSameVectorWhenNoTasks()
    {
        var strategy = new VariationalContinualLearning<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        var result = strategy.ModifyGradients(network, gradients);

        Assert.Same(gradients, result);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_Reset_ClearsStateAndTaskCount()
    {
        var strategy = new VariationalContinualLearning<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_SampleWeights_ThrowsOnNullNetwork()
    {
        var strategy = new VariationalContinualLearning<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.SampleWeights(null));
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_SampleWeights_ReturnsNetworkParamsWhenNoPrior()
    {
        var strategy = new VariationalContinualLearning<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var samples = strategy.SampleWeights(network);

        Assert.NotNull(samples);
        Assert.Equal(network.ParameterCount, samples.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_UpdateVariance_ThrowsOnNullGradients()
    {
        var strategy = new VariationalContinualLearning<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.UpdateVariance(null, 0.01));
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_GetPrior_ReturnsEmptyVectorsInitially()
    {
        var strategy = new VariationalContinualLearning<double>();

        var (mean, logVar) = strategy.GetPrior();

        Assert.Equal(0, mean.Length);
        Assert.Equal(0, logVar.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task VariationalContinualLearning_GetPosterior_ReturnsEmptyVectorsInitially()
    {
        var strategy = new VariationalContinualLearning<double>();

        var (mean, logVar) = strategy.GetPosterior();

        Assert.Equal(0, mean.Length);
        Assert.Equal(0, logVar.Length);
    }

    #endregion

}
