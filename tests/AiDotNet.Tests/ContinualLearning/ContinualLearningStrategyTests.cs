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
    #region ElasticWeightConsolidation Tests

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_Constructor_DefaultParameters()
    {
        var strategy = new ElasticWeightConsolidation<double>();

        Assert.Equal(400.0, strategy.Lambda);
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_Constructor_CustomLambda()
    {
        var strategy = new ElasticWeightConsolidation<double>(lambda: 1000.0);

        Assert.Equal(1000.0, strategy.Lambda);
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_Lambda_CanBeModified()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        strategy.Lambda = 500.0;

        Assert.Equal(500.0, strategy.Lambda);
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new ElasticWeightConsolidation<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_AfterTask_ThrowsOnNullNetwork()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(null, taskData, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_AfterTask_ThrowsOnNullInputs()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var (_, targets) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(network, (null, targets), 0));
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_AfterTask_ThrowsOnNullTargets()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var (inputs, _) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(network, (inputs, null), 0));
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_ComputeLoss_ReturnsZeroBeforeAnyTasks()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_ModifyGradients_ThrowsOnNullNetwork()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(null, gradients));
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_ModifyGradients_ThrowsOnNullGradients()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(network, null));
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_ModifyGradients_ReturnsSameVectorWhenNoTasks()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        var result = strategy.ModifyGradients(network, gradients);

        Assert.Same(gradients, result);
    }

    [Fact(Timeout = 60000)]
    public async Task ElasticWeightConsolidation_Reset_ClearsState()
    {
        var strategy = new ElasticWeightConsolidation<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        var loss = strategy.ComputeLoss(network);
        Assert.Equal(0.0, loss);
    }

    #endregion

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

    #region SynapticIntelligence Tests

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_Constructor_DefaultParameters()
    {
        var strategy = new SynapticIntelligence<double>();

        Assert.Equal(1.0, strategy.Lambda);
    }

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_Constructor_CustomParameters()
    {
        var strategy = new SynapticIntelligence<double>(lambda: 2.0, damping: 0.2);

        Assert.Equal(2.0, strategy.Lambda);
    }

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new SynapticIntelligence<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_AfterTask_ThrowsOnNullNetwork()
    {
        var strategy = new SynapticIntelligence<double>();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(null, taskData, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_ComputeLoss_ReturnsZeroBeforeAnyTasks()
    {
        var strategy = new SynapticIntelligence<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_ModifyGradients_ThrowsOnNullNetwork()
    {
        var strategy = new SynapticIntelligence<double>();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(null, gradients));
    }

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_ModifyGradients_ThrowsOnNullGradients()
    {
        var strategy = new SynapticIntelligence<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(network, null));
    }

    [Fact(Timeout = 60000)]
    public async Task SynapticIntelligence_Reset_ClearsState()
    {
        var strategy = new SynapticIntelligence<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        var loss = strategy.ComputeLoss(network);
        Assert.Equal(0.0, loss);
    }

    #endregion

    #region MemoryAwareSynapses Tests

    [Fact(Timeout = 60000)]
    public async Task MemoryAwareSynapses_Constructor_DefaultParameters()
    {
        var strategy = new MemoryAwareSynapses<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task MemoryAwareSynapses_Constructor_CustomLambda()
    {
        var strategy = new MemoryAwareSynapses<double>(lambda: 5.0);

        Assert.Equal(5.0, strategy.Lambda);
    }

    [Fact(Timeout = 60000)]
    public async Task MemoryAwareSynapses_TaskCount_IncreasesAfterTask()
    {
        var strategy = new MemoryAwareSynapses<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        Assert.Equal(1, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task MemoryAwareSynapses_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new MemoryAwareSynapses<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task MemoryAwareSynapses_ComputeLoss_ReturnsZeroBeforeAnyTasks()
    {
        var strategy = new MemoryAwareSynapses<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task MemoryAwareSynapses_Reset_ClearsStateAndTaskCount()
    {
        var strategy = new MemoryAwareSynapses<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.TaskCount);
    }

    #endregion

    #region LearningWithoutForgetting Tests

    [Fact(Timeout = 60000)]
    public async Task LearningWithoutForgetting_Constructor_DefaultParameters()
    {
        var strategy = new LearningWithoutForgetting<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.Equal(2.0, strategy.Temperature);
        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task LearningWithoutForgetting_Constructor_CustomParameters()
    {
        var strategy = new LearningWithoutForgetting<double>(lambda: 2.0, temperature: 4.0);

        Assert.Equal(2.0, strategy.Lambda);
        Assert.Equal(4.0, strategy.Temperature);
    }

    [Fact(Timeout = 60000)]
    public async Task LearningWithoutForgetting_TaskCount_IncreasesAfterPrepareDistillation()
    {
        var strategy = new LearningWithoutForgetting<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        // LwF tracks tasks via PrepareDistillation, not BeforeTask/AfterTask
        strategy.PrepareDistillation(network, taskData.inputs, 0);

        Assert.Equal(1, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task LearningWithoutForgetting_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new LearningWithoutForgetting<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task LearningWithoutForgetting_ComputeLoss_ReturnsZeroBeforeAnyTasks()
    {
        var strategy = new LearningWithoutForgetting<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task LearningWithoutForgetting_Reset_ClearsStateAndTaskCount()
    {
        var strategy = new LearningWithoutForgetting<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        // LwF tracks tasks via PrepareDistillation
        strategy.PrepareDistillation(network, taskData.inputs, 0);
        Assert.Equal(1, strategy.TaskCount);

        strategy.Reset();

        Assert.Equal(0, strategy.TaskCount);
    }

    #endregion

    #region GradientEpisodicMemory Tests

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_Constructor_DefaultParameters()
    {
        var strategy = new GradientEpisodicMemory<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.Equal(0.5, strategy.Margin);
        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_Constructor_CustomParameters()
    {
        var strategy = new GradientEpisodicMemory<double>(memorySize: 512, margin: 0.3, lambda: 2.0);

        Assert.Equal(2.0, strategy.Lambda);
        Assert.Equal(0.3, strategy.Margin);
    }

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_TaskCount_IncreasesAfterTask()
    {
        var strategy = new GradientEpisodicMemory<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        Assert.Equal(1, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new GradientEpisodicMemory<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_AfterTask_ThrowsOnNullInputs()
    {
        var strategy = new GradientEpisodicMemory<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var (_, targets) = ContinualLearningTestHelper.CreateTaskData();

        Assert.Throws<ArgumentNullException>(() => strategy.AfterTask(network, (null, targets), 0));
    }

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_ComputeLoss_ReturnsZeroAlways()
    {
        var strategy = new GradientEpisodicMemory<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_ModifyGradients_ReturnsSameVectorWhenNoTasks()
    {
        var strategy = new GradientEpisodicMemory<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        var result = strategy.ModifyGradients(network, gradients);

        Assert.Same(gradients, result);
    }

    [Fact(Timeout = 60000)]
    public async Task GradientEpisodicMemory_Reset_ClearsMemoryAndTaskCount()
    {
        var strategy = new GradientEpisodicMemory<double>();
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

    #region PackNet Tests

    [Fact(Timeout = 60000)]
    public async Task PackNet_Constructor_DefaultParameters()
    {
        var strategy = new PackNet<double>();

        Assert.Equal(1000.0, strategy.Lambda);
        Assert.Equal(0.5, strategy.PruningRatio);
        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_Constructor_CustomParameters()
    {
        var strategy = new PackNet<double>(pruningRatio: 0.7, lambda: 500.0);

        Assert.Equal(500.0, strategy.Lambda);
        Assert.Equal(0.7, strategy.PruningRatio);
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_Constructor_ThrowsOnInvalidPruningRatio_Zero()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PackNet<double>(pruningRatio: 0.0));
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_Constructor_ThrowsOnInvalidPruningRatio_One()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PackNet<double>(pruningRatio: 1.0));
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_Constructor_ThrowsOnInvalidPruningRatio_Negative()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PackNet<double>(pruningRatio: -0.5));
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_Constructor_ThrowsOnInvalidPruningRatio_GreaterThanOne()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PackNet<double>(pruningRatio: 1.5));
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_TaskCount_IncreasesAfterTask()
    {
        var strategy = new PackNet<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);

        Assert.Equal(1, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new PackNet<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_ComputeLoss_ReturnsZeroAlways()
    {
        var strategy = new PackNet<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_ModifyGradients_ReturnsSameVectorWhenNoTasks()
    {
        var strategy = new PackNet<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        var result = strategy.ModifyGradients(network, gradients);

        Assert.Same(gradients, result);
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_Reset_ClearsStateAndTaskCount()
    {
        var strategy = new PackNet<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.TaskCount);
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_GetTaskMask_ReturnsEmptyForInvalidTaskId()
    {
        var strategy = new PackNet<double>();

        // GetTaskMask returns an empty HashSet for non-existent task IDs (defensive design)
        var mask = strategy.GetTaskMask(999);

        Assert.NotNull(mask);
        Assert.Empty(mask);
    }

    [Fact(Timeout = 60000)]
    public async Task PackNet_GetWeightAllocationStats_ReturnsStats()
    {
        var strategy = new PackNet<double>();

        // GetWeightAllocationStats returns Dictionary<int, int> mapping task IDs to weight counts
        // Key -1 represents free (unallocated) weights
        var stats = strategy.GetWeightAllocationStats();

        Assert.NotNull(stats);
        // Initially contains only free weights entry (-1 key) with 0 free weights
        Assert.Single(stats);
        Assert.True(stats.ContainsKey(-1));
        Assert.Equal(0, stats[-1]);
    }

    #endregion

    #region ProgressiveNeuralNetworks Tests

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_Constructor_DefaultParameters()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();

        Assert.Equal(1.0, strategy.Lambda);
        Assert.True(strategy.UseLateralConnections);
        Assert.Equal(0, strategy.ColumnCount);
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_Constructor_CustomParameters()
    {
        var strategy = new ProgressiveNeuralNetworks<double>(useLateralConnections: false, lambda: 2.0);

        Assert.Equal(2.0, strategy.Lambda);
        Assert.False(strategy.UseLateralConnections);
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_ColumnCount_IncreasesAfterTask()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        Assert.Equal(1, strategy.ColumnCount);

        strategy.AfterTask(network, taskData, 0);
        Assert.Equal(1, strategy.ColumnCount);

        strategy.BeforeTask(network, 1);
        Assert.Equal(2, strategy.ColumnCount);
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_BeforeTask_ThrowsOnNullNetwork()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();

        Assert.Throws<ArgumentNullException>(() => strategy.BeforeTask(null, 0));
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_ComputeLoss_ReturnsZeroAlways()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        var loss = strategy.ComputeLoss(network);

        Assert.Equal(0.0, loss);
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_ModifyGradients_ThrowsOnNullNetwork()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(null, gradients));
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_ModifyGradients_ThrowsOnNullGradients()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();

        Assert.Throws<ArgumentNullException>(() => strategy.ModifyGradients(network, null));
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_Reset_ClearsColumnsAndState()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();
        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();

        strategy.BeforeTask(network, 0);
        strategy.AfterTask(network, taskData, 0);
        strategy.Reset();

        Assert.Equal(0, strategy.ColumnCount);
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_GetColumnParameters_ReturnsNullForInvalidTaskId()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();

        var parameters = strategy.GetColumnParameters(999);

        Assert.Null(parameters);
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_GetNetworkStats_ReturnsStats()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();

        var stats = strategy.GetNetworkStats();

        Assert.NotNull(stats);
        Assert.True(stats.ContainsKey("ColumnCount"));
        Assert.True(stats.ContainsKey("UseLateralConnections"));
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_EstimateMemoryUsage_ReturnsZeroWhenEmpty()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();

        var memory = strategy.EstimateMemoryUsage();

        Assert.Equal(0, memory);
    }

    [Fact(Timeout = 60000)]
    public async Task ProgressiveNeuralNetworks_ComputeLateralInput_ReturnsEmptyWhenNoActivations()
    {
        var strategy = new ProgressiveNeuralNetworks<double>();
        var activations = new List<Tensor<double>>();
        var weights = new List<Tensor<double>>();

        var result = strategy.ComputeLateralInput(activations, weights);

        Assert.NotNull(result);
        Assert.Equal(0, result.Shape[0]);
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

    #region Integration Tests

    [Fact(Timeout = 60000)]
    public async Task AllStrategies_ImplementIContinualLearningStrategy()
    {
        var strategies = new List<IContinualLearningStrategy<double>>
        {
            new ElasticWeightConsolidation<double>(),
            new OnlineEWC<double>(),
            new SynapticIntelligence<double>(),
            new MemoryAwareSynapses<double>(),
            new LearningWithoutForgetting<double>(),
            new GradientEpisodicMemory<double>(),
            new AveragedGEM<double>(),
            new ExperienceReplay<double>(),
            new GenerativeReplay<double>(),
            new PackNet<double>(),
            new ProgressiveNeuralNetworks<double>(),
            new VariationalContinualLearning<double>()
        };

        Assert.Equal(12, strategies.Count);
        Assert.All(strategies, s => Assert.NotNull(s));
    }

    [Fact(Timeout = 60000)]
    public async Task AllStrategies_HaveLambdaProperty()
    {
        var strategies = new List<IContinualLearningStrategy<double>>
        {
            new ElasticWeightConsolidation<double>(),
            new OnlineEWC<double>(),
            new SynapticIntelligence<double>(),
            new MemoryAwareSynapses<double>(),
            new LearningWithoutForgetting<double>(),
            new GradientEpisodicMemory<double>(),
            new AveragedGEM<double>(),
            new ExperienceReplay<double>(),
            new GenerativeReplay<double>(),
            new PackNet<double>(),
            new ProgressiveNeuralNetworks<double>(),
            new VariationalContinualLearning<double>()
        };

        foreach (var strategy in strategies)
        {
            var originalLambda = strategy.Lambda;
            strategy.Lambda = 999.0;
            Assert.Equal(999.0, strategy.Lambda);
            strategy.Lambda = originalLambda;
        }
    }

    [Fact(Timeout = 60000)]
    public async Task AllStrategies_CanProcessMultipleTasks()
    {
        var strategies = new List<IContinualLearningStrategy<double>>
        {
            new ElasticWeightConsolidation<double>(),
            new OnlineEWC<double>(),
            new SynapticIntelligence<double>(),
            new MemoryAwareSynapses<double>(),
            new LearningWithoutForgetting<double>(),
            new GradientEpisodicMemory<double>(),
            new AveragedGEM<double>(),
            new ExperienceReplay<double>(),
            new GenerativeReplay<double>(),
            new PackNet<double>(),
            new ProgressiveNeuralNetworks<double>(),
            new VariationalContinualLearning<double>()
        };

        var network = ContinualLearningTestHelper.CreateMockNetwork();
        var taskData = ContinualLearningTestHelper.CreateTaskData();
        var gradients = ContinualLearningTestHelper.CreateTestGradients();

        foreach (var strategy in strategies)
        {
            strategy.BeforeTask(network, 0);
            strategy.AfterTask(network, taskData, 0);

            var loss = strategy.ComputeLoss(network);
            Assert.True(loss >= 0.0);

            var modifiedGrads = strategy.ModifyGradients(network, gradients);
            Assert.NotNull(modifiedGrads);

            strategy.BeforeTask(network, 1);
            strategy.AfterTask(network, taskData, 1);

            strategy.Reset();
        }
    }

    #endregion
}
