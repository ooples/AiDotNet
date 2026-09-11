using System;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NER.Options;
using AiDotNet.NER.TransformerBased;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class TransformerNERSmokeFixtureContractTests
{
    [Theory]
    [InlineData(4, 0.0, 0.0025)]
    [InlineData(4, 0.001, 0.001)]
    [InlineData(0, 0.0, 0.0)]
    public void ExplicitSmokeProfile_PreservesPositiveStartsAndDisabledWarmup(
        int warmupSteps, double initialRate, double expectedRate)
    {
        var options = CreateOptions();
        options.WarmupSteps = warmupSteps;
        options.WarmupInitialLearningRate = initialRate;

        var configured = FixtureProbe.Configure(options);

        Assert.Same(options, configured);
        Assert.Equal(expectedRate, configured.WarmupInitialLearningRate);
        Assert.Equal(warmupSteps, configured.WarmupSteps);
        Assert.Equal(0.01, configured.LearningRate);
        Assert.Equal(8, configured.HiddenDimension);
    }

    [Fact]
    public void ZeroStartPreparation_ConsumesOnlyTheVerifiedZeroStepWithoutChangingWeights()
    {
        using var arena = TensorArena.Create();
        using var model = new CountingTinyNER(CreateOptions());
        var (input, target) = CreateInputAndTarget();
        model.Predict(input);
        var before = model.GetParameters().ToArray();
        Assert.NotEmpty(before);
        var fixture = new FixtureProbe();

        fixture.Prepare(model, input, target);

        Assert.Equal(1, model.TrainingCalls);
        Assert.Equal(before, model.GetParameters().ToArray());
        var scheduler = Assert.IsType<LinearWarmupScheduler>(GetOptimizer(model).LearningRateScheduler);
        Assert.Equal(1, scheduler.CurrentStep);
        Assert.Equal(0.0025, scheduler.CurrentLearningRate);
        fixture.Prepare(model, input, target);
        Assert.Equal(1, model.TrainingCalls);

        model.Train(input, target);
        var after = model.GetParameters().ToArray();
        Assert.Contains(Enumerable.Range(0, before.Length), index => before[index] != after[index]);
        Assert.All(after, value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
    }

    public enum GuardControl
    {
        PositiveConfiguredStart,
        DisabledWarmup,
        InjectedPositiveScheduler,
        InjectedOptimizerWithoutScheduler,
        AlreadyAdvanced,
        EpochSteppedScheduler,
        SchedulerResetAfterAnOptimizerStep
    }

    [Theory]
    [InlineData(GuardControl.PositiveConfiguredStart)]
    [InlineData(GuardControl.DisabledWarmup)]
    [InlineData(GuardControl.InjectedPositiveScheduler)]
    [InlineData(GuardControl.InjectedOptimizerWithoutScheduler)]
    [InlineData(GuardControl.AlreadyAdvanced)]
    [InlineData(GuardControl.EpochSteppedScheduler)]
    [InlineData(GuardControl.SchedulerResetAfterAnOptimizerStep)]
    public void Preparation_DoesNotAddStepsForOtherOptimizerStates(GuardControl control)
    {
        using var arena = TensorArena.Create();
        var options = CreateOptions();
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? injected = null;
        switch (control)
        {
            case GuardControl.PositiveConfiguredStart:
                options.WarmupInitialLearningRate = 0.001;
                break;
            case GuardControl.DisabledWarmup:
                options.WarmupSteps = 0;
                break;
            case GuardControl.InjectedPositiveScheduler:
                injected = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        LearningRate = 0.01,
                        SchedulerStepMode = SchedulerStepMode.StepPerBatch,
                        LearningRateScheduler = new LinearWarmupScheduler(0.01, 4, warmupInitLr: 0.003)
                    });
                break;
            case GuardControl.InjectedOptimizerWithoutScheduler:
                injected = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    { LearningRate = 0.01, SchedulerStepMode = SchedulerStepMode.StepPerBatch });
                break;
            case GuardControl.EpochSteppedScheduler:
                injected = new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
                    new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
                    {
                        LearningRate = 0.01,
                        SchedulerStepMode = SchedulerStepMode.StepPerEpoch,
                        LearningRateScheduler = new LinearWarmupScheduler(0.01, 4)
                    });
                break;
            case GuardControl.AlreadyAdvanced:
            case GuardControl.SchedulerResetAfterAnOptimizerStep:
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(control));
        }
        using var model = new CountingTinyNER(options, injected);
        var (input, target) = CreateInputAndTarget();
        model.Predict(input);
        if (control is GuardControl.AlreadyAdvanced or GuardControl.SchedulerResetAfterAnOptimizerStep)
            model.Train(input, target);
        if (control == GuardControl.SchedulerResetAfterAnOptimizerStep)
        {
            var advancedOptimizer = GetOptimizer(model);
            var resetScheduler = Assert.IsType<LinearWarmupScheduler>(advancedOptimizer.LearningRateScheduler);
            resetScheduler.Reset();
            Assert.Equal(0, resetScheduler.CurrentStep);
            Assert.Equal(0.0, resetScheduler.CurrentLearningRate);
            Assert.True(advancedOptimizer.CurrentStep > 0);
            Assert.True(advancedOptimizer.GetCurrentLearningRate() > 0);
        }
        if (control == GuardControl.EpochSteppedScheduler)
        {
            var epochOptimizer = GetOptimizer(model);
            Assert.Equal(SchedulerStepMode.StepPerEpoch, epochOptimizer.SchedulerStepMode);
            Assert.Equal(0, epochOptimizer.CurrentStep);
            Assert.Equal(0.0, epochOptimizer.GetCurrentLearningRate());
            var epochScheduler = Assert.IsType<LinearWarmupScheduler>(epochOptimizer.LearningRateScheduler);
            Assert.Equal(0, epochScheduler.CurrentStep);
            Assert.Equal(0.0, epochScheduler.CurrentLearningRate);
            Assert.True(epochScheduler.GetLearningRateAtStep(1) > 0);
        }
        int initialCalls = model.TrainingCalls;
        var parameters = model.GetParameters().ToArray();
        var scheduler = GetOptimizer(model).LearningRateScheduler;
        int? initialStep = scheduler?.CurrentStep;
        double? initialRate = scheduler?.CurrentLearningRate;

        new FixtureProbe().Prepare(model, input, target);

        Assert.Equal(initialCalls, model.TrainingCalls);
        Assert.Equal(parameters, model.GetParameters().ToArray());
        Assert.Equal(initialStep, scheduler?.CurrentStep);
        Assert.Equal(initialRate, scheduler?.CurrentLearningRate);
    }

    [Fact]
    public void Preparation_DoesNotAdvanceACloneThatAlreadyHasAPositiveRate()
    {
        using var arena = TensorArena.Create();
        using var original = new TinyBERTNER<float>(CreateArchitecture(), CreateOptions());
        var (input, target) = CreateInputAndTarget();
        original.Train(input, target);
        using var clone = Assert.IsType<TinyBERTNER<float>>(original.Clone());
        var scheduler = Assert.IsType<LinearWarmupScheduler>(GetOptimizer(clone).LearningRateScheduler);
        if (scheduler.CurrentStep == 0) clone.Train(input, target);
        Assert.True(scheduler.CurrentStep > 0);
        Assert.True(scheduler.CurrentLearningRate > 0);
        int initialStep = scheduler.CurrentStep;
        var parameters = clone.GetParameters().ToArray();

        new FixtureProbe().Prepare(clone, input, target);

        Assert.Equal(initialStep, scheduler.CurrentStep);
        Assert.Equal(parameters, clone.GetParameters().ToArray());
    }

    [Fact]
    public async Task LegacyTinyBERTFixture_RetainsTheSharedNonzeroFiniteGradientAssertions()
    {
        var fixture = new ModelFamilyTests.NeuralNetworks.TinyBERTNERTests();
        await fixture.InitializeAsync();
        try
        {
            await fixture.GradientFlow_ShouldBeNonZeroAndFinite();
        }
        finally
        {
            await fixture.DisposeAsync();
        }
    }

    public enum ZeroStepDefect
    {
        ParameterMutation,
        SchedulerDoesNotAdvance
    }

    [Theory]
    [InlineData(ZeroStepDefect.ParameterMutation)]
    [InlineData(ZeroStepDefect.SchedulerDoesNotAdvance)]
    public void ZeroStartPreparation_RejectsABrokenZeroStepInsteadOfHidingIt(ZeroStepDefect defect)
    {
        using var arena = TensorArena.Create();
        using var model = new CountingTinyNER(CreateOptions(), defect: defect);
        var (input, target) = CreateInputAndTarget();
        model.Predict(input);

        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => new FixtureProbe().Prepare(model, input, target));
        Assert.Equal(1, model.TrainingCalls);
        var scheduler = Assert.IsType<LinearWarmupScheduler>(GetOptimizer(model).LearningRateScheduler);
        Assert.Equal(defect == ZeroStepDefect.SchedulerDoesNotAdvance ? 0 : 1, scheduler.CurrentStep);
    }

    [Fact]
    public async Task SharedGradientInvariant_StillRejectsABrokenFirstPositiveUpdate()
    {
        var fixture = new BrokenPositiveUpdateFixture();
        await fixture.InitializeAsync();
        try
        {
            var failure = await Assert.ThrowsAsync<Xunit.Sdk.TrueException>(
                fixture.GradientFlow_ShouldBeNonZeroAndFinite);
            Assert.Contains("No parameters changed after training", failure.Message);
        }
        finally
        {
            await fixture.DisposeAsync();
        }
    }

    private static TransformerNEROptions CreateOptions() => new()
    {
        HiddenDimension = 8, NumAttentionHeads = 2, NumTransformerLayers = 1,
        IntermediateDimension = 16, NumLabels = 9, MaxSequenceLength = 4,
        DropoutRate = 0, LearningRate = 0.01, WarmupSteps = 4, WarmupInitialLearningRate = 0.0
    };

    private static NeuralNetworkArchitecture<float> CreateArchitecture() => new(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.SequenceToSequence,
        inputSize: 8, outputSize: 9) { RandomSeed = 1337 };

    private static (Tensor<float> Input, Tensor<float> Target) CreateInputAndTarget()
    {
        var input = new Tensor<float>(new[] { 4, 8 });
        for (int i = 0; i < input.Length; i++) input[i] = (i % 7 - 3) * 0.1f;
        var target = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < target.Length; i++) target[i] = i;
        return (input, target);
    }

    private static GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>> GetOptimizer(
        TransformerNERBase<float> model)
    {
        FieldInfo? field = typeof(TransformerNERBase<float>).GetField("_optimizer",
            BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(field);
        return Assert.IsAssignableFrom<GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>>>(field.GetValue(model));
    }

    private sealed class CountingTinyNER : TinyBERTNER<float>
    {
        private readonly ZeroStepDefect? _defect;

        public CountingTinyNER(TransformerNEROptions options,
            IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>? optimizer = null,
            ZeroStepDefect? defect = null)
            : base(CreateArchitecture(), options, optimizer) { _defect = defect; }

        public int TrainingCalls { get; private set; }

        public override void Train(Tensor<float> input, Tensor<float> expected)
        {
            TrainingCalls++;
            if (_defect == ZeroStepDefect.SchedulerDoesNotAdvance) return;
            base.Train(input, expected);
            if (_defect == ZeroStepDefect.ParameterMutation)
            {
                var parameters = GetParameters();
                parameters[0] += 1.0f;
                UpdateParameters(parameters);
            }
        }
    }

    private sealed class BrokenPositiveUpdateFixture : TransformerNERTestBase<float>
    {
        protected override int[] InputShape => new[] { 4, 8 };
        protected override int[] OutputShape => new[] { 4, 9 };

        protected override INeuralNetworkModel<float> CreateNetwork()
        {
            var options = CreateOptions();
            options.WarmupInitialLearningRate = 0.001;
            return new CountingTinyNER(options, defect: ZeroStepDefect.SchedulerDoesNotAdvance);
        }
    }

    private sealed class FixtureProbe : TransformerNERTestBase<float>
    {
        protected override INeuralNetworkModel<float> CreateNetwork() =>
            throw new InvalidOperationException("This probe inspects only explicit supplied models.");

        public void Prepare(INeuralNetworkModel<float> model, Tensor<float> input, Tensor<float> target) =>
            PrepareForGradientFlowInvariant(model, input, target);

        public static TransformerNEROptions Configure(TransformerNEROptions options) => WithPositiveSmokeWarmup(options);
    }
}
