using System.Reflection;
using AiDotNet.ComputerVision.Detection;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

public sealed class DetrSemanticTrainingModelTests
{
    public DetrSemanticTrainingModelTests() => TestModuleInitializer.EnsureInitialized();

    public enum StepMutation { NoUpdate, DoubleUpdate, RawMse }

    [Theory(Timeout = 180000)]
    [InlineData(false, 0.0)]
    [InlineData(true, 0.0)]
    [InlineData(false, 0.7)]
    [InlineData(true, 0.7)]
    public async Task ActualDetr_UsesOneSharedUpdateAndRestoresTrainingMode(bool emptyTargets, double boxLogit)
    {
        await Task.Yield();
        using var model = new ObservedDetr();
        ObjectDetectionTestBase<double>.VerifyDetrSemanticStep(model, emptyTargets, boxLogit: boxLogit);
        Assert.Equal(new[] { false, false, true, true }, model.ForwardModes);
        Assert.False(model.TrainingMode);
        int previous = model.ForwardModes.Count;
        model.TrainDetections(new Tensor<double>(new[] { 1, 3, 64, 64 }), EmptyBatch());
        Assert.Equal(previous + 1, model.ForwardModes.Count); // Successful warmup is cached.
        Assert.False(model.TrainingMode);
    }

    [Fact(Timeout = 180000)]
    public async Task ActualFloatDetr_UpdatesBothHeadsWithTheSameObjective()
    {
        await Task.Yield();
        using var model = new DETR<float>(new ObjectDetectionOptions<float>
        {
            InputSize = new[] { 64, 64 }, Size = ModelSize.Nano, NumClasses = 2
        });
        ObjectDetectionTestBase<float>.VerifyDetrSemanticStep(model, emptyTargets: false);
    }

    [Fact(Timeout = 180000)]
    public async Task Facade_UsesTheCallerSelectedModelAndTypedTargets()
    {
        await Task.Yield();
        using var first = new ObservedDetr();
        using var selected = new ObservedDetr();
        IAiModelBuilder<double, Tensor<double>, Tensor<double>> builder =
            new AiModelBuilder<double, Tensor<double>, Tensor<double>>().ConfigureModel(first).ConfigureModel(selected);
        ObjectDetectionTestBase<double>.VerifyDetrSemanticStep(selected, emptyTargets: false,
            (input, targets) => Assert.Same(builder, builder.TrainDetections(input, targets)));
        Assert.Empty(first.ForwardModes);
        Assert.Equal(0, first.GetLastLoss());
        Assert.Equal(4, selected.ForwardModes.Count);
    }

    [Fact(Timeout = 180000)]
    public async Task Facade_CocoAdapterConvertsToTheSameSemanticUpdate()
    {
        await Task.Yield();
        using var model = new ObservedDetr();
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>().ConfigureModel(model);
        ObjectDetectionTestBase<double>.VerifyDetrSemanticStep(model, emptyTargets: false, (input, targets) =>
        {
            var target = Assert.Single(targets[0]);
            var coco = new Tensor<double>(new[]
            {
                (double)target.ClassId, target.CenterX - target.Width / 2,
                target.CenterY - target.Height / 2, target.Width, target.Height,
                0.0, 0, 0, 0, 0
            }, new[] { 1, 2, 5 });
            Assert.Same(builder, builder.TrainCocoDetections(input, coco));
        });
    }

    [Theory(Timeout = 180000)]
    [InlineData(StepMutation.NoUpdate)]
    [InlineData(StepMutation.DoubleUpdate)]
    [InlineData(StepMutation.RawMse)]
    public async Task SharedInvariant_RejectsWrongActualTrainingRoutes(StepMutation mutation)
    {
        await Task.Yield();
        using var model = new ObservedDetr();
        bool mutationApplied = false;
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() =>
            ObjectDetectionTestBase<double>.VerifyDetrSemanticStep(model, emptyTargets: false, (input, targets) =>
            {
                mutationApplied = true;
                switch (mutation)
                {
                    case StepMutation.NoUpdate: break;
                    case StepMutation.DoubleUpdate:
                        model.TrainDetections(input, targets);
                        model.TrainDetections(input, targets);
                        break;
                    case StepMutation.RawMse:
                        model.Train(input, model.Predict(input));
                        break;
                    default: throw new ArgumentOutOfRangeException(nameof(mutation));
                }
            }));
        Assert.True(mutationApplied); // Construction/registry failures cannot count as mutant proof.
    }

    [Fact]
    public void InvalidBatchAndExcessTargets_FailBeforeForwardOrTrainingMutation()
    {
        using var model = new ObservedDetr();
        var input = new Tensor<double>(new[] { 1, 3, 64, 64 });
        var tooMany = new DetectionTrainingBatch<double>(new[]
        {
            Enumerable.Range(0, 51).Select(_ => new DetectionTrainingTarget<double>(0, 0.5, 0.5, 0.2, 0.2))
        });
        Assert.Throws<ArgumentException>(() => model.TrainDetections(input, tooMany));
        Assert.Throws<ArgumentException>(() => model.TrainDetections(input,
            new DetectionTrainingBatch<double>(new[] { Array.Empty<DetectionTrainingTarget<double>>(), Array.Empty<DetectionTrainingTarget<double>>() })));
        Assert.Throws<ArgumentException>(() => model.TrainDetections(input,
            new DetectionTrainingBatch<double>(new[] { new[] { new DetectionTrainingTarget<double>(2, 0.5, 0.5, 0.2, 0.2) } })));
        Assert.Empty(model.ForwardModes);
        Assert.False(model.TrainingMode);
        Assert.Equal(0, model.GetLastLoss());
    }

    [Fact]
    public void EveryDetectorFamilyImplementsItsPublishedDetectionObjective()
    {
        foreach (Type family in new[]
                 {
                     typeof(DETR<double>), typeof(RTDETR<double>), typeof(DINO<double>),
                     typeof(YOLOv8<double>), typeof(YOLOv9<double>), typeof(YOLOv10<double>), typeof(YOLOv11<double>),
                     typeof(FasterRCNN<double>), typeof(CascadeRCNN<double>)
                 })
            Assert.True(typeof(IDetectionTrainingModel<double>).IsAssignableFrom(family), family.Name);
    }

    [Fact(Timeout = 180000)]
    public async Task FacadeRejectsANonDetectionModelBeforeAnyParameterMutation()
    {
        await Task.Yield();
        // Semantic detection training never falls back to raw regression: a model without the capability is refused.
        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            inputType: AiDotNet.Enums.InputType.OneDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        using var model = new AiDotNet.NeuralNetworks.FeedForwardNeuralNetwork<double>(architecture);
        Assert.False(model is IDetectionTrainingModel<double>);
        using (model.Predict(new Tensor<double>(new[] { 1, 4 }))) { }
        var before = model.GetParameters().ToArray();
        Assert.NotEmpty(before);

        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>().ConfigureModel(model);
        using var input = new Tensor<double>(new[] { 1, 3, 64, 64 });
        using var coco = new Tensor<double>(new[] { 1, 1, 5 });
        Assert.Throws<NotSupportedException>(() => builder.TrainDetections(input, EmptyBatch()));
        Assert.Throws<NotSupportedException>(() => builder.TrainCocoDetections(input, coco));
        Assert.Equal(before, model.GetParameters().ToArray());
    }

    [Fact(Timeout = 180000)]
    public async Task RawTrain_PreservesMseAndItsUnambiguousNullContract()
    {
        await Task.Yield();
        using var model = new ObservedDetr();
        var input = new Tensor<double>(new[] { 1, 3, 64, 64 });
        Action<Tensor<double>, Tensor<double>> rawTrain = model.Train;
        var exception = Assert.Throws<TargetInvocationException>(() => rawTrain.DynamicInvoke(input, null));
        Assert.Equal("expectedOutput", Assert.IsType<ArgumentNullException>(exception.InnerException).ParamName);
        Assert.Empty(model.ForwardModes);
        model.Predict(input);
        foreach (var chunk in model.GetParameterStateChunks().Where(chunk => chunk.Role == ParameterSlotRole.Trainable))
            chunk.Tensor.Fill(0);
        Assert.Single(model.GetParameterStateChunks(), chunk => chunk.Role == ParameterSlotRole.Trainable
            && chunk.Tensor.Rank == 1 && chunk.Tensor.Length == 4).Tensor.Fill(0.5);
        rawTrain(input, new Tensor<double>(new[] { 1, 350 }));
        Assert.Equal(1.0 / 7, model.GetLastLoss(), 12); // 200 RAW box outputs at .5, 150 logits at0.
        Assert.False(model.TrainingMode);
    }

    private static DetectionTrainingBatch<double> EmptyBatch() => new(new[] { Array.Empty<DetectionTrainingTarget<double>>() });

    private sealed class ObservedDetr : DETR<double>
    {
        internal ObservedDetr() : base(ObjectDetectionPositiveFixture<double>.CreateOptions()) { }
        internal List<bool> ForwardModes { get; } = new();
        internal bool TrainingMode => IsTrainingMode;
        protected override List<Tensor<double>> Forward(Tensor<double> input)
        {
            ForwardModes.Add(IsTrainingMode);
            return base.Forward(input); // Observe the real numerical path; do not replace its outputs.
        }
    }
}
