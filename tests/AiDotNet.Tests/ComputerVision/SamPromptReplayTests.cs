using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

[Collection("EngineCurrentGlobalState")]
public class SamPromptReplayTests : IDisposable
{
    private readonly IEngine _previousEngine = AiDotNetEngine.Current;
    public SamPromptReplayTests() => AiDotNetEngine.Current = new CpuEngine();
    public void Dispose() => AiDotNetEngine.Current = _previousEngine;

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void IgnoredSlotsMatchRaggedPromptsAndKeepLearnedNotAPoint(bool hasBox, bool allIgnored)
    {
        using var fixture = new TinySam();
        var net = fixture.Network;
        var embedding = net.EncodeImage(net.Preprocess(Image(0.3)));
        var positional = net.DensePositional();
        var box = hasBox ? new Tensor<double>(new[] { 1.0, 1.0, 6.0, 6.0 }, [1, 4]) : null;
        var points = allIgnored ? null : new Tensor<double>(new[] { 2.0, 3.0, double.NaN, double.NaN }, [1, 2, 2]);
        var labels = allIgnored ? null : new Tensor<double>(new[] { 1.0, -1.0 }, [1, 2]);
        var paddedPoints = new Tensor<double>(new[] { 2.0, 3.0, double.NaN, double.PositiveInfinity,
            double.NaN, double.NaN, double.NegativeInfinity, double.NaN }, [1, 4, 2]);
        var paddedLabels = new Tensor<double>(allIgnored ? new[] { -2.0, -2.0, -2.0, -2.0 }
            : new[] { 1.0, -2.0, -1.0, -2.0 }, [1, 4]);
        var (sparse, dense) = net.EncodePrompts(points, labels, box, null);
        var (paddedSparse, paddedDense) = net.EncodePrompts(paddedPoints, paddedLabels, box, null);
        foreach (bool multimask in new[] { false, true })
        {
            var expected = net.DecodeMasks(embedding, positional, sparse, dense, multimask);
            var actual = net.DecodeMasks(embedding, positional, paddedSparse, paddedDense, multimask,
                net.SparsePromptValidity(paddedLabels, hasBox));
            Close(expected.Logits.ToArray(), actual.Logits.ToArray());
            Close(expected.Iou.ToArray(), actual.Iou.ToArray());
        }
        // -1 is a real learned token, not the ignored-slot sentinel.
        var validity = net.SparsePromptValidity(new Tensor<double>(new[] { -1.0, -2.0 }, [1, 2]), hasBox)!.ToArray();
        Assert.Equal(1.0, validity[0]);
        Assert.Equal(0.0, validity[1]);
    }

    [Fact]
    public void ErrorPointPlanRebindsErrorsAndDrawsAndRepresentsAnEmptyRegion()
    {
        var engine = AiDotNetEngine.Current;
        var error = new Tensor<double>(new[] { 0.0, 1.0, 0.0, -1.0 }, [4]);
        var draw = new Tensor<double>(new[] { 0.0 }, [1]);
        CompiledInferencePlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var point = SamTraining<double>.SampleErrorPoint(engine, error, draw, width: 2);
            plan = scope.CompileInference(point, new[] { error, draw });
        }
        using (plan)
        {
            Assert.Equal(new[] { 1.0, 0.0, 1.0 }, plan.Execute().ToArray());
            plan.SetInputs(new[] { error, new Tensor<double>(new[] { 0.75 }, [1]) });
            Assert.Equal(new[] { 1.0, 1.0, 0.0 }, plan.Execute().ToArray());
            plan.SetInputs(new[] { new Tensor<double>(new[] { -1.0, 0.0, 0.0, 0.0 }, [4]), draw });
            Assert.Equal(new[] { 0.0, 0.0, 0.0 }, plan.Execute().ToArray());
            plan.SetInputs(new[] { new Tensor<double>([4]), draw });
            Assert.Equal(new[] { 0.0, 0.0, -2.0 }, plan.Execute().ToArray());
        }
    }

    [Fact]
    public void IouUsesTheMaskUnionAndDefinesTheEmptyUnionAsOne()
    {
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<double>(new[] { 1.0, 1.0, -1.0, -1.0 }, [1, 1, 2, 2]);
        var truth = new Tensor<double>(new[] { 1.0, 0.0, 1.0, 0.0 }, [1, 1, 2, 2]);
        Assert.Equal(1.0 / 3.0, SamTraining<double>.ActualIou(engine, logits, truth)[0], 12);
        Assert.Equal(1.0, SamTraining<double>.ActualIou(engine,
            Tensor<double>.CreateDefault([1, 1, 2, 2], -1.0), new Tensor<double>([1, 1, 2, 2]))[0]);
    }

    [Fact]
    public void DoubleIouTraceDoesNotMaterializePredicatesAndReplaysChangedMasks()
    {
        var engine = AiDotNetEngine.Current;
        var logits = new Tensor<double>(new[] { 1.0, 1.0, -1.0, -1.0 }, [1, 1, 2, 2]);
        var truth = new Tensor<double>(new[] { 1.0, 0.0, 1.0, 0.0 }, [1, 1, 2, 2]);
        CompiledInferencePlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var iou = SamTraining<double>.ActualIou(engine, logits, truth);
            Assert.Null(scope.HostReadOp);
            plan = scope.CompileInference(iou, new[] { logits, truth });
        }
        using (plan)
        {
            Assert.Equal(1.0 / 3.0, plan.Execute()[0], 12);
            plan.SetInputs(new[] { Tensor<double>.CreateDefault([1, 1, 2, 2], -1.0), new Tensor<double>([1, 1, 2, 2]) });
            Assert.Equal(1.0, plan.Execute()[0]);
            // SetInputs refills the original leaves in place; use fresh truth data after the empty-mask replay.
            plan.SetInputs(new[] { Tensor<double>.CreateDefault([1, 1, 2, 2], 1.0),
                new Tensor<double>(new[] { 1.0, 0.0, 1.0, 0.0 }, [1, 1, 2, 2]) });
            Assert.Equal(0.5, plan.Execute()[0]);
        }
    }

    [Fact]
    public void HalfIouCountsWithoutOverflowingTheImageArea()
    {
        var ones = Tensor<Half>.CreateDefault([1, 1, 256, 256], (Half)1);
        Assert.Equal((Half)1, SamTraining<Half>.ActualIou(AiDotNetEngine.Current, ones, ones)[0]);
    }

    [Fact]
    public void InteractiveObjectiveReplaysCurrentDataAndBackpropagatesThroughTheImageEncoder()
    {
        using var fixture = new TinySam();
        var engine = AiDotNetEngine.Current;
        var training = new SamTraining<double>(engine, fixture.Network, 20.0, 1.0, 2.0, 0.25);
        var random = new CountingRandom(7);
        var objective = training.Objective(Image(0.3), Mask(false), random);
        int drawsAfterConstruction = random.Draws;
        var parameters = TapeTrainingStep<double>.CollectParameters(fixture.Layers).ToArray();
        var encoderWeight = fixture.Network.PatchProjection.GetTrainableParameters().Single(p => p.Rank == 4);
        int encoderIndex = Array.IndexOf(parameters, encoderWeight);
        Assert.True(encoderIndex >= 0);
        CompiledTrainingPlan<double> plan;
        ClosureObjectiveGuard<double>? guard;
        using (ClosureObjectiveGuard<double>.BeginRecording())
        {
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                var loss = objective();
                Assert.Null(scope.HostReadOp);
                plan = scope.CompileTraining(parameters, loss);
            }
            guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
        }
        using (plan)
        {
            double expected = objective()[0];
            Close(new[] { expected }, plan.Step().ToArray());
            var gradient = plan.Gradients[encoderIndex].ToArray();
            Assert.All(gradient, value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
            Assert.Contains(gradient, value => value != 0.0);
            Assert.Equal(drawsAfterConstruction, random.Draws);

            var next = training.Objective(Image(0.7), Mask(true), new Random(7));
            Assert.Equal(ClosureGuardResult.Matched, guard!.TryRebind(next, parameters, out _));
            double nextExpected = next()[0];
            Close(new[] { nextExpected }, plan.Step().ToArray());
            Assert.NotEqual(expected, nextExpected);
        }
    }

    private static Tensor<double> Image(double scale) => new(
        Enumerable.Range(0, 3 * 8 * 8).Select(i => scale * Math.Sin(i * 0.13)).ToArray(), [1, 3, 8, 8]);

    private static Tensor<double> Mask(bool shifted) => new(
        Enumerable.Range(0, 64).Select(i => ((i % 8 >= (shifted ? 3 : 1)) && i / 8 < 5) ? 1.0 : 0.0).ToArray(), [64]);

    private static void Close(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1.0, Math.Abs(expected[i])),
                $"[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private sealed class CountingRandom(int seed) : Random(seed)
    {
        public int Draws { get; private set; }
        public override int Next(int maxValue) { Draws++; return base.Next(maxValue); }
        public override double NextDouble() { Draws++; return base.NextDouble(); }
    }

    private sealed class TinySam : IDisposable
    {
        public List<LayerBase<double>> Layers { get; } = new();
        public SamNetwork<double> Network { get; }
        public TinySam()
        {
            Network = new SamNetwork<double>(AiDotNetEngine.Current, Layers, new SamConfiguration
            {
                ImageSize = 8, PatchSize = 4, EncoderDim = 8, EncoderDepth = 1, EncoderHeads = 2,
                WindowSize = 2, GlobalAttention = new[] { 0 }, OutChannels = 8, MaskInChannels = 4,
                DecoderDepth = 2, DecoderHeads = 2, DecoderMlp = 16, IouHidden = 8,
            });
            Network.InitializeLikeReference(new Random(17));
        }
        public void Dispose() { foreach (var layer in Layers) layer.Dispose(); }
    }
}
