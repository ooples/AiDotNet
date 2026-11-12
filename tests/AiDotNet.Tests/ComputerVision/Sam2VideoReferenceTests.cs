using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

/// <summary>
/// SAM 2's video model against Hugging Face's port of the official code (tools/reference-data/sam2_video_reference.py):
/// tracking through six frames from a click, a mask and an empty mask (prompted frames decoded alone, the object propagated
/// forward and backward through the memory encoder, memory attention and object pointers), as SAM 2 and as SAM 2.1; and the
/// training objective (a click and a corrective click, tracked frames with and without the object) before and after one SGD
/// step.
/// </summary>
public class Sam2VideoReferenceTests
{
    private const int Size = 64;

    private static readonly Lazy<JsonElement> Data = new(() =>
    {
        const string name = "sam2_video_reference.json";
        string relative = Path.Combine("ComputerVision", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Values(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static void AssertClose(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < tolerance, $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    internal static Sam2Configuration SmallConfiguration(bool sam21) => new()
    {
        ImageSize = Size, BlocksPerStage = new[] { 1, 1, 2, 1 }, EmbedDims = new[] { 8, 16, 32, 64 }, HeadsPerStage = new[] { 1, 2, 4, 8 },
        WindowSizes = new[] { 4, 4, 4, 2 }, GlobalAttentionBlocks = new[] { 3 }, Dim = 128, MaskInChannels = 8, DecoderHeads = 2,
        DecoderMlp = 64, IouHidden = 16,
        Memory = (sam21 ? Sam2MemoryConfiguration.Sam21 : new Sam2MemoryConfiguration()) with
        {
            MemoryDim = 64, MaskMemories = 3, AttentionLayers = 2, AttentionFeedForward = 64, FuserHidden = 64, MaxObjectPointers = 4, Dropout = 0.0,
        },
    };

    private static (Sam2Network<double> Network, List<LayerBase<double>> Layers) Build(bool sam21)
    {
        var k = Data.Value;
        var layers = new List<LayerBase<double>>();
        var net = new Sam2Network<double>(AiDotNetEngine.Current, layers, SmallConfiguration(sam21));
        var state = k.GetProperty("state");
        var extra = k.GetProperty("state21_extra");
        net.LoadTorchWeights((name, shape) =>
        {
            var v = Values(state.TryGetProperty(name, out var e) ? e : extra.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        return (net, layers);
    }

    private static Tensor<double> Frame(double[] frames, int t)
        => new(new[] { 1, 3, Size, Size }, new Vector<double>(frames.Skip(t * 3 * Size * Size).Take(3 * Size * Size).ToArray()));

    private static Sam2Prompt<double> Click(double x, double y)
    {
        var points = new Tensor<double>(new[] { 1, 1, 2 }, new Vector<double>(new[] { x, y }));
        var labels = new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { 1.0 }));
        return new Sam2Prompt<double>(points, labels, null);
    }

    [Theory(Timeout = 600000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Tracking_MatchesTheOfficialCode(bool sam21)
    {
        await Task.Yield();
        var k = Data.Value;
        var (net, _) = Build(sam21);
        var frames = Values(k.GetProperty("frames"));
        var mask = new Tensor<double>(new[] { 1, 1, Size, Size }, new Vector<double>(Values(k.GetProperty("mask"))));
        var prompts = new Dictionary<int, Sam2Prompt<double>>
        {
            [1] = Click(30, 25),
            [3] = new(null, null, mask),
            [5] = new(null, null, new Tensor<double>(new[] { 1, 1, Size, Size })),
        };
        var tracker = new Sam2VideoTracker<double>(AiDotNetEngine.Current, net);
        Sam2FrameOutput<double>[] outputs;
        using (new NoGradScope<double>())
            outputs = tracker.Propagate(6, t => net.EncodeFeatures(Frame(frames, t)), prompts);

        var expected = k.GetProperty("variants").GetProperty(sam21 ? "sam21" : "sam2");
        var expectedFrames = expected.GetProperty("frames").EnumerateArray().ToArray();
        for (int t = 0; t < 6; t++)
        {
            AssertClose(Values(expectedFrames[t].GetProperty("logits")), outputs[t].LowRes, 1e-9, $"frame {t} logits");
            double score = Values(expectedFrames[t].GetProperty("object_score"))[0];
            Assert.True(Math.Abs(score - outputs[t].ObjectScore) < 1e-9, $"frame {t} object score: {score} vs {outputs[t].ObjectScore}");
        }
        var detail = expected.GetProperty("detail");
        foreach (int t in new[] { 1, 2 })
        {
            AssertClose(Values(detail.GetProperty($"{t}").GetProperty("memory")), outputs[t].Memory!, 1e-9, $"frame {t} memory");
            AssertClose(Values(detail.GetProperty($"{t}").GetProperty("pointer")), outputs[t].Pointer, 1e-9, $"frame {t} pointer");
        }
    }

    [Fact(Timeout = 600000)]
    public async Task TrainingObjective_MatchesTheOfficialLoss_AndAfterOneSgdStep()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var k = Data.Value;
        var training = k.GetProperty("training");
        var (net, layers) = Build(false);
        net.Training = true;
        var frames = Values(k.GetProperty("frames"));
        var images = Enumerable.Range(0, 3).Select(t => Frame(frames, t)).ToList();
        var allMasks = Values(training.GetProperty("masks"));
        var masks = Enumerable.Range(0, 3).Select(t => allMasks.Skip(t * Size * Size).Take(Size * Size).ToArray()).ToList();
        var point = Values(training.GetProperty("point"));
        var click = Values(training.GetProperty("click"));
        var objective = new Sam2VideoTraining<double>(engine, net, new Sam2TrainingSettings { CorrectionPoints = 1 });
        var plan = new Sam2VideoTraining<double>.Plan(new[] { 0, 1, 2 }, new List<int> { 0 },
            new Dictionary<int, Sam2Prompt<double>> { [0] = Click(point[0], point[1]) }, new HashSet<int> { 0 }, false);
        Tensor<double> Loss() => objective.Loss(images, masks, plan, new Random(0), (_, _) => (click[0], click[1], (int)click[2]));

        var parameters = AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers);
        Tensor<double> loss;
        Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            loss = Loss();
            grads = tape.ComputeGradients(loss, sources: parameters);
        }
        double expected = training.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(expected - loss[0]) < 1e-9, $"loss: {expected} vs {loss[0]}");
        foreach (var p in parameters)
        {
            if (!grads.TryGetValue(p, out var g)) continue;
            for (int i = 0; i < p.Length; i++) p[i] -= 0.01 * g[i];
            engine.InvalidatePersistentTensor(p);
        }
        double after;
        using (new NoGradScope<double>())
            after = Loss()[0];
        double expectedAfter = training.GetProperty("loss_after_sgd").GetDouble();
        Assert.True(Math.Abs(expectedAfter - after) < 1e-9, $"loss after SGD: {expectedAfter} vs {after}");
    }
}
