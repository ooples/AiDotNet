using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.FrameInterpolation;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Zhong et al. 2024, Clearer Frames, Anytime (arXiv 2311.08007), and the release's DR-RIFE (zzh-tech/InterpAny-Clearer,
/// models/DI-RIFE: IFNet_sdi_recur.py, refine.py, RIFE_sdi_recur.py, train_sdi_m_mask_recur.py): the configuration and its
/// parameter count, the padding, the uniform index as a time, per-pixel indices, the training inputs, and tape gradients
/// reaching every weight including the privileged teacher (#2397).
/// </summary>
public class InterpAnyClearerPaperTests
{
    private static InterpAnyClearerOptions Small() => new()
    {
        Seed = 1, BlockWidths = new[] { 4, 4, 4 }, TeacherWidth = 4, ContextChannels = 2, WarmupSteps = 0,
    };

    private static InterpAnyClearer<double> Model(InterpAnyClearerOptions options) =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 3), options);

    private static Tensor<double> Random(int[] shape, int seed, double scale = 1.0, double offset = 0.0)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = offset + scale * random.NextDouble();
        return t;
    }

    private static Tensor<double> Concat(params Tensor<double>[] parts)
    {
        int h = parts[0].Shape[1], w = parts[0].Shape[2];
        var result = new Tensor<double>(new[] { parts.Sum(p => p.Shape[0]), h, w });
        int offset = 0;
        foreach (var part in parts)
        {
            for (int i = 0; i < part.Length; i++) result[offset + i] = part[i];
            offset += part.Length;
        }
        return result;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheReleaseDrRifeAndTraining()
    {
        await Task.Yield();
        var o = new InterpAnyClearerOptions();
        Assert.Equal(new[] { 240, 150, 90 }, o.BlockWidths);
        Assert.Equal((90, 16, true, 2), (o.TeacherWidth, o.ContextChannels, o.UseDistillation, o.Iterations));
        Assert.Equal((3e-4, 3e-6, 1e-3, 2000, 302700), (o.LearningRate, o.MinLearningRate, o.WeightDecay, o.WarmupSteps, o.TotalSteps));
        Assert.Equal((0.002, 5), (o.DistillationWeight, o.LaplacianLevels));
    }

    // The release's modules counted layer by layer: the three students 6,830,475 (the index map adds five input channels
    // to each), the teacher 638,690, the Contextnet 294,000 and the U-Net 2,957,875.
    [Theory(Timeout = 240000)]
    [InlineData(true, 10721040)]
    [InlineData(false, 10082350)]
    public async Task ParameterCount_MatchesTheRelease(bool distill, int expected)
    {
        await Task.Yield();
        var model = Model(new InterpAnyClearerOptions { UseDistillation = distill });
        Assert.Equal(expected, model.NetworkParameters.Sum(t => t.Length));
    }

    // inference_img_plus_sdi_recur.interpolate: zeros on the right and bottom to multiples of 32.
    [Fact(Timeout = 60000)]
    public async Task Padding_IsTheReleaseMultiple()
    {
        await Task.Yield();
        Assert.Equal((30, 28), InterpAnyClearer<double>.Padding(100, 130));
        Assert.Equal((0, 0), InterpAnyClearer<double>.Padding(64, 96));
    }

    // A uniform index is the time: the per-pixel entry point with a constant map reproduces Interpolate exactly, and a
    // map that varies over the image changes the frame.
    [Fact(Timeout = 240000)]
    public async Task UniformDistanceMap_IsTheTime()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame0 = Random(new[] { 3, 20, 28 }, 1, 255.0);
        var frame1 = Random(new[] { 3, 20, 28 }, 2, 255.0);
        var uniform = new Tensor<double>(new[] { 20, 28 });
        uniform.Fill(0.3);
        var timed = model.Interpolate(frame0, frame1, 0.3);
        Assert.Equal(new[] { 3, 20, 28 }, timed.Shape.ToArray());
        Assert.Equal(timed.ToArray(), model.InterpolateWithDistanceMap(frame0, frame1, uniform).ToArray());
        var ramp = new Tensor<double>(new[] { 1, 20, 28 });
        for (int y = 0; y < 20; y++)
            for (int x = 0; x < 28; x++)
                ramp[0, y, x] = x / 27.0;
        Assert.False(timed.ToArray().SequenceEqual(model.InterpolateWithDistanceMap(frame0, frame1, ramp).ToArray()));
        Assert.All(timed.ToArray(), v => Assert.True(v >= 0 && v <= 255));
    }

    [Fact(Timeout = 60000)]
    public async Task DistanceMap_IsValidated()
    {
        await Task.Yield();
        var model = Model(Small());
        var frame = Random(new[] { 3, 32, 32 }, 3, 255.0);
        Assert.Throws<ArgumentException>(() => model.InterpolateWithDistanceMap(frame, frame, new Tensor<double>(new[] { 2, 32, 32 })));
        var outside = new Tensor<double>(new[] { 32, 32 });
        outside.Fill(1.5);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.InterpolateWithDistanceMap(frame, frame, outside));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Interpolate(frame, frame, -0.1));
    }

    // A training input is the pair, optionally with the target's index or with the index, the reference's index and the
    // reference frame (the release's sdi_map); other widths and crops off the multiple of 32 are rejected.
    [Fact(Timeout = 240000)]
    public async Task Training_TakesTheReleaseInputs()
    {
        await Task.Yield();
        var model = Model(Small());
        var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var pair = Random(new[] { 6, 32, 32 }, 4, 255.0);
        var target = Random(new[] { 3, 32, 32 }, 5, 255.0);
        var index = Random(new[] { 1, 32, 32 }, 6);
        var reference = Random(new[] { 3, 32, 32 }, 7, 255.0);
        double plain = objective.EvaluateTrainingObjective(pair, target);
        double indexed = objective.EvaluateTrainingObjective(Concat(pair, index), target);
        double referenced = objective.EvaluateTrainingObjective(Concat(pair, index, index, reference), target);
        Assert.All(new[] { plain, indexed, referenced }, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
        Assert.NotEqual(plain, indexed);
        Assert.NotEqual(indexed, referenced);
        // The default sample is the target at 0.5 from frame 0 at 0.
        var half = new Tensor<double>(new[] { 1, 32, 32 });
        half.Fill(0.5);
        var zero = new Tensor<double>(new[] { 1, 32, 32 });
        Assert.Equal(plain, objective.EvaluateTrainingObjective(Concat(pair, half, zero, Slice(pair, 0, 3)), target));
        Assert.Throws<ArgumentException>(() => objective.EvaluateTrainingObjective(Concat(pair, index, index), target));
        Assert.Throws<ArgumentException>(() => model.Train(Random(new[] { 6, 48, 48 }, 8, 255.0), Random(new[] { 3, 48, 48 }, 9, 255.0)));
    }

    private static Tensor<double> Slice(Tensor<double> x, int start, int count)
    {
        int plane = x.Shape[1] * x.Shape[2];
        var result = new Tensor<double>(new[] { count, x.Shape[1], x.Shape[2] });
        for (int i = 0; i < count * plane; i++) result[i] = x[start * plane + i];
        return result;
    }

    // The Laplacian losses and the distillation reach the students, the teacher, the Contextnet and the U-Net; read from
    // the tape. A PReLU slope is exempt only when setting it to 3 leaves the loss unchanged (it never sees a negative).
    [Fact(Timeout = 240000)]
    public async Task Loss_ReachesEveryWeight()
    {
        await Task.Yield();
        var model = Model(Small());
        var parameters = model.NetworkParameters.ToArray();
        var frame0 = Random(new[] { 1, 3, 32, 32 }, 10);
        var frame1 = Random(new[] { 1, 3, 32, 32 }, 11);
        var target = Random(new[] { 1, 3, 32, 32 }, 12);
        var index = new Tensor<double>(new[] { 1, 5, 32, 32 });
        for (int i = 0; i < 32 * 32; i++) index[i] = 0.5;
        for (int i = 0; i < 3 * 32 * 32; i++) index[2 * 32 * 32 + i] = frame0[i];
        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            gradients = tape.ComputeGradients(model.ReleaseLoss(frame0, frame1, index, target), sources: parameters);
        }
        double Loss()
        {
            using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
            return model.ReleaseLoss(frame0, frame1, index, target)[0];
        }
        double loss = Loss();
        bool IsSlope(int i) => parameters[i].ToArray().All(v => v == 0.25);
        bool LossIgnores(int i)
        {
            var saved = parameters[i].ToArray();
            for (int k = 0; k < saved.Length; k++) parameters[i][k] = 3.0;
            bool ignores = Loss() == loss;
            for (int k = 0; k < saved.Length; k++) parameters[i][k] = saved[k];
            return ignores;
        }
        var unreached = Enumerable.Range(0, parameters.Length)
            .Where(i => !gradients.TryGetValue(parameters[i], out var g) || g.ToArray().All(v => v == 0.0)).ToArray();
        var unexplained = unreached.Where(i => !(IsSlope(i) && LossIgnores(i))).ToArray();
        Assert.True(unexplained.Length == 0, $"{unexplained.Length} of {parameters.Length} weight tensors received no gradient: "
            + string.Join(", ", unexplained.Select(i => $"{i} [{string.Join("x", parameters[i].Shape.ToArray())}]")));
        int slopes = Enumerable.Range(0, parameters.Length).Count(IsSlope);
        Assert.True(unreached.Length < slopes / 4, $"{unreached.Length} of {slopes} PReLU slopes never see a negative input");
    }

    [Fact(Timeout = 240000)]
    public async Task TrainingStep_UpdatesTheNetwork()
    {
        await Task.Yield();
        var model = Model(Small());
        var before = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        model.Train(Random(new[] { 6, 32, 32 }, 13, 255.0), Random(new[] { 3, 32, 32 }, 14, 255.0));
        var after = model.NetworkParameters.SelectMany(t => t.ToArray()).ToArray();
        Assert.False(before.SequenceEqual(after));
        Assert.All(after, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }
}
