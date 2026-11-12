using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.Video;

/// <summary>
/// Zhao et al. 2025, RealisVSR (arXiv 2507.19138), Sec. 3-4: the Wan 2.1-1.3B configuration, the Consistency-Preserved
/// ControlNet's initialization and training scope, the High-Frequency Rectified Diffusion Loss's wavelet and HOG terms, and
/// the sampling pipeline (#2397).
/// </summary>
public class RealisVSRPaperTests
{
    private const int Low = 8;

    private static RealisVSR<double> Small() =>
        new(new NeuralNetworkArchitecture<double>(inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputFrames: 2, inputDepth: 3, inputHeight: Low, inputWidth: Low, outputSize: 3 * 16 * Low * Low),
            new RealisVSROptions
            {
                Seed = 1, ModelDim = 24, FfnDim = 32, NumHeads = 2, NumLayers = 2, ControlNetLayers = 1, FrequencyDim = 8, TextDim = 8,
                TextLength = 4, VaeDim = 4, VaeDimMultipliers = new[] { 1, 1, 1, 1 }, VaeResidualBlocks = 1, ScaleFactor = 4,
                NumFrames = 2, SamplingSteps = 2, HogCellSize = 2, HogBlockSize = 2,
            });

    private static Tensor<double> Pixels(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = 255.0 * random.NextDouble();
        return t;
    }

    private static Tensor<double> Gaussian(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var random = new System.Random(seed);
        for (int i = 0; i < t.Length; i++) t[i] = random.NextDouble() * 2.0 - 1.0;
        return t;
    }

    private static Tensor<double> Input() => Pixels(new[] { 2, 3, Low, Low }, 3);

    private static Tensor<double> Target() => Pixels(new[] { 2, 3, 4 * Low, 4 * Low }, 4);

    private static List<double[]> Snapshot(IReadOnlyList<Tensor<double>> parameters)
        => parameters.Select(p => Enumerable.Range(0, p.Length).Select(i => p[i]).ToArray()).ToList();

    private static bool Changed(List<double[]> before, IReadOnlyList<Tensor<double>> after)
        => before.Select((values, k) => values.Where((v, i) => v != after[k][i]).Any()).Any(changed => changed);

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePaperConfiguration()
    {
        await Task.Yield();
        var o = new RealisVSROptions();
        Assert.Equal((1536, 8960, 12, 30, 256), (o.ModelDim, o.FfnDim, o.NumHeads, o.NumLayers, o.FrequencyDim));
        Assert.Equal((4096, 512, 1e-6), (o.TextDim, o.TextLength, o.NormEpsilon));
        Assert.Equal((96, 2), (o.VaeDim, o.VaeResidualBlocks));
        Assert.Equal(new[] { 1, 2, 4, 4 }, o.VaeDimMultipliers);
        Assert.Equal((6, 1.0), (o.ControlNetLayers, o.ConditionScale));
        Assert.Equal(new[] { 1.0, 2.0, 2.0, 2.0 }, o.WaveletWeights);
        Assert.Equal(9, o.HogBins);
        Assert.Equal((1e-5, 1e-4), (o.LearningRate, o.WeightDecay));
        Assert.Equal((4, 5.0, 1000), (o.ScaleFactor, o.SigmaShift, o.NumTrainTimesteps));
    }

    // Eq. 4 on one 2x2 patch: the Haar bands of the error are (a + b + c + d)/2, (−a + b − c + d)/2, (a + b − c − d)/2 and
    // (a − b − c + d)/2 up to sign, each squared and weighted {1, 2, 2, 2}.
    [Fact(Timeout = 60000)]
    public async Task WaveletLoss_IsTheWeightedHaarBandError()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var prediction = new Tensor<double>(new[] { 1, 1, 1, 2, 2 }, new Vector<double>(new[] { 1.0, 4.0, -2.0, 3.0 }));
        var target = new Tensor<double>(new[] { 1, 1, 1, 2, 2 }, new Vector<double>(new[] { 0.5, 1.0, 1.0, -1.0 }));
        double a = 0.5, b = 3.0, c = -3.0, d = 4.0;
        double ll = (a + b + c + d) / 2, lh = (-a + b - c + d) / 2, hl = (a + b - c - d) / 2, hh = (a - b - c + d) / 2;
        double expected = ll * ll + 2 * (lh * lh + hl * hl + hh * hh);
        var loss = RealisLoss.Wavelet(engine, prediction, target, new[] { 1.0, 2.0, 2.0, 2.0 });
        Assert.Equal(expected, loss[0], 10);
    }

    // Eq. 5's HOG descriptor is L2-Hys normalized per block, so it sees the gradient structure and not its scale: a scaled
    // copy of the target costs nothing, a different pattern does.
    [Fact(Timeout = 60000)]
    public async Task HogLoss_ComparesNormalizedGradientStructure()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var target = Gaussian(new[] { 1, 2, 1, 8, 8 }, 5);
        var scaled = engine.TensorMultiplyScalar(target, 3.0);
        var other = Gaussian(new[] { 1, 2, 1, 8, 8 }, 6);
        Assert.Equal(0.0, RealisLoss.Hog(engine, target, target, 9, 2, 2)[0], 12);
        Assert.True(RealisLoss.Hog(engine, scaled, target, 9, 2, 2)[0] < 1e-8);
        Assert.True(RealisLoss.Hog(engine, other, target, 9, 2, 2)[0] > 1e-3);
        // A plane smaller than one block contributes nothing.
        Assert.Equal(0.0, RealisLoss.Hog(engine, other, target, 9, 8, 3)[0]);
    }

    // The ControlNet convention: the branch's blocks start as copies of the main network's first blocks.
    [Fact(Timeout = 300000)]
    public async Task ControlNet_StartsAsTheMainNetworksFirstBlocks()
    {
        await Task.Yield();
        var model = Small();
        model.Predict(Input());
        var branch = model.BranchBlockParameters;
        var main = model.LeadingMainBlockParameters;
        Assert.NotEmpty(branch);
        Assert.Equal(main.Count, branch.Count);
        for (int k = 0; k < main.Count; k++) Assert.Equal(main[k].ToArray(), branch[k].ToArray());
    }

    // Sec. 3.3: only the ControlNet branch trains; Wan and its prompt stay frozen.
    [Fact(Timeout = 600000)]
    public async Task Training_UpdatesOnlyTheControlNet()
    {
        await Task.Yield();
        var model = Small();
        var (input, target) = (Input(), Target());
        model.Predict(input);
        var (backbone, branch) = (Snapshot(model.BackboneParameters), Snapshot(model.ControlNetParameters));
        model.Train(input, target);
        Assert.True(Changed(branch, model.ControlNetParameters));
        Assert.False(Changed(backbone, model.BackboneParameters));
    }

    [Fact(Timeout = 300000)]
    public async Task Predict_ReturnsTheUpscaledClip()
    {
        await Task.Yield();
        var output = Small().Predict(Input());
        Assert.Equal(new[] { 2, 3, 4 * Low, 4 * Low }, output.Shape.ToArray());
        for (int i = 0; i < output.Length; i++) Assert.True(output[i] >= 0 && output[i] <= 255, $"value {output[i]} at {i}");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheObjective()
    {
        await Task.Yield();
        var model = Small();
        var (input, target) = (Input(), Target());
        var provider = (ITrainingObjectiveProvider<double>)model;
        double before = provider.EvaluateTrainingObjective(input, target);
        for (int step = 0; step < 10; step++) model.Train(input, target);
        double after = provider.EvaluateTrainingObjective(input, target);
        Assert.True(after < before, $"objective {before} -> {after}");
    }
}
