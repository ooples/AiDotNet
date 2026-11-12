using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Classification;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Hershey et al. 2017's audio ResNet-50 (sections 3.1 and 3.3.5): 96 x 64 log-mel patches of 960 ms, a stride-1 7x7
/// stem, 6 x 4 final activations, and a sigmoid unit per label (#2397).
/// </summary>
public class AudioEventDetectorPaperTests
{
    private static readonly string[] Labels = { "Speech", "Music", "Dog", "Siren" };

    private static AudioEventDetector<double> Small(int patchFrames = 32, int mels = 16, int[]? depths = null) =>
        new(new NeuralNetworkArchitecture<double>(InputType.TwoDimensional, NeuralNetworkTaskType.MultiLabelClassification,
                inputHeight: patchFrames, inputWidth: mels, outputSize: Labels.Length),
            new AudioEventDetectorOptions
            {
                NumMels = mels, PatchFrames = patchFrames, BaseChannels = 4, StageDepths = depths ?? new[] { 1, 1, 1, 1 },
                CustomLabels = Labels, Seed = 5,
            });

    private static Tensor<double> Patch(int frames, int mels, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var patch = new Tensor<double>(new[] { frames, mels });
        for (int i = 0; i < patch.Length; i++) patch[i] = rng.NextDouble() - 0.5;
        return patch;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreSectionThree()
    {
        await Task.Yield();
        var o = new AudioEventDetectorOptions();
        Assert.Equal((16000, 400, 160, 64), (o.SampleRate, o.FftSize, o.HopLength, o.NumMels));
        Assert.Equal((0.96, 0.0, 96), (o.WindowSize, o.WindowOverlap, o.PatchFrames));
        Assert.Equal(64, o.BaseChannels);
        Assert.Equal(new[] { 3, 4, 6, 3 }, o.StageDepths);
    }

    [Fact(Timeout = 120000)]
    public async Task Network_IsResNet50WithAStrideOneStem_EndingInSixByFourActivations()
    {
        await Task.Yield();
        // ResNet-50's depth over the paper's 96 x 64 patch, at 4 base channels to keep the test fast.
        using var model = Small(96, 64, new[] { 3, 4, 6, 3 });
        var stem = Assert.IsType<ConvolutionalLayer<double>>(model.Layers[0]);
        Assert.Equal((7, 1), (stem.KernelSize, stem.Stride));
        Assert.Equal(16, model.Layers.OfType<BottleneckBlock<double>>().Count());
        var trunk = model.TrunkActivations(Patch(96, 64, 1));
        // 96 x 64 -> 48 x 32 after the stem pooling -> 6 x 4 after three stride-2 stages; 4 << 3 = 32 base channels in
        // the last stage, expanded four times.
        Assert.Equal(new[] { 1, 128, 6, 4 }, trunk.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task Output_IsAnIndependentSigmoidPerLabel()
    {
        await Task.Yield();
        using var model = Small();
        var probabilities = model.Predict(Patch(32, 16, 2));
        Assert.Equal(new[] { 4 }, probabilities.Shape.ToArray());
        Assert.All(Enumerable.Range(0, 4), k => Assert.InRange(probabilities[k], 0.0, 1.0));
        // Several labels can be present at once: nothing ties the scores to a sum of one.
        var batch = new Tensor<double>(new[] { 2, 32, 16 });
        var a = Patch(32, 16, 2);
        var b = Patch(32, 16, 3);
        for (int i = 0; i < a.Length; i++) { batch[i] = a[i]; batch[a.Length + i] = b[i]; }
        var both = model.Predict(batch);
        Assert.Equal(new[] { 2, 4 }, both.Shape.ToArray());
        for (int k = 0; k < 4; k++) Assert.Equal(probabilities[k], both[0, k], 9);
    }

    [Fact(Timeout = 120000)]
    public async Task Detection_ClassifiesNonOverlapping960MsWindows()
    {
        await Task.Yield();
        using var model = new AudioEventDetector<double>(new NeuralNetworkArchitecture<double>(InputType.TwoDimensional,
                NeuralNetworkTaskType.MultiLabelClassification, inputHeight: 96, inputWidth: 64, outputSize: Labels.Length),
            new AudioEventDetectorOptions { BaseChannels = 4, StageDepths = new[] { 1, 1, 1, 1 }, CustomLabels = Labels, Seed = 5 });
        var audio = new Tensor<double>(new[] { 3 * 15360 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.3 * Math.Sin(2 * Math.PI * 440 * i / 16000.0);
        // 3 x 960 ms of audio: three windows, each one 96 x 64 patch.
        var probabilities = model.GetEventProbabilities(audio);
        Assert.Equal(new[] { 3, 4 }, probabilities.Shape.ToArray());
        Assert.Equal(0.96, model.TimeResolution, 12);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_WithSigmoidCrossEntropy_LearnsMultipleLabelsAtOnce()
    {
        await Task.Yield();
        using var model = Small();
        var inputs = new Tensor<double>(new[] { 2, 32, 16 });
        var a = Patch(32, 16, 4);
        var b = Patch(32, 16, 5);
        for (int i = 0; i < a.Length; i++) { inputs[i] = a[i]; inputs[a.Length + i] = b[i]; }
        // The first patch has Speech and Dog, the second Music and Siren.
        var targets = new Tensor<double>(new[] { 2, 4 }, new Vector<double>(new[] { 1.0, 0, 1, 0, 0, 1, 0, 1 }));
        for (int step = 0; step < 60; step++) model.Train(inputs, targets);
        var output = model.Predict(inputs);
        for (int i = 0; i < 8; i++)
            Assert.True(targets[i] > 0.5 ? output[i] > 0.5 : output[i] < 0.5, $"output {i} = {output[i]} for target {targets[i]}");
    }
}
