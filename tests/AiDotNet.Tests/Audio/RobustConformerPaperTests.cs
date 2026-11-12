using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Robust;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Wang et al. 2021's noise-robust wav2vec 2.0: the BASE encoder, the reconstruction module of section 2.2, the
/// continual pre-training loss of Equation 4, and CTC fine-tuning (#2397).
/// </summary>
public class RobustConformerPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 3200, outputSize: 16);

    private static RobustConformerOptions Tiny() => new()
    {
        MaxAudioLengthSeconds = 1, EncoderDim = 32, NumEncoderLayers = 1, NumAttentionHeads = 4, FeedForwardDim = 64, ReconstructionHidden = 8,
        VocabSize = 16, MaxTextLength = 8, DropoutRate = 0, ActivationDropout = 0, FeatureProjectionDropout = 0, FinalDropout = 0,
        LayerDrop = 0, MaskTimeProbability = 0, Seed = 5,
    };

    private static Tensor<double> Voice(int samples, double pitch, double noise = 0, int seed = 1)
    {
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var audio = new Tensor<double>(new[] { 1, samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[0, i] = 0.3 * Math.Sin(2 * Math.PI * pitch * t) + 0.15 * Math.Sin(2 * Math.PI * 3 * pitch * t) * Math.Sin(2 * Math.PI * 4 * t)
                + noise * (2 * random.NextDouble() - 1);
        }
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreSectionThreeTwosBaseModel()
    {
        await Task.Yield();
        var o = new RobustConformerOptions();
        Assert.Equal((768, 12, 12, 3072), (o.EncoderDim, o.NumEncoderLayers, o.NumAttentionHeads, o.FeedForwardDim));
        Assert.Equal((0.1, 0.1), (o.DiversityWeight, o.ReconstructionWeight));
        var c = o.ToConfiguration();
        // BASE: group-normalized feature encoder and a post-norm Transformer.
        Assert.False(c.LayerNormFeatureEncoder);
        Assert.False(c.StableLayerNorm);
    }

    [Fact(Timeout = 120000)]
    public async Task ReconstructionModule_IsTwoBlstmLayersAndAMirroredCnnDecoder()
    {
        await Task.Yield();
        using var model = new RobustConformer<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        Assert.Equal(2, network.Bottleneck.Count);
        var c = network.Stack.Network.Configuration;
        Assert.Equal(c.ConvDims.Length, network.Decoder.Count);
        var noisy = Voice(3200, 180, noise: 0.1);
        var hidden = network.Stack.Network.Forward(noisy, training: false, Tensors.Helpers.RandomHelper.CreateSeededRandom(1)).Hidden;
        // The prediction has the input's length.
        Assert.Equal(new[] { 1, 3200 }, network.Reconstruct(hidden, 3200).Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task PretrainingLoss_IsWav2Vec2sPlusTenthOfTheReconstructionL1()
    {
        await Task.Yield();
        using var model = new RobustConformer<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var noisy = Voice(3200, 180, noise: 0.1);
        var clean = Voice(3200, 180);
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(3);
        var encoder = network.Stack.Network;
        var mask = encoder.DrawMask(1, encoder.Frames(3200), 0.5, random);
        var negatives = network.Pretraining.SampleNegatives(mask, random);
        var (total, reconstruction) = network.ContinualPretrainingLoss(noisy, clean, training: false, Tensors.Helpers.RandomHelper.CreateSeededRandom(4), mask, negatives);

        var output = encoder.Forward(noisy, training: false, Tensors.Helpers.RandomHelper.CreateSeededRandom(4), mask: mask);
        double wav2vec2 = network.Pretraining.Loss(output, negatives, training: false, Tensors.Helpers.RandomHelper.CreateSeededRandom(4)).Total[0];
        var predicted = network.Reconstruct(output.Hidden, 3200);
        double l1 = Enumerable.Range(0, 3200).Average(i => Math.Abs(predicted[0, i] - clean[0, i]));
        Assert.Equal(l1, reconstruction[0], 9);
        Assert.Equal(wav2vec2 + 0.1 * l1, total[0], 9);
    }

    [Fact(Timeout = 300000)]
    public async Task ContinualPretraining_LearnsToReconstructTheCleanSpeech()
    {
        await Task.Yield();
        using var model = new RobustConformer<double>(Architecture(), Tiny());
        var noisy = Voice(3200, 180, noise: 0.1);
        var clean = Voice(3200, 180);
        model.ContinualPretrain(noisy, clean);
        double first = model.LastReconstructionLoss;
        for (int step = 0; step < 30; step++) model.ContinualPretrain(noisy, clean);
        Assert.True(model.LastReconstructionLoss < 0.8 * first, $"L_r {first} -> {model.LastReconstructionLoss}");
    }

    [Fact(Timeout = 300000)]
    public async Task FineTuning_LowersTheCtcLoss_AndLeavesTheReconstructionModule()
    {
        await Task.Yield();
        using var model = new RobustConformer<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Voice(3200, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 2 }));
        var decoder = network.Decoder[0].GetParameters().ToArray();
        var objective = (Interfaces.ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"CTC loss {before} -> {after}");
        Assert.Equal(decoder, network.Decoder[0].GetParameters().ToArray());
        Assert.Equal(new[] { 9, 16 }, model.Predict(new Tensor<double>(new[] { 3200 }, new Vector<double>(audio.ToArray()))).Shape.ToArray());
    }
}
