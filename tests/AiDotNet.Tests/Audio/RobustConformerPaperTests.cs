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
        var losses = network.Pretraining.Loss(output, negatives, training: false, Tensors.Helpers.RandomHelper.CreateSeededRandom(4));
        double wav2vec2 = losses.Total[0];
        Assert.True(losses.Masked > 1, $"masked steps {losses.Masked}");
        var predicted = network.Reconstruct(output.Hidden, 3200);
        double l1 = Enumerable.Range(0, 3200).Average(i => Math.Abs(predicted[0, i] - clean[0, i]));
        Assert.Equal(l1, reconstruction[0], 9);
        // wav2vec 2.0's terms are summed over the masked steps (fairseq's criterion), and so is λ2 L_r.
        Assert.Equal(wav2vec2 + 0.1 * losses.Masked * l1, total[0], 9);
    }

    // Why two tests rather than one "continual pre-training lowers L_r by 20%": the paper continues from wav2vec_small,
    // whose contrastive objective has converged and whose gradients are small. From random weights the wav2vec 2.0 terms
    // swamp L_r at the shared encoder: measured gradient norms were 8.2 against 0.001 at the first step on a 0.2 s clip
    // (2.4 against 0.11 by step 31), and 55 against 0.006 on a 2 s clip (108 against 1.4 by step 31), so 31 steps of
    // Equation 4 from scratch lower L_r by only 16-20%. That is the paper's objective, not a defect. The tests check the two
    // claims that hold at this scale: masked reconstruction learns, and the λ2 L_r term is what makes it learn inside
    // Equation 4.
    //
    // Both measure L_r in evaluation mode under one fixed, seeded mask before and after training. Each training step
    // draws its own mask, so a step's training-mode L_r is a random variable that swings by tens of percent with the mask
    // alone; comparing two such draws says little about learning.

    private static double FixedMaskReconstructionLoss(NoiseRobustWav2Vec2Network<double> network, Tensor<double> noisy, Tensor<double> clean)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
        var encoder = network.Stack.Network;
        var mask = encoder.DrawPretrainingMask(1, encoder.Frames(noisy.Shape[1]), network.Pretraining.Configuration.MaskProbability, random);
        var negatives = network.Pretraining.SampleNegatives(mask, random);
        return network.ContinualPretrainingLoss(noisy, clean, training: false, random, mask, negatives).Reconstruction[0];
    }

    /// <summary>
    /// Masked reconstruction learns: with the input masked as in pre-training, 31 Adam steps on L_r alone (Equation 3, the
    /// reconstruction term of the network's continual pre-training loss) lower the fixed-mask L_r by at least 20%. The clip
    /// is 1 s (16000 samples, 49 frames), long enough that the spans leave much of the input visible.
    /// </summary>
    [Fact(Timeout = 300000)]
    public async Task MaskedReconstruction_LearnsTheCleanSpeech()
    {
        await Task.Yield();
        const int samples = 16000;
        var options = Tiny();
        options.MaxAudioLengthSeconds = 2;
        using var model = new RobustConformer<double>(
            new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: samples, outputSize: 16), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var encoder = network.Stack.Network;
        var noisy = Voice(samples, 180, noise: 0.1);
        var clean = Voice(samples, 180);
        double first = FixedMaskReconstructionLoss(network, noisy, clean);

        // Adam at 1e-3 (the model's default rate) on every parameter, driven by the reconstruction term alone.
        var parameters = network.Layers.SelectMany(l => l.GetTrainableParameters()).Distinct().ToList();
        var moments = parameters.ToDictionary(p => p, p => (M: new double[p.Length], V: new double[p.Length]));
        var random = Tensors.Helpers.RandomHelper.CreateSeededRandom(6);
        for (int step = 1; step <= 31; step++)
        {
            var mask = encoder.DrawPretrainingMask(1, encoder.Frames(samples), network.Pretraining.Configuration.MaskProbability, random);
            var negatives = network.Pretraining.SampleNegatives(mask, random);
            System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> gradients;
            using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
            {
                var reconstruction = network.ContinualPretrainingLoss(noisy, clean, training: true, random, mask, negatives).Reconstruction;
                gradients = tape.ComputeGradients(reconstruction, parameters.ToArray());
            }
            foreach (var p in parameters)
            {
                if (!gradients.TryGetValue(p, out var g)) continue;
                var gv = g.ToArray();
                var pv = p.AsWritableSpan();
                var (m, v) = moments[p];
                for (int i = 0; i < gv.Length; i++)
                {
                    m[i] = 0.9 * m[i] + 0.1 * gv[i];
                    v[i] = 0.999 * v[i] + 0.001 * gv[i] * gv[i];
                    pv[i] -= 1e-3 * (m[i] / (1 - Math.Pow(0.9, step))) / (Math.Sqrt(v[i] / (1 - Math.Pow(0.999, step))) + 1e-8);
                }
            }
        }
        double last = FixedMaskReconstructionLoss(network, noisy, clean);
        Assert.True(last < 0.8 * first, $"L_r {first} -> {last}");
    }

    /// <summary>
    /// The paper's term contributes: two identically seeded models trained by <c>ContinualPretrain</c> on Equation 4 for
    /// the same 31 steps, one at λ2 = 0.1 and one at λ2 = 0, end with a lower fixed-mask L_r at λ2 = 0.1.
    /// </summary>
    [Theory(Timeout = 300000)]
    [InlineData(5)]
    [InlineData(11)]
    public async Task ContinualPretraining_ReconstructionTermLowersTheReconstructionLoss(int seed)
    {
        await Task.Yield();
        var noisy = Voice(3200, 180, noise: 0.1);
        var clean = Voice(3200, 180);
        double Trained(double weight)
        {
            var options = Tiny();
            options.Seed = seed;
            options.ReconstructionWeight = weight;
            using var model = new RobustConformer<double>(Architecture(), options);
            var network = model.Network ?? throw new InvalidOperationException("native model");
            for (int step = 0; step < 31; step++) model.ContinualPretrain(noisy, clean);
            return FixedMaskReconstructionLoss(network, noisy, clean);
        }
        double with = Trained(0.1), without = Trained(0.0);
        Assert.True(with < without, $"L_r with λ2 = 0.1: {with}; with λ2 = 0: {without}");
    }

    [Fact(Timeout = 120000)]
    public async Task ContinualPretraining_WithLayerDrop_StepsOnTheEagerTape()
    {
        await Task.Yield();
        // LayerDrop > 0 trains on the eager tape (a skipped layer is not stepped), and the step hands its reconstruction
        // loss out through a retained output, which needs the AdamW update to report whether it was applied.
        var options = Tiny();
        options.LayerDrop = 0.1;
        using var model = new RobustConformer<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var noisy = Voice(3200, 180, noise: 0.1);
        var clean = Voice(3200, 180);
        var decoder = network.Decoder[^1].GetParameters().ToArray();
        model.ContinualPretrain(noisy, clean);
        Assert.True(model.LastReconstructionLoss > 0 && !double.IsNaN(model.LastReconstructionLoss), $"L_r {model.LastReconstructionLoss}");
        Assert.NotEqual(decoder, network.Decoder[^1].GetParameters().ToArray());
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
