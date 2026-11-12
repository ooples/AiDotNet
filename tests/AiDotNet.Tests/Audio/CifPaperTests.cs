using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.CTCVariants;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The CIF-based recognizer of Dong and Xu (2020): the paper's Mandarin configuration, the encoder's 1/8 frame rate and
/// training on Equation 1's loss (#2397).
/// </summary>
public class CifPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static CIFDecoderOptions Tiny() => new()
    {
        NumMels = 16, NumFilters = 8, EncoderDim = 16, NumAttentionHeads = 2, FeedForwardDim = 32, NumLayersPerResolution = 1,
        NumDecoderLayers = 1, VocabSize = 16, MaxTextLength = 8, MaxAudioLengthSeconds = 1, Seed = 5,
    };

    private static Tensor<double> Voice(int samples, double pitch)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * pitch * t) + 0.2 * Math.Sin(2 * Math.PI * 2 * pitch * t);
        }
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreThePapersMandarinModel()
    {
        await Task.Yield();
        var o = new CIFDecoderOptions();
        // Section 4: h = 4, d = 640, dff = 2560, n = 5, two decoder SANs, window 3, λ1 = 0.5, λ2 = 1, smoothing 0.2.
        Assert.Equal(40, o.NumMels);
        Assert.Equal(64, o.NumFilters);
        Assert.Equal(640, o.EncoderDim);
        Assert.Equal(4, o.NumAttentionHeads);
        Assert.Equal(2560, o.FeedForwardDim);
        Assert.Equal(5, o.NumLayersPerResolution);
        Assert.Equal(3, o.NumResolutions);
        Assert.Equal(3, o.PredictorWindow);
        Assert.Equal(2, o.NumDecoderLayers);
        Assert.Equal(5230, o.VocabSize);
        Assert.Equal(0.5, o.CtcWeight);
        Assert.Equal(1.0, o.QuantityWeight);
        Assert.Equal(0.2, o.LabelSmoothing);
        Assert.Equal(0.5, o.TailThreshold);
    }

    [Fact(Timeout = 60000)]
    public async Task Encoder_RunsAtAnEighthOfTheFrameRate()
    {
        await Task.Yield();
        using var model = new CIFDecoder<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var features = network.Features(Voice(16000, 180));
        int frames = features.Shape[2];
        Assert.Equal(98, frames);
        // The strided convolution halves 98 frames to 49; the two poolings give 25, then 13.
        Assert.Equal(13, network.Encode(features).Shape[0]);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersThePapersLoss()
    {
        await Task.Yield();
        using var model = new CIFDecoder<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
        var target = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-3 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(x => x.ToString("F3")))}");
        // The predictor learns to count the three labels and <EOS>.
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var alphas = network.Alphas(network.Encode(network.Features(audio)));
        double count = 0;
        for (int i = 0; i < alphas.Length; i++) count += alphas.GetFlat(i);
        Assert.InRange(count, 3.0, 5.0);
    }
}
