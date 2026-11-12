using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.LLMIntegrated;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Seed-ASR (Bai et al. 2024, §3.2–3.4): LUISE's 32 blocks at 40 ms, the converter's four-frame splicing to 160 ms, the
/// context placed before the speech, the loss over the transcript only, and SFT's fixed LLM (#2397).
/// </summary>
public class SeedASRPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 16000, outputSize: 16);

    private static SeedASROptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, ConvKernel = 3, LlmDim = 16, LlmFeedForwardDim = 32,
        NumLlmLayers = 1, NumLlmAttentionHeads = 2, NumLlmKvHeads = 1, VocabSize = 16, MaxTextLength = 8, Seed = 5,
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

    private static double[] Samples(Tensor<double> audio) => Enumerable.Range(0, audio.Length).Select(i => audio[i]).ToArray();

    private static Tensor<double> Transcript(params double[] ids) => new(new[] { ids.Length }, new Vector<double>(ids));

    [Fact(Timeout = 60000)]
    public async Task Defaults_FollowThePaper()
    {
        await Task.Yield();
        var o = new SeedASROptions();
        Assert.Equal((80, 32, 4, SpeechLlmTrainingScope.AllParameters), (o.NumMels, o.NumEncoderLayers, o.SpliceFrames, o.TrainingScope));
    }

    [Fact(Timeout = 120000)]
    public async Task Converter_SplicesFourFortyMillisecondFrames()
    {
        await Task.Yield();
        using var model = new SeedASR<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var audio = Samples(Voice(16000, 180));
        var encoded = network.Encoder.Encode(network.Encoder.Features(audio), training: false);
        // 4x subsampling of 10 ms frames: about 25 frames a second.
        Assert.InRange(encoded.Shape[0], 23, 25);
        var speech = network.Speech(audio, training: false);
        Assert.Equal(new[] { encoded.Shape[0] / 4, 16 }, speech.Shape.ToArray());
        // Token r is the converter's projection of frames 4r..4r+3 laid side by side.
        var spliced = new Tensor<double>(new[] { 1, 64 });
        for (int f = 0; f < 4; f++)
            for (int j = 0; j < 16; j++) spliced[0, f * 16 + j] = encoded[4 + f, j];
        var expected = network.Converter.Forward(spliced);
        for (int j = 0; j < 16; j++) Assert.Equal(expected[0, j], speech[1, j], 9);
    }

    [Fact(Timeout = 120000)]
    public async Task Context_PrecedesTheSpeech()
    {
        await Task.Yield();
        using var model = new SeedASR<double>(Architecture(), Tiny());
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var speech = network.Speech(Samples(Voice(16000, 180)), training: false);
        var context = new[] { 4, 9 };
        var prompt = network.Prompt(speech, context);
        Assert.Equal(context.Length + speech.Shape[0], prompt.Shape[0]);
        var embedded = network.TokenEmbeddings(context);
        for (int j = 0; j < 16; j++)
        {
            Assert.Equal(embedded[1, j], prompt[1, j], 12);
            Assert.Equal(speech[0, j], prompt[2, j], 12);
        }
        Assert.Equal(speech.Shape[0], network.Prompt(speech, Array.Empty<int>()).Shape[0]);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = new SeedASR<double>(Architecture(), Tiny());
        var audio = Voice(16000, 180);
        var target = Transcript(3, 5, 7);
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new List<double>();
        for (int step = 0; step < 40; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(t => t.ToString("F3")))}");
        Assert.Equal("\u0003\u0005\u0007", model.Transcribe(audio).Text);
    }

    [Fact(Timeout = 300000)]
    public async Task SftScope_KeepsTheLlmFixed()
    {
        await Task.Yield();
        var options = Tiny();
        options.TrainingScope = SpeechLlmTrainingScope.EncoderAdapterAndLora;
        using var model = new SeedASR<double>(Architecture(), options);
        var network = model.Network ?? throw new InvalidOperationException("native model");
        var query = network.Llm.Layers[0].Query.GetParameters().ToArray();
        var converter = network.Converter.GetParameters().ToArray();
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        model.Train(Voice(16000, 180), Transcript(3, 5, 7));
        Assert.Equal(query, network.Llm.Layers[0].Query.GetParameters().ToArray());
        Assert.NotEqual(converter, network.Converter.GetParameters().ToArray());
    }
}
