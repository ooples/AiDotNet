using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.SpeechRecognition.AlibabaASR;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// SenseVoice-Large (FunAudioLLM 2024): the Whisper-large geometry, task selection by decoder input tokens, and
/// teacher-forced training on the task outputs and transcript (#2397).
/// </summary>
public class SenseVoiceLargePaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static SenseVoiceLargeOptions Tiny() => new()
    {
        NumMels = 80, EncoderDim = 16, NumEncoderLayers = 1, NumDecoderLayers = 1, NumAttentionHeads = 2, VocabSize = 16,
        MaxTextLength = 8, Seed = 5,
    };

    private static Tensor<double> Voice(int samples, double pitch)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) audio[i] = 0.4 * Math.Sin(2 * Math.PI * pitch * i / 16000.0);
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreTheWhisperLargeGeometry()
    {
        await Task.Yield();
        var o = new SenseVoiceLargeOptions();
        Assert.Equal(128, o.NumMels);
        Assert.Equal(1280, o.EncoderDim);
        Assert.Equal(20, o.NumAttentionHeads);
        Assert.Equal(32, o.NumEncoderLayers);
        Assert.Equal(32, o.NumDecoderLayers);
        Assert.True(o.PredictLanguage);
        Assert.False(o.PredictEmotion);
        Assert.False(o.PredictEvents);
    }

    [Fact(Timeout = 60000)]
    public async Task Prompt_CarriesTheSelectedTaskTokens()
    {
        await Task.Yield();
        var options = Tiny();
        options.PredictEmotion = true;
        options.PredictEvents = true;
        using var model = new SenseVoiceLarge<double>(Architecture(), options);
        // Start of transcript, then LID, SER and AED, at the top of the vocabulary under end of text.
        Assert.Equal(new[] { 14, 13, 12, 11 }, model.Prompt());
        Assert.Equal(15, model.EndOfText);
        options.PredictLanguage = false;
        options.PredictEmotion = false;
        options.PredictEvents = false;
        Assert.Equal(new[] { 14 }, model.Prompt());
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheTeacherForcedLoss()
    {
        await Task.Yield();
        using var model = new SenseVoiceLarge<double>(Architecture(), Tiny());
        var audio = Voice(8000, 180);
        // A language token, then the transcript.
        var target = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 8, 3, 5, 7 }));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        model.SetBaseTrainOptimizer(new AdamOptimizer<double, Tensor<double>, Tensor<double>>(model,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 }));
        var trace = new System.Collections.Generic.List<double>();
        for (int step = 0; step < 30; step++) { model.Train(audio, target); trace.Add(objective.EvaluateTrainingObjective(audio, target)); }
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}: {string.Join(", ", trace.Select(x => x.ToString("F3")))}");
        // Greedy decoding reproduces the memorized outputs (rendered as code points without a vocabulary).
        Assert.Equal("\u0008\u0003\u0005\u0007", model.Transcribe(audio).Text);
    }
}
