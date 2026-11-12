using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// YourTTS trains and synthesizes as its paper specifies (Casanova et al. 2022): VITS conditioned on d-vectors from a
/// frozen H/ASP speaker encoder — of a reference recording at synthesis (zero-shot) — with language embeddings for
/// multilingual training and the speaker consistency loss −α·cos(φ(g), φ(h)).
/// </summary>
/// <remarks>
/// Before this change YourTTS ran a stack of generic layers once and trained it by regression; it had no speaker
/// encoder, no external speaker conditioning, no language embedding and none of VITS's components.
/// </remarks>
public class YourTTSPaperTests
{
    private static YourTTSOptions Options(bool scl = false, int languages = 0) => new()
    {
        VocabSize = 32, HiddenDim = 16, InterChannels = 8, FilterChannels = 32, NumHeads = 2, NumEncoderLayers = 2,
        DropoutRate = 0.0, PosteriorLayers = 2, FlowLayers = 2, NumFlowSteps = 2, DurationPredictorDropout = 0.0,
        DurationPredictorFlows = 2, UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
        ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], DiscriminatorWidthDivisor = 32,
        FftSize = 64, WindowSize = 64, HopSize = 16, SampleRate = 4000, MelChannels = 8, SegmentSize = 128,
        SpeakerEncoderDim = 8, SpeakerEncoderFilters = [8, 8, 8, 8], UseSpeakerConsistencyLoss = scl,
        NumLanguages = languages, LanguageEmbeddingDim = 4,
    };

    private static YourTTS<double> CreateModel(YourTTSOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 3, outputSize: 64) { RandomSeed = 11 },
        options,
        new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null));

    private static (Tensor<double> Tokens, Tensor<double> Audio) Example(double pitch = 200)
    {
        var tokens = new Tensor<double>(new[] { 3 });
        for (int i = 0; i < 3; i++) tokens[i] = 4 + i * 7;
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < 512; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * (pitch + i / 4.0) * i / 4000.0);
        return (tokens, audio);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_ReducesTheVitsObjective_WithTheUtterancesOwnDVector()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var (tokens, audio) = Example();
        double before = provider.EvaluateTrainingObjective(tokens, audio);
        for (int i = 0; i < 25; i++) model.Train(tokens, audio);
        double after = provider.EvaluateTrainingObjective(tokens, audio);
        Assert.True(after < before, $"Objective did not fall ({before} -> {after}).");
    }

    [Fact(Timeout = 120000)]
    public async Task ZeroShot_SpeaksInTheVoiceOfTheReferenceRecording()
    {
        await Task.Yield();
        var model = CreateModel(Options());
        var (tokens, low) = Example(150);
        var (_, high) = Example(600);
        Assert.Throws<InvalidOperationException>(() => model.Predict(tokens));
        model.Voice = new TtsVoice<double> { ReferenceAudio = low };
        var first = model.Predict(tokens);
        Assert.Equal(first.ToVector().ToArray(), model.Predict(tokens).ToVector().ToArray());
        model.Voice = new TtsVoice<double> { ReferenceAudio = high };
        var second = model.Predict(tokens);
        int shared = Math.Min(first.Length, second.Length);
        Assert.Contains(Enumerable.Range(0, shared), i => Math.Abs(first[i] - second[i]) > 1e-12);
    }

    [Fact(Timeout = 300000)]
    public async Task SpeakerConsistencyLoss_AddsMinusAlphaTimesACosine()
    {
        await Task.Yield();
        var plain = CreateModel(Options());
        var withScl = CreateModel(Options(scl: true));
        withScl.SetParameters(plain.GetParameters());
        var (tokens, audio) = Example();
        double without = ((AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)plain).EvaluateTrainingObjective(tokens, audio);
        double with = ((AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)withScl).EvaluateTrainingObjective(tokens, audio);
        double cosine = (without - with) / 9.0;
        Assert.InRange(cosine, -1.0 - 1e-9, 1.0 + 1e-9);
        Assert.NotEqual(0.0, cosine);

        // Training with the loss follows its gradient into the generated audio without touching the frozen encoder.
        var encoderBefore = withScl.ComputeSpeakerEmbedding(audio).ToVector().ToArray();
        withScl.Train(tokens, audio);
        Assert.Equal(encoderBefore, withScl.ComputeSpeakerEmbedding(audio).ToVector().ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Multilingual_TrainsOnNamedLanguages_AndTheLanguageChangesTheSpeech()
    {
        await Task.Yield();
        var model = CreateModel(Options(languages: 2));
        var (tokens, audio) = Example();
        Assert.Throws<NotSupportedException>(() => model.Train(tokens, audio));
        Assert.Throws<ArgumentException>(() => model.Train(new TtsTrainingSample<double> { Tokens = tokens, Audio = audio }));
        double loss = model.Train(new TtsTrainingSample<double> { Tokens = tokens, Audio = audio, LanguageId = 1 });
        Assert.False(double.IsNaN(loss));

        model.Voice = new TtsVoice<double> { ReferenceAudio = audio, LanguageId = 0 };
        var first = model.Predict(tokens);
        model.Voice = new TtsVoice<double> { ReferenceAudio = audio, LanguageId = 1 };
        var second = model.Predict(tokens);
        int shared = Math.Min(first.Length, second.Length);
        Assert.Contains(Enumerable.Range(0, shared), i => Math.Abs(first[i] - second[i]) > 1e-12);
    }

    [Fact(Timeout = 60000)]
    public async Task Resampling_MatchesTorchaudiosSincInterpolation_OnASine()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var x = new Tensor<double>(new[] { 400 });
        for (int i = 0; i < 400; i++) x[i] = Math.Sin(2 * Math.PI * 300 * i / 4000.0);
        var y = HaspSpeakerEncoder<double>.Resample(engine, x, 4000, 16000);
        Assert.Equal(1600, y.Length);
        // Away from the zero-padded ends the band-limited interpolation reproduces the sine at the new rate.
        for (int i = 200; i < 1400; i++)
            Assert.True(Math.Abs(Math.Sin(2 * Math.PI * 300 * i / 16000.0) - y[i]) < 5e-3, $"Sample {i}: {y[i]}.");
    }

    [Fact(Timeout = 120000)]
    public async Task SpeakerEncoder_LoadsACoquiStateDictionary_AndRejectsAnIncompleteOne()
    {
        await Task.Yield();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var encoder = new HaspSpeakerEncoder<double>(engine, 8, new[] { 8, 8, 8, 8 });
        Assert.Throws<ArgumentException>(() => encoder.LoadState(new Dictionary<string, Tensor<double>>
        {
            ["conv1.weight"] = new Tensor<double>(new[] { 8, 1, 3, 3 }),
        }));
        Assert.Throws<ArgumentException>(() => encoder.LoadState(new Dictionary<string, Tensor<double>>
        {
            ["not.a.parameter"] = new Tensor<double>(new[] { 1 }),
        }));

        // A d-vector is the mean of normalized window embeddings: its norm is at most one.
        var audio = new Tensor<double>(new[] { 3200 });
        for (int i = 0; i < audio.Length; i++) audio[i] = Math.Sin(0.05 * i) + 0.1 * Math.Sin(0.31 * i);
        var dvector = encoder.DVector(audio);
        double norm = Math.Sqrt(dvector.ToVector().ToArray().Sum(v => v * v));
        Assert.InRange(norm, 0.0, 1.0 + 1e-9);
        var embedding = encoder.Embed(audio);
        Assert.Equal(1.0, Math.Sqrt(embedding.ToVector().ToArray().Sum(v => v * v)), 9);
    }
}
