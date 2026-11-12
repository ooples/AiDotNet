using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Chatterbox Turbo and Nano: the release's sizes, the reference loudness normalization, the three silence tokens,
/// and the mean-flow decoder's distillation from its teacher flow.</summary>
public class ChatterboxTurboTests
{
    private static Chatterbox<double> Model(ChatterboxOptions options) =>
        new(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    private static ChatterboxOptions TinyTurbo()
    {
        var options = ChatterboxFixture.Options();
        options.Variant = ChatterboxVariant.Turbo;
        options.FlowTimesteps = 2;
        return options;
    }

    [Fact]
    public void Turbo_UsesTheReleaseConfiguration()
    {
        var turbo = ChatterboxOptions.Turbo();
        Assert.Equal((ChatterboxVariant.Turbo, 50276, 6563, 1024, 24, 16), (turbo.Variant, turbo.VocabSize, turbo.SpeechVocabularySize, turbo.LLMDim, turbo.NumLLMLayers, turbo.NumHeads));
        Assert.Equal((375, 15, 1000, 0.95, 0.0, 2), (turbo.SpeechConditionPromptTokens, turbo.EncoderReferenceSeconds, turbo.GenerationTopK, turbo.GenerationTopP,
            turbo.ClassifierFreeGuidanceWeight, turbo.FlowTimesteps));
        Assert.Equal(-27.0, turbo.ReferenceLoudness);
        var nano = ChatterboxOptions.Turbo(nano: true);
        Assert.Equal((ChatterboxVariant.Nano, 768, 12, 12), (nano.Variant, nano.LLMDim, nano.NumLLMLayers, nano.NumHeads));
        Assert.Null(new ChatterboxOptions().ReferenceLoudness);
        Assert.Equal(0, new ChatterboxOptions().GenerationTopK);
    }

    [Fact]
    public void ReferenceLoudness_NormalizesTheReferenceRecording()
    {
        var recording = ChatterboxFixture.Recording<double>(6);
        var plain = Model(ChatterboxFixture.Options()).CreateVoice(recording, 24000).ReferenceAudio!;
        Assert.Equal(recording.ToArray(), plain.ToArray());

        var options = ChatterboxFixture.Options();
        options.ReferenceLoudness = -27;
        var normalized = Model(options).CreateVoice(recording, 24000).ReferenceAudio!;
        double loudness = AiDotNet.Audio.Effects.LoudnessMeter.IntegratedLoudness(normalized.ToArray(), 24000);
        Assert.True(Math.Abs(loudness - (-27)) < 1e-6, $"normalized to {loudness} LUFS");
    }

    [Fact]
    public void Turbo_RefusesAReferenceOfFiveSecondsOrLess() =>
        Assert.Throws<ArgumentException>(() => Model(TinyTurbo()).CreateVoice(ChatterboxFixture.Recording<double>(5), 24000));

    [Fact]
    public void DistillationTeacherSteps_SetTheTeachersIntegral()
    {
        var model = Model(TinyTurbo());
        var flow = model.FlowModel!;
        var tokens = Enumerable.Range(0, 6).Select(i => (i * 37) % 50).ToArray();
        int frames = flow.MelFrames(tokens.Length);
        var mel = new Tensor<double>(new[] { 1, flow.Configuration.OutputSize, frames });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Sin(0.37 * i);
        var xvector = new Tensor<double>(new[] { flow.Configuration.SpeakerDim });
        for (int i = 0; i < xvector.Length; i++) xvector[i] = Math.Cos(0.5 * i);
        double Loss(int steps) => flow.DistillationLoss(model.TeacherFlow, tokens, mel, xvector, false, new Random(11), steps)[0];
        Assert.Equal(Loss(4), Loss(4));
        Assert.NotEqual(Loss(1), Loss(4));
        Assert.Throws<InvalidOperationException>(() => flow.TrainingLoss(tokens, mel, xvector, false, new Random(11)));
    }
}
