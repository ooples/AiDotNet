using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.AlibabaASR;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// SenseVoice-Small (An et al. 2024; the release's <c>SenseVoiceSmall</c>): the SAN-M encoder over four task queries and
/// the stacked filterbanks, the CTC head after the queries, and training on the CTC plus rich-token loss (#2397).
/// </summary>
public class SenseVoicePaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 8000, outputSize: 16);

    private static SenseVoiceOptions Tiny() => new()
    {
        NumMels = 16, EncoderDim = 16, NumEncoderLayers = 1, NumTpEncoderLayers = 1, NumAttentionHeads = 2, FeedForwardDim = 32,
        FsmnKernelSize = 3, VocabSize = 16, MaxTextLength = 8, MaxAudioLengthSeconds = 1, LearningRate = 1e-3, Seed = 7,
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
    public async Task Defaults_AreTheRelease()
    {
        await Task.Yield();
        var o = new SenseVoiceOptions();
        Assert.Equal(80, o.NumMels);
        Assert.Equal(7, o.LfrM);
        Assert.Equal(6, o.LfrN);
        Assert.Equal(512, o.EncoderDim);
        Assert.Equal(4, o.NumAttentionHeads);
        Assert.Equal(2048, o.FeedForwardDim);
        Assert.Equal(50, o.NumEncoderLayers);
        Assert.Equal(20, o.NumTpEncoderLayers);
        Assert.Equal(11, o.FsmnKernelSize);
        Assert.Equal(25055, o.VocabSize);
        Assert.Equal("auto", o.Language);
        Assert.False(o.UseInverseTextNormalization);
    }

    [Fact(Timeout = 120000)]
    public async Task Output_IsTheCtcLogitsAfterTheFourQueries()
    {
        await Task.Yield();
        using var model = new SenseVoice<double>(Architecture(), Tiny());
        // 0.5 s: 48 filterbank frames, stacked every six into 8; the four query positions are not transcript frames.
        var logits = model.Predict(Voice(8000, 180));
        Assert.Equal(new[] { 8, 16 }, logits.Shape.ToArray());
        Assert.Equal(4, model.RichTokenIds(Voice(8000, 180)).Length);
        // The text-normalization query reaches the encoder, so it changes the transcript logits.
        var options = Tiny();
        options.UseInverseTextNormalization = true;
        using var withItn = new SenseVoice<double>(Architecture(), options);
        withItn.Predict(Voice(8000, 180));                                                  // materializes the lazy layers
        withItn.SetParameters(model.GetParameters());
        var other = withItn.Predict(Voice(8000, 180));
        Assert.Contains(Enumerable.Range(0, logits.Length), i => Math.Abs(logits.GetFlat(i) - other.GetFlat(i)) > 1e-9);
    }

    [Fact(Timeout = 300000)]
    public async Task Training_LowersTheCtcPlusRichLoss()
    {
        await Task.Yield();
        var options = Tiny();
        options.TargetsIncludeRichTokens = true;
        using var model = new SenseVoice<double>(Architecture(), options);
        var audio = Voice(8000, 180);
        // The release's target layout: language, emotion, event and text-normalization tokens, then the transcript.
        var labels = new[] { 9, 10, 11, 12, 3, 5, 2 };
        var target = new Tensor<double>(new[] { labels.Length }, new Vector<double>(labels.Select(l => (double)l).ToArray()));
        var objective = (ITrainingObjectiveProvider<double>)model;
        double before = objective.EvaluateTrainingObjective(audio, target);
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        double after = objective.EvaluateTrainingObjective(audio, target);
        Assert.True(after < 0.5 * before, $"loss {before} -> {after}");
        Assert.Equal(new[] { 9, 10, 11, 12 }, model.RichTokenIds(audio));
    }
}
