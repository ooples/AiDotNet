using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.ProprietaryAPI;
using AiDotNet.SpeechRecognition.Specialized;
using AiDotNet.SpeechRecognition.WhisperFamily;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The Whisper-family recognizers run openai-whisper's network (Radford et al. 2022) at each checkpoint's dimensions:
/// large-v3, its 4-layer-decoder turbo, Distil-Whisper's 2-layer decoder, and whisper.cpp's base model (#2397).
/// </summary>
public class WhisperFamilyPaperTests
{
    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.SequenceToSequence, inputSize: 3200, outputSize: 51865);

    private static Tensor<double> Voice(int samples)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
        {
            double t = i / 16000.0;
            audio[i] = 0.3 * Math.Sin(2 * Math.PI * 200 * t) + 0.1 * Math.Sin(2 * Math.PI * 600 * t);
        }
        return audio;
    }

    [Fact(Timeout = 60000)]
    public async Task Defaults_AreEachCheckpointsDimensions()
    {
        await Task.Yield();
        // openai-whisper ModelDimensions: large-v3 is 128 mels, 32 + 32 layers, width 1280, 20 heads, 51,866 tokens; turbo
        // keeps the encoder with 4 decoder layers; distil-large-v3 has 2; base is 80 mels, 6 + 6 layers, width 512, 8 heads.
        var large = new WhisperLargeV3Options();
        Assert.Equal((128, 32, 32, 1280, 20, 51866), (large.NumMels, large.NumEncoderLayers, large.NumDecoderLayers, large.EncoderDim, large.NumAttentionHeads, large.VocabSize));
        Assert.Equal(4, new WhisperLargeV3TurboOptions().NumDecoderLayers);
        Assert.Equal(2, new DistilWhisperOptions().NumDecoderLayers);
        Assert.Equal((2, "ja"), (new KotobaWhisperOptions().NumDecoderLayers, new KotobaWhisperOptions().Language));
        var cpp = new WhisperCPPOptions();
        Assert.Equal((80, 6, 6, 512, 8, 51865), (cpp.NumMels, cpp.NumEncoderLayers, cpp.NumDecoderLayers, cpp.EncoderDim, cpp.NumAttentionHeads, cpp.VocabSize));
        Assert.Equal(32, new GroqWhisperOptions().NumDecoderLayers);
        Assert.True(new WhisperXOptions().WordTimestamps);
    }

    private static DistilWhisper<double> Tiny() => new(Architecture(), new DistilWhisperOptions
    {
        NumMels = 80, EncoderDim = 16, DecoderDim = 16, NumEncoderLayers = 1, NumDecoderLayers = 1, NumAttentionHeads = 2,
        VocabSize = 51865, Language = "en", Seed = 3,
    });

    [Fact(Timeout = 120000)]
    public async Task Predict_ReturnsTheDecodersLogitsOverTheStartSequence()
    {
        await Task.Yield();
        using var model = Tiny();
        var logits = model.Predict(Voice(3200));
        // <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>, each over the 51,865-token vocabulary.
        Assert.Equal(new[] { 4, 51865 }, logits.Shape.ToArray());
        var whisper = model.Whisper ?? throw new InvalidOperationException("native model");
        Assert.Equal(new[] { 50258, 50259, 50359, 50363 }, whisper.Prompt("en"));
    }

    [Fact(Timeout = 300000)]
    public async Task TeacherForcedTraining_LowersTheTranscriptLoss()
    {
        await Task.Yield();
        using var model = Tiny();
        var whisper = model.Whisper ?? throw new InvalidOperationException("native model");
        var audio = Voice(3200);
        var transcript = whisper.Transcriber.Encode(" hello");
        var target = new Tensor<double>(new[] { transcript.Length }, new Vector<double>(transcript.Select(t => (double)t).ToArray()));
        var window = whisper.Window(audio.ToArray());
        double Loss() => whisper.TranscriptLoss(window, transcript, "en")[0];
        double before = Loss();
        for (int step = 0; step < 30; step++) model.Train(audio, target);
        double after = Loss();
        Assert.True(after < 0.5 * before, $"transcript loss {before} -> {after}");
    }
}
