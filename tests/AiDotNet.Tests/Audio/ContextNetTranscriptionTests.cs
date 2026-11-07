using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.ConformerFamily;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class ContextNetTranscriptionTests
{
    [Fact]
    public void NativeTranscriptionMatchesChannelsFirstEncoderAndSingleUtteranceCtc()
    {
        var options = new ContextNetOptions
        {
            NumBlocks = 2, NumSubBlocks = 1, NumMels = 4, VocabSize = 5,
            WidthScaling = 0.015625, KernelSize = 3, SqueezeExcitationRatio = 2,
            DropoutRate = 0, MaxAudioLengthSeconds = 1, Vocabulary = new[] { "", "a", "b", "c", "d" }
        };
        using var model = new Probe(options);
        model.SetTrainingMode(false);
        var audio = new Tensor<double>(new[] { 20000 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.4 * Math.Sin(i * 0.13);
        var features = model.Features(audio);
        Assert.Equal(2, features.Rank);
        Assert.Equal(options.NumMels, features.Shape[1]);
        var packed = new Tensor<double>(new[] { 1, options.NumMels, features.Shape[0] });
        for (int frame = 0; frame < features.Shape[0]; frame++)
            for (int mel = 0; mel < options.NumMels; mel++) packed[0, mel, frame] = features[frame, mel];
        var logits = model.Predict(packed);
        var expected = new System.Text.StringBuilder();
        int previous = -1;
        for (int frame = 0; frame < logits.Shape[1]; frame++)
        {
            // The first index of the largest logit, as MaxBy would pick (MaxBy is not on net471).
            int f = frame;
            int token = Enumerable.Range(0, options.VocabSize).Aggregate((best, v) => logits[0, f, v] > logits[0, f, best] ? v : best);
            if (token != 0 && token != previous) expected.Append(options.Vocabulary[token]);
            previous = token;
        }
        var result = model.Transcribe(audio, includeTimestamps: false);
        Assert.Equal(expected.ToString(), result.Text);
        Assert.Equal(1.25, result.DurationSeconds);
        Assert.Equal(1, model.GetModelMetadata().AdditionalInfo["MaxAudioLengthSeconds"]);
        Assert.True(TtsPaperNumerics.IsFinite(result.Confidence));
    }

    private sealed class Probe : ContextNet<double>
    {
        internal Probe(ContextNetOptions options)
            : base(new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 5), options) { }
        internal Tensor<double> Features(Tensor<double> audio) => PreprocessAudio(audio);
    }
}
