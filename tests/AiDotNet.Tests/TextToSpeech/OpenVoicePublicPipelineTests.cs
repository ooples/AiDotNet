using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class OpenVoicePublicPipelineTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublicSynthesis_UsesSeparateBaseVoice_AndConverterUsesReferenceEncoder(bool versionTwo)
    {
        var architecture = new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression, inputSize: 32, outputSize: 1);
        var source = new Tensor<double>(new[] { 1024 });
        for (int i = 0; i < source.Length; i++) source[i] = Math.Sin(i * .071) * .07;
        var synthesizer = new RecordedBaseVoice(source);
        Tensor<double> embedding, converted;
        if (versionTwo)
        {
            using var model = new OpenVoiceV2<double>(architecture,
                new OpenVoiceV2Options { HiddenDim = 4, DecoderDim = 4, SpeakerEmbeddingDim = 3 });
            model.BaseSynthesizer = synthesizer;
            Assert.Equal(3, model.SpeakerEmbeddingDim);
            Assert.Same(source, model.Synthesize("separate base voice"));
            embedding = model.ExtractSpeakerEmbedding(source);
            converted = model.ConvertVoice(source, embedding, embedding, tau: 0);
            Assert.Equal(4, model.NumFlowSteps);
        }
        else
        {
            using var model = new OpenVoice<double>(architecture,
                new OpenVoiceOptions { HiddenDim = 4, InterChannels = 4, SpeakerEmbeddingDim = 3 });
            model.BaseSynthesizer = synthesizer;
            Assert.Equal(3, model.SpeakerEmbeddingDim);
            Assert.Same(source, model.Synthesize("separate base voice"));
            embedding = model.ExtractSpeakerEmbedding(source);
            converted = model.ConvertVoice(source, embedding, embedding, tau: 0);
            Assert.Equal(4, model.NumFlowSteps);
        }
        Assert.Equal("separate base voice", synthesizer.LastText);
        Assert.Equal(3, embedding.Length);
        Assert.Equal(1024, converted.Length);
        Assert.All(converted.ToArray(), x => Assert.False(double.IsNaN(x) || double.IsInfinity(x)));
        Assert.True(source.ToArray().Zip(converted.ToArray(), (a, b) => Math.Abs(a - b)).Max() > 1e-8);
    }

    [Fact]
    public void OpenVoiceOptions_CopyRetainsInheritedSpectrogramAndModelSettings()
    {
        var original = new OpenVoiceOptions { SampleRate = 22050, FftSize = 512, HopSize = 256,
            HiddenDim = 24, InterChannels = 16, SpeakerEmbeddingDim = 12, LearningRate = .0007 };
        var copy = new OpenVoiceOptions(original);
        Assert.Equal(original.FftSize, copy.FftSize); Assert.Equal(original.HopSize, copy.HopSize);
        Assert.Equal(original.HiddenDim, copy.HiddenDim); Assert.Equal(original.InterChannels, copy.InterChannels);
        Assert.Equal(original.SpeakerEmbeddingDim, copy.SpeakerEmbeddingDim);
        Assert.Equal(original.LearningRate, copy.LearningRate);
    }

    private sealed class RecordedBaseVoice : ITtsModel<double>
    {
        private readonly Tensor<double> _recording;
        public RecordedBaseVoice(Tensor<double> recording) => _recording = recording;
        public string? LastText { get; private set; }
        public Tensor<double> Synthesize(string text) { LastText = text; return _recording; }
        public int SampleRate => 22050;
        public int MaxTextLength => 512;
    }
}
