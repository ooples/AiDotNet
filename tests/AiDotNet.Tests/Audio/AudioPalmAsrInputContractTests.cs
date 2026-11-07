using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.TextToSpeech;
using Xunit;

namespace AiDotNet.Tests.Audio;

public class AudioPalmAsrInputContractTests
{
    [Fact]
    public void PublicPredictionAcceptsWaveformsAndMatchesTranscription()
    {
        using var model = AudioPaLMFixture.Asr<double>();
        var recording = AudioPaLMFixture.Tones<double>(0.25);
        Assert.Equal(LayerInputDomainKind.Continuous, model.GetInputDomain(recording.Shape.ToArray()).Kind);
        var ids = model.Predict(recording).Select(value => (int)value).ToArray();
        var transcript = model.Transcribe(recording, includeTimestamps: false);
        Assert.Equal(model.Core.DecodeText(ids), transcript.Text);
        Assert.Equal(0.25, transcript.DurationSeconds);
        Assert.True(double.IsFinite(transcript.Confidence));
    }
}
