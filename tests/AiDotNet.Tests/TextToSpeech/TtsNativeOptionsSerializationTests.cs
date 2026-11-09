using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class TtsNativeOptionsSerializationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void E3NonuniformStagesRoundTripRegardlessOfPropertyOrder(bool reverse)
    {
        var options = E3TTSPaperTests.TinyOptions();
        options.BlockChannels = [3, 3, 4, 4, 5, 6, 6, 6, 6, 6];
        var json = JObject.FromObject(options);
        if (reverse) json = new JObject(json.Properties().Reverse().Select(property => new JProperty(property.Name, property.Value.DeepClone())));
        var restored = JsonConvert.DeserializeObject<E3TTSOptions>(json.ToString())!;
        Assert.Equal(options.BlockChannels, restored.BlockChannels);
        Assert.Equal(6, restored.HiddenDim);
        Assert.Equal(6, restored.EncoderDim);
        Assert.Equal(6, restored.DecoderDim);
    }

    [Fact]
    public void E3LegacyIndependentWidthMetadataDoesNotResizeStoredStages()
    {
        var options = E3TTSPaperTests.TinyOptions();
        var json = JObject.FromObject(options);
        json["HiddenDim"] = json["EncoderDim"] = json["DecoderDim"] = 1024;
        var restored = JsonConvert.DeserializeObject<E3TTSOptions>(json.ToString())!;
        Assert.Equal(options.BlockChannels, restored.BlockChannels);
        Assert.Equal(16, restored.HiddenDim);
    }

    [Fact]
    public void E3PartialSharedWidthScalesTheDefaultStages()
    {
        var restored = JsonConvert.DeserializeObject<E3TTSOptions>("{\"HiddenDim\":512}")!;
        Assert.Equal(new E3TTSOptions().BlockChannels.Select(width => width / 2), restored.BlockChannels);
        Assert.Equal(512, restored.HiddenDim);
    }

    [Fact]
    public void E3PopulatingAnUnrelatedSettingPreservesCompactStages()
    {
        var options = E3TTSPaperTests.TinyOptions();
        var before = (int[])options.BlockChannels.Clone();
        JsonConvert.PopulateObject("{\"DropoutRate\":0.2}", options);
        Assert.Equal(before, options.BlockChannels);
        Assert.Equal(.2, options.BertDropout);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void UniAudioNativeStridesRoundTripAcrossSampleRatesAndPropertyOrders(bool reverse)
    {
        var options = new UniAudioOptions
        {
            SampleRate = 44100, HopSize = 294, CodecFrameRate = 150, CodecRatios = [7, 7, 3, 2],
        };
        var json = JObject.FromObject(options);
        if (reverse) json = new JObject(json.Properties().Reverse().Select(property => new JProperty(property.Name, property.Value.DeepClone())));
        var restored = JsonConvert.DeserializeObject<UniAudioOptions>(json.ToString())!;
        Assert.Equal(options.CodecRatios, restored.CodecRatios);
        Assert.Equal((44100, 294, 150), (restored.SampleRate, restored.HopSize, restored.CodecFrameRate));
    }

    [Theory]
    [InlineData("{\"SampleRate\":320,\"HopSize\":32}", 320, 32, 10)]
    [InlineData("{\"HopSize\":32,\"SampleRate\":320}", 320, 32, 10)]
    [InlineData("{\"CodecFrameRate\":25}", 16000, 640, 25)]
    public void UniAudioPartialGeometryConfiguresNativeStrides(string json, int sampleRate, int hop, int frameRate)
    {
        var restored = JsonConvert.DeserializeObject<UniAudioOptions>(json)!;
        Assert.Equal((sampleRate, hop, frameRate), (restored.SampleRate, restored.HopSize, restored.CodecFrameRate));
        Assert.Equal(hop, restored.CodecRatios.Aggregate(1, (a, b) => a * b));
    }

    [Fact]
    public void UniAudioConflictingPartialAliasesAreRejected()
    {
        Assert.ThrowsAny<Exception>(() => JsonConvert.DeserializeObject<UniAudioOptions>(
            "{\"SampleRate\":320,\"HopSize\":32,\"CodecFrameRate\":12}"));
    }
}
