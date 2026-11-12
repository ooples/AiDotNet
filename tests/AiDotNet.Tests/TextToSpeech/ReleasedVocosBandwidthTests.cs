using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class ReleasedVocosBandwidthTests
{
    [Fact(Timeout = 120000)]
    public async Task AuthenticNormalizationState_ChangesActualBandwidthConditionedDecoding()
    {
        await Task.Yield();
        var decoder = new VocosEncodecDecoder<double>(AiDotNetEngine.Current, new Random(1337), VocosEncodecConfiguration.Released());
        ReleasedVocosBandwidthFixture.Load(decoder);
        var codes = new int[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 } };
        var first = decoder.Decode(codes, 0).ToArray();
        var second = decoder.Decode(codes, 1).ToArray();
        Assert.Equal(960, first.Length);
        Assert.Equal(first, decoder.Decode(codes, 0).ToArray());
        Assert.False(first.SequenceEqual(second));
        Assert.All(first.Concat(second), sample => Assert.True(AiDotNet.Helpers.NumericalStabilityHelper.IsFinite(sample)));
    }
}
