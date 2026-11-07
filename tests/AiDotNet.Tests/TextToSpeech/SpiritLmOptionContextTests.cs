using System;
using System.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class SpiritLmOptionContextTests
{
    [Theory]
    [InlineData(2560, 4)]
    [InlineData(5120, 8)]
    [InlineData(100000, 12)]
    public void VocoderSegmentKeepsCodesProsodyAndWaveformAligned(int segmentSize, int frames)
    {
        var options = SpiritLmFixture.Options();
        options.VocoderSegmentSize = segmentSize;
        using var model = SpiritLmFixture.Tts<double>(options);
        var codes = Enumerable.Range(0, 12).ToArray();
        var pitch = codes.Select(i => i + 20).ToArray();
        var styles = codes.Select(i => i + 40).ToArray();
        var recording = SpiritLmFixture.Tones<double>(0.5);
        int start = new Random(1337).Next(codes.Length - frames + 1);
        var actual = model.VocoderTrainingSegment(recording, codes, pitch, styles, new Random(1337));
        Assert.Equal(codes.Skip(start).Take(frames), actual.Codes);
        Assert.Equal(pitch.Skip(start).Take(frames), actual.Pitch);
        Assert.Equal(styles.Skip(start).Take(frames), actual.Styles);
        Assert.Equal(recording.Skip(start * 640).Take(frames * 640), actual.Recording);
        Assert.Equal(frames * 640, actual.Recording.Length);
    }
}
