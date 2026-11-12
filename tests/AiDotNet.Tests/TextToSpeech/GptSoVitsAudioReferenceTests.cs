using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GPT-SoVITS's reference-audio resampler against the release's <c>resample(match_librosa=True)</c>
/// (tools/reference-data/gptsovits_resample_reference.py) from 44.1, 32 and 8 kHz to 16 kHz.
/// </summary>
public class GptSoVitsAudioReferenceTests
{
    [Fact(Timeout = 120000)]
    public async Task Resample_MatchesTheRelease()
    {
        await Task.Yield();
        const string name = "gptsovits_resample_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        var r = JsonDocument.Parse(File.ReadAllText(path)).RootElement;
        foreach (var c in r.GetProperty("cases").EnumerateArray())
        {
            var wav = c.GetProperty("wav").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            var expected = c.GetProperty("out").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            var actual = GptSoVitsAudio.Resample(wav, c.GetProperty("rate").GetInt32(), 16000);
            Assert.Equal(expected.Length, actual.Length);
            // The release sums its taps in float32.
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-6, $"{c.GetProperty("rate")}[{i}]: {expected[i]} vs {actual[i]}");
        }
    }
}
