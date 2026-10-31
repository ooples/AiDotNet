using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Pitch;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio.Pitch;

/// <summary>
/// The pitch spectrogram matches the FastSpeech 2 authors' implementation.
/// </summary>
/// <remarks>
/// <c>ReferenceData/natspeech_pitch_cwt.json</c> was produced by NATSpeech's <c>utils/audio/cwt.py</c>
/// (<c>get_cont_lf0</c>, <c>get_lf0_cwt</c> with pycwt 0.4 on its scipy.fftpack path, <c>inverse_cwt</c>, and the
/// numpy branch of <c>cwt2f0</c>) applied to the PyWorld StoneMask contours in
/// <c>pyworld_dio_stonemask.json</c>, with the per-utterance log-F0 mean and standard deviation.
/// </remarks>
public class PitchWaveletTransformTests
{
    private static readonly Lazy<JObject> Reference = new(() => JObject.Parse(File.ReadAllText(ResolveReferencePath())));

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    [Theory(Timeout = 60000)]
    [InlineData("22050")]
    [InlineData("16000")]
    public async Task MatchesNatSpeech(string key)
    {
        await Task.Yield();
        var r = (JObject)Reference.Value[key]!;
        var f0 = Values(r["f0"]!);

        var (uv, contLf0) = PitchWaveletTransform.ContinuousLogF0(f0);
        AssertClose(Values(r["uv"]!), uv, 0, "uv");
        AssertClose(Values(r["cont_lf0"]!), contLf0, 1e-12, "continuous log-F0");

        double mean = (double)r["mean"]!, std = (double)r["std"]!;
        var normalized = contLf0.Select(v => (v - mean) / std).ToArray();
        var spectrogram = PitchWaveletTransform.Forward(normalized);

        var expected = r["cwt"]!.Select(row => Values(row)).ToArray();
        Assert.Equal(expected.Length, spectrogram.GetLength(0));
        Assert.Equal(PitchWaveletTransform.ScaleCount, spectrogram.GetLength(1));
        for (int t = 0; t < expected.Length; t++)
            for (int s = 0; s < PitchWaveletTransform.ScaleCount; s++)
                Assert.True(Math.Abs(expected[t][s] - spectrogram[t, s]) <= 1e-9,
                    $"cwt[{t},{s}]: NATSpeech {expected[t][s]:R}, port {spectrogram[t, s]:R}.");

        AssertClose(Values(r["rec"]!), PitchWaveletTransform.Inverse(spectrogram), 1e-9, "inverse");
        AssertClose(Values(r["f0_rec"]!), PitchWaveletTransform.ToF0(spectrogram, mean, std), 1e-7, "F0 reconstruction");
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{what}[{i}]: NATSpeech {expected[i]:R}, port {actual[i]:R}.");
    }

    private static string ResolveReferencePath()
    {
        const string fileName = "natspeech_pitch_cwt.json";
        string output = Path.Combine(AppContext.BaseDirectory, "Audio", "Pitch", "ReferenceData", fileName);
        if (File.Exists(output)) return output;
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string source = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Audio", "Pitch", "ReferenceData", fileName);
            if (File.Exists(source)) return source;
            dir = dir.Parent;
        }
        throw new FileNotFoundException($"Reference data '{fileName}' was not found.");
    }
}
