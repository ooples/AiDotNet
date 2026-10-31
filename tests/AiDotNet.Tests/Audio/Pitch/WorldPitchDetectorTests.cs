using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Pitch;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio.Pitch;

/// <summary>
/// The WORLD port reproduces PyWorld's <c>dio</c> and <c>stonemask</c> contours.
/// </summary>
/// <remarks>
/// <para>
/// FastSpeech 2 and the acoustic models built on it take their pitch targets from PyWorld (Ren et al. 2021,
/// Appendix C.2), so the reference is PyWorld itself rather than a property of a good pitch tracker.
/// </para>
/// <para>
/// <c>ReferenceData/pyworld_dio_stonemask.json</c> was produced with pyworld 0.3.5 / numpy 2.5.3 from the signal
/// in <see cref="Signal"/>, written identically in Python: a harmonic voice gliding from 120 to 220 Hz, an
/// unvoiced gap between 55 % and 70 % of the clip, and a small deterministic LCG noise floor. Two settings are
/// covered: 22050 Hz with FastSpeech 2's hop (256 samples, 11.61 ms) and 16 kHz with WORLD's default 5 ms.
/// </para>
/// </remarks>
public class WorldPitchDetectorTests
{
    private static readonly Lazy<JObject> Reference = new(() => JObject.Parse(File.ReadAllText(ResolveReferencePath())));

    private static double[] Signal(int fs, double seconds)
    {
        int n = (int)(fs * seconds);
        var x = new double[n];
        double phase = 0.0;
        long state = 12345;
        for (int i = 0; i < n; i++)
        {
            double t = (double)i / fs;
            double f0 = 120.0 + 100.0 * t / seconds;
            phase += 2 * Math.PI * f0 / fs;
            double v = 0.0;
            if (!(0.55 * seconds <= t && t < 0.70 * seconds))
            {
                for (int h = 1; h < 6; h++) v += Math.Sin(h * phase) / h;
                v *= 0.3;
            }
            state = (1103515245L * state + 12345L) % 2147483648L;
            v += 0.01 * ((state / 2147483648.0) - 0.5);
            x[i] = v;
        }
        return x;
    }

    private static double[] Values(JObject setting, string key) => setting[key]!.Select(v => (double)v).ToArray();

    [Theory(Timeout = 120000)]
    [InlineData(22050)]
    [InlineData(16000)]
    public async Task Dio_MatchesPyWorld(int fs)
    {
        await Task.Yield();
        var setting = (JObject)Reference.Value[fs.ToString(System.Globalization.CultureInfo.InvariantCulture)]!;
        var x = Signal(fs, (double)setting["seconds"]!);
        Assert.Equal((int)setting["n"]!, x.Length);

        var detector = new WorldPitchDetector<double>(sampleRate: fs, refineWithStoneMask: false);
        var (f0, times) = detector.EstimateF0(x, (double)setting["frame_period"]!);

        AssertContoursMatch(Values(setting, "dio"), f0, "dio");
        var expectedTimes = Values(setting, "t");
        for (int i = 0; i < times.Length; i++) Assert.Equal(expectedTimes[i], times[i], 12);
    }

    [Theory(Timeout = 120000)]
    [InlineData(22050)]
    [InlineData(16000)]
    public async Task StoneMask_MatchesPyWorld(int fs)
    {
        await Task.Yield();
        var setting = (JObject)Reference.Value[fs.ToString(System.Globalization.CultureInfo.InvariantCulture)]!;
        var x = Signal(fs, (double)setting["seconds"]!);

        var detector = new WorldPitchDetector<double>(sampleRate: fs);
        var (f0, _) = detector.EstimateF0(x, (double)setting["frame_period"]!);

        AssertContoursMatch(Values(setting, "stonemask"), f0, "stonemask");
    }

    private static void AssertContoursMatch(double[] expected, double[] actual, string stage)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True((expected[i] > 0) == (actual[i] > 0),
                $"{stage} frame {i}: PyWorld voicing {expected[i]}, port {actual[i]}.");
            double tolerance = 1e-6 * Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{stage} frame {i}: PyWorld {expected[i]:R}, port {actual[i]:R}.");
        }
    }

    private static string ResolveReferencePath()
    {
        const string fileName = "pyworld_dio_stonemask.json";
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
