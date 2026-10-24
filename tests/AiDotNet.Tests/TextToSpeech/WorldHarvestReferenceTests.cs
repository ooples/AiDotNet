using System;
using System.IO;
using System.Linq;
using AiDotNet.Audio.Pitch;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// WORLD's Harvest and StoneMask against pyworld 0.3.5 (<c>tools/reference-data/world_harvest_reference.py</c>) at
/// three sample rates (decimation 3, 2 and 1) and three frame periods.
/// </summary>
public class WorldHarvestReferenceTests
{
    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string fileName = "world_harvest_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    });

    private static void AssertClose(JToken expected, double[] actual, string what)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-8 * Math.Max(1, Math.Abs(values[i])),
                $"{what}[{i}]: pyworld {values[i]:R}, port {actual[i]:R}.");
    }

    [Theory(Timeout = 300000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    public async System.Threading.Tasks.Task HarvestAndStoneMask_MatchPyWorld(int index)
    {
        await System.Threading.Tasks.Task.Yield();
        var c = Data.Value["cases"]![index]!;
        int fs = (int)c["fs"]!;
        double period = (double)c["frame_period"]!;
        var x = Data.Value["signals"]![(string?)c["signal"] ?? fs.ToString()]!.Select(v => (double)v).ToArray();
        var detector = new WorldPitchDetector<double>(fs);
        var (f0, t) = detector.Harvest(x, period);
        AssertClose(c["t"]!, t, $"{fs} Hz / {period} ms times");
        AssertClose(c["f0"]!, f0, $"{fs} Hz / {period} ms harvest");
        AssertClose(c["stonemask"]!, detector.StoneMask(x, f0, t), $"{fs} Hz / {period} ms stonemask");
    }
}
