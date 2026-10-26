using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Resampling;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// librosa's default resampler (libsoxr HQ through python-soxr) against libsoxr itself in double precision
/// (tools/reference-data/soxr_reference.py): rate pairs that take each of libsoxr's stage kinds, on LCG noise.
/// </summary>
public class SoxrResamplerReferenceTests
{
    // The fixture's noise: a 64-bit LCG's top 24 bits, centred, scaled and rounded to float32.
    private static double[] LcgNoise(int count, ulong seed, double scale)
    {
        var values = new double[count];
        ulong state = seed;
        for (int i = 0; i < count; i++)
        {
            state = unchecked(state * 6364136223846793005UL + 1442695040888963407UL);
            values[i] = (float)(((state >> 40) / 16777216.0 - 0.5) * scale);
        }
        return values;
    }

    [Fact(Timeout = 300000)]
    public async Task LibrosaResample_MatchesLibsoxr()
    {
        await Task.Yield();
        const string fileName = "soxr_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "Audio", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Audio", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        foreach (var c in f["cases"]!)
        {
            int orig = (int)c["orig"]!, target = (int)c["target"]!;
            var input = LcgNoise((int)c["length"]!, (ulong)c["seed"]!, (double)c["scale"]!);
            var expected = c["output"]!.Select(v => (double)v).ToArray();
            var actual = SoxrResampler.LibrosaResample(input, orig, target);
            string what = $"{orig}->{target} {(string)c["signal"]!}";
            Assert.True(expected.Length == actual.Length, $"{what}: {expected.Length} vs {actual.Length} samples");
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"{what}[{i}]: {expected[i]} vs {actual[i]}");
        }
    }
}
