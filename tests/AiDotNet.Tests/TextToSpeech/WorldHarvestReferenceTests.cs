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
        var signal = new Tensor<double>(new[] { x.Length }, new Vector<double>(x));
        var (f0, t) = detector.Harvest(signal, period);
        AssertClose(c["t"]!, t.ToArray(), $"{fs} Hz / {period} ms times");
        AssertClose(c["f0"]!, f0.ToArray(), $"{fs} Hz / {period} ms harvest");
        AssertClose(c["stonemask"]!, detector.StoneMask(signal, f0, t).ToArray(), $"{fs} Hz / {period} ms stonemask");
    }

    /// <summary>
    /// harvest.cpp GetFilteredSignal writes each product to bin fftSize − i − 1 while it goes, so bin fftSize/2 is
    /// multiplied by the signal spectrum twice and bin fftSize/2 − 1 is then overwritten with it (the same in-place
    /// write as dio.cpp). The port must reproduce that, not the textbook product: compared here against a line-by-line
    /// transcription of the C loop with direct DFTs.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async System.Threading.Tasks.Task FilteredSignal_ReproducesHarvestInPlaceSpectrumWrite()
    {
        await System.Threading.Tasks.Task.Yield();
        const double fs = 8000, boundaryF0 = 1000;
        const int fftSize = 64, yLength = 20;
        var y = new double[fftSize];
        for (int i = 0; i < yLength; i++) y[i] = Math.Sin(0.7 * i) + 0.3 * Math.Cos(1.9 * i);
        var (yRe, yIm) = RealDft(y);

        // GetFilteredSignal, transcribed.
        int filterLengthHalf = (int)Math.Round(fs / boundaryF0 * 2.0, MidpointRounding.AwayFromZero);
        var bandPass = new double[fftSize];
        int length = filterLengthHalf * 2 + 1;
        for (int i = 0; i < length; i++)
        {
            double x = i / (length - 1.0);
            bandPass[i] = 0.355768 - 0.487396 * Math.Cos(2 * Math.PI * x) + 0.144232 * Math.Cos(4 * Math.PI * x)
                - 0.012604 * Math.Cos(6 * Math.PI * x);
        }
        for (int i = -filterLengthHalf; i <= filterLengthHalf; ++i)
            bandPass[i + filterLengthHalf] *= Math.Cos(2 * Math.PI * boundaryF0 * i / fs);
        var (re, im) = RealDft(bandPass);
        Array.Resize(ref re, fftSize);
        Array.Resize(ref im, fftSize);
        double tmp = yRe[0] * re[0] - yIm[0] * im[0];
        im[0] = yRe[0] * im[0] + yIm[0] * re[0];
        re[0] = tmp;
        for (int i = 1; i <= fftSize / 2; ++i)
        {
            tmp = yRe[i] * re[i] - yIm[i] * im[i];
            im[i] = yRe[i] * im[i] + yIm[i] * re[i];
            re[i] = tmp;
            re[fftSize - i - 1] = re[i];
            im[fftSize - i - 1] = im[i];
        }
        var filtered = InverseRealDft(re, im, fftSize);
        var expected = new double[yLength];
        Array.Copy(filtered, filterLengthHalf + 1, expected, 0, yLength);

        var actual = WorldPitchDetector<double>.HarvestFilteredSignal(boundaryF0, fftSize, fs,
            new Tensor<double>(new[] { yRe.Length }, new Vector<double>(yRe)),
            new Tensor<double>(new[] { yIm.Length }, new Vector<double>(yIm)), yLength).ToArray();
        Assert.Equal(yLength, actual.Length);
        for (int i = 0; i < yLength; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])),
                $"filtered[{i}]: harvest.cpp {expected[i]:R}, port {actual[i]:R}.");
    }

    // r2c: bins 0 … n/2 of Σ x_t e^{−2πikt/n}.
    private static (double[] Re, double[] Im) RealDft(double[] x)
    {
        int n = x.Length, bins = n / 2 + 1;
        var re = new double[bins];
        var im = new double[bins];
        for (int k = 0; k < bins; k++)
            for (int t = 0; t < n; t++)
            {
                double a = -2 * Math.PI * k * (long)t / n;
                re[k] += x[t] * Math.Cos(a);
                im[k] += x[t] * Math.Sin(a);
            }
        return (re, im);
    }

    // c2r (unnormalized): bins 0 … n/2 extended Hermitian, the imaginary parts of bins 0 and n/2 ignored.
    private static double[] InverseRealDft(double[] re, double[] im, int n)
    {
        var x = new double[n];
        for (int t = 0; t < n; t++)
        {
            double sum = re[0] + re[n / 2] * ((t & 1) == 0 ? 1 : -1);
            for (int k = 1; k < n / 2; k++)
            {
                double a = 2 * Math.PI * k * (long)t / n;
                sum += 2 * (re[k] * Math.Cos(a) - im[k] * Math.Sin(a));
            }
            x[t] = sum;
        }
        return x;
    }
}
