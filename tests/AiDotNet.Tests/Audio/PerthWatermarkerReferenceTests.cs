using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Watermarking;
using AiDotNet.ModelLoading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.TextToSpeech;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>Resemble's Perth implicit watermarker against resemble-perth 1.0.1 (<c>tools/reference-data/perth_reference.py</c>):
/// the watermarked signal and the detections of a tiny model and the actual released model. The hash-verified release
/// checkpoint is cached automatically, or supplied through <c>PERTH_CHECKPOINT</c>.</summary>
public class PerthWatermarkerReferenceTests
{
    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static void Close(double[] expected, double[] actual, string what, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]:R} vs {actual[i]:R}");
    }

    private static void Check(PerthWatermarker<double> perth, JToken expected, double[] signal, double tolerance, double resampledTolerance)
    {
        // At 32 kHz nothing is resampled: the model's arithmetic alone.
        int n32 = signal.Length * 4 / 3;
        var native = Enumerable.Range(0, n32).Select(i =>
        {
            double t = i / 32000.0;
            return 0.3 * Math.Sin(2 * Math.PI * 220 * t * (1 + 0.3 * t)) * Math.Min(1, 4 * t) + 0.05 * Math.Sin(2 * Math.PI * 1700 * t);
        }).ToArray();
        Close(Doubles(expected["watermarked_32k"]!), perth.Apply(native, 32000), "watermarked at 32 kHz", tolerance);
        // python-soxr runs libsoxr's HQ quality (a 20-bit, about -120 dB, design) in single precision; the port's libsoxr runs
        // it in double; each resampling is good to about 2^-20 of full scale, the watermark passes through two, and the
        // encoder's residual carries the first one's error into the second.
        var watermarked = perth.Apply(signal, 24000);
        Close(Doubles(expected["watermarked"]!), watermarked, "watermarked", resampledTolerance);
        var resampled = PerthWatermarker<double>.ResamplePoly(signal, 32000, 24000).Take(32000).ToArray();
        var magnitude = perth.MagnitudeAndPhase(new Tensor<double>(new[] { resampled.Length }, new Vector<double>(resampled))).Magnitude;
        Close(new[] { (double)expected["raw_decoder"]! }, new[] { perth.Decode(magnitude) }, "raw decoder", tolerance);
        Close(new[] { (double)expected["detected_plain"]!, (double)expected["detected"]! }, new[] { perth.Detect(signal, 24000), perth.Detect(watermarked, 24000) },
            "detections", Math.Max(tolerance, 1e-5));
    }

    [Fact(Timeout = 300000)]
    public async Task Watermark_MatchesResemblePerth()
    {
        await Task.Yield();
        var f = WhisperSpeechT2SReferenceTests.Load("perth_reference.json");
        var signal = Doubles(f["signal"]!);
        var tiny = new PerthWatermarker<double>(AiDotNetEngine.Current, hidden: 8);
        var state = (JObject)f["tiny_state"]!;
        tiny.LoadTorchWeights(TorchTensorSource.From(n => state[n] is { } v ? Doubles(v) : null));
        Check(tiny, f["tiny"]!, signal, 1e-9, 3e-6);

        string checkpoint = PerthReleasedCheckpointFixture.CheckpointPath;
        var released = PerthWatermarker<double>.Load(AiDotNetEngine.Current, checkpoint);
        Check(released, f["released"]!, signal, 1e-6, 1e-5);
    }
}
