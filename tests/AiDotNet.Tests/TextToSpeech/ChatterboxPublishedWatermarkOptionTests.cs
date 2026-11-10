using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.Audio;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class ChatterboxPublishedWatermarkOptionTests
{
    [Theory(Timeout = 300000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task WatermarkFlagChangesTheRealReleasedPerthOutput(bool singlePrecision)
    {
        await Task.Yield();
        if (singlePrecision) Check<float>(); else Check<double>();
    }

    private static void Check<T>()
    {
        var options = ChatterboxFixture.Options();
        options.WatermarkModelPath = PerthReleasedCheckpointFixture.CheckpointPath;
        options.Watermark = false;
        var model = new Chatterbox<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);
        var recording = ChatterboxFixture.Recording<T>();
        var clean = model.Watermarked(recording);
        Assert.Equal(recording.ToArray(), clean.ToArray());
        options.Watermark = true;
        var marked = model.Watermarked(recording);
        Assert.Equal(256, model.Watermarker.Hidden);
        Assert.Equal(recording.Length, marked.Length);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        double difference = Enumerable.Range(0, marked.Length)
            .Max(i => Math.Abs(ops.ToDouble(marked[i]) - ops.ToDouble(clean[i])));
        Assert.True(difference > 1e-8, $"The released Perth checkpoint did not alter the waveform ({difference:R}).");
        Assert.All(marked, v => Assert.True(TtsPaperNumerics.IsFinite(ops.ToDouble(v))));
        options.Watermark = false;
        Assert.Equal(recording.ToArray(), model.Watermarked(recording).ToArray());
    }
}
