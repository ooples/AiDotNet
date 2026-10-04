using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.OCR;
using AiDotNet.ComputerVision.OCR.Recognition;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// CRNN's native weight file omitted batch norms 5 and 6, so a save/load round trip lost their learned scale and
/// shift and their running statistics, and the reloaded model predicted differently.
/// </summary>
public class CRNNNativeWeightsTests
{
    private static CRNN<double> Model(int seed) => new(new OCROptions<double>
    {
        RecognitionHeight = 32, MaxRecognitionWidth = 64, CharacterSet = "0123456789", RandomSeed = seed,
    });

    private static Tensor<double> Image(int seed)
    {
        var image = new Tensor<double>(new[] { 1, 3, 32, 64 });
        for (int i = 0; i < image.Length; i++)
            image[i] = ((i * 13 + seed * 7) % 89) / 89.0;
        return image;
    }

    [Fact(Timeout = 120000)]
    public async Task RoundTrip_ReproducesThePredictions_IncludingBatchNormState()
    {
        await Task.Yield();
        var source = Model(1);
        // Training-mode forwards move the running statistics away from a fresh model's defaults.
        source.SetTrainingMode(true);
        for (int step = 0; step < 3; step++)
            source.Recognize(Image(step));
        source.SetTrainingMode(false);
        var probe = Image(42);
        var expected = source.RecognizeText(probe);

        string path = Path.Combine(Path.GetTempPath(), $"crnn-{Guid.NewGuid():N}.bin");
        try
        {
            source.SaveWeights(path);
            var copy = Model(2);
            await copy.LoadWeightsAsync(path);
            copy.SetTrainingMode(false);
            var actual = copy.RecognizeText(probe);

            Assert.Equal(expected.text, actual.text);
            Assert.Equal(expected.confidence, actual.confidence);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
