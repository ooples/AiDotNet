using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class MaskGctOptionConstraintTests
{
    [Fact]
    public void QuantizerCountRequiresOneDecodingSchedulePerLayer()
    {
        var options = Options(2);
        options.NumQuantizers = 3;
        var error = Assert.Throws<ArgumentException>(() => new MaskGCT<double>(Architecture(), options));
        Assert.Contains("SemanticToAcousticSteps", error.Message);
    }

    [Fact]
    public void PairedQuantizerAndScheduleChangesResizeAndRunTheActualAcousticModels()
    {
        long Run(int quantizers)
        {
            using var model = new MaskGCT<double>(Architecture(), Options(quantizers));
            model.SetTrainingMode(false);
            var random = new Random(1337);
            var codes = model.Acoustic(new[] { 0, 1, 2, 3, 4, 5 }, new int[2, quantizers],
                count => Enumerable.Range(0, count).Select(_ => random.NextDouble()).ToArray());
            Assert.Equal(4, codes.GetLength(0));
            Assert.Equal(quantizers, codes.GetLength(1));
            Assert.All(codes.Cast<int>(), code => Assert.InRange(code, 0, 15));
            return model.GetParameters().Length;
        }
        Assert.True(Run(3) > Run(2));
    }

    [Fact]
    public void HopRequiresMatchingCodecStrides()
    {
        var options = Options(2);
        options.HopSize = 320;
        var error = Assert.Throws<ArgumentException>(() => new MaskGCT<double>(Architecture(), options));
        Assert.Contains("HopSize", error.Message);
    }

    [Theory]
    [InlineData(3, 480)]
    [InlineData(2, 320)]
    public void PairedHopAndStrideChangesControlTheActualDecodedWaveform(int firstStride, int hop)
    {
        var options = Options(2);
        options.CodecEncoderRates = new[] { firstStride, 4, 5, 8 };
        options.HopSize = hop;
        using var model = new MaskGCT<double>(Architecture(), options);
        model.SetTrainingMode(false);
        var waveform = model.DecodeFromTokens(new Tensor<double>(new[] { 4, 2 }));
        Assert.Equal(4 * hop, waveform.Length);
        Assert.All(waveform, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputFeatures: 1, outputSize: 1) { RandomSeed = 1337 };

    internal static MaskGCTOptions Options(int quantizers = 2) => new()
    {
        SemanticHiddenSize = 32, SemanticHeads = 4, SemanticIntermediateSize = 64, SemanticLayer = 2,
        SemanticCodebookSize = 32, SemanticCodebookDim = 4, SemanticCodecDim = 16,
        SemanticCodecIntermediate = 32, SemanticCodecLayers = 1, CodecEncoderDim = 2,
        CodecLatentDim = 16, NumQuantizers = quantizers, CodebookSize = 16, CodebookDim = 4,
        VocosDim = 16, VocosIntermediate = 32, VocosLayers = 1, HiddenDim = 16,
        TextToSemanticDim = 16, TextToSemanticLayers = 1, TextToSemanticHeads = 2,
        SemanticToAcousticLayers = 1, SemanticToAcousticHeads = 2,
        DurationDim = 16, DurationLayers = 1, DurationHeads = 2,
        TextToSemanticSteps = 2, SemanticToAcousticSteps = Enumerable.Range(0, quantizers).Select(i => i == 0 ? 2 : 1).ToArray()
    };
}
