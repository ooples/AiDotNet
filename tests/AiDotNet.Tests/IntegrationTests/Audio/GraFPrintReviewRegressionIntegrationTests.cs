#nullable disable
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.NeuralNetworks;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Audio;

public class GraFPrintReviewRegressionIntegrationTests
{
    [Fact(Timeout = 120000)]
    public async Task NativeMode_ArchitectureOutputSize_NormalizesEmbeddingDimensionConsumers()
    {
        await Task.Yield();
        var options = new GraFPrintOptions
        {
            EmbeddingDim = 16,
            NumMels = 8,
            GnnHiddenDim = 8,
            NumGnnLayers = 1,
            NumAttentionHeads = 1
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: options.NumMels,
            outputSize: 6);

        var model = new GraFPrint<double>(architecture, options);
        var metadata = model.GetModelMetadata();

        Assert.Equal(6, model.FingerprintLength);
        Assert.Equal(6, options.EmbeddingDim);
        Assert.Equal("6", metadata.AdditionalInfo["EmbeddingDim"]);
    }
}
