using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AudioVisualEventLocalizationNetworkTests : NeuralNetworkModelTestBase<float>
{
    // Default: inputSize=512, outputSize=1 (binary classification)
    protected override int[] InputShape => [512];
    protected override int[] OutputShape => [1];

    // The VGGish audio embedding defaults to the PUBLISHED widths (FC 4096 x2, ~67M parameters),
    // which is right for fidelity and far too heavy for a fixture that constructs a fresh model per
    // test: at paper scale this suite pushed single tests past a minute and crashed the test host.
    // The widths are constructor parameters precisely so a fixture can build a small variant without
    // a second, divergent implementation existing anywhere.
    private const int FixtureEmbeddingWidth = 64;
    private const int FixtureEmbeddingSize = 32;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new AudioVisualEventLocalizationNetwork<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.BinaryClassification,
                inputSize: 512,
                outputSize: 1),
            audioEmbeddingFullyConnectedWidth: FixtureEmbeddingWidth,
            audioEmbeddingSize: FixtureEmbeddingSize);
}
