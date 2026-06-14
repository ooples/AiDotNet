using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class OmniGen2ModelTests : DiffusionModelTestBase
{
    protected override int[] InputShape => [1, 16, 32, 32];
    protected override int[] OutputShape => [1, 16, 32, 32];

    // OmniGen-2 defaults to a foundation-scale SiT predictor (hidden 3072, 32 layers, Phi-3
    // backbone): it trips disk-backed weight streaming and cannot complete a single forward
    // within the 120s test budget. Inject a tiny SiT + VAE so these model-family invariants run
    // against a REAL instance of the same architecture at a runnable scale — latentChannels (16)
    // stays paper-correct; only the hidden width / depth / VAE base channels are shrunk, keeping
    // the model well below the streaming threshold.
    protected override IDiffusionModel<double> CreateModel()
        => new OmniGen2Model<double>(
            predictor: new SiTPredictor<double>(
                inputChannels: 16, hiddenSize: 64, numLayers: 2, numHeads: 2, seed: 42),
            vae: new StandardVAE<double>(
                inputChannels: 3, latentChannels: 16, baseChannels: 16,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1,
                latentScaleFactor: 1.5305, seed: 42),
            seed: 42);
}
