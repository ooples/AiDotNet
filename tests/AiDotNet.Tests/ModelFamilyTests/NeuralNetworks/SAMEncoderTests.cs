using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// The vision-language model-family invariants for <see cref="SAM{T}"/> in AiDotNet.VisionLanguage.Encoders.
/// </summary>
/// <remarks>
/// The generated "SAMTests" class belongs to AiDotNet.ComputerVision.Segmentation.Foundation.SAM (see the
/// generator's CollisionOwners), so this namesake gets no generated class. This scaffold covers it by type
/// through its CreateNetwork override. Its name deliberately does not match "SAM" + a test suffix, so it never
/// claims the owner's name and the segmentation model's generated suite keeps running.
/// </remarks>
public class SAMEncoderTests : VisionLanguageTestBase<double>
{
    protected override int[] InputShape => [1, 3, 32, 32];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 32);

        // The paper's ViT-B image encoder (1024px, 768 wide, 12 layers, 12 heads, 16px patches) at test scale:
        // the same patch-embedding ViT, with 4 patches per side.
        var options = new SAMOptions
        {
            ImageSize = 32,
            PatchSize = 8,
            EmbeddingDim = 32,
            NumLayers = 2,
            NumHeads = 4,
            DropoutRate = 0.0,
        };
        return new SAM<double>(architecture, options);
    }

    // SAM trains on an image and its ground-truth mask through the paper's interactive prompt simulation, not on a
    // target for its embedding. Every invariant trains it on a mask derived from the image, so a given image always
    // carries the same mask.
    protected override void TrainOn(INeuralNetworkModel<double> network, Tensor<double> input, Tensor<double> target)
    {
        if (network is SAM<double> sam)
        {
            sam.Train(input, MaskFor(input));
            return;
        }
        base.TrainOn(network, input, target);
    }

    // Measured on the objective SAM trains (focal + dice + IoU over simulated prompts), with the prompts drawn from a
    // fixed seed so two measurements of the same weights agree.
    protected override double MeasureLoss(
        INeuralNetworkModel<double> network, Tensor<double> input, Tensor<double> output, Tensor<double> target)
    {
        if (network is not SAM<double> sam)
            return base.MeasureLoss(network, input, output, target);
        using var noGrad = new NoGradScope<double>();
        var training = new AiDotNet.ComputerVision.Segmentation.Foundation.SamTraining<double>(
            AiDotNetEngine.Current, sam.Network, 20.0, 1.0, 2.0, 0.25);
        var loss = training.Objective(input, MaskFor(input), new Random(MeasurementSeed))();
        return loss[0];
    }

    // GetLastLoss is one random draw of the prompts; the memorization probe measures the fixed-seed objective instead.
    protected override bool MemorizationTaskUsesDeterministicEvalLoss => true;

    private const int MeasurementSeed = 7;

    // Foreground where the pixel's channel mean is above one half: one H*W mask per image.
    private static Tensor<double> MaskFor(Tensor<double> image)
    {
        int channels = image.Shape[1], height = image.Shape[2], width = image.Shape[3];
        var mask = new double[height * width];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                double sum = 0;
                for (int c = 0; c < channels; c++) sum += image[0, c, y, x];
                mask[y * width + x] = sum / channels > 0.5 ? 1.0 : 0.0;
            }
        return new Tensor<double>(new[] { height * width }, new Vector<double>(mask));
    }
}
