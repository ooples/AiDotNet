using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// The R-CNN family is built as its papers build it, with parameter counts checked against torchvision 0.29's
/// <c>fasterrcnn_resnet50_fpn(num_classes=91)</c>: backbone 23,508,032, FPN 3,344,384, box head 14,362,055.
/// </summary>
public class DetectorPaperLayoutTests
{
    // AiDotNet's BatchNormalizationLayer also counts its running mean and variance: four values per channel against
    // torchvision's two trainable ones. ResNet-50 has 26,560 normalized channels.
    private const long ResNet50RunningStatistics = 2L * 26_560;

    [Fact]
    public void ResNet50Backbone_NormalizesEveryConvolution_AsHeEtAlDo()
    {
        // Bias-free convolutions plus a batch normalization after each: torchvision's ResNet-50 body.
        var backbone = new ResNet<double>(options: new ResNetBackboneOptions { Variant = ResNetVariant.ResNet50 });

        Assert.Equal(23_508_032L + ResNet50RunningStatistics, backbone.ParameterCount);
        // The trainable count leaves the running statistics out and is torchvision's exactly.
        Assert.Equal(23_508_032L, backbone.TrainableParameterCount);
    }

    [Fact]
    public void Fpn_SmoothsEachLevelWithA3x3Convolution()
    {
        var fpn = new FPN<double>(new[] { 256, 512, 1024, 2048 }, outputChannels: 256);

        // Four 1x1 laterals plus four 3x3 output convolutions, all biased.
        Assert.Equal(3_344_384L, fpn.GetParameterCount());
    }

    [Fact]
    public void FpnOutputs_AreLinear()
    {
        // "There are no non-linearities in these extra layers" (Lin et al. 2017, Sec. 3): a ReLU on the outputs
        // would make every pyramid value nonnegative.
        int[] channels = { 4, 8, 16, 32 };
        var fpn = new FPN<double>(channels, outputChannels: 8);
        var random = new Random(3);
        var features = new List<Tensor<double>>();
        for (int level = 0; level < channels.Length; level++)
        {
            int size = 16 >> level;
            var map = new Tensor<double>(new[] { 1, channels[level], size, size });
            for (int i = 0; i < map.Length; i++) map[i] = random.NextDouble() * 2 - 1;
            features.Add(map);
        }

        var outputs = fpn.Forward(features);

        Assert.Contains(outputs, output => output.ToArray().Any(value => value < 0));
    }

    [Fact]
    public void FasterRcnn_HasTheFpnBoxHeadAndRpnWidth()
    {
        var model = new FasterRCNN<double>(new ObjectDetectionOptions<double>
        {
            Size = ModelSize.Medium,
            NumClasses = 90,
            InputSize = new[] { 64, 64 },
        });
        // The convolutions resolve their shapes on the first forward; count after it.
        model.Predict(new Tensor<double>(new[] { 1, 3, 64, 64 }));

        // torchvision's 41,808,406, plus the BatchNorm running statistics, plus 771 for the RPN's 2k-way softmax
        // objectness (Ren et al. 2015) where torchvision uses k sigmoid scores.
        Assert.Equal(41_808_406L + ResNet50RunningStatistics + 771, model.ParameterCount);
        Assert.Equal(41_808_406L + 771, model.TrainableParameterCount);
    }
}
