using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.ComputerVision.Segmentation.Diffusion;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Video (DEVA, EfficientTAM, UniVS) and
/// Diffusion-based (DiffCutSegmentation, ODISESegmentation, MedSegDiffV2Segmentation) segmentation models.
/// </summary>
public class VideoDiffusionSegmentationIntegrationTests
{
    private static NeuralNetworkArchitecture<double> Arch(int h = 32, int w = 32, int d = 3)
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
               NetworkComplexity.Deep, 0, h, w, d, 0);

    private static Tensor<double> Rand(params int[] shape)
    {
        int total = 1; foreach (int s in shape) total *= s;
        var data = new double[total];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    #region DEVA

    [Fact]
    public void DEVA_Construction_Succeeds()
    {
        var model = new DEVA<double>(Arch(), modelSize: DEVAModelSize.Base);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void DEVA_Predict_ReturnsOutput()
    {
        var model = new DEVA<double>(Arch(), modelSize: DEVAModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void DEVA_Train_DoesNotThrow()
    {
        var model = new DEVA<double>(Arch(), modelSize: DEVAModelSize.Base);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void DEVA_Dispose_DoesNotThrow()
    {
        var model = new DEVA<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region EfficientTAM

    [Fact]
    public void EfficientTAM_Construction_Succeeds()
    {
        var model = new EfficientTAM<double>(Arch(), modelSize: EfficientTAMModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void EfficientTAM_Predict_ReturnsOutput()
    {
        var model = new EfficientTAM<double>(Arch(), modelSize: EfficientTAMModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void EfficientTAM_Dispose_DoesNotThrow()
    {
        var model = new EfficientTAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region UniVS

    [Fact]
    public void UniVS_Construction_Succeeds()
    {
        var model = new UniVS<double>(Arch(), modelSize: UniVSModelSize.R50);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void UniVS_Predict_ReturnsOutput()
    {
        var model = new UniVS<double>(Arch(), modelSize: UniVSModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void UniVS_Dispose_DoesNotThrow()
    {
        var model = new UniVS<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region DiffCutSegmentation

    [Fact]
    public void DiffCutSegmentation_Construction_Succeeds()
    {
        var model = new DiffCutSegmentation<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void DiffCutSegmentation_Predict_ReturnsOutput()
    {
        var model = new DiffCutSegmentation<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void DiffCutSegmentation_Dispose_DoesNotThrow()
    {
        var model = new DiffCutSegmentation<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region ODISESegmentation

    [Fact]
    public void ODISESegmentation_Construction_Succeeds()
    {
        var model = new ODISESegmentation<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void ODISESegmentation_Predict_ReturnsOutput()
    {
        var model = new ODISESegmentation<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ODISESegmentation_Dispose_DoesNotThrow()
    {
        var model = new ODISESegmentation<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MedSegDiffV2Segmentation

    [Fact]
    public void MedSegDiffV2Segmentation_Construction_Succeeds()
    {
        var model = new MedSegDiffV2Segmentation<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void MedSegDiffV2Segmentation_Predict_ReturnsOutput()
    {
        var model = new MedSegDiffV2Segmentation<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MedSegDiffV2Segmentation_Dispose_DoesNotThrow()
    {
        var model = new MedSegDiffV2Segmentation<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion
}
