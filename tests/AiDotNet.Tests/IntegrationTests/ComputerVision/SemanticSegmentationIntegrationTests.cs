using AiDotNet.ComputerVision.Segmentation.Semantic;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Semantic segmentation models (excluding SegFormer which has its own tests):
/// SegNeXt, InternImage, ViTAdapter, ViTCoMer, DiffCut, DiffSeg.
/// </summary>
public class SemanticSegmentationIntegrationTests
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

    #region SegNeXt

    [Fact]
    public void SegNeXt_Construction_Succeeds()
    {
        var model = new SegNeXt<double>(Arch(), modelSize: SegNeXtModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SegNeXt_Predict_ReturnsOutput()
    {
        var model = new SegNeXt<double>(Arch(), modelSize: SegNeXtModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SegNeXt_Train_DoesNotThrow()
    {
        var model = new SegNeXt<double>(Arch(), modelSize: SegNeXtModelSize.Tiny);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void SegNeXt_Dispose_DoesNotThrow()
    {
        var model = new SegNeXt<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region InternImage

    [Fact]
    public void InternImage_Construction_Succeeds()
    {
        var model = new InternImage<double>(Arch(), modelSize: InternImageModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void InternImage_Predict_ReturnsOutput()
    {
        var model = new InternImage<double>(Arch(), modelSize: InternImageModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void InternImage_Dispose_DoesNotThrow()
    {
        var model = new InternImage<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region ViTAdapter

    [Fact]
    public void ViTAdapter_Construction_Succeeds()
    {
        var model = new ViTAdapter<double>(Arch(), modelSize: ViTAdapterModelSize.Base);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void ViTAdapter_Predict_ReturnsOutput()
    {
        var model = new ViTAdapter<double>(Arch(), modelSize: ViTAdapterModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ViTAdapter_Dispose_DoesNotThrow()
    {
        var model = new ViTAdapter<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region ViTCoMer

    [Fact]
    public void ViTCoMer_Construction_Succeeds()
    {
        var model = new ViTCoMer<double>(Arch(), modelSize: ViTCoMerModelSize.Small);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void ViTCoMer_Predict_ReturnsOutput()
    {
        var model = new ViTCoMer<double>(Arch(), modelSize: ViTCoMerModelSize.Small);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ViTCoMer_Dispose_DoesNotThrow()
    {
        var model = new ViTCoMer<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region DiffCut

    [Fact]
    public void DiffCut_Construction_Succeeds()
    {
        var model = new DiffCut<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void DiffCut_Predict_ReturnsOutput()
    {
        var model = new DiffCut<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void DiffCut_Dispose_DoesNotThrow()
    {
        var model = new DiffCut<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region DiffSeg

    [Fact]
    public void DiffSeg_Construction_Succeeds()
    {
        var model = new DiffSeg<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void DiffSeg_Predict_ReturnsOutput()
    {
        var model = new DiffSeg<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void DiffSeg_Dispose_DoesNotThrow()
    {
        var model = new DiffSeg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion
}
