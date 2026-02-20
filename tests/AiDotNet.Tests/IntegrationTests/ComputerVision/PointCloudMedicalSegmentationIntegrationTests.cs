using AiDotNet.ComputerVision.Segmentation.PointCloud;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Point Cloud (PointTransformerV3, Sonata, Concerto) and
/// Medical (MedSAM, MedSAM2, MedNeXt, NnUNet, SwinUNETR, TransUNet, UMamba,
/// UniverSeg, SegMamba, BiomedParse, MedSegDiffV2) segmentation models.
/// </summary>
public class PointCloudMedicalSegmentationIntegrationTests
{
    private static NeuralNetworkArchitecture<double> Arch(int h = 32, int w = 32, int d = 3)
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
               NetworkComplexity.Deep, 0, h, w, d, 0);

    private static Tensor<double> Rand(params int[] shape)
    {
        int total = 1; foreach (int s in shape) total *= s;
        var data = new double[total];
        var rng = new Random(42);
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    #region PointTransformerV3

    [Fact]
    public void PointTransformerV3_Construction_Succeeds()
    {
        var model = new PointTransformerV3<double>(Arch(), modelSize: PointTransformerV3ModelSize.Base);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void PointTransformerV3_Predict_ReturnsOutput()
    {
        var model = new PointTransformerV3<double>(Arch(), modelSize: PointTransformerV3ModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void PointTransformerV3_Dispose_DoesNotThrow()
    {
        var model = new PointTransformerV3<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region Sonata

    [Fact]
    public void Sonata_Construction_Succeeds()
    {
        var model = new Sonata<double>(Arch(), modelSize: SonataModelSize.Base);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void Sonata_Predict_ReturnsOutput()
    {
        var model = new Sonata<double>(Arch(), modelSize: SonataModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void Sonata_Dispose_DoesNotThrow()
    {
        var model = new Sonata<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region Concerto

    [Fact]
    public void Concerto_Construction_Succeeds()
    {
        var model = new Concerto<double>(Arch(), modelSize: ConcertoModelSize.Base);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void Concerto_Predict_ReturnsOutput()
    {
        var model = new Concerto<double>(Arch(), modelSize: ConcertoModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void Concerto_Dispose_DoesNotThrow()
    {
        var model = new Concerto<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MedSAM

    [Fact]
    public void MedSAM_Construction_Succeeds()
    {
        var model = new MedSAM<double>(Arch(), modelSize: MedSAMModelSize.ViTBase);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void MedSAM_Predict_ReturnsOutput()
    {
        var model = new MedSAM<double>(Arch(), modelSize: MedSAMModelSize.ViTBase);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MedSAM_Train_DoesNotThrow()
    {
        var model = new MedSAM<double>(Arch(), modelSize: MedSAMModelSize.ViTBase);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void MedSAM_Dispose_DoesNotThrow()
    {
        var model = new MedSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MedSAM2

    [Fact]
    public void MedSAM2_Construction_Succeeds()
    {
        var model = new MedSAM2<double>(Arch(), modelSize: MedSAM2ModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void MedSAM2_Predict_ReturnsOutput()
    {
        var model = new MedSAM2<double>(Arch(), modelSize: MedSAM2ModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MedSAM2_Dispose_DoesNotThrow()
    {
        var model = new MedSAM2<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MedNeXt

    [Fact]
    public void MedNeXt_Construction_Succeeds()
    {
        var model = new MedNeXt<double>(Arch(), modelSize: MedNeXtModelSize.Small);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void MedNeXt_Predict_ReturnsOutput()
    {
        var model = new MedNeXt<double>(Arch(), modelSize: MedNeXtModelSize.Small);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MedNeXt_Dispose_DoesNotThrow()
    {
        var model = new MedNeXt<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region NnUNet

    [Fact]
    public void NnUNet_Construction_Succeeds()
    {
        var model = new NnUNet<double>(Arch(), modelSize: NnUNetModelSize.UNet2D);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void NnUNet_Predict_ReturnsOutput()
    {
        var model = new NnUNet<double>(Arch(), modelSize: NnUNetModelSize.UNet2D);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void NnUNet_Dispose_DoesNotThrow()
    {
        var model = new NnUNet<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SwinUNETR

    [Fact]
    public void SwinUNETR_Construction_Succeeds()
    {
        var model = new SwinUNETR<double>(Arch(), modelSize: SwinUNETRModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SwinUNETR_Predict_ReturnsOutput()
    {
        var model = new SwinUNETR<double>(Arch(), modelSize: SwinUNETRModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SwinUNETR_Dispose_DoesNotThrow()
    {
        var model = new SwinUNETR<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region TransUNet

    [Fact]
    public void TransUNet_Construction_Succeeds()
    {
        var model = new TransUNet<double>(Arch(), modelSize: TransUNetModelSize.Base);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void TransUNet_Predict_ReturnsOutput()
    {
        var model = new TransUNet<double>(Arch(), modelSize: TransUNetModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void TransUNet_Dispose_DoesNotThrow()
    {
        var model = new TransUNet<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region UMamba

    [Fact]
    public void UMamba_Construction_Succeeds()
    {
        var model = new UMamba<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void UMamba_Predict_ReturnsOutput()
    {
        var model = new UMamba<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void UMamba_Dispose_DoesNotThrow()
    {
        var model = new UMamba<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region UniverSeg

    [Fact]
    public void UniverSeg_Construction_Succeeds()
    {
        var model = new UniverSeg<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void UniverSeg_Predict_ReturnsOutput()
    {
        var model = new UniverSeg<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void UniverSeg_Dispose_DoesNotThrow()
    {
        var model = new UniverSeg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SegMamba

    [Fact]
    public void SegMamba_Construction_Succeeds()
    {
        var model = new SegMamba<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SegMamba_Predict_ReturnsOutput()
    {
        var model = new SegMamba<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SegMamba_Dispose_DoesNotThrow()
    {
        var model = new SegMamba<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region BiomedParse

    [Fact]
    public void BiomedParse_Construction_Succeeds()
    {
        var model = new BiomedParse<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void BiomedParse_Predict_ReturnsOutput()
    {
        var model = new BiomedParse<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void BiomedParse_Dispose_DoesNotThrow()
    {
        var model = new BiomedParse<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MedSegDiffV2

    [Fact]
    public void MedSegDiffV2_Construction_Succeeds()
    {
        var model = new MedSegDiffV2<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void MedSegDiffV2_Predict_ReturnsOutput()
    {
        var model = new MedSegDiffV2<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MedSegDiffV2_Dispose_DoesNotThrow()
    {
        var model = new MedSegDiffV2<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion
}
