using AiDotNet.ComputerVision.Segmentation.Panoptic;
using AiDotNet.ComputerVision.Segmentation.Mamba;
using AiDotNet.ComputerVision.Segmentation.Efficient;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Panoptic (KMaXDeepLab, ODISE, CUPS),
/// Mamba (VisionMamba, VMamba, ViMUNet), and
/// Efficient (PIDNet, FastSAM, MobileSAM, EdgeSAM, SlimSAM, EfficientSAM, RepViTSAM) segmentation models.
/// </summary>
public class PanopticMambaEfficientSegmentationIntegrationTests
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

    #region KMaXDeepLab

    [Fact]
    public void KMaXDeepLab_Construction_Succeeds()
    {
        var model = new KMaXDeepLab<double>(Arch(), modelSize: KMaXDeepLabModelSize.R50);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void KMaXDeepLab_Predict_ReturnsOutput()
    {
        var model = new KMaXDeepLab<double>(Arch(), modelSize: KMaXDeepLabModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void KMaXDeepLab_Dispose_DoesNotThrow()
    {
        var model = new KMaXDeepLab<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region ODISE

    [Fact]
    public void ODISE_Construction_Succeeds()
    {
        var model = new ODISE<double>(Arch(), modelSize: ODISEModelSize.Base);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void ODISE_Predict_ReturnsOutput()
    {
        var model = new ODISE<double>(Arch(), modelSize: ODISEModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ODISE_Dispose_DoesNotThrow()
    {
        var model = new ODISE<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region CUPS

    [Fact]
    public void CUPS_Construction_Succeeds()
    {
        var model = new CUPS<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void CUPS_Predict_ReturnsOutput()
    {
        var model = new CUPS<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void CUPS_Dispose_DoesNotThrow()
    {
        var model = new CUPS<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region VisionMamba

    [Fact]
    public void VisionMamba_Construction_Succeeds()
    {
        var model = new VisionMamba<double>(Arch(), modelSize: VisionMambaModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void VisionMamba_Predict_ReturnsOutput()
    {
        var model = new VisionMamba<double>(Arch(), modelSize: VisionMambaModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void VisionMamba_Dispose_DoesNotThrow()
    {
        var model = new VisionMamba<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region VMamba

    [Fact]
    public void VMamba_Construction_Succeeds()
    {
        var model = new VMamba<double>(Arch(), modelSize: VMambaModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void VMamba_Predict_ReturnsOutput()
    {
        var model = new VMamba<double>(Arch(), modelSize: VMambaModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void VMamba_Dispose_DoesNotThrow()
    {
        var model = new VMamba<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region ViMUNet

    [Fact]
    public void ViMUNet_Construction_Succeeds()
    {
        var model = new ViMUNet<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void ViMUNet_Predict_ReturnsOutput()
    {
        var model = new ViMUNet<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void ViMUNet_Dispose_DoesNotThrow()
    {
        var model = new ViMUNet<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region PIDNet

    [Fact]
    public void PIDNet_Construction_Succeeds()
    {
        var model = new PIDNet<double>(Arch(), modelSize: PIDNetModelSize.Small);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void PIDNet_Predict_ReturnsOutput()
    {
        var model = new PIDNet<double>(Arch(), modelSize: PIDNetModelSize.Small);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void PIDNet_Dispose_DoesNotThrow()
    {
        var model = new PIDNet<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region FastSAM

    [Fact]
    public void FastSAM_Construction_Succeeds()
    {
        var model = new FastSAM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void FastSAM_Predict_ReturnsOutput()
    {
        var model = new FastSAM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void FastSAM_Dispose_DoesNotThrow()
    {
        var model = new FastSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MobileSAM

    [Fact]
    public void MobileSAM_Construction_Succeeds()
    {
        var model = new MobileSAM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void MobileSAM_Predict_ReturnsOutput()
    {
        var model = new MobileSAM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MobileSAM_Dispose_DoesNotThrow()
    {
        var model = new MobileSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region EdgeSAM

    [Fact]
    public void EdgeSAM_Construction_Succeeds()
    {
        var model = new EdgeSAM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void EdgeSAM_Predict_ReturnsOutput()
    {
        var model = new EdgeSAM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void EdgeSAM_Dispose_DoesNotThrow()
    {
        var model = new EdgeSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SlimSAM

    [Fact]
    public void SlimSAM_Construction_Succeeds()
    {
        var model = new SlimSAM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SlimSAM_Predict_ReturnsOutput()
    {
        var model = new SlimSAM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SlimSAM_Dispose_DoesNotThrow()
    {
        var model = new SlimSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region EfficientSAM

    [Fact]
    public void EfficientSAM_Construction_Succeeds()
    {
        var model = new EfficientSAM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void EfficientSAM_Predict_ReturnsOutput()
    {
        var model = new EfficientSAM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void EfficientSAM_Dispose_DoesNotThrow()
    {
        var model = new EfficientSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region RepViTSAM

    [Fact]
    public void RepViTSAM_Construction_Succeeds()
    {
        var model = new RepViTSAM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void RepViTSAM_Predict_ReturnsOutput()
    {
        var model = new RepViTSAM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void RepViTSAM_Dispose_DoesNotThrow()
    {
        var model = new RepViTSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion
}
