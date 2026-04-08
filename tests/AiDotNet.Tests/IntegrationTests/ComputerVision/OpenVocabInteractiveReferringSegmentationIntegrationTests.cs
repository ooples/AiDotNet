using AiDotNet.ComputerVision.Segmentation.OpenVocabulary;
using AiDotNet.ComputerVision.Segmentation.Interactive;
using AiDotNet.ComputerVision.Segmentation.Referring;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using AiDotNet.Tensors.Helpers;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Open-Vocabulary (OpenVocabSAM, GroundedSAM2, CATSeg, MaskAdapter, SAN, SED),
/// Interactive (SegGPT, SEEM), and
/// Referring (LISA, VideoLISA, GLaMM, OMGLLaVA, PixelLM) segmentation models.
/// </summary>
public class OpenVocabInteractiveReferringSegmentationIntegrationTests
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

    #region OpenVocabSAM

    [Fact(Timeout = 120000)]
    public async Task OpenVocabSAM_Construction_Succeeds()
    {
        var model = new OpenVocabSAM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task OpenVocabSAM_Predict_ReturnsOutput()
    {
        var model = new OpenVocabSAM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task OpenVocabSAM_Dispose_DoesNotThrow()
    {
        var model = new OpenVocabSAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region GroundedSAM2

    [Fact(Timeout = 120000)]
    public async Task GroundedSAM2_Construction_Succeeds()
    {
        var model = new GroundedSAM2<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task GroundedSAM2_Predict_ReturnsOutput()
    {
        var model = new GroundedSAM2<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task GroundedSAM2_Dispose_DoesNotThrow()
    {
        var model = new GroundedSAM2<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region CATSeg

    [Fact(Timeout = 120000)]
    public async Task CATSeg_Construction_Succeeds()
    {
        var model = new CATSeg<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task CATSeg_Predict_ReturnsOutput()
    {
        var model = new CATSeg<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task CATSeg_Dispose_DoesNotThrow()
    {
        var model = new CATSeg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MaskAdapter

    [Fact(Timeout = 120000)]
    public async Task MaskAdapter_Construction_Succeeds()
    {
        var model = new MaskAdapter<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskAdapter_Predict_ReturnsOutput()
    {
        var model = new MaskAdapter<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskAdapter_Dispose_DoesNotThrow()
    {
        var model = new MaskAdapter<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SAN

    [Fact(Timeout = 120000)]
    public async Task SAN_Construction_Succeeds()
    {
        var model = new SAN<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SAN_Predict_ReturnsOutput()
    {
        var model = new SAN<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SAN_Dispose_DoesNotThrow()
    {
        var model = new SAN<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SED

    [Fact(Timeout = 120000)]
    public async Task SED_Construction_Succeeds()
    {
        var model = new SED<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SED_Predict_ReturnsOutput()
    {
        var model = new SED<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SED_Dispose_DoesNotThrow()
    {
        var model = new SED<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SegGPT

    [Fact(Timeout = 120000)]
    public async Task SegGPT_Construction_Succeeds()
    {
        var model = new SegGPT<double>(Arch(), modelSize: SegGPTModelSize.ViTLarge);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SegGPT_Predict_ReturnsOutput()
    {
        var model = new SegGPT<double>(Arch(), modelSize: SegGPTModelSize.ViTLarge);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SegGPT_Dispose_DoesNotThrow()
    {
        var model = new SegGPT<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SEEM

    [Fact(Timeout = 120000)]
    public async Task SEEM_Construction_Succeeds()
    {
        var model = new SEEM<double>(Arch(), modelSize: SEEMModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SEEM_Predict_ReturnsOutput()
    {
        var model = new SEEM<double>(Arch(), modelSize: SEEMModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SEEM_Dispose_DoesNotThrow()
    {
        var model = new SEEM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region LISA

    [Fact(Timeout = 120000)]
    public async Task LISA_Construction_Succeeds()
    {
        var model = new LISA<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task LISA_Predict_ReturnsOutput()
    {
        var model = new LISA<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task LISA_Dispose_DoesNotThrow()
    {
        var model = new LISA<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region VideoLISA

    [Fact(Timeout = 120000)]
    public async Task VideoLISA_Construction_Succeeds()
    {
        var model = new VideoLISA<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoLISA_Predict_ReturnsOutput()
    {
        var model = new VideoLISA<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoLISA_Dispose_DoesNotThrow()
    {
        var model = new VideoLISA<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region GLaMM

    [Fact(Timeout = 120000)]
    public async Task GLaMM_Construction_Succeeds()
    {
        var model = new GLaMM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task GLaMM_Predict_ReturnsOutput()
    {
        var model = new GLaMM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task GLaMM_Dispose_DoesNotThrow()
    {
        var model = new GLaMM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region OMGLLaVA

    [Fact(Timeout = 120000)]
    public async Task OMGLLaVA_Construction_Succeeds()
    {
        var model = new OMGLLaVA<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task OMGLLaVA_Predict_ReturnsOutput()
    {
        var model = new OMGLLaVA<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task OMGLLaVA_Dispose_DoesNotThrow()
    {
        var model = new OMGLLaVA<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region PixelLM

    [Fact(Timeout = 120000)]
    public async Task PixelLM_Construction_Succeeds()
    {
        var model = new PixelLM<double>(Arch());
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task PixelLM_Predict_ReturnsOutput()
    {
        var model = new PixelLM<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task PixelLM_Dispose_DoesNotThrow()
    {
        var model = new PixelLM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion
}
