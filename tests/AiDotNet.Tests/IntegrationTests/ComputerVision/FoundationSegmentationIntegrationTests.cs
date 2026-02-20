using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Foundation segmentation models: SAM, SAM 2.1, SAM-HQ,
/// Mask2Former, OneFormer, MaskDINO, OMGSeg, EoMT, QueryMeldNet, UNINEXT, U2Seg, XDecoder.
/// </summary>
public class FoundationSegmentationIntegrationTests
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

    #region SAM

    [Fact]
    public void SAM_Construction_NativeMode_Succeeds()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SAM_Predict_ReturnsOutput()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SAM_Train_DoesNotThrow()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void SAM_Metadata_ReturnsCorrectModelType()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var meta = model.GetModelMetadata();
        Assert.Equal("SAM", meta.AdditionalInfo["ModelName"]);
    }

    [Fact]
    public void SAM_Dispose_DoesNotThrow()
    {
        var model = new SAM<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact]
    public void SAM_PromptableSegmentation_SetImageAndSegmentFromPoints()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<double>)model;
        Assert.True(promptable.SupportsPointPrompts);
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
        Assert.True(result.Masks.Length > 0);
    }

    #endregion

    #region SAM 2.1

    [Fact]
    public void SAM21_Construction_NativeMode_Succeeds()
    {
        var model = new SAM21<double>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SAM21_Predict_ReturnsOutput()
    {
        var model = new SAM21<double>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SAM21_Train_DoesNotThrow()
    {
        var model = new SAM21<double>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.Tiny);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void SAM21_Dispose_DoesNotThrow()
    {
        var model = new SAM21<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SAM-HQ

    [Fact]
    public void SAMHQ_Construction_NativeMode_Succeeds()
    {
        var model = new SAMHQ<double>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SAMHQ_Predict_ReturnsOutput()
    {
        var model = new SAMHQ<double>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SAMHQ_Dispose_DoesNotThrow()
    {
        var model = new SAMHQ<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region Mask2Former

    [Fact]
    public void Mask2Former_Construction_Succeeds()
    {
        var model = new Mask2Former<double>(Arch(), modelSize: Mask2FormerModelSize.SwinTiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void Mask2Former_Predict_ReturnsOutput()
    {
        var model = new Mask2Former<double>(Arch(), modelSize: Mask2FormerModelSize.SwinTiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void Mask2Former_Dispose_DoesNotThrow()
    {
        var model = new Mask2Former<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region OneFormer

    [Fact]
    public void OneFormer_Construction_Succeeds()
    {
        var model = new OneFormer<double>(Arch(), modelSize: OneFormerModelSize.SwinLarge);
        Assert.NotNull(model);
    }

    [Fact]
    public void OneFormer_Predict_ReturnsOutput()
    {
        var model = new OneFormer<double>(Arch(), modelSize: OneFormerModelSize.SwinLarge);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void OneFormer_Dispose_DoesNotThrow()
    {
        var model = new OneFormer<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MaskDINO

    [Fact]
    public void MaskDINO_Construction_Succeeds()
    {
        var model = new MaskDINO<double>(Arch(), modelSize: MaskDINOModelSize.R50);
        Assert.NotNull(model);
    }

    [Fact]
    public void MaskDINO_Predict_ReturnsOutput()
    {
        var model = new MaskDINO<double>(Arch(), modelSize: MaskDINOModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void MaskDINO_Dispose_DoesNotThrow()
    {
        var model = new MaskDINO<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region OMGSeg

    [Fact]
    public void OMGSeg_Construction_Succeeds()
    {
        var model = new OMGSeg<double>(Arch(), modelSize: OMGSegModelSize.Large);
        Assert.NotNull(model);
    }

    [Fact]
    public void OMGSeg_Predict_ReturnsOutput()
    {
        var model = new OMGSeg<double>(Arch(), modelSize: OMGSegModelSize.Large);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void OMGSeg_Dispose_DoesNotThrow()
    {
        var model = new OMGSeg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region EoMT

    [Fact]
    public void EoMT_Construction_Succeeds()
    {
        var model = new EoMT<double>(Arch(), modelSize: EoMTModelSize.Base);
        Assert.NotNull(model);
    }

    [Fact]
    public void EoMT_Predict_ReturnsOutput()
    {
        var model = new EoMT<double>(Arch(), modelSize: EoMTModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void EoMT_Dispose_DoesNotThrow()
    {
        var model = new EoMT<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region QueryMeldNet

    [Fact]
    public void QueryMeldNet_Construction_Succeeds()
    {
        var model = new QueryMeldNet<double>(Arch(), modelSize: QueryMeldNetModelSize.R50);
        Assert.NotNull(model);
    }

    [Fact]
    public void QueryMeldNet_Predict_ReturnsOutput()
    {
        var model = new QueryMeldNet<double>(Arch(), modelSize: QueryMeldNetModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void QueryMeldNet_Dispose_DoesNotThrow()
    {
        var model = new QueryMeldNet<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region UNINEXT

    [Fact]
    public void UNINEXT_Construction_Succeeds()
    {
        var model = new UNINEXT<double>(Arch(), modelSize: UNINEXTModelSize.R50);
        Assert.NotNull(model);
    }

    [Fact]
    public void UNINEXT_Predict_ReturnsOutput()
    {
        var model = new UNINEXT<double>(Arch(), modelSize: UNINEXTModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void UNINEXT_Dispose_DoesNotThrow()
    {
        var model = new UNINEXT<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region U2Seg

    [Fact]
    public void U2Seg_Construction_Succeeds()
    {
        var model = new U2Seg<double>(Arch());
        Assert.NotNull(model);
    }

    [Fact]
    public void U2Seg_Predict_ReturnsOutput()
    {
        var model = new U2Seg<double>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void U2Seg_Dispose_DoesNotThrow()
    {
        var model = new U2Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region XDecoder

    [Fact]
    public void XDecoder_Construction_Succeeds()
    {
        var model = new XDecoder<double>(Arch(), modelSize: XDecoderModelSize.Tiny);
        Assert.NotNull(model);
    }

    [Fact]
    public void XDecoder_Predict_ReturnsOutput()
    {
        var model = new XDecoder<double>(Arch(), modelSize: XDecoderModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void XDecoder_Dispose_DoesNotThrow()
    {
        var model = new XDecoder<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion
}
