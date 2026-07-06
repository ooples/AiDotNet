using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Foundation segmentation models: SAM, SAM 2.1, SAM-HQ,
/// Mask2Former, OneFormer, MaskDINO, OMGSeg, EoMT, QueryMeldNet, UNINEXT, U2Seg, XDecoder.
/// </summary>
public class FoundationSegmentationIntegrationTests : IDisposable
{
    // Heavy SAM/ViT-family models are constructed one per test; reclaim process-global retention
    // (InferenceWeightCache weight-pack pins + uncompacted LOH) between tests so committed memory does
    // not accumulate across the class until a later test OOMs. Pure hygiene — same teardown the
    // NeuralNetworks/Diffusion model-family bases use; changes no assertion, scale, or iteration count.
    public void Dispose() => ModelFamilyTestGcGate.ReclaimBetweenTests();

    private static NeuralNetworkArchitecture<float> Arch(int h = 32, int w = 32, int d = 3)
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
               NetworkComplexity.Deep, 0, h, w, d, 0);

    private static Tensor<float> Rand(params int[] shape)
    {
        int total = 1; foreach (int s in shape) total *= s;
        var data = new float[total];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < total; i++) data[i] = (float)rng.NextDouble();
        return new Tensor<float>(shape, new Vector<float>(data));
    }

    #region SAM

    [Fact(Timeout = 120000)]
    public async Task SAM_Construction_NativeMode_Succeeds()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_Predict_ReturnsOutput()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_Train_DoesNotThrow()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_Metadata_ReturnsCorrectModelType()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var meta = model.GetModelMetadata();
        Assert.Equal("SAM", meta.AdditionalInfo["ModelName"]);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_Dispose_DoesNotThrow()
    {
        var model = new SAM<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_PromptableSegmentation_SetImageAndSegmentFromPoints()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<float>)model;
        Assert.True(promptable.SupportsPointPrompts);
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
        Assert.True(result.Masks.Length > 0);
    }

    #endregion

    #region SAM 2.1

    [Fact(Timeout = 120000)]
    public async Task SAM21_Construction_NativeMode_Succeeds()
    {
        var model = new SAM21<float>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.Tiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM21_Predict_ReturnsOutput()
    {
        var model = new SAM21<float>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM21_Train_DoesNotThrow()
    {
        var model = new SAM21<float>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.Tiny);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task SAM21_Dispose_DoesNotThrow()
    {
        var model = new SAM21<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region SAM-HQ

    [Fact(Timeout = 120000)]
    public async Task SAMHQ_Construction_NativeMode_Succeeds()
    {
        var model = new SAMHQ<float>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SAMHQ_Predict_ReturnsOutput()
    {
        var model = new SAMHQ<float>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SAMHQ_Dispose_DoesNotThrow()
    {
        var model = new SAMHQ<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region Mask2Former

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_Construction_Succeeds()
    {
        var model = new Mask2Former<float>(Arch(), modelSize: Mask2FormerModelSize.SwinTiny);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_Predict_ReturnsOutput()
    {
        var model = new Mask2Former<float>(Arch(), modelSize: Mask2FormerModelSize.SwinTiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_Dispose_DoesNotThrow()
    {
        var model = new Mask2Former<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region OneFormer

    [Fact(Timeout = 120000)]
    public async Task OneFormer_Construction_Succeeds()
    {
        var model = new OneFormer<float>(Arch(), modelSize: OneFormerModelSize.SwinLarge);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task OneFormer_Predict_ReturnsOutput()
    {
        var model = new OneFormer<float>(Arch(), modelSize: OneFormerModelSize.SwinLarge);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task OneFormer_Dispose_DoesNotThrow()
    {
        var model = new OneFormer<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MaskDINO

    [Fact(Timeout = 120000)]
    public async Task MaskDINO_Construction_Succeeds()
    {
        var model = new MaskDINO<float>(Arch(), modelSize: MaskDINOModelSize.R50);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskDINO_Predict_ReturnsOutput()
    {
        var model = new MaskDINO<float>(Arch(), modelSize: MaskDINOModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskDINO_Dispose_DoesNotThrow()
    {
        var model = new MaskDINO<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region OMGSeg

    [Fact(Timeout = 120000)]
    public async Task OMGSeg_Construction_Succeeds()
    {
        var model = new OMGSeg<float>(Arch(), modelSize: OMGSegModelSize.Large);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task OMGSeg_Predict_ReturnsOutput()
    {
        var model = new OMGSeg<float>(Arch(), modelSize: OMGSegModelSize.Large);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task OMGSeg_Dispose_DoesNotThrow()
    {
        var model = new OMGSeg<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region EoMT

    [Fact(Timeout = 120000)]
    public async Task EoMT_Construction_Succeeds()
    {
        var model = new EoMT<float>(Arch(), modelSize: EoMTModelSize.Base);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task EoMT_Predict_ReturnsOutput()
    {
        var model = new EoMT<float>(Arch(), modelSize: EoMTModelSize.Base);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task EoMT_Dispose_DoesNotThrow()
    {
        var model = new EoMT<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region QueryMeldNet

    [Fact(Timeout = 120000)]
    public async Task QueryMeldNet_Construction_Succeeds()
    {
        var model = new QueryMeldNet<float>(Arch(), modelSize: QueryMeldNetModelSize.R50);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task QueryMeldNet_Predict_ReturnsOutput()
    {
        var model = new QueryMeldNet<float>(Arch(), modelSize: QueryMeldNetModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task QueryMeldNet_Dispose_DoesNotThrow()
    {
        var model = new QueryMeldNet<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region UNINEXT

    [Fact(Timeout = 120000)]
    public async Task UNINEXT_Construction_Succeeds()
    {
        var model = new UNINEXT<float>(Arch(), modelSize: UNINEXTModelSize.R50);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task UNINEXT_Predict_ReturnsOutput()
    {
        var model = new UNINEXT<float>(Arch(), modelSize: UNINEXTModelSize.R50);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task UNINEXT_Dispose_DoesNotThrow()
    {
        var model = new UNINEXT<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region U2Seg

    [Fact(Timeout = 120000)]
    public async Task U2Seg_Construction_Succeeds()
    {
        var model = new U2Seg<float>(Arch());
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task U2Seg_Predict_ReturnsOutput()
    {
        var model = new U2Seg<float>(Arch());
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task U2Seg_Dispose_DoesNotThrow()
    {
        var model = new U2Seg<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region XDecoder

    [Fact(Timeout = 120000)]
    public async Task XDecoder_Construction_Succeeds()
    {
        var model = new XDecoder<float>(Arch(), modelSize: XDecoderModelSize.Tiny);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task XDecoder_Predict_ReturnsOutput()
    {
        var model = new XDecoder<float>(Arch(), modelSize: XDecoderModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task XDecoder_Dispose_DoesNotThrow()
    {
        var model = new XDecoder<float>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion
}
