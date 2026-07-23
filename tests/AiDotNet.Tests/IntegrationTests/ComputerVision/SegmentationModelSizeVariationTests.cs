using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.ComputerVision.Segmentation.Semantic;
using AiDotNet.ComputerVision.Segmentation.Efficient;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.ComputerVision.Segmentation.Panoptic;
using AiDotNet.ComputerVision.Segmentation.Mamba;
using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.ComputerVision.Segmentation.OpenVocabulary;
using AiDotNet.ComputerVision.Segmentation.Referring;
using AiDotNet.ComputerVision.Segmentation.Diffusion;
using AiDotNet.ComputerVision.Segmentation.Interactive;
using AiDotNet.ComputerVision.Segmentation.PointCloud;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Tests that every model size variant constructs correctly, produces output,
/// and that larger variants have more parameters than smaller ones.
/// Catches bugs in model size configuration (wrong channel dimensions, depths, etc).
/// </summary>
public class SegmentationModelSizeVariationTests : IDisposable
{
    // The largest size variants (XL/Huge/B5) construct big models one-per-test; reclaim process-global
    // retention (InferenceWeightCache pins + uncompacted LOH) between tests so committed memory does not
    // accumulate across the class until a later variant OOMs. Pure hygiene — same teardown the
    // model-family bases use; changes no assertion, scale, or iteration count.
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

    #region SegFormer — All 6 sizes

    [Theory]
    [InlineData(SegFormerModelSize.B0)]
    [InlineData(SegFormerModelSize.B1)]
    [InlineData(SegFormerModelSize.B2)]
    [InlineData(SegFormerModelSize.B3)]
    [InlineData(SegFormerModelSize.B4)]
    [InlineData(SegFormerModelSize.B5)]
    public void SegFormer_AllModelSizes_ConstructAndPredict(SegFormerModelSize size)
    {
        var model = new SegFormer<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0, $"SegFormer-{size} has no parameters");
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Mask2Former — All sizes

    [Theory]
    [InlineData(Mask2FormerModelSize.R50)]
    [InlineData(Mask2FormerModelSize.SwinTiny)]
    [InlineData(Mask2FormerModelSize.SwinSmall)]
    [InlineData(Mask2FormerModelSize.SwinBase)]
    [InlineData(Mask2FormerModelSize.SwinLarge)]
    public void Mask2Former_AllModelSizes_ConstructAndPredict(Mask2FormerModelSize size)
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region SAM — All sizes

    [Theory]
    [InlineData(SAMModelSize.ViTBase)]
    [InlineData(SAMModelSize.ViTLarge)]
    [InlineData(SAMModelSize.ViTHuge)]
    public void SAM_AllModelSizes_ConstructAndPredict(SAMModelSize size)
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region SAM21 — All sizes

    [Theory]
    [InlineData(SAM21ModelSize.Tiny)]
    [InlineData(SAM21ModelSize.Small)]
    [InlineData(SAM21ModelSize.BasePlus)]
    [InlineData(SAM21ModelSize.Large)]
    public void SAM21_AllModelSizes_ConstructAndPredict(SAM21ModelSize size)
    {
        var model = new SAM21<float>(Arch(), numClasses: 1, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region SegNeXt — All sizes

    [Theory]
    [InlineData(SegNeXtModelSize.Tiny)]
    [InlineData(SegNeXtModelSize.Small)]
    [InlineData(SegNeXtModelSize.Base)]
    [InlineData(SegNeXtModelSize.Large)]
    public void SegNeXt_AllModelSizes_ConstructAndPredict(SegNeXtModelSize size)
    {
        var model = new SegNeXt<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region YOLO Seg Models — All sizes

    [Theory]
    [InlineData(YOLOv8SegModelSize.N)]
    [InlineData(YOLOv8SegModelSize.S)]
    [InlineData(YOLOv8SegModelSize.M)]
    [InlineData(YOLOv8SegModelSize.L)]
    [InlineData(YOLOv8SegModelSize.X)]
    public void YOLOv8Seg_AllModelSizes_ConstructAndPredict(YOLOv8SegModelSize size)
    {
        var model = new YOLOv8Seg<float>(Arch(), modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(YOLOv9SegModelSize.C)]
    [InlineData(YOLOv9SegModelSize.E)]
    public void YOLOv9Seg_AllModelSizes_ConstructAndPredict(YOLOv9SegModelSize size)
    {
        var model = new YOLOv9Seg<float>(Arch(), modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(YOLO11SegModelSize.N)]
    [InlineData(YOLO11SegModelSize.S)]
    [InlineData(YOLO11SegModelSize.M)]
    [InlineData(YOLO11SegModelSize.L)]
    [InlineData(YOLO11SegModelSize.X)]
    public void YOLO11Seg_AllModelSizes_ConstructAndPredict(YOLO11SegModelSize size)
    {
        var model = new YOLO11Seg<float>(Arch(), modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(YOLOv12SegModelSize.N)]
    [InlineData(YOLOv12SegModelSize.S)]
    [InlineData(YOLOv12SegModelSize.M)]
    [InlineData(YOLOv12SegModelSize.L)]
    [InlineData(YOLOv12SegModelSize.X)]
    public void YOLOv12Seg_AllModelSizes_ConstructAndPredict(YOLOv12SegModelSize size)
    {
        var model = new YOLOv12Seg<float>(Arch(), modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(YOLO26SegModelSize.N)]
    [InlineData(YOLO26SegModelSize.S)]
    [InlineData(YOLO26SegModelSize.M)]
    [InlineData(YOLO26SegModelSize.L)]
    [InlineData(YOLO26SegModelSize.X)]
    public void YOLO26Seg_AllModelSizes_ConstructAndPredict(YOLO26SegModelSize size)
    {
        var model = new YOLO26Seg<float>(Arch(), modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Medical Models — All sizes

    [Theory]
    [InlineData(NnUNetModelSize.UNet2D)]
    [InlineData(NnUNetModelSize.UNet3DFull)]
    [InlineData(NnUNetModelSize.UNet3DCascade)]
    public void NnUNet_AllModelSizes_ConstructAndPredict(NnUNetModelSize size)
    {
        var model = new NnUNet<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(MedNeXtModelSize.Small)]
    [InlineData(MedNeXtModelSize.Base)]
    [InlineData(MedNeXtModelSize.Medium)]
    [InlineData(MedNeXtModelSize.Large)]
    public void MedNeXt_AllModelSizes_ConstructAndPredict(MedNeXtModelSize size)
    {
        var model = new MedNeXt<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Mamba Models — All sizes

    [Theory]
    [InlineData(VisionMambaModelSize.Tiny)]
    [InlineData(VisionMambaModelSize.Small)]
    [InlineData(VisionMambaModelSize.Base)]
    public void VisionMamba_AllModelSizes_ConstructAndPredict(VisionMambaModelSize size)
    {
        var model = new VisionMamba<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(VMambaModelSize.Tiny)]
    [InlineData(VMambaModelSize.Small)]
    [InlineData(VMambaModelSize.Base)]
    public void VMamba_AllModelSizes_ConstructAndPredict(VMambaModelSize size)
    {
        var model = new VMamba<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Efficient Models — All sizes (where applicable)

    [Theory]
    [InlineData(PIDNetModelSize.Small)]
    [InlineData(PIDNetModelSize.Medium)]
    [InlineData(PIDNetModelSize.Large)]
    public void PIDNet_AllModelSizes_ConstructAndPredict(PIDNetModelSize size)
    {
        var model = new PIDNet<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Panoptic Models — All sizes

    [Theory]
    [InlineData(KMaXDeepLabModelSize.R50)]
    [InlineData(KMaXDeepLabModelSize.ConvNeXtLarge)]
    public void KMaXDeepLab_AllModelSizes_ConstructAndPredict(KMaXDeepLabModelSize size)
    {
        var model = new KMaXDeepLab<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(OneFormerModelSize.SwinLarge)]
    [InlineData(OneFormerModelSize.DiNATLarge)]
    public void OneFormer_AllModelSizes_ConstructAndPredict(OneFormerModelSize size)
    {
        var model = new OneFormer<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Interactive Models — All sizes

    [Theory]
    [InlineData(SegGPTModelSize.ViTLarge)]
    public void SegGPT_AllModelSizes_ConstructAndPredict(SegGPTModelSize size)
    {
        var model = new SegGPT<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(SEEMModelSize.Tiny)]
    [InlineData(SEEMModelSize.Large)]
    public void SEEM_AllModelSizes_ConstructAndPredict(SEEMModelSize size)
    {
        var model = new SEEM<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Video Models — All sizes

    [Theory]
    [InlineData(DEVAModelSize.Base)]
    [InlineData(DEVAModelSize.Large)]
    public void DEVA_AllModelSizes_ConstructAndPredict(DEVAModelSize size)
    {
        var model = new DEVA<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(EfficientTAMModelSize.Tiny)]
    [InlineData(EfficientTAMModelSize.Small)]
    public void EfficientTAM_AllModelSizes_ConstructAndPredict(EfficientTAMModelSize size)
    {
        var model = new EfficientTAM<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Point Cloud Models — All sizes

    [Theory]
    [InlineData(PointTransformerV3ModelSize.Base)]
    [InlineData(PointTransformerV3ModelSize.Large)]
    public void PointTransformerV3_AllModelSizes_ConstructAndPredict(PointTransformerV3ModelSize size)
    {
        var model = new PointTransformerV3<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(SonataModelSize.Base)]
    [InlineData(SonataModelSize.Large)]
    public void Sonata_AllModelSizes_ConstructAndPredict(SonataModelSize size)
    {
        var model = new Sonata<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(ConcertoModelSize.Base)]
    [InlineData(ConcertoModelSize.Large)]
    public void Concerto_AllModelSizes_ConstructAndPredict(ConcertoModelSize size)
    {
        var model = new Concerto<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Foundation Models — All sizes

    [Theory]
    [InlineData(SAMHQModelSize.ViTBase)]
    [InlineData(SAMHQModelSize.ViTLarge)]
    [InlineData(SAMHQModelSize.ViTHuge)]
    public void SAMHQ_AllModelSizes_ConstructAndPredict(SAMHQModelSize size)
    {
        var model = new SAMHQ<float>(Arch(), numClasses: 1, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(MaskDINOModelSize.R50)]
    [InlineData(MaskDINOModelSize.SwinLarge)]
    public void MaskDINO_AllModelSizes_ConstructAndPredict(MaskDINOModelSize size)
    {
        var model = new MaskDINO<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(EoMTModelSize.Small)]
    [InlineData(EoMTModelSize.Base)]
    [InlineData(EoMTModelSize.Large)]
    public void EoMT_AllModelSizes_ConstructAndPredict(EoMTModelSize size)
    {
        var model = new EoMT<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(InternImageModelSize.Tiny)]
    [InlineData(InternImageModelSize.Small)]
    [InlineData(InternImageModelSize.Base)]
    [InlineData(InternImageModelSize.XL)]
    [InlineData(InternImageModelSize.Huge)]
    public void InternImage_AllModelSizes_ConstructAndPredict(InternImageModelSize size)
    {
        var model = new InternImage<float>(Arch(), numClasses: 5, modelSize: size);
        Assert.True(model.ParameterCount > 0);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Larger Size Has More Parameters (scaling regression test)

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_LargerSizeHasMoreParameters()
    {
        var small = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var large = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinLarge);
        Assert.True(large.ParameterCount > small.ParameterCount,
            $"SwinLarge ({large.ParameterCount}) should have more params than SwinTiny ({small.ParameterCount})");
    }

    [Fact(Timeout = 120000)]
    public async Task SegNeXt_LargerSizeHasMoreParameters()
    {
        var small = new SegNeXt<float>(Arch(), numClasses: 5, modelSize: SegNeXtModelSize.Tiny);
        var large = new SegNeXt<float>(Arch(), numClasses: 5, modelSize: SegNeXtModelSize.Large);
        Assert.True(large.ParameterCount > small.ParameterCount,
            $"Large ({large.ParameterCount}) should have more params than Tiny ({small.ParameterCount})");
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_LargerSizeHasMoreParameters()
    {
        var small = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var large = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTHuge);
        Assert.True(large.ParameterCount > small.ParameterCount,
            $"ViTHuge ({large.ParameterCount}) should have more params than ViTBase ({small.ParameterCount})");
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_LargerSizeHasMoreParameters()
    {
        var small = new NnUNet<float>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var large = new NnUNet<float>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet3DCascade);
        Assert.True(large.ParameterCount > small.ParameterCount,
            $"UNet3DCascade ({large.ParameterCount}) should have more params than UNet2D ({small.ParameterCount})");
    }

    [Fact(Timeout = 120000)]
    public async Task VisionMamba_LargerSizeHasMoreParameters()
    {
        var small = new VisionMamba<float>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        var large = new VisionMamba<float>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Base);
        Assert.True(large.ParameterCount > small.ParameterCount,
            $"Base ({large.ParameterCount}) should have more params than Tiny ({small.ParameterCount})");
    }

    [Fact(Timeout = 120000)]
    public async Task PIDNet_LargerSizeHasMoreParameters()
    {
        var small = new PIDNet<float>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        var large = new PIDNet<float>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Large);
        Assert.True(large.ParameterCount > small.ParameterCount,
            $"Large ({large.ParameterCount}) should have more params than Small ({small.ParameterCount})");
    }

    #endregion
}
