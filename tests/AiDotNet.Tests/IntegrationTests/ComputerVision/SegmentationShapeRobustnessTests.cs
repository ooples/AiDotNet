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
/// Tests tensor shape robustness across all segmentation models.
/// Verifies models handle 3D [C,H,W] vs 4D [B,C,H,W] inputs,
/// non-square images, and different spatial sizes without crashing.
/// </summary>
public class SegmentationShapeRobustnessTests : IDisposable
{
    // Reclaim process-global retention (InferenceWeightCache pins + uncompacted LOH) between tests so
    // heavy models constructed one-per-test don't accumulate committed memory until a later test OOMs.
    // Pure hygiene — same teardown the model-family bases use; changes no assertion or scale.
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

    #region 3D (unbatched) vs 4D (batched) Input — Semantic Models

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]       // 3D unbatched
    [InlineData(new[] { 1, 3, 32, 32 })]     // 4D batched B=1
    public void SegFormer_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void SegNeXt_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new SegNeXt<float>(Arch(), numClasses: 5, modelSize: SegNeXtModelSize.Tiny);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void InternImage_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new InternImage<float>(Arch(), numClasses: 5, modelSize: InternImageModelSize.Tiny);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void ViTAdapter_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new ViTAdapter<float>(Arch(), numClasses: 5, modelSize: ViTAdapterModelSize.Small);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void ViTCoMer_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new ViTCoMer<float>(Arch(), numClasses: 5, modelSize: ViTCoMerModelSize.Small);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void DiffCut_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new DiffCut<float>(Arch(), numClasses: 5);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void DiffSeg_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new DiffSeg<float>(Arch(), numClasses: 5);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region 3D vs 4D Input — Foundation Models

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void Mask2Former_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void OneFormer_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new OneFormer<float>(Arch(), numClasses: 5, modelSize: OneFormerModelSize.SwinLarge);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void MaskDINO_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new MaskDINO<float>(Arch(), numClasses: 5, modelSize: MaskDINOModelSize.R50);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void SAM_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void SAM21_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new SAM21<float>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.BasePlus);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void SAMHQ_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new SAMHQ<float>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void EoMT_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new EoMT<float>(Arch(), numClasses: 5, modelSize: EoMTModelSize.Small);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void OMGSeg_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new OMGSeg<float>(Arch(), numClasses: 5, modelSize: OMGSegModelSize.Base);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region 3D vs 4D Input — Efficient Models

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void PIDNet_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new PIDNet<float>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void FastSAM_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new FastSAM<float>(Arch(), numClasses: 1);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void MobileSAM_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new MobileSAM<float>(Arch(), numClasses: 1);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void EfficientSAM_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new EfficientSAM<float>(Arch(), numClasses: 1);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region 3D vs 4D Input — Medical Models

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void NnUNet_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new NnUNet<float>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void TransUNet_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new TransUNet<float>(Arch(), numClasses: 5, modelSize: TransUNetModelSize.Base);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void SwinUNETR_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new SwinUNETR<float>(Arch(), numClasses: 5, modelSize: SwinUNETRModelSize.Base);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void MedSAM_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new MedSAM<float>(Arch(), numClasses: 5, modelSize: MedSAMModelSize.ViTBase);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void MedNeXt_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new MedNeXt<float>(Arch(), numClasses: 5, modelSize: MedNeXtModelSize.Small);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region 3D vs 4D Input — Mamba Models

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void VisionMamba_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new VisionMamba<float>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void VMamba_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new VMamba<float>(Arch(), numClasses: 5, modelSize: VMambaModelSize.Tiny);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void ViMUNet_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new ViMUNet<float>(Arch(), numClasses: 5);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region 3D vs 4D Input — Video, OpenVocab, Referring, Diffusion

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void DEVA_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new DEVA<float>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void EfficientTAM_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new EfficientTAM<float>(Arch(), numClasses: 5, modelSize: EfficientTAMModelSize.Small);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void SAN_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new SAN<float>(Arch(), numClasses: 5);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void LISA_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new LISA<float>(Arch(), numClasses: 5);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void DiffCutSegmentation_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new DiffCutSegmentation<float>(Arch(), numClasses: 5);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Non-Square Images

    [Theory]
    [InlineData(48, 32)]  // wider than tall
    [InlineData(32, 48)]  // taller than wide
    [InlineData(64, 32)]  // 2:1 aspect ratio
    public void SegFormer_Predict_NonSquare_ReturnsOutput(int h, int w)
    {
        var model = new SegFormer<float>(Arch(h, w, 3), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var output = model.Predict(Rand(1, 3, h, w));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void Mask2Former_Predict_NonSquare_ReturnsOutput(int h, int w)
    {
        var model = new Mask2Former<float>(Arch(h, w, 3), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var output = model.Predict(Rand(1, 3, h, w));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void NnUNet_Predict_NonSquare_ReturnsOutput(int h, int w)
    {
        var model = new NnUNet<float>(Arch(h, w, 3), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var output = model.Predict(Rand(1, 3, h, w));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void PIDNet_Predict_NonSquare_ReturnsOutput(int h, int w)
    {
        var model = new PIDNet<float>(Arch(h, w, 3), numClasses: 5, modelSize: PIDNetModelSize.Small);
        var output = model.Predict(Rand(1, 3, h, w));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void VisionMamba_Predict_NonSquare_ReturnsOutput(int h, int w)
    {
        var model = new VisionMamba<float>(Arch(h, w, 3), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        var output = model.Predict(Rand(1, 3, h, w));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Non-Square Train (backward pass with non-square shapes)

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void SegFormer_Train_NonSquare_DoesNotThrow(int h, int w)
    {
        var model = new SegFormer<float>(Arch(h, w, 3), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var input = Rand(1, 3, h, w);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void Mask2Former_Train_NonSquare_DoesNotThrow(int h, int w)
    {
        var model = new Mask2Former<float>(Arch(h, w, 3), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var input = Rand(1, 3, h, w);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void NnUNet_Train_NonSquare_DoesNotThrow(int h, int w)
    {
        var model = new NnUNet<float>(Arch(h, w, 3), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var input = Rand(1, 3, h, w);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Theory]
    [InlineData(48, 32)]
    [InlineData(32, 48)]
    public void SAM_Train_NonSquare_DoesNotThrow(int h, int w)
    {
        var model = new SAM<float>(Arch(h, w, 3), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var input = Rand(1, 3, h, w);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    #endregion

    #region Single-Channel Inputs

    [Fact(Timeout = 120000)]
    public async Task SegFormer_Predict_SingleChannel_ReturnsOutput()
    {
        var model = new SegFormer<float>(Arch(32, 32, 1), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var output = model.Predict(Rand(1, 1, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_Predict_SingleChannel_ReturnsOutput()
    {
        // Medical images are often single-channel (grayscale CT/MRI)
        var model = new NnUNet<float>(Arch(32, 32, 1), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var output = model.Predict(Rand(1, 1, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task TransUNet_Predict_SingleChannel_ReturnsOutput()
    {
        var model = new TransUNet<float>(Arch(32, 32, 1), numClasses: 5, modelSize: TransUNetModelSize.Base);
        var output = model.Predict(Rand(1, 1, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task MedSAM_Predict_SingleChannel_ReturnsOutput()
    {
        var model = new MedSAM<float>(Arch(32, 32, 1), numClasses: 5, modelSize: MedSAMModelSize.ViTBase);
        var output = model.Predict(Rand(1, 1, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Instance Segmentation Models — 3D vs 4D

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void YOLOv8Seg_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new YOLOv8Seg<float>(Arch(), modelSize: YOLOv8SegModelSize.N);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void YOLOv9Seg_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new YOLOv9Seg<float>(Arch(), modelSize: YOLOv9SegModelSize.C);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void YOLO11Seg_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new YOLO11Seg<float>(Arch(), modelSize: YOLO11SegModelSize.N);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void YOLOv12Seg_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new YOLOv12Seg<float>(Arch(), modelSize: YOLOv12SegModelSize.N);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void YOLO26Seg_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new YOLO26Seg<float>(Arch(), modelSize: YOLO26SegModelSize.N);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion

    #region Point Cloud Models — Shape Robustness

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void PointTransformerV3_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new PointTransformerV3<float>(Arch(), numClasses: 5, modelSize: PointTransformerV3ModelSize.Base);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void Sonata_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new Sonata<float>(Arch(), numClasses: 5, modelSize: SonataModelSize.Base);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Theory]
    [InlineData(new[] { 3, 32, 32 })]
    [InlineData(new[] { 1, 3, 32, 32 })]
    public void Concerto_Predict_DifferentRanks_ReturnsOutput(int[] shape)
    {
        var model = new Concerto<float>(Arch(), numClasses: 5, modelSize: ConcertoModelSize.Base);
        var output = model.Predict(Rand(shape));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    #endregion
}
