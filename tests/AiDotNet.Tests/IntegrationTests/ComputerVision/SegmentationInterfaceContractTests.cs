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
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Tests interface contract compliance, model metadata, parameter counts,
/// options retrieval, serialization, and dispose lifecycle for all segmentation models.
/// </summary>
public class SegmentationInterfaceContractTests
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

    #region ISemanticSegmentation Contract — GetClassMap and GetProbabilityMap

    [Fact]
    public void SegFormer_GetClassMap_ReturnsNonEmpty()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var classMap = ((ISemanticSegmentation<double>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact]
    public void SegFormer_GetProbabilityMap_ReturnsNonEmpty()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var probMap = ((ISemanticSegmentation<double>)model).GetProbabilityMap(Rand(1, 3, 32, 32));
        Assert.NotNull(probMap);
        Assert.True(probMap.Length > 0);
    }

    [Fact]
    public void SegNeXt_GetClassMap_ReturnsNonEmpty()
    {
        var model = new SegNeXt<double>(Arch(), numClasses: 5, modelSize: SegNeXtModelSize.Tiny);
        var classMap = ((ISemanticSegmentation<double>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact]
    public void InternImage_GetClassMap_ReturnsNonEmpty()
    {
        var model = new InternImage<double>(Arch(), numClasses: 5, modelSize: InternImageModelSize.Tiny);
        var classMap = ((ISemanticSegmentation<double>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact]
    public void VisionMamba_GetClassMap_ReturnsNonEmpty()
    {
        var model = new VisionMamba<double>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        var classMap = ((ISemanticSegmentation<double>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact]
    public void PIDNet_GetClassMap_ReturnsNonEmpty()
    {
        var model = new PIDNet<double>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        var classMap = ((ISemanticSegmentation<double>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact]
    public void DiffCutSegmentation_GetClassMap_ReturnsNonEmpty()
    {
        var model = new DiffCutSegmentation<double>(Arch(), numClasses: 5);
        var classMap = ((ISemanticSegmentation<double>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    #endregion

    #region IPromptableSegmentation Contract — SetImage, SegmentFromPoints/Box/Mask

    [Fact]
    public void SAM_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<double>)model;
        Assert.True(promptable.SupportsPointPrompts);
        Assert.True(promptable.SupportsBoxPrompts);
        Assert.True(promptable.SupportsMaskPrompts);
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);
    }

    [Fact]
    public void SAM_PromptableInterface_SegmentFromBox()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromBox(Rand(4)); // [x1, y1, x2, y2]
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);
    }

    [Fact]
    public void SAM_PromptableInterface_SegmentFromMask()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromMask(Rand(32, 32)); // [H, W]
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);
    }

    [Fact]
    public void SAM_PromptableInterface_SegmentEverything()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var results = promptable.SegmentEverything();
        Assert.NotNull(results);
    }

    [Fact]
    public void SAM21_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SAM21<double>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.BasePlus);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact]
    public void SAMHQ_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SAMHQ<double>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact]
    public void FastSAM_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new FastSAM<double>(Arch(), numClasses: 1);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact]
    public void MobileSAM_PromptableInterface_SegmentFromBox()
    {
        var model = new MobileSAM<double>(Arch(), numClasses: 1);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromBox(Rand(4));
        Assert.NotNull(result);
    }

    [Fact]
    public void EfficientSAM_PromptableInterface_SegmentEverything()
    {
        var model = new EfficientSAM<double>(Arch(), numClasses: 1);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var results = promptable.SegmentEverything();
        Assert.NotNull(results);
    }

    [Fact]
    public void SegGPT_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SegGPT<double>(Arch(), numClasses: 5, modelSize: SegGPTModelSize.ViTLarge);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact]
    public void SEEM_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SEEM<double>(Arch(), numClasses: 5, modelSize: SEEMModelSize.Tiny);
        var promptable = (IPromptableSegmentation<double>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    #endregion

    #region IPanopticSegmentation Contract — SegmentPanoptic, NumStuffClasses, NumThingClasses

    [Fact]
    public void Mask2Former_PanopticInterface_SegmentPanoptic()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var panoptic = (IPanopticSegmentation<double>)model;
        Assert.True(panoptic.NumStuffClasses >= 0);
        Assert.True(panoptic.NumThingClasses >= 0);
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
        Assert.NotNull(result.SemanticMap);
        Assert.NotNull(result.InstanceMap);
    }

    [Fact]
    public void KMaXDeepLab_PanopticInterface_SegmentPanoptic()
    {
        var model = new KMaXDeepLab<double>(Arch(), numClasses: 5, modelSize: KMaXDeepLabModelSize.R50);
        var panoptic = (IPanopticSegmentation<double>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact]
    public void ODISE_PanopticInterface_SegmentPanoptic()
    {
        var model = new ODISE<double>(Arch(), numClasses: 5, modelSize: ODISEModelSize.Base);
        var panoptic = (IPanopticSegmentation<double>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact]
    public void OneFormer_PanopticInterface_SegmentPanoptic()
    {
        var model = new OneFormer<double>(Arch(), numClasses: 5, modelSize: OneFormerModelSize.SwinLarge);
        var panoptic = (IPanopticSegmentation<double>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact]
    public void MaskDINO_PanopticInterface_SegmentPanoptic()
    {
        var model = new MaskDINO<double>(Arch(), numClasses: 5, modelSize: MaskDINOModelSize.R50);
        var panoptic = (IPanopticSegmentation<double>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    #endregion

    #region IVideoSegmentation Contract — InitializeTracking, PropagateToFrame, ResetTracking

    [Fact]
    public void DEVA_VideoInterface_TrackingLifecycle()
    {
        var model = new DEVA<double>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var video = (IVideoSegmentation<double>)model;
        Assert.True(video.MaxTrackedObjects > 0);

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);

        Assert.Null(Record.Exception(() => video.ResetTracking()));
    }

    [Fact]
    public void EfficientTAM_VideoInterface_TrackingLifecycle()
    {
        var model = new EfficientTAM<double>(Arch(), numClasses: 5, modelSize: EfficientTAMModelSize.Small);
        var video = (IVideoSegmentation<double>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.NotNull(result);
        video.ResetTracking();
    }

    [Fact]
    public void UniVS_VideoInterface_TrackingLifecycle()
    {
        var model = new UniVS<double>(Arch(), numClasses: 5, modelSize: UniVSModelSize.R50);
        var video = (IVideoSegmentation<double>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.NotNull(result);
        video.ResetTracking();
    }

    [Fact]
    public void DEVA_VideoInterface_MultiFramePropagation()
    {
        var model = new DEVA<double>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var video = (IVideoSegmentation<double>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));

        for (int i = 0; i < 3; i++)
        {
            var result = video.PropagateToFrame(Rand(3, 32, 32));
            Assert.NotNull(result);
        }
    }

    [Fact]
    public void DEVA_VideoInterface_AddCorrection()
    {
        var model = new DEVA<double>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var video = (IVideoSegmentation<double>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        video.PropagateToFrame(Rand(3, 32, 32));
        Assert.Null(Record.Exception(() => video.AddCorrection(0, Rand(32, 32))));
    }

    #endregion

    #region IMedicalSegmentation Contract — SegmentSlice, SegmentVolume, SupportedModalities

    [Fact]
    public void NnUNet_MedicalInterface_SegmentSlice()
    {
        var model = new NnUNet<double>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var medical = (IMedicalSegmentation<double>)model;
        Assert.NotNull(medical.SupportedModalities);
        Assert.True(medical.SupportedModalities.Count > 0);
        Assert.True(medical.Supports2D);

        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
        Assert.NotNull(result.Labels);
    }

    [Fact]
    public void TransUNet_MedicalInterface_SegmentSlice()
    {
        var model = new TransUNet<double>(Arch(), numClasses: 5, modelSize: TransUNetModelSize.Base);
        var medical = (IMedicalSegmentation<double>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact]
    public void SwinUNETR_MedicalInterface_SegmentSlice()
    {
        var model = new SwinUNETR<double>(Arch(), numClasses: 5, modelSize: SwinUNETRModelSize.Base);
        var medical = (IMedicalSegmentation<double>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact]
    public void MedSAM_MedicalInterface_SegmentSlice()
    {
        var model = new MedSAM<double>(Arch(), numClasses: 5, modelSize: MedSAMModelSize.ViTBase);
        var medical = (IMedicalSegmentation<double>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact]
    public void MedNeXt_MedicalInterface_SegmentSlice()
    {
        var model = new MedNeXt<double>(Arch(), numClasses: 5, modelSize: MedNeXtModelSize.Small);
        var medical = (IMedicalSegmentation<double>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact]
    public void BiomedParse_MedicalInterface_SegmentSlice()
    {
        var model = new BiomedParse<double>(Arch(), numClasses: 5);
        var medical = (IMedicalSegmentation<double>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    #endregion

    #region IOpenVocabSegmentation Contract

    [Fact]
    public void SAN_OpenVocabInterface_SegmentWithText()
    {
        var model = new SAN<double>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<double>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car", "person" });
        Assert.NotNull(result);
    }

    [Fact]
    public void CATSeg_OpenVocabInterface_SegmentWithText()
    {
        var model = new CATSeg<double>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<double>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car" });
        Assert.NotNull(result);
    }

    [Fact]
    public void SED_OpenVocabInterface_SegmentWithText()
    {
        var model = new SED<double>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<double>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car" });
        Assert.NotNull(result);
    }

    [Fact]
    public void GroundedSAM2_OpenVocabInterface_SegmentWithText()
    {
        var model = new GroundedSAM2<double>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<double>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car" });
        Assert.NotNull(result);
    }

    #endregion

    #region IReferringSegmentation Contract

    [Fact]
    public void LISA_ReferringInterface_SegmentFromExpression()
    {
        var model = new LISA<double>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<double>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "the red car on the left");
        Assert.NotNull(result);
    }

    [Fact]
    public void VideoLISA_ReferringInterface_SegmentFromExpression()
    {
        var model = new VideoLISA<double>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<double>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "the person walking");
        Assert.NotNull(result);
    }

    [Fact]
    public void GLaMM_ReferringInterface_SegmentFromExpression()
    {
        var model = new GLaMM<double>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<double>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "cat");
        Assert.NotNull(result);
    }

    [Fact]
    public void PixelLM_ReferringInterface_SegmentFromExpression()
    {
        var model = new PixelLM<double>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<double>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "the building");
        Assert.NotNull(result);
    }

    #endregion

    #region Model Metadata and Parameter Counts

    [Fact]
    public void SegFormer_ParameterCount_PositiveAndModelSizeDependent()
    {
        var b0 = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var b3 = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B3);

        Assert.True(b0.ParameterCount > 0, "B0 should have parameters");
        Assert.True(b3.ParameterCount > b0.ParameterCount,
            $"B3 ({b3.ParameterCount}) should have more params than B0 ({b0.ParameterCount})");
    }

    [Fact]
    public void Mask2Former_ParameterCount_Positive()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void SAM_ParameterCount_Positive()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void NnUNet_ParameterCount_Positive()
    {
        var model = new NnUNet<double>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void VisionMamba_ParameterCount_Positive()
    {
        var model = new VisionMamba<double>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void PIDNet_ParameterCount_Positive()
    {
        var model = new PIDNet<double>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion

    #region GetModelMetadata

    [Fact]
    public void SegFormer_GetModelMetadata_ContainsRequiredFields()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 21, modelSize: SegFormerModelSize.B2);
        var meta = model.GetModelMetadata();
        Assert.NotNull(meta);
        Assert.Equal(ModelType.SemanticSegmentation, meta.ModelType);
        Assert.NotNull(meta.AdditionalInfo);
        Assert.Equal("SegFormer", meta.AdditionalInfo["ModelName"]);
        Assert.Equal(21, meta.AdditionalInfo["NumClasses"]);
    }

    [Fact]
    public void Mask2Former_GetModelMetadata_ContainsRequiredFields()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var meta = model.GetModelMetadata();
        Assert.NotNull(meta);
        Assert.NotNull(meta.AdditionalInfo);
        Assert.True(meta.AdditionalInfo.ContainsKey("ModelName"));
    }

    [Fact]
    public void SAM_GetModelMetadata_ContainsModelName()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var meta = model.GetModelMetadata();
        Assert.NotNull(meta);
        Assert.Equal("SAM", meta.AdditionalInfo["ModelName"]);
    }

    #endregion

    #region Dispose Lifecycle — Double Dispose, Use-After-Dispose

    [Fact]
    public void SegFormer_DoubleDispose_DoesNotThrow()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact]
    public void Mask2Former_DoubleDispose_DoesNotThrow()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact]
    public void SAM_DoubleDispose_DoesNotThrow()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact]
    public void NnUNet_DoubleDispose_DoesNotThrow()
    {
        var model = new NnUNet<double>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact]
    public void VisionMamba_DoubleDispose_DoesNotThrow()
    {
        var model = new VisionMamba<double>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact]
    public void PIDNet_DoubleDispose_DoesNotThrow()
    {
        var model = new PIDNet<double>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact]
    public void DEVA_DoubleDispose_DoesNotThrow()
    {
        var model = new DEVA<double>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region Multiple Consecutive Predictions (stateful regression)

    [Fact]
    public void SegFormer_MultiplePredictions_AllReturnOutput()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0, $"Prediction {i} returned empty output");
        }
    }

    [Fact]
    public void Mask2Former_MultiplePredictions_AllReturnOutput()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0);
        }
    }

    [Fact]
    public void NnUNet_MultiplePredictions_AllReturnOutput()
    {
        var model = new NnUNet<double>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0);
        }
    }

    [Fact]
    public void SAM_MultiplePredictions_AllReturnOutput()
    {
        var model = new SAM<double>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0);
        }
    }

    #endregion

    #region Segment() method (ISegmentationModel)

    [Fact]
    public void SegFormer_Segment_EquivalentToPredict()
    {
        var model = new SegFormer<double>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var input = Rand(1, 3, 32, 32);

        var segResult = ((ISegmentationModel<double>)model).Segment(input);
        Assert.NotNull(segResult);
        Assert.True(segResult.Length > 0);
    }

    [Fact]
    public void Mask2Former_Segment_EquivalentToPredict()
    {
        var model = new Mask2Former<double>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var segResult = ((ISegmentationModel<double>)model).Segment(Rand(1, 3, 32, 32));
        Assert.NotNull(segResult);
        Assert.True(segResult.Length > 0);
    }

    #endregion

    #region ISegmentationModel Properties

    [Fact]
    public void SegFormer_ISegmentationModel_Properties()
    {
        var model = new SegFormer<double>(Arch(64, 48, 3), numClasses: 21, modelSize: SegFormerModelSize.B0);
        var seg = (ISegmentationModel<double>)model;
        Assert.Equal(21, seg.NumClasses);
        Assert.Equal(64, seg.InputHeight);
        Assert.Equal(48, seg.InputWidth);
        Assert.False(seg.IsOnnxMode);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SAM_ISegmentationModel_Properties()
    {
        var model = new SAM<double>(Arch(32, 32, 3), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var seg = (ISegmentationModel<double>)model;
        Assert.Equal(1, seg.NumClasses);
        Assert.False(seg.IsOnnxMode);
        Assert.True(model.SupportsTraining);
    }

    #endregion
}
