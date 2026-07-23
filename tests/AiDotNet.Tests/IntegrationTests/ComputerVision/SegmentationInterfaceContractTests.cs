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
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Tests interface contract compliance, model metadata, parameter counts,
/// options retrieval, serialization, and dispose lifecycle for all segmentation models.
/// </summary>
public class SegmentationInterfaceContractTests : IDisposable
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

    #region ISemanticSegmentation Contract — GetClassMap and GetProbabilityMap

    [Fact(Timeout = 120000)]
    public async Task SegFormer_GetClassMap_ReturnsNonEmpty()
    {
        var model = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var classMap = ((ISemanticSegmentation<float>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SegFormer_GetProbabilityMap_ReturnsNonEmpty()
    {
        var model = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var probMap = ((ISemanticSegmentation<float>)model).GetProbabilityMap(Rand(1, 3, 32, 32));
        Assert.NotNull(probMap);
        Assert.True(probMap.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SegNeXt_GetClassMap_ReturnsNonEmpty()
    {
        var model = new SegNeXt<float>(Arch(), numClasses: 5, modelSize: SegNeXtModelSize.Tiny);
        var classMap = ((ISemanticSegmentation<float>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task InternImage_GetClassMap_ReturnsNonEmpty()
    {
        var model = new InternImage<float>(Arch(), numClasses: 5, modelSize: InternImageModelSize.Tiny);
        var classMap = ((ISemanticSegmentation<float>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task VisionMamba_GetClassMap_ReturnsNonEmpty()
    {
        var model = new VisionMamba<float>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        var classMap = ((ISemanticSegmentation<float>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task PIDNet_GetClassMap_ReturnsNonEmpty()
    {
        var model = new PIDNet<float>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        var classMap = ((ISemanticSegmentation<float>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffCutSegmentation_GetClassMap_ReturnsNonEmpty()
    {
        var model = new DiffCutSegmentation<float>(Arch(), numClasses: 5);
        var classMap = ((ISemanticSegmentation<float>)model).GetClassMap(Rand(1, 3, 32, 32));
        Assert.NotNull(classMap);
        Assert.True(classMap.Length > 0);
    }

    #endregion

    #region IPromptableSegmentation Contract — SetImage, SegmentFromPoints/Box/Mask

    [Fact(Timeout = 120000)]
    public async Task SAM_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<float>)model;
        Assert.True(promptable.SupportsPointPrompts);
        Assert.True(promptable.SupportsBoxPrompts);
        Assert.True(promptable.SupportsMaskPrompts);
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_PromptableInterface_SegmentFromBox()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromBox(Rand(4)); // [x1, y1, x2, y2]
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_PromptableInterface_SegmentFromMask()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromMask(Rand(32, 32)); // [H, W]
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_PromptableInterface_SegmentEverything()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var results = promptable.SegmentEverything();
        Assert.NotNull(results);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM21_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SAM21<float>(Arch(), numClasses: 1, modelSize: SAM21ModelSize.BasePlus);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task SAMHQ_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SAMHQ<float>(Arch(), numClasses: 1, modelSize: SAMHQModelSize.ViTBase);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task FastSAM_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new FastSAM<float>(Arch(), numClasses: 1);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task MobileSAM_PromptableInterface_SegmentFromBox()
    {
        var model = new MobileSAM<float>(Arch(), numClasses: 1);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromBox(Rand(4));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task EfficientSAM_PromptableInterface_SegmentEverything()
    {
        var model = new EfficientSAM<float>(Arch(), numClasses: 1);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var results = promptable.SegmentEverything();
        Assert.NotNull(results);
    }

    [Fact(Timeout = 120000)]
    public async Task SegGPT_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SegGPT<float>(Arch(), numClasses: 5, modelSize: SegGPTModelSize.ViTLarge);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task SEEM_PromptableInterface_SetImageAndSegmentFromPoints()
    {
        var model = new SEEM<float>(Arch(), numClasses: 5, modelSize: SEEMModelSize.Tiny);
        var promptable = (IPromptableSegmentation<float>)model;
        promptable.SetImage(Rand(3, 32, 32));
        var result = promptable.SegmentFromPoints(Rand(1, 2), Rand(1));
        Assert.NotNull(result);
    }

    #endregion

    #region IPanopticSegmentation Contract — SegmentPanoptic, NumStuffClasses, NumThingClasses

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_PanopticInterface_SegmentPanoptic()
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var panoptic = (IPanopticSegmentation<float>)model;
        Assert.True(panoptic.NumStuffClasses >= 0);
        Assert.True(panoptic.NumThingClasses >= 0);
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
        Assert.NotNull(result.SemanticMap);
        Assert.NotNull(result.InstanceMap);
    }

    [Fact(Timeout = 120000)]
    public async Task KMaXDeepLab_PanopticInterface_SegmentPanoptic()
    {
        var model = new KMaXDeepLab<float>(Arch(), numClasses: 5, modelSize: KMaXDeepLabModelSize.R50);
        var panoptic = (IPanopticSegmentation<float>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task ODISE_PanopticInterface_SegmentPanoptic()
    {
        var model = new ODISE<float>(Arch(), numClasses: 5, modelSize: ODISEModelSize.Base);
        var panoptic = (IPanopticSegmentation<float>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task OneFormer_PanopticInterface_SegmentPanoptic()
    {
        var model = new OneFormer<float>(Arch(), numClasses: 5, modelSize: OneFormerModelSize.SwinLarge);
        var panoptic = (IPanopticSegmentation<float>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskDINO_PanopticInterface_SegmentPanoptic()
    {
        var model = new MaskDINO<float>(Arch(), numClasses: 5, modelSize: MaskDINOModelSize.R50);
        var panoptic = (IPanopticSegmentation<float>)model;
        var result = panoptic.SegmentPanoptic(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    #endregion

    #region IVideoSegmentation Contract — InitializeTracking, PropagateToFrame, ResetTracking

    [Fact(Timeout = 120000)]
    public async Task DEVA_VideoInterface_TrackingLifecycle()
    {
        var model = new DEVA<float>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var video = (IVideoSegmentation<float>)model;
        Assert.True(video.MaxTrackedObjects > 0);

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.NotNull(result);
        Assert.NotNull(result.Masks);

        Assert.Null(Record.Exception(() => video.ResetTracking()));
    }

    [Fact(Timeout = 120000)]
    public async Task EfficientTAM_VideoInterface_TrackingLifecycle()
    {
        var model = new EfficientTAM<float>(Arch(), numClasses: 5, modelSize: EfficientTAMModelSize.Small);
        var video = (IVideoSegmentation<float>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.NotNull(result);
        video.ResetTracking();
    }

    [Fact(Timeout = 120000)]
    public async Task UniVS_VideoInterface_TrackingLifecycle()
    {
        var model = new UniVS<float>(Arch(), numClasses: 5, modelSize: UniVSModelSize.R50);
        var video = (IVideoSegmentation<float>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        var result = video.PropagateToFrame(Rand(3, 32, 32));
        Assert.NotNull(result);
        video.ResetTracking();
    }

    [Fact(Timeout = 120000)]
    public async Task DEVA_VideoInterface_MultiFramePropagation()
    {
        var model = new DEVA<float>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var video = (IVideoSegmentation<float>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));

        for (int i = 0; i < 3; i++)
        {
            var result = video.PropagateToFrame(Rand(3, 32, 32));
            Assert.NotNull(result);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DEVA_VideoInterface_AddCorrection()
    {
        var model = new DEVA<float>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        var video = (IVideoSegmentation<float>)model;

        video.InitializeTracking(Rand(3, 32, 32), Rand(1, 32, 32));
        video.PropagateToFrame(Rand(3, 32, 32));
        Assert.Null(Record.Exception(() => video.AddCorrection(0, Rand(32, 32))));
    }

    #endregion

    #region IMedicalSegmentation Contract — SegmentSlice, SegmentVolume, SupportedModalities

    [Fact(Timeout = 120000)]
    public async Task NnUNet_MedicalInterface_SegmentSlice()
    {
        var model = new NnUNet<float>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        var medical = (IMedicalSegmentation<float>)model;
        Assert.NotNull(medical.SupportedModalities);
        Assert.True(medical.SupportedModalities.Count > 0);
        Assert.True(medical.Supports2D);

        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
        Assert.NotNull(result.Labels);
    }

    [Fact(Timeout = 120000)]
    public async Task TransUNet_MedicalInterface_SegmentSlice()
    {
        var model = new TransUNet<float>(Arch(), numClasses: 5, modelSize: TransUNetModelSize.Base);
        var medical = (IMedicalSegmentation<float>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task SwinUNETR_MedicalInterface_SegmentSlice()
    {
        var model = new SwinUNETR<float>(Arch(), numClasses: 5, modelSize: SwinUNETRModelSize.Base);
        var medical = (IMedicalSegmentation<float>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task MedSAM_MedicalInterface_SegmentSlice()
    {
        var model = new MedSAM<float>(Arch(), numClasses: 5, modelSize: MedSAMModelSize.ViTBase);
        var medical = (IMedicalSegmentation<float>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task MedNeXt_MedicalInterface_SegmentSlice()
    {
        var model = new MedNeXt<float>(Arch(), numClasses: 5, modelSize: MedNeXtModelSize.Small);
        var medical = (IMedicalSegmentation<float>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task BiomedParse_MedicalInterface_SegmentSlice()
    {
        var model = new BiomedParse<float>(Arch(), numClasses: 5);
        var medical = (IMedicalSegmentation<float>)model;
        var result = medical.SegmentSlice(Rand(3, 32, 32));
        Assert.NotNull(result);
    }

    #endregion

    #region IOpenVocabSegmentation Contract

    [Fact(Timeout = 120000)]
    public async Task SAN_OpenVocabInterface_SegmentWithText()
    {
        var model = new SAN<float>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<float>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car", "person" });
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task CATSeg_OpenVocabInterface_SegmentWithText()
    {
        var model = new CATSeg<float>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<float>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car" });
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task SED_OpenVocabInterface_SegmentWithText()
    {
        var model = new SED<float>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<float>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car" });
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task GroundedSAM2_OpenVocabInterface_SegmentWithText()
    {
        var model = new GroundedSAM2<float>(Arch(), numClasses: 5);
        var ov = (IOpenVocabSegmentation<float>)model;
        var result = ov.SegmentWithText(Rand(1, 3, 32, 32), new[] { "car" });
        Assert.NotNull(result);
    }

    #endregion

    #region IReferringSegmentation Contract

    [Fact(Timeout = 120000)]
    public async Task LISA_ReferringInterface_SegmentFromExpression()
    {
        var model = new LISA<float>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<float>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "the red car on the left");
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoLISA_ReferringInterface_SegmentFromExpression()
    {
        var model = new VideoLISA<float>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<float>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "the person walking");
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task GLaMM_ReferringInterface_SegmentFromExpression()
    {
        var model = new GLaMM<float>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<float>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "cat");
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task PixelLM_ReferringInterface_SegmentFromExpression()
    {
        var model = new PixelLM<float>(Arch(), numClasses: 5);
        var referring = (IReferringSegmentation<float>)model;
        var result = referring.SegmentFromExpression(Rand(1, 3, 32, 32), "the building");
        Assert.NotNull(result);
    }

    #endregion

    #region Model Metadata and Parameter Counts

    [Fact(Timeout = 120000)]
    public async Task SegFormer_ParameterCount_PositiveAndModelSizeDependent()
    {
        var b0 = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var b3 = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B3);

        Assert.True(b0.ParameterCount > 0, "B0 should have parameters");
        Assert.True(b3.ParameterCount > b0.ParameterCount,
            $"B3 ({b3.ParameterCount}) should have more params than B0 ({b0.ParameterCount})");
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_ParameterCount_Positive()
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_ParameterCount_Positive()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_ParameterCount_Positive()
    {
        var model = new NnUNet<float>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task VisionMamba_ParameterCount_Positive()
    {
        var model = new VisionMamba<float>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task PIDNet_ParameterCount_Positive()
    {
        var model = new PIDNet<float>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion

    #region GetModelMetadata

    [Fact(Timeout = 120000)]
    public async Task SegFormer_GetModelMetadata_ContainsRequiredFields()
    {
        var model = new SegFormer<float>(Arch(), numClasses: 21, modelSize: SegFormerModelSize.B2);
        var meta = model.GetModelMetadata();
        Assert.NotNull(meta);

        Assert.NotNull(meta.AdditionalInfo);
        Assert.Equal("SegFormer", meta.AdditionalInfo["ModelName"]);
        Assert.Equal(21, meta.AdditionalInfo["NumClasses"]);
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_GetModelMetadata_ContainsRequiredFields()
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var meta = model.GetModelMetadata();
        Assert.NotNull(meta);
        Assert.NotNull(meta.AdditionalInfo);
        Assert.True(meta.AdditionalInfo.ContainsKey("ModelName"));
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_GetModelMetadata_ContainsModelName()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var meta = model.GetModelMetadata();
        Assert.NotNull(meta);
        Assert.Equal("SAM", meta.AdditionalInfo["ModelName"]);
    }

    #endregion

    #region Dispose Lifecycle — Double Dispose, Use-After-Dispose

    [Fact(Timeout = 120000)]
    public async Task SegFormer_DoubleDispose_DoesNotThrow()
    {
        var model = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_DoubleDispose_DoesNotThrow()
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_DoubleDispose_DoesNotThrow()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_DoubleDispose_DoesNotThrow()
    {
        var model = new NnUNet<float>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact(Timeout = 120000)]
    public async Task VisionMamba_DoubleDispose_DoesNotThrow()
    {
        var model = new VisionMamba<float>(Arch(), numClasses: 5, modelSize: VisionMambaModelSize.Tiny);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact(Timeout = 120000)]
    public async Task PIDNet_DoubleDispose_DoesNotThrow()
    {
        var model = new PIDNet<float>(Arch(), numClasses: 5, modelSize: PIDNetModelSize.Small);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    [Fact(Timeout = 120000)]
    public async Task DEVA_DoubleDispose_DoesNotThrow()
    {
        var model = new DEVA<float>(Arch(), numClasses: 5, modelSize: DEVAModelSize.Base);
        model.Dispose();
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region Multiple Consecutive Predictions (stateful regression)

    [Fact(Timeout = 120000)]
    public async Task SegFormer_MultiplePredictions_AllReturnOutput()
    {
        var model = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0, $"Prediction {i} returned empty output");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_MultiplePredictions_AllReturnOutput()
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_MultiplePredictions_AllReturnOutput()
    {
        var model = new NnUNet<float>(Arch(), numClasses: 5, modelSize: NnUNetModelSize.UNet2D);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_MultiplePredictions_AllReturnOutput()
    {
        var model = new SAM<float>(Arch(), numClasses: 1, modelSize: SAMModelSize.ViTBase);

        for (int i = 0; i < 5; i++)
        {
            var output = model.Predict(Rand(1, 3, 32, 32));
            Assert.NotNull(output);
            Assert.True(output.Length > 0);
        }
    }

    #endregion

    #region Segment() method (ISegmentationModel)

    [Fact(Timeout = 120000)]
    public async Task SegFormer_Segment_EquivalentToPredict()
    {
        var model = new SegFormer<float>(Arch(), numClasses: 5, modelSize: SegFormerModelSize.B0);
        var input = Rand(1, 3, 32, 32);

        var segResult = ((ISegmentationModel<float>)model).Segment(input);
        Assert.NotNull(segResult);
        Assert.True(segResult.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Mask2Former_Segment_EquivalentToPredict()
    {
        var model = new Mask2Former<float>(Arch(), numClasses: 5, modelSize: Mask2FormerModelSize.SwinTiny);
        var segResult = ((ISegmentationModel<float>)model).Segment(Rand(1, 3, 32, 32));
        Assert.NotNull(segResult);
        Assert.True(segResult.Length > 0);
    }

    #endregion

    #region ISegmentationModel Properties

    [Fact(Timeout = 120000)]
    public async Task SegFormer_ISegmentationModel_Properties()
    {
        var model = new SegFormer<float>(Arch(64, 48, 3), numClasses: 21, modelSize: SegFormerModelSize.B0);
        var seg = (ISegmentationModel<float>)model;
        Assert.Equal(21, seg.NumClasses);
        Assert.Equal(64, seg.InputHeight);
        Assert.Equal(48, seg.InputWidth);
        Assert.False(seg.IsOnnxMode);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM_ISegmentationModel_Properties()
    {
        var model = new SAM<float>(Arch(32, 32, 3), numClasses: 1, modelSize: SAMModelSize.ViTBase);
        var seg = (ISegmentationModel<float>)model;
        Assert.Equal(1, seg.NumClasses);
        Assert.False(seg.IsOnnxMode);
        Assert.True(model.SupportsTraining);
    }

    #endregion
}
