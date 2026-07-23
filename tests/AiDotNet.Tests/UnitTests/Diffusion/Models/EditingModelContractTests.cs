using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Diffusion.VirtualTryOn;
using AiDotNet.Diffusion.Panorama;
using AiDotNet.Diffusion.MotionGeneration;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.Alignment;
using AiDotNet.Diffusion.Safety;
using AiDotNet.Diffusion.Distillation;
using AiDotNet.Diffusion.MaskUtilities;
using AiDotNet.Models;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Contract tests for Phase 5 editing, inpainting, style transfer, virtual try-on,
/// panorama, and motion generation models, plus Phase 8 SR, alignment, and safety.
/// </summary>
public class EditingModelContractTests : DiffusionUnitTestBase
{
    #region Inpainting Models (Phase 5b)

    [Fact(Timeout = 120000)]
    public async Task BrushNetModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new BrushNetModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task BrushNetXModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new BrushNetXModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task PowerPaintModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PowerPaintModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task TurboFillModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TurboFillModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task HDPainterModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new HDPainterModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task FluxInpaintingModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FluxInpaintingModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SD3InpaintingModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SD3InpaintingModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SDXLInpaintingModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SDXLInpaintingModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Editing Models (Phase 5c)

    [Fact(Timeout = 120000)]
    public async Task OmniGen2Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new OmniGen2Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task ICEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ICEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task AnyEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new AnyEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task FlowEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FlowEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task TurboEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TurboEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task UltraEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new UltraEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task Pix2PixZeroModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new Pix2PixZeroModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task CycleGANTurboModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new CycleGANTurboModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Style Transfer Models (Phase 5d)

    [Fact(Timeout = 120000)]
    public async Task StyDiffModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StyDiffModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task StyleStudioModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StyleStudioModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task RBModulationModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new RBModulationModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task InstantStyleModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new InstantStyleModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Virtual Try-On Models (Phase 5e)

    [Fact(Timeout = 120000)]
    public async Task IDMVTONModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new IDMVTONModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task CatVTONModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new CatVTONModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task StableVITONModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StableVITONModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Panorama Models (Phase 5f)

    [Fact(Timeout = 120000)]
    public async Task MultiDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MultiDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task SyncDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SyncDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task StitchDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StitchDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Motion Generation Models (Phase 5g)

    [Fact(Timeout = 120000)]
    public async Task MotionDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MotionDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    [Fact(Timeout = 120000)]
    public async Task MoMaskModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MoMaskModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
    }

    #endregion

    #region Super-Resolution Models (Phase 8)

    [Fact(Timeout = 120000)]
    public async Task SeeSRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SeeSRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task PASDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new PASDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task CCSRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new CCSRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task TSDSRModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TSDSRModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.Equal(4, model.LatentChannels);
    }

    #endregion

    #region Alignment (Phase 8)

    [Fact(Timeout = 120000)]
    public async Task DiffusionDPO_Constructor_CreatesValid()
    {
        var model = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var reference = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();

        var dpo = new DiffusionDPO<double>(model, reference);

        Assert.NotNull(dpo);
        Assert.NotNull(dpo.Model);
        Assert.NotNull(dpo.ReferenceModel);
        Assert.Equal(5000.0, dpo.Beta);
    }

    [Fact(Timeout = 120000)]
    public async Task DiffusionRLHF_Constructor_CreatesValid()
    {
        var model = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var reference = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();

        var rlhf = new DiffusionRLHF<double>(model, reference);

        Assert.NotNull(rlhf);
        Assert.Equal(0.01, rlhf.KLWeight);
        Assert.Equal(1.0, rlhf.RewardScale);
    }

    [Fact(Timeout = 120000)]
    public async Task RewardGuidance_DefaultConstructor_CreatesValid()
    {
        var guidance = new RewardGuidance<double>();

        Assert.NotNull(guidance);
        Assert.Equal(5.0, guidance.GuidanceScale);
        Assert.Equal(0.5, guidance.TruncationTimestep);
    }

    #endregion

    #region Safety (Phase 8)

    [Fact(Timeout = 120000)]
    public async Task ConceptEraser_DefaultConstructor_CreatesValid()
    {
        var eraser = new ConceptEraser<double>();

        Assert.NotNull(eraser);
        Assert.Equal(0, eraser.ConceptCount);
        Assert.Equal(1.0, eraser.ErasureStrength);
    }

    [Fact(Timeout = 120000)]
    public async Task ConceptEraser_AddAndErase_Works()
    {
        var eraser = new ConceptEraser<double>();
        var concept = new Vector<double>(new double[] { 1.0, 0.0, 0.0, 0.0 });
        eraser.AddConceptDirection(concept);

        Assert.Equal(1, eraser.ConceptCount);

        var embedding = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });
        var result = eraser.EraseFromEmbedding(embedding);

        Assert.Equal(4, result.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task SCOREFramework_DefaultConstructor_CreatesValid()
    {
        var score = new SCOREFramework<double>();

        Assert.NotNull(score);
        Assert.Equal(0, score.RemappingCount);
        Assert.Equal(1.0, score.RemappingStrength);
    }

    [Fact(Timeout = 120000)]
    public async Task SCOREFramework_AddRemapping_Works()
    {
        var score = new SCOREFramework<double>();
        score.AddConceptRemapping("unsafe", "neutral");

        Assert.Equal(1, score.RemappingCount);
        Assert.True(score.IsConceptErased("unsafe"));
        Assert.Equal("neutral", score.GetReplacement("unsafe"));
    }

    [Fact(Timeout = 120000)]
    public async Task SGRACEEraser_DefaultConstructor_CreatesValid()
    {
        var eraser = new SGRACEEraser<double>();

        Assert.NotNull(eraser);
        Assert.Equal(0, eraser.StyleCount);
        Assert.Equal(1000, eraser.NumIterations);
    }

    [Fact(Timeout = 120000)]
    public async Task RACEEraser_DefaultConstructor_CreatesValid()
    {
        var eraser = new RACEEraser<double>();

        Assert.NotNull(eraser);
        Assert.Equal(1.0, eraser.AdversarialWeight);
        Assert.Equal(5, eraser.InnerSteps);
        Assert.Equal(1000, eraser.OuterSteps);
    }

    #endregion

    #region Mask Utilities (Phase 5a)

    [Fact(Timeout = 120000)]
    public async Task MaskFeatherer_DefaultConstructor_CreatesValid()
    {
        var featherer = new MaskFeatherer<double>();

        Assert.NotNull(featherer);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskBinarizer_DefaultConstructor_CreatesValid()
    {
        var binarizer = new MaskBinarizer<double>();

        Assert.NotNull(binarizer);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskInverter_DefaultConstructor_CreatesValid()
    {
        var inverter = new MaskInverter<double>();

        Assert.NotNull(inverter);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskBlur_DefaultConstructor_CreatesValid()
    {
        var blur = new MaskBlur<double>();

        Assert.NotNull(blur);
    }

    #endregion

    #region Distillation Trainers (Phase 7c)

    [Fact(Timeout = 120000)]
    public async Task ScoreDistillationTrainer_Constructor_CreatesValid()
    {
        var teacher = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var trainer = new ScoreDistillationTrainer<double>(teacher);

        Assert.NotNull(trainer);
        Assert.NotNull(trainer.Teacher);
        Assert.Equal(100.0, trainer.GuidanceScale);
    }

    [Fact(Timeout = 120000)]
    public async Task ConsistencyDistillationTrainer_Constructor_CreatesValid()
    {
        var teacher = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var trainer = new ConsistencyDistillationTrainer<double>(teacher);

        Assert.NotNull(trainer);
        Assert.NotNull(trainer.Teacher);
    }

    [Fact(Timeout = 120000)]
    public async Task AdversarialDistillationTrainer_Constructor_CreatesValid()
    {
        var teacher = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var trainer = new AdversarialDistillationTrainer<double>(teacher);

        Assert.NotNull(trainer);
        Assert.NotNull(trainer.Teacher);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgressiveDistillationTrainer_Constructor_CreatesValid()
    {
        var teacher = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var trainer = new ProgressiveDistillationTrainer<double>(teacher);

        Assert.NotNull(trainer);
        Assert.NotNull(trainer.Teacher);
        Assert.True(trainer.NumRounds > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task TrajectoryConsistencyDistiller_Constructor_CreatesValid()
    {
        var teacher = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var trainer = new TrajectoryConsistencyDistiller<double>(teacher);

        Assert.NotNull(trainer);
        Assert.NotNull(trainer.Teacher);
    }

    #endregion

    #region Gap Analysis — Additional Panorama Models

    [Fact(Timeout = 120000)]
    public async Task DiffPanoModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new DiffPanoModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion

    #region Gap Analysis — Alignment Variants

    [Fact(Timeout = 120000)]
    public async Task D3PO_Constructor_CreatesValid()
    {
        var model = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var refModel = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var d3po = new D3PO<double>(model, refModel);

        Assert.NotNull(d3po);
        Assert.Equal(5000.0, d3po.Beta);
    }

    [Fact(Timeout = 120000)]
    public async Task AsyncOnlineDPO_Constructor_CreatesValid()
    {
        var model = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var refModel = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var trainer = new AsyncOnlineDPO<double>(model, refModel);

        Assert.NotNull(trainer);
        Assert.Equal(5000.0, trainer.Beta);
        Assert.Equal(0.5, trainer.OnlineMixingRatio);
        Assert.Equal(0, trainer.TotalUpdates);
    }

    [Fact(Timeout = 120000)]
    public async Task RefinerStage_Constructor_CreatesValid()
    {
        var model = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var refiner = new RefinerStage<double>(model);

        Assert.NotNull(refiner);
        Assert.Equal(0.3, refiner.Strength);
        Assert.Equal(7.5, refiner.GuidanceScale);
        Assert.Equal(20, refiner.RefinerSteps);
    }

    #endregion

    #region Gap Analysis — Distillation Infrastructure

    [Fact(Timeout = 120000)]
    public async Task StudentTeacherFramework_Constructor_CreatesValid()
    {
        var teacher = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var student = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var framework = new StudentTeacherFramework<double>(teacher, student);

        Assert.NotNull(framework);
        Assert.NotNull(framework.Teacher);
        Assert.NotNull(framework.Student);
        Assert.Equal(0.9999, framework.EMADecay);
    }

    [Fact(Timeout = 120000)]
    public async Task ConsistencyTrainingTrainer_Constructor_CreatesValid()
    {
        var trainer = new ConsistencyTrainingTrainer<double>();

        Assert.NotNull(trainer);
        Assert.Equal(2, trainer.InitialDiscretizationSteps);
        Assert.Equal(0, trainer.CurrentStep);
    }

    [Fact(Timeout = 120000)]
    public async Task DistributionMatchingDistiller_Constructor_CreatesValid()
    {
        var teacher = new AiDotNet.Diffusion.TextToImage.StableDiffusion15Model<double>();
        var distiller = new DistributionMatchingDistiller<double>(teacher);

        Assert.NotNull(distiller);
        Assert.Equal(1.0, distiller.RegressionWeight);
        Assert.Equal(1.0, distiller.DistributionWeight);
    }

    #endregion

    #region Gap Analysis — Mask Utilities (Internal Components)

    [Fact(Timeout = 120000)]
    public async Task LatentMaskBlender_DefaultConstructor_CreatesValid()
    {
        var blender = new LatentMaskBlender<double>();

        Assert.NotNull(blender);
        Assert.Equal(1.0, blender.BlendSharpness);
    }

    [Fact(Timeout = 120000)]
    public async Task AlphaCompositor_DefaultConstructor_CreatesValid()
    {
        var compositor = new AlphaCompositor<double>();

        Assert.NotNull(compositor);
        Assert.False(compositor.PremultipliedAlpha);
    }

    [Fact(Timeout = 120000)]
    public async Task SeamlessBlender_DefaultConstructor_CreatesValid()
    {
        var blender = new SeamlessBlender<double>();

        Assert.NotNull(blender);
        Assert.Equal(BlendProfile.Cosine, blender.Profile);
        Assert.Equal(32, blender.OverlapSize);
    }

    #endregion

    #region Gap Analysis — Scheduler Utilities (Internal Components)

    [Fact(Timeout = 120000)]
    public async Task LatentInitializer_DefaultConstructor_CreatesValid()
    {
        var initializer = new AiDotNet.Diffusion.Schedulers.LatentInitializer<double>();

        Assert.NotNull(initializer);
        Assert.Equal(1.0, initializer.InitNoiseSigma);
    }

    [Fact(Timeout = 120000)]
    public async Task StrengthBasedScheduling_DefaultConstructor_CreatesValid()
    {
        var scheduling = new AiDotNet.Diffusion.Schedulers.StrengthBasedScheduling<double>();

        Assert.NotNull(scheduling);
        Assert.Equal(1000, scheduling.TotalTimesteps);
        Assert.Equal(0.8, scheduling.DefaultStrength);
    }

    #endregion

    #region Missing Model Coverage Tests

    [Fact(Timeout = 120000)]
    public async Task BrushEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new BrushEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task FreeInpaintModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FreeInpaintModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task RADModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new RADModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ReplaceAnythingModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ReplaceAnythingModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Step1XEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new Step1XEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SeedEdit3Model_DefaultConstructor_CreatesValidModel()
    {
        var model = new SeedEdit3Model<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task StyleAlignedEditModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new StyleAlignedEditModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task CACTIModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new CACTIModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task ConsisLoRAModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new ConsisLoRAModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task KLoRAStyleModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new KLoRAStyleModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SASTDModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SASTDModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task TLoRAModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new TLoRAModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task CATDMModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new CATDMModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task FashionVDMModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new FashionVDMModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task CubeDiffModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new CubeDiffModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SpotDiffusionModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new SpotDiffusionModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task MotionDiffuseModel_DefaultConstructor_CreatesValidModel()
    {
        var model = new MotionDiffuseModel<double>();

        Assert.NotNull(model);
        Assert.NotNull(model.NoisePredictor);
        Assert.NotNull(model.VAE);
        Assert.True(model.ParameterCount > 0);
    }

    #endregion
}
