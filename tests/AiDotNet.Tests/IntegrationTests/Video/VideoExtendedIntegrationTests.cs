using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Video.ActionRecognition;
using AiDotNet.Video.Denoising;
using AiDotNet.Video.Depth;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.FrameInterpolation;
using AiDotNet.Video.Generation;
using AiDotNet.Video.Inpainting;
using AiDotNet.Video.Interfaces;
using AiDotNet.Video.Matting;
using AiDotNet.Video.Motion;
using AiDotNet.Video.Options;
using AiDotNet.Video.Restoration;
using AiDotNet.Video.Segmentation;
using AiDotNet.Video.Stabilization;
using AiDotNet.Video.Tracking;
using AiDotNet.Video.Understanding;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Video;

/// <summary>
/// Extended integration tests for Video module classes beyond RealESRGAN.
/// Covers all model construction, options defaults, parameter validation,
/// metadata, predictions, and enum values.
/// </summary>
public class VideoExtendedIntegrationTests
{
    private static NeuralNetworkArchitecture<double> CreateArch(
        int height = 32, int width = 32, int depth = 3) =>
        new(inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: height, inputWidth: width, inputDepth: depth);

    #region VideoModelOptions

    [Fact]
    public void VideoModelOptions_DefaultsAreNull()
    {
        var opts = new VideoModelOptions<double>();

        Assert.Null(opts.HiddenDimension);
        Assert.Null(opts.NumAttentionHeads);
        Assert.Null(opts.NumLayers);
        Assert.Null(opts.DropoutRate);
        Assert.Null(opts.NumFrames);
        Assert.Null(opts.InputHeight);
        Assert.Null(opts.InputWidth);
        Assert.Null(opts.InputChannels);
        Assert.Null(opts.LearningRate);
        Assert.Null(opts.BatchSize);
        Assert.Null(opts.WeightDecay);
        Assert.Null(opts.UseGradientClipping);
        Assert.Null(opts.MaxGradientNorm);
        Assert.Null(opts.UseGpu);
        Assert.Null(opts.UseMixedPrecision);
        Assert.Null(opts.RandomSeed);
    }

    [Fact]
    public void VideoModelOptions_SetAllProperties()
    {
        var opts = new VideoModelOptions<double>
        {
            HiddenDimension = 512,
            NumAttentionHeads = 8,
            NumLayers = 6,
            DropoutRate = 0.2,
            NumFrames = 32,
            InputHeight = 256,
            InputWidth = 256,
            InputChannels = 1,
            LearningRate = 0.001,
            BatchSize = 4,
            WeightDecay = 0.05,
            UseGradientClipping = false,
            MaxGradientNorm = 2.0,
            UseGpu = true,
            UseMixedPrecision = true,
            RandomSeed = 42
        };

        Assert.Equal(512, opts.HiddenDimension);
        Assert.Equal(8, opts.NumAttentionHeads);
        Assert.Equal(6, opts.NumLayers);
        Assert.Equal(0.2, opts.DropoutRate);
        Assert.Equal(32, opts.NumFrames);
        Assert.Equal(256, opts.InputHeight);
        Assert.Equal(256, opts.InputWidth);
        Assert.Equal(1, opts.InputChannels);
        Assert.Equal(0.001, opts.LearningRate);
        Assert.Equal(4, opts.BatchSize);
        Assert.Equal(0.05, opts.WeightDecay);
        Assert.False(opts.UseGradientClipping);
        Assert.Equal(2.0, opts.MaxGradientNorm);
        Assert.True(opts.UseGpu);
        Assert.True(opts.UseMixedPrecision);
        Assert.Equal(42, opts.RandomSeed);
    }

    #endregion

    #region VideoEnhancementOptions

    [Fact]
    public void VideoEnhancementOptions_DefaultsAreNull()
    {
        var opts = new VideoEnhancementOptions<double>();

        Assert.Null(opts.ScaleFactor);
        Assert.Null(opts.TemporalScaleFactor);
        Assert.Null(opts.EnhancementType);
        Assert.Null(opts.UseTemporalConsistency);
        Assert.Null(opts.RecurrentIterations);
        Assert.Null(opts.PerceptualLossWeight);
        Assert.Null(opts.L1LossWeight);
        Assert.Null(opts.AdversarialLossWeight);
        Assert.Null(opts.TemporalLossWeight);
        Assert.Null(opts.FlowLossWeight);
        Assert.Null(opts.NumResidualBlocks);
        Assert.Null(opts.NumFeatureChannels);
        Assert.Null(opts.UseAttention);
        Assert.Null(opts.UseBidirectional);
    }

    [Fact]
    public void VideoEnhancementOptions_SetCustomValues()
    {
        var opts = new VideoEnhancementOptions<double>
        {
            ScaleFactor = 4,
            TemporalScaleFactor = 2,
            EnhancementType = VideoEnhancementType.Denoising,
            UseTemporalConsistency = false,
            RecurrentIterations = 5,
            PerceptualLossWeight = 0.5,
            L1LossWeight = 2.0,
            AdversarialLossWeight = 0.2,
            TemporalLossWeight = 0.8,
            FlowLossWeight = 0.3,
            NumResidualBlocks = 32,
            NumFeatureChannels = 128,
            UseAttention = false,
            UseBidirectional = false
        };

        Assert.Equal(4, opts.ScaleFactor);
        Assert.Equal(2, opts.TemporalScaleFactor);
        Assert.Equal(VideoEnhancementType.Denoising, opts.EnhancementType);
        Assert.False(opts.UseTemporalConsistency);
        Assert.Equal(5, opts.RecurrentIterations);
        Assert.Equal(0.5, opts.PerceptualLossWeight);
        Assert.Equal(2.0, opts.L1LossWeight);
        Assert.Equal(0.2, opts.AdversarialLossWeight);
        Assert.Equal(0.8, opts.TemporalLossWeight);
        Assert.Equal(0.3, opts.FlowLossWeight);
        Assert.Equal(32, opts.NumResidualBlocks);
        Assert.Equal(128, opts.NumFeatureChannels);
        Assert.False(opts.UseAttention);
        Assert.False(opts.UseBidirectional);
    }

    [Fact]
    public void VideoEnhancementOptions_InheritsFromVideoModelOptions()
    {
        var opts = new VideoEnhancementOptions<double>
        {
            HiddenDimension = 1024,
            LearningRate = 0.01
        };

        Assert.Equal(1024, opts.HiddenDimension);
        Assert.Equal(0.01, opts.LearningRate);
    }

    #endregion

    #region Enums

    [Fact]
    public void VideoEnhancementType_HasExpectedValues()
    {
        Assert.Equal(0, (int)VideoEnhancementType.SuperResolution);
        Assert.Equal(1, (int)VideoEnhancementType.Denoising);
        Assert.Equal(2, (int)VideoEnhancementType.Stabilization);
        Assert.Equal(3, (int)VideoEnhancementType.FrameInterpolation);
    }

    [Fact]
    public void SAM2ModelSize_HasExpectedValues()
    {
        Assert.Equal(0, (int)SAM2ModelSize.Tiny);
        Assert.Equal(1, (int)SAM2ModelSize.Small);
        Assert.Equal(2, (int)SAM2ModelSize.Base);
        Assert.Equal(3, (int)SAM2ModelSize.Large);
    }

    [Fact]
    public void AttentionType_HasExpectedValues()
    {
        Assert.Equal(0, (int)AttentionType.JointSpaceTime);
        Assert.Equal(1, (int)AttentionType.DividedSpaceTime);
    }

    #endregion

    #region SlowFast - Action Recognition

    [Fact]
    public void SlowFast_Construction_DefaultParams()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch);

        Assert.True(model.SupportsTraining);
        Assert.Equal(400, model.NumClasses);
        Assert.Equal(4, model.SlowFrames);
        Assert.Equal(32, model.FastFrames);
        Assert.Equal(8, model.Alpha);
    }

    [Fact]
    public void SlowFast_Construction_CustomParams()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch, numClasses: 200, slowFrames: 8, alpha: 4);

        Assert.Equal(200, model.NumClasses);
        Assert.Equal(8, model.SlowFrames);
        Assert.Equal(32, model.FastFrames);
        Assert.Equal(4, model.Alpha);
    }

    [Fact]
    public void SlowFast_InvalidNumClasses_Throws()
    {
        var arch = CreateArch();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlowFast<double>(arch, numClasses: 0));
    }

    [Fact]
    public void SlowFast_InvalidSlowFrames_Throws()
    {
        var arch = CreateArch();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlowFast<double>(arch, slowFrames: 0));
    }

    [Fact]
    public void SlowFast_InvalidAlpha_Throws()
    {
        var arch = CreateArch();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlowFast<double>(arch, alpha: 0));
    }

    [Fact]
    public void SlowFast_Predict_ProducesOutput()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch, numClasses: 10);
        var input = new Tensor<double>([1, 3, 32, 32]);
        var output = model.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void SlowFast_GetModelMetadata_ContainsCorrectInfo()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch, numClasses: 100);
        var metadata = model.GetModelMetadata();

        Assert.Equal(ModelType.VideoActionRecognition, metadata.ModelType);
        Assert.Equal("SlowFast", metadata.AdditionalInfo["ModelName"]);
        Assert.Equal(100, metadata.AdditionalInfo["NumClasses"]);
    }

    #endregion

    #region Model Construction Tests (Lightweight)

    [Fact]
    public void RAFT_Construction()
    {
        var arch = CreateArch();
        var model = new RAFT<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void RAFT_InputChannels_DefaultsToThree()
    {
        var arch = CreateArch();
        var model = new RAFT<double>(arch);
        Assert.Equal(3, model.InputChannels);
    }

    [Fact]
    public void GMFlow_Construction()
    {
        var arch = CreateArch();
        var model = new GMFlow<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void FlowFormer_Construction()
    {
        var arch = CreateArch();
        var model = new FlowFormer<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void FILM_Construction()
    {
        var arch = CreateArch();
        var model = new FILM<double>(arch, numScales: 3, numFeatures: 32);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void RIFE_Construction()
    {
        var arch = CreateArch();
        var model = new RIFE<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void FLAVR_Construction()
    {
        var arch = CreateArch();
        var model = new FLAVR<double>(arch, numFeatures: 32);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void VRT_Construction()
    {
        var arch = CreateArch();
        var model = new VRT<double>(arch, embedDim: 64, numFrames: 4, scaleFactor: 2);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void BasicVSRPlusPlus_Construction()
    {
        var arch = CreateArch();
        var model = new BasicVSRPlusPlus<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void EDVR_Construction()
    {
        var arch = CreateArch();
        var model = new EDVR<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void FastDVDNet_Construction()
    {
        var arch = CreateArch();
        var model = new FastDVDNet<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void DepthAnythingV2_Construction()
    {
        var arch = CreateArch();
        var model = new DepthAnythingV2<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void MiDaS_Construction()
    {
        var arch = CreateArch();
        var model = new MiDaS<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void AnimateDiff_Construction()
    {
        var arch = CreateArch();
        var model = new AnimateDiff<double>(arch, numLayers: 2, numFrames: 4);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void CogVideo_Construction()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new CogVideo<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void OpenSora_Construction()
    {
        var arch = CreateArch();
        var model = new OpenSora<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void StableVideoDiffusion_Construction()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new StableVideoDiffusion<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void E2FGVI_Construction()
    {
        var arch = CreateArch();
        var model = new E2FGVI<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void ProPainter_Construction()
    {
        var arch = CreateArch();
        var model = new ProPainter<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void RVM_Construction()
    {
        var arch = CreateArch();
        var model = new RVM<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SAM2_Construction_DefaultParams()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new SAM2<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void SAM2_Construction_CustomModelSize()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new SAM2<double>(arch, modelSize: SAM2ModelSize.Small);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void Cutie_Construction()
    {
        var arch = CreateArch();
        var model = new Cutie<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void XMem_Construction()
    {
        var arch = CreateArch();
        var model = new XMem<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void DIFRINT_Construction()
    {
        var arch = CreateArch();
        var model = new DIFRINT<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void ByteTrack_Construction()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new ByteTrack<double>(arch, numFeatures: 64);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void InternVideo2_Construction()
    {
        var arch = CreateArch();
        var model = new InternVideo2<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void VideoCLIP_Construction()
    {
        var arch = CreateArch();
        var model = new VideoCLIP<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void TimeSformer_Construction()
    {
        var arch = CreateArch();
        var model = new TimeSformer<double>(arch, numClasses: 10, embedDim: 64, numHeads: 4, numLayers: 2, numFrames: 4, patchSize: 8);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void VideoMAE_Construction()
    {
        var arch = CreateArch();
        var model = new VideoMAE<double>(arch, numClasses: 10, numFrames: 4, numFeatures: 64);
        Assert.True(model.SupportsTraining);
    }

    #endregion

    #region Model Metadata Tests

    [Fact]
    public void SlowFast_Metadata_HasVideoActionRecognitionType()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch);
        var metadata = model.GetModelMetadata();
        Assert.Equal(ModelType.VideoActionRecognition, metadata.ModelType);
    }

    [Fact]
    public void RAFT_Metadata_NotNull()
    {
        var arch = CreateArch();
        var model = new RAFT<double>(arch);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void FILM_Metadata_NotNull()
    {
        var arch = CreateArch();
        var model = new FILM<double>(arch, numScales: 3, numFeatures: 32);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact]
    public void VRT_Metadata_NotNull()
    {
        var arch = CreateArch();
        var model = new VRT<double>(arch, embedDim: 64, numFrames: 4, scaleFactor: 2);
        Assert.NotNull(model.GetModelMetadata());
    }

    #endregion

    #region Options Classes

    [Fact]
    public void SlowFastOptions_IsNeuralNetworkOptions()
    {
        var opts = new SlowFastOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void TimeSformerOptions_IsNeuralNetworkOptions()
    {
        var opts = new TimeSformerOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void VideoMAEOptions_IsNeuralNetworkOptions()
    {
        var opts = new VideoMAEOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void RAFTOptions_IsNeuralNetworkOptions()
    {
        var opts = new RAFTOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void RIFEOptions_IsNeuralNetworkOptions()
    {
        var opts = new RIFEOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void FILMOptions_IsNeuralNetworkOptions()
    {
        var opts = new FILMOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void FLAVROptions_IsNeuralNetworkOptions()
    {
        var opts = new FLAVROptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void SAM2Options_IsNeuralNetworkOptions()
    {
        var opts = new SAM2Options();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void ByteTrackOptions_IsNeuralNetworkOptions()
    {
        var opts = new ByteTrackOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void VRTOptions_IsNeuralNetworkOptions()
    {
        var opts = new VRTOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void AnimateDiffOptions_IsNeuralNetworkOptions()
    {
        var opts = new AnimateDiffOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void CogVideoOptions_IsNeuralNetworkOptions()
    {
        var opts = new CogVideoOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void OpenSoraOptions_IsNeuralNetworkOptions()
    {
        var opts = new OpenSoraOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void StableVideoDiffusionOptions_IsNeuralNetworkOptions()
    {
        var opts = new StableVideoDiffusionOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void BasicVSRPlusPlusOptions_IsNeuralNetworkOptions()
    {
        var opts = new BasicVSRPlusPlusOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void EDVROptions_IsNeuralNetworkOptions()
    {
        var opts = new EDVROptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void FastDVDNetOptions_IsNeuralNetworkOptions()
    {
        var opts = new FastDVDNetOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void DepthAnythingV2Options_IsNeuralNetworkOptions()
    {
        var opts = new DepthAnythingV2Options();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void MiDaSOptions_IsNeuralNetworkOptions()
    {
        var opts = new MiDaSOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void E2FGVIOptions_IsNeuralNetworkOptions()
    {
        var opts = new E2FGVIOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void ProPainterOptions_IsNeuralNetworkOptions()
    {
        var opts = new ProPainterOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void RVMOptions_IsNeuralNetworkOptions()
    {
        var opts = new RVMOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void DIFRINTOptions_IsNeuralNetworkOptions()
    {
        var opts = new DIFRINTOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void CutieOptions_IsNeuralNetworkOptions()
    {
        var opts = new CutieOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void XMemOptions_IsNeuralNetworkOptions()
    {
        var opts = new XMemOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void InternVideo2Options_IsNeuralNetworkOptions()
    {
        var opts = new InternVideo2Options();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void VideoCLIPVideoOptions_IsNeuralNetworkOptions()
    {
        var opts = new VideoCLIPVideoOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void FlowFormerOptions_IsNeuralNetworkOptions()
    {
        var opts = new FlowFormerOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact]
    public void GMFlowOptions_IsNeuralNetworkOptions()
    {
        var opts = new GMFlowOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    #endregion
}
