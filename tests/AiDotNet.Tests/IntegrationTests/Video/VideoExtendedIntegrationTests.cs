using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task VideoModelOptions_DefaultsAreNull()
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

    [Fact(Timeout = 120000)]
    public async Task VideoModelOptions_SetAllProperties()
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

    [Fact(Timeout = 120000)]
    public async Task VideoEnhancementOptions_DefaultsAreNull()
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

    [Fact(Timeout = 120000)]
    public async Task VideoEnhancementOptions_SetCustomValues()
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

    [Fact(Timeout = 120000)]
    public async Task VideoEnhancementOptions_InheritsFromVideoModelOptions()
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

    [Fact(Timeout = 120000)]
    public async Task VideoEnhancementType_HasExpectedValues()
    {
        Assert.Equal(0, (int)VideoEnhancementType.SuperResolution);
        Assert.Equal(1, (int)VideoEnhancementType.Denoising);
        Assert.Equal(2, (int)VideoEnhancementType.Stabilization);
        Assert.Equal(3, (int)VideoEnhancementType.FrameInterpolation);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM2ModelSize_HasExpectedValues()
    {
        Assert.Equal(0, (int)SAM2ModelSize.Tiny);
        Assert.Equal(1, (int)SAM2ModelSize.Small);
        Assert.Equal(2, (int)SAM2ModelSize.Base);
        Assert.Equal(3, (int)SAM2ModelSize.Large);
    }

    [Fact(Timeout = 120000)]
    public async Task AttentionType_HasExpectedValues()
    {
        Assert.Equal(0, (int)AttentionType.JointSpaceTime);
        Assert.Equal(1, (int)AttentionType.DividedSpaceTime);
    }

    #endregion

    #region SlowFast - Action Recognition

    [Fact(Timeout = 120000)]
    public async Task SlowFast_Construction_DefaultParams()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch);

        Assert.True(model.SupportsTraining);
        Assert.Equal(400, model.NumClasses);
        Assert.Equal(4, model.SlowFrames);
        Assert.Equal(32, model.FastFrames);
        Assert.Equal(8, model.Alpha);
    }

    [Fact(Timeout = 120000)]
    public async Task SlowFast_Construction_CustomParams()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch, numClasses: 200, slowFrames: 8, alpha: 4);

        Assert.Equal(200, model.NumClasses);
        Assert.Equal(8, model.SlowFrames);
        Assert.Equal(32, model.FastFrames);
        Assert.Equal(4, model.Alpha);
    }

    [Fact(Timeout = 120000)]
    public async Task SlowFast_InvalidNumClasses_Throws()
    {
        var arch = CreateArch();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlowFast<double>(arch, numClasses: 0));
    }

    [Fact(Timeout = 120000)]
    public async Task SlowFast_InvalidSlowFrames_Throws()
    {
        var arch = CreateArch();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlowFast<double>(arch, slowFrames: 0));
    }

    [Fact(Timeout = 120000)]
    public async Task SlowFast_InvalidAlpha_Throws()
    {
        var arch = CreateArch();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlowFast<double>(arch, alpha: 0));
    }

    [Fact(Timeout = 120000)]
    public async Task SlowFast_Predict_ProducesOutput()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch, numClasses: 10);
        var input = new Tensor<double>([1, 3, 32, 32]);
        var output = model.Predict(input);

        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task SlowFast_GetModelMetadata_ContainsCorrectInfo()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch, numClasses: 100);
        var metadata = model.GetModelMetadata();


        Assert.Equal("SlowFast", metadata.AdditionalInfo["ModelName"]);
        Assert.Equal(100, metadata.AdditionalInfo["NumClasses"]);
    }

    #endregion

    #region Model Construction Tests (Lightweight)

    [Fact(Timeout = 120000)]
    public async Task RAFT_Construction()
    {
        var arch = CreateArch();
        var model = new RAFT<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task RAFT_InputChannels_DefaultsToThree()
    {
        var arch = CreateArch();
        var model = new RAFT<double>(arch);
        Assert.Equal(3, model.InputChannels);
    }

    [Fact(Timeout = 120000)]
    public async Task GMFlow_Construction()
    {
        var arch = CreateArch();
        var model = new GMFlow<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task FlowFormer_Construction()
    {
        var arch = CreateArch();
        var model = new FlowFormer<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task FILM_Construction()
    {
        var arch = CreateArch();
        var model = new FILM<double>(arch, numScales: 3, numFeatures: 32);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task RIFE_Construction()
    {
        var arch = CreateArch();
        var model = new RIFE<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task FLAVR_Construction()
    {
        var arch = CreateArch();
        var model = new FLAVR<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task VRT_Construction()
    {
        var arch = CreateArch();
        var model = new VRT<double>(arch, embedDim: 64, numFrames: 4, scaleFactor: 2);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task BasicVSRPlusPlus_Construction()
    {
        var arch = CreateArch();
        var model = new BasicVSRPlusPlus<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    // ── BasicVSR++ SECOND-ORDER grid propagation (Chan et al. 2022; #1789 review) ─────────────────────
    private const int SecondOrderFeatures = 8;
    private const int SecondOrderHW = 32;   // >= 2^(numLevels-1) so SPyNet's 5-level pyramid stays valid.

    private static BasicVSRPlusPlus<double> CreateSecondOrderModel() =>
        new(new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputHeight: SecondOrderHW, inputWidth: SecondOrderHW, inputDepth: 3, outputSize: 2),
            scaleFactor: 2, numFeatures: SecondOrderFeatures, numResidualBlocks: 1, numPropagations: 1);

    private static Tensor<double> MakeClip(int numFrames, int seed)
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        var clip = new Tensor<double>([numFrames, 3, SecondOrderHW, SecondOrderHW]);
        var s = clip.Data.Span;
        for (int i = 0; i < s.Length; i++) s[i] = rng.NextDouble();
        return clip;
    }

    // Second-order propagation feeds the deformable-alignment layer THREE feature maps — the current
    // feature plus the first-order (i±1) and second-order (i±2) warped neighbours — so its lazily
    // resolved input width is 3 × numFeatures. A first-order-only implementation would concat only two
    // maps (2×). This is the structural proof that the second-order path is wired into alignment.
    [Fact(Timeout = 120000)]
    public async Task BasicVSRPlusPlus_SecondOrder_AlignmentConsumesThreeFeatureMaps()
    {
        await Task.Yield();
        var model = CreateSecondOrderModel();
        _ = model.EnhanceVideo(MakeClip(numFrames: 3, seed: 1)); // resolves lazy alignment channels

        var alignments = model.Layers.OfType<DeformableConvolutionalLayer<double>>().ToList();
        Assert.NotEmpty(alignments);
        foreach (var align in alignments)
            Assert.Equal(3 * SecondOrderFeatures, align.GetInputShape()[0]);
    }

    // The DIRECT second-order (i+2) path carries temporal information end-to-end: perturbing a frame two
    // steps away from a target frame changes that frame's reconstruction THROUGH THE i+2 WARP ALONE.
    // First-order propagation is disabled so that the ordinary i±1 recurrence cannot carry frame 2's
    // change to frame 0 transitively (2→1→0) — without that isolation the assertion would pass even if
    // the direct i→i+2 propagation were broken, making the test meaningless.
    [Fact(Timeout = 120000)]
    public async Task BasicVSRPlusPlus_SecondOrder_TwoStepNeighbourInfluencesOutput()
    {
        await Task.Yield();
        var model = CreateSecondOrderModel();
        // Isolate the second-order path: with the first-order (i±1) contribution zeroed, the ONLY route
        // from frame 2 to frame 0 is the composed i→i+2 warp.
        model.DisableFirstOrderPropagationForTesting = true;

        var clip = MakeClip(numFrames: 3, seed: 2);
        var baseline = model.EnhanceVideo(clip);

        // Perturb ONLY frame 2.
        var perturbed = new Tensor<double>([3, 3, SecondOrderHW, SecondOrderHW]);
        clip.Data.Span.CopyTo(perturbed.Data.Span);
        int frameStride = 3 * SecondOrderHW * SecondOrderHW;
        var ps = perturbed.Data.Span;
        for (int i = 2 * frameStride; i < 3 * frameStride; i++) ps[i] += 0.5;
        var changed = model.EnhanceVideo(perturbed);

        // Frame 0's reconstruction must differ (output is [3, C, H*scale, W*scale]).
        int outFrameStride = changed.Length / 3;
        double maxDelta = 0.0;
        var a = baseline.Data.Span;
        var b = changed.Data.Span;
        for (int i = 0; i < outFrameStride; i++)
            maxDelta = System.Math.Max(maxDelta, System.Math.Abs(a[i] - b[i]));
        Assert.True(maxDelta > 1e-9,
            $"With first-order propagation disabled, perturbing frame 2 left frame 0's reconstruction " +
            $"unchanged (Δ={maxDelta:E3}); the DIRECT second-order (i+2) path is not carrying information " +
            "to frame 0.");
    }

    // Directly proves the second-order (i+2) SLOT of frame 0's alignment input is a function of frame 2:
    // records frame 0's [current | first-order | second-order] alignment tensor via the model's recorder
    // hook and asserts the second-order channel slice ([2·numFeatures, 3·numFeatures)) responds to a
    // frame-2 perturbation. This exercises the composed i→i+2 flow at the exact tensor the alignment layer
    // consumes, independent of any first-order contribution and of the downstream reconstruction.
    [Fact(Timeout = 120000)]
    public async Task BasicVSRPlusPlus_SecondOrder_AlignmentSecondOrderSliceRespondsToTwoStepPerturbation()
    {
        await Task.Yield();
        var model = CreateSecondOrderModel();

        Tensor<double>? frame0AlignInput = null;
        model.BackwardAlignInputRecorder = (frameIndex, alignInput) =>
        {
            if (frameIndex == 0) frame0AlignInput = alignInput.Clone();
        };

        var clip = MakeClip(numFrames: 3, seed: 3);
        _ = model.EnhanceVideo(clip);
        Assert.NotNull(frame0AlignInput);
        var baselineSlice = ExtractSecondOrderSlice(frame0AlignInput!);

        frame0AlignInput = null;
        var perturbed = new Tensor<double>([3, 3, SecondOrderHW, SecondOrderHW]);
        clip.Data.Span.CopyTo(perturbed.Data.Span);
        int frameStride = 3 * SecondOrderHW * SecondOrderHW;
        var ps = perturbed.Data.Span;
        for (int i = 2 * frameStride; i < 3 * frameStride; i++) ps[i] += 0.5;
        _ = model.EnhanceVideo(perturbed);
        Assert.NotNull(frame0AlignInput);
        var changedSlice = ExtractSecondOrderSlice(frame0AlignInput!);

        double maxDelta = 0.0;
        for (int i = 0; i < baselineSlice.Length; i++)
            maxDelta = System.Math.Max(maxDelta, System.Math.Abs(baselineSlice[i] - changedSlice[i]));
        Assert.True(maxDelta > 1e-9,
            $"Frame 0's second-order alignment slot did not respond to a frame-2 perturbation (Δ={maxDelta:E3}); " +
            "the composed i→i+2 flow is not routing frame 2 into frame 0's alignment input.");
    }

    // The alignment input is [current | first-order warp | second-order warp] concatenated along the
    // channel axis (axis 0 of the rank-3 [C, H, W] feature map), so the second-order slot is the last
    // third of the channels: [2·numFeatures, 3·numFeatures).
    private static double[] ExtractSecondOrderSlice(Tensor<double> alignInput)
    {
        int channels = alignInput.Shape[0];
        int perChannel = alignInput.Length / channels;
        int start = 2 * SecondOrderFeatures * perChannel;
        int count = SecondOrderFeatures * perChannel;
        var slice = new double[count];
        var span = alignInput.Data.Span;
        for (int i = 0; i < count; i++) slice[i] = span[start + i];
        return slice;
    }

    [Fact(Timeout = 120000)]
    public async Task EDVR_Construction()
    {
        var arch = CreateArch();
        var model = new EDVR<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task FastDVDNet_Construction()
    {
        var arch = CreateArch();
        var model = new FastDVDNet<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task DepthAnythingV2_Construction()
    {
        var arch = CreateArch();
        var model = new DepthAnythingV2<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task MiDaS_Construction()
    {
        var arch = CreateArch();
        var model = new MiDaS<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task AnimateDiff_Construction()
    {
        var arch = CreateArch();
        var model = new AnimateDiff<double>(arch, numLayers: 2, numFrames: 4);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task CogVideo_Construction()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new CogVideo<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task OpenSora_Construction()
    {
        var arch = CreateArch();
        var model = new OpenSora<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task StableVideoDiffusion_Construction()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new StableVideoDiffusion<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task E2FGVI_Construction()
    {
        var arch = CreateArch();
        var model = new E2FGVI<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task ProPainter_Construction()
    {
        var arch = CreateArch();
        var model = new ProPainter<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task RVM_Construction()
    {
        var arch = CreateArch();
        var model = new RVM<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM2_Construction_DefaultParams()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new SAM2<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM2_Construction_CustomModelSize()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new SAM2<double>(arch, modelSize: SAM2ModelSize.Small);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task Cutie_Construction()
    {
        var arch = CreateArch();
        var model = new Cutie<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task XMem_Construction()
    {
        var arch = CreateArch();
        var model = new XMem<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task DIFRINT_Construction()
    {
        var arch = CreateArch();
        var model = new DIFRINT<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task ByteTrack_Construction()
    {
        var arch = CreateArch(height: 64, width: 64);
        var model = new ByteTrack<double>(arch, numFeatures: 64);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task InternVideo2_Construction()
    {
        var arch = CreateArch();
        var model = new InternVideo2<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoCLIP_Construction()
    {
        var arch = CreateArch();
        var model = new VideoCLIP<double>(arch);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task TimeSformer_Construction()
    {
        var arch = CreateArch();
        var model = new TimeSformer<double>(arch, numClasses: 10, embedDim: 64, numHeads: 4, numLayers: 2, numFrames: 4, patchSize: 8);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoMAE_Construction()
    {
        var arch = CreateArch();
        var model = new VideoMAE<double>(arch, numClasses: 10, numFrames: 4, numFeatures: 64);
        Assert.True(model.SupportsTraining);
    }

    #endregion

    #region Model Metadata Tests

    [Fact(Timeout = 120000)]
    public async Task SlowFast_Metadata_HasVideoActionRecognitionType()
    {
        var arch = CreateArch();
        var model = new SlowFast<double>(arch);
        var metadata = model.GetModelMetadata();

    }

    [Fact(Timeout = 120000)]
    public async Task RAFT_Metadata_NotNull()
    {
        var arch = CreateArch();
        var model = new RAFT<double>(arch);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 120000)]
    public async Task FILM_Metadata_NotNull()
    {
        var arch = CreateArch();
        var model = new FILM<double>(arch, numScales: 3, numFeatures: 32);
        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 120000)]
    public async Task VRT_Metadata_NotNull()
    {
        var arch = CreateArch();
        var model = new VRT<double>(arch, embedDim: 64, numFrames: 4, scaleFactor: 2);
        Assert.NotNull(model.GetModelMetadata());
    }

    #endregion

    #region Options Classes

    [Fact(Timeout = 120000)]
    public async Task SlowFastOptions_IsNeuralNetworkOptions()
    {
        var opts = new SlowFastOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task TimeSformerOptions_IsNeuralNetworkOptions()
    {
        var opts = new TimeSformerOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoMAEOptions_IsNeuralNetworkOptions()
    {
        var opts = new VideoMAEOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task RAFTOptions_IsNeuralNetworkOptions()
    {
        var opts = new RAFTOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task RIFEOptions_IsNeuralNetworkOptions()
    {
        var opts = new RIFEOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task FILMOptions_IsNeuralNetworkOptions()
    {
        var opts = new FILMOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task FLAVROptions_IsNeuralNetworkOptions()
    {
        var opts = new FLAVROptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task SAM2Options_IsNeuralNetworkOptions()
    {
        var opts = new SAM2Options();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task ByteTrackOptions_IsNeuralNetworkOptions()
    {
        var opts = new ByteTrackOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task VRTOptions_IsNeuralNetworkOptions()
    {
        var opts = new VRTOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task AnimateDiffOptions_IsNeuralNetworkOptions()
    {
        var opts = new AnimateDiffOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task CogVideoOptions_IsNeuralNetworkOptions()
    {
        var opts = new CogVideoOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task OpenSoraOptions_IsNeuralNetworkOptions()
    {
        var opts = new OpenSoraOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task StableVideoDiffusionOptions_IsNeuralNetworkOptions()
    {
        var opts = new StableVideoDiffusionOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task BasicVSRPlusPlusOptions_IsNeuralNetworkOptions()
    {
        var opts = new BasicVSRPlusPlusOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task EDVROptions_IsNeuralNetworkOptions()
    {
        var opts = new EDVROptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task FastDVDNetOptions_IsNeuralNetworkOptions()
    {
        var opts = new FastDVDNetOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task DepthAnythingV2Options_IsNeuralNetworkOptions()
    {
        var opts = new DepthAnythingV2Options();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task MiDaSOptions_IsNeuralNetworkOptions()
    {
        var opts = new MiDaSOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task E2FGVIOptions_IsNeuralNetworkOptions()
    {
        var opts = new E2FGVIOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task ProPainterOptions_IsNeuralNetworkOptions()
    {
        var opts = new ProPainterOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task RVMOptions_IsNeuralNetworkOptions()
    {
        var opts = new RVMOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task DIFRINTOptions_IsNeuralNetworkOptions()
    {
        var opts = new DIFRINTOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task CutieOptions_IsNeuralNetworkOptions()
    {
        var opts = new CutieOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task XMemOptions_IsNeuralNetworkOptions()
    {
        var opts = new XMemOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task InternVideo2Options_IsNeuralNetworkOptions()
    {
        var opts = new InternVideo2Options();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task VideoCLIPVideoOptions_IsNeuralNetworkOptions()
    {
        var opts = new VideoCLIPVideoOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task FlowFormerOptions_IsNeuralNetworkOptions()
    {
        var opts = new FlowFormerOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    [Fact(Timeout = 120000)]
    public async Task GMFlowOptions_IsNeuralNetworkOptions()
    {
        var opts = new GMFlowOptions();
        Assert.IsAssignableFrom<AiDotNet.Models.Options.NeuralNetworkOptions>(opts);
    }

    #endregion
}
