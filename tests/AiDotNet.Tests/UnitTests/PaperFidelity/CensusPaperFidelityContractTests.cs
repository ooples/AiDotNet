using System.Reflection;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Document.OCR.TextRecognition;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Enhancement;
using AiDotNet.Video.FrameInterpolation;
using AiDotNet.Video.Options;
using AiDotNet.VisionLanguage.Encoders;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

/// <summary>
/// Paper contracts for the census models corrected by PR #2006. These intentionally inspect the
/// production defaults; bounded generated fixtures are tested separately and may not redefine them.
/// </summary>
public sealed class CensusPaperFidelityContractTests
{
    [Fact]
    public void SvtrTiny_DefaultTopologyMatchesReleasedConfiguration()
    {
        using var model = new SVTR<double>();
        var layers = ((ILayeredModel<double>)model).Layers;

        Assert.Equal(30, layers.Count);
        var tps = Assert.IsType<SVTRThinPlateSplineLayer<double>>(layers[0]);
        Assert.Equal(32, tps.LocalizationHeight);
        Assert.Equal(64, tps.LocalizationWidth);
        Assert.Equal(32, tps.OutputHeight);
        Assert.Equal(100, tps.OutputWidth);
        Assert.Equal(20, tps.ControlPointCount);
        Assert.Equal(0.05, tps.MarginX, 12);
        Assert.Equal(0.05, tps.MarginY, 12);

        var blocks = layers.OfType<SVTRMixingBlockLayer<double>>().ToArray();
        Assert.Equal(12, blocks.Length);
        Assert.All(blocks, block => Assert.True(block.UsesPreNormalization));
        Assert.All(blocks.Take(6), block => Assert.True(block.UsesLocalMixing));
        Assert.All(blocks.Skip(6), block => Assert.False(block.UsesLocalMixing));
        Assert.All(blocks.Take(6), block =>
        {
            Assert.Equal(7, block.WindowHeight);
            Assert.Equal(11, block.WindowWidth);
        });
        Assert.Equal([64, 64, 64, 128, 128, 128, 128, 128, 128, 256, 256, 256],
            blocks.Select(block => block.HiddenSize).ToArray());
        Assert.Equal([2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8, 8],
            blocks.Select(block => block.NumHeads).ToArray());
        Assert.Equal(0.0, blocks[0].DropPathRate, 12);
        Assert.Equal(0.1, blocks[^1].DropPathRate, 12);
        Assert.Single(layers.OfType<BiasFreeLinearLayer<double>>());
    }

    [Fact]
    public void SvtrTps_ReleasedInitializationUsesDistinctSourceAndTargetMargins()
    {
        var layer = new SVTRThinPlateSplineLayer<double>();
        var trainable = layer.GetTrainableParameters();
        var localizationWeights = Assert.Single(trainable.Where(tensor => tensor.Length == 512 * 40));
        var localizationBias = Assert.Single(trainable.Where(tensor => tensor.Length == 40)).AsSpan();

        Assert.Equal(0.05, layer.MarginX, 12);
        Assert.Equal(0.05, layer.MarginY, 12);
        Assert.Equal(0.01, localizationBias[0], 12);
        Assert.Equal(0.01, localizationBias[1], 12);
        Assert.Equal(0.99, localizationBias[18], 12);
        Assert.Equal(0.01, localizationBias[19], 12);
        Assert.Equal(0.01, localizationBias[20], 12);
        Assert.Equal(0.99, localizationBias[21], 12);
        Assert.Equal(0.99, localizationBias[38], 12);
        Assert.Equal(0.99, localizationBias[39], 12);
        Assert.All(localizationWeights.AsSpan().ToArray(),
            value => Assert.Equal(0.0, value, 12));
    }

    [Fact]
    public void UprNet_DefaultGraphUsesOneSharedFortySevenLayerPaperTopology()
    {
        using var model = new UPRNet<double>(VideoArchitecture(16, 16));
        var layers = ((ILayeredModel<double>)model).Layers;

        Assert.Equal(47, layers.Count);
        Assert.Equal(12, layers.Take(12).Count(layer => layer is ConvolutionalLayer<double>));
        Assert.Equal(6, layers.Skip(12).Take(6).Count(layer => layer is ConvolutionalLayer<double>));
        Assert.Equal(2, layers.OfType<DeconvolutionalLayer<double>>().Count());
        Assert.Equal(14, layers.OfType<PReLULayer<double>>().Count());
        Assert.Equal(5, ((ConvolutionalLayer<double>)layers[^1]).OutputDepth);
        Assert.Equal(47, model.GetModelMetadata().AdditionalInfo["LayerCount"]);
    }

    [Fact]
    public void UprNet_ArbitraryTimestepForwardHasPaperOutputShape()
    {
        using var model = new UPRNet<double>(VideoArchitecture(16, 16));
        var frame0 = Filled([3, 16, 16], 0.1, 0.0003);
        var frame1 = Filled([3, 16, 16], 0.7, -0.0002);

        var output = model.Interpolate(frame0, frame1, 0.25);

        Assert.Equal([3, 16, 16], output.Shape.ToArray());
        Assert.All(output.AsSpan().ToArray(), value =>
            Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        Assert.Equal(1_653_961, model.ParameterCount);

        var payload = model.Serialize();
        using var restored = new UPRNet<double>(VideoArchitecture(16, 16));
        restored.Deserialize(payload);
        Assert.Equal(model.ParameterCount, restored.ParameterCount);
        Assert.Equal(model.GetParameters().AsSpan().ToArray(), restored.GetParameters().AsSpan().ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void UprNet_ReleasedSkippedLevelScheduleStillRunsCoarsestAndFinestLevels(int skippedLevels)
    {
        using var model = new UPRNet<double>(VideoArchitecture(16, 16), new UPRNetOptions
        {
            NumPyramidLevels = 3,
            NumLevelsSkipped = skippedLevels
        });

        var output = model.Interpolate(
            Filled([3, 16, 16], 0.1, 0.0003),
            Filled([3, 16, 16], 0.7, -0.0002),
            0.5);

        Assert.Equal([3, 16, 16], output.Shape.ToArray());
        Assert.All(output.AsSpan().ToArray(), value =>
            Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
    }

    [Fact]
    public void MedClip_LogitScaleIsLearnedAndTapeConnected()
    {
        var layer = new LearnableLogitScaleLayer<double>(0.07);
        var parameter = Assert.Single(layer.GetTrainableParameters());
        var input = new Tensor<double>([0.25, -0.5, 1.0, 2.0], [1, 4]);
        var engine = new CpuEngine();

        using var tape = new GradientTape<double>();
        var loss = engine.ReduceSum(layer.Forward(input), null);
        var gradients = tape.ComputeGradients(loss, [parameter]);

        Assert.True(gradients.TryGetValue(parameter, out var gradient));
        Assert.NotNull(gradient);
        Assert.NotEqual(0.0, gradient![0]);
        Assert.Equal(1.0 / 0.07, layer.Scale, 10);
    }

    [Fact]
    public void MedClip_NativeTopologyRetainsReferenceResNetAndClinicalBertContracts()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 16, inputWidth: 16, inputDepth: 3, outputSize: 4);
        using var model = new MedCLIP<float>(architecture, new MedCLIPOptions
        {
            ImageSize = 16,
            ProjectionDim = 4,
            TextEmbeddingDim = 8,
            NumTextLayers = 2,
            NumTextHeads = 2,
            VocabSize = 64,
            MaxSequenceLength = 8
        });

        Assert.Equal(16, model.VisionBottleneckBlockCount);
        Assert.Equal(2, model.TextTransformerBlockCount);
        Assert.True(model.UsesTokenTypeEmbeddings);
        Assert.True(model.UsesBiasFreeReferenceProjections);
    }

    [Fact]
    public void MedClip_SemanticObjectiveNormalizesEmbeddingsAndClampsLabelSimilarity()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 2, inputWidth: 2, inputDepth: 3, outputSize: 2,
            layers: [new BiasFreeLinearLayer<double>(12, 2)]);
        using var model = new MedCLIP<double>(architecture, new MedCLIPOptions
        {
            ImageSize = 2,
            ProjectionDim = 2,
            TextEmbeddingDim = 4,
            NumTextLayers = 1,
            NumTextHeads = 1,
            VocabSize = 16,
            MaxSequenceLength = 4
        });
        var images = new Tensor<double>([2.0, 0.0, 0.0, 3.0], [2, 2]);
        var texts = new Tensor<double>([4.0, 0.0, 0.0, 5.0], [2, 2]);
        var unboundedScores = new Tensor<double>([10.0, -10.0, -4.0, 8.0], [2, 2]);
        var boundedScores = new Tensor<double>([1.0, -1.0, -1.0, 1.0], [2, 2]);

        var unboundedLoss = model.ComputeSemanticMatchingLoss(
            images, texts, unboundedScores);
        var boundedLoss = model.ComputeSemanticMatchingLoss(
            images, texts, boundedScores);

        Assert.Equal(boundedLoss, unboundedLoss, 10);
    }

    [Fact]
    public void MedClip_CustomGraphRoundTripsAllOwnedParameterStreams()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 4, inputWidth: 4, inputDepth: 3, outputSize: 4,
            layers: [new BiasFreeLinearLayer<double>(48, 4)]);
        var callerOptions = new MedCLIPOptions
        {
            ImageSize = 4,
            ProjectionDim = 4,
            TextEmbeddingDim = 8,
            NumTextLayers = 2,
            NumTextHeads = 2,
            VocabSize = 64,
            MaxSequenceLength = 8
        };
        using var source = new MedCLIP<double>(architecture, callerOptions);
        callerOptions.ProjectionDim = 99;
        Assert.Equal(4, ((MedCLIPOptions)source.GetOptions()).ProjectionDim);
        Assert.Throws<ArgumentException>(() => source.UpdateParameters(new Vector<double>(1)));

        var parameters = source.GetParameters();
        var payload = source.Serialize();
        using var restored = new MedCLIP<double>(architecture, new MedCLIPOptions
        {
            ImageSize = 4,
            ProjectionDim = 4,
            TextEmbeddingDim = 8,
            NumTextLayers = 2,
            NumTextHeads = 2,
            VocabSize = 64,
            MaxSequenceLength = 8
        });
        restored.Deserialize(payload);

        Assert.Equal(parameters.AsSpan().ToArray(), restored.GetParameters().AsSpan().ToArray());
        Assert.Equal(source.ParameterCount, restored.ParameterCount);
    }

    [Fact]
    public void UpscaleAVideo_DefaultPredictorMatchesReleasedSevenChannelContract()
    {
        using var model = new UpscaleAVideoModel<double>();
        var predictor = Assert.IsType<VideoUNetPredictor<double>>(model.NoisePredictor);

        Assert.Equal(4, predictor.InputChannels);
        Assert.Equal(3, predictor.ImageConditionChannels);
        Assert.True(predictor.ConcatenatesImageCondition);
        Assert.Equal(256, predictor.BaseChannels);
        Assert.Equal([1, 2, 2, 4], predictor.ChannelMultipliers.ToArray());
        Assert.Equal(1000, predictor.NumClassEmbeddings);
        Assert.Equal(2, predictor.NumResBlocks);
        Assert.Equal(VideoUNetArchitectureProfile.UpscaleAVideo, predictor.ArchitectureProfile);
        Assert.Equal(9, predictor.TemporalModuleCount);
        Assert.Equal(22, predictor.SpatialResBlockCount);
        Assert.Equal(16, predictor.VideoTransformerCount);
        Assert.Equal(10, predictor.OnlyCrossAttentionTransformerCount);
        Assert.Equal(3, predictor.DownsampleCount);
        Assert.Equal(3, predictor.UpsampleCount);
        Assert.True(predictor.UsesTemporalTransformerAttention);
        Assert.Equal(105, predictor.TemporalTrainingLayers.Count);
        Assert.All(predictor.TemporalTrainingLayers, layer =>
            Assert.True(
                layer is TemporalModule3DLayer<double> ||
                layer is TemporalConv3DLayer<double> ||
                layer is GroupNormalizationLayer<double> ||
                layer is LayerNormalizationLayer<double> ||
                layer is DiffusionAttentionLayer<double>,
                $"Unexpected spatial layer {layer.GetType().Name} in temporal fine-tuning surface."));
        var vae = Assert.IsType<TemporalVAE<double>>(model.VAE);
        Assert.Equal(4, vae.DownsampleFactor);
        Assert.True(model.ParameterCount > 0);
    }

    [Fact]
    public void UpscaleAVideo_DoesNotSilentlyDisablePaperGuidance()
    {
        using var model = new UpscaleAVideoModel<double>();
        var input = new Tensor<double>([1, 1, 3, 4, 4]);

        var error = Assert.Throws<InvalidOperationException>(() =>
            model.Upscale(input, numInferenceSteps: 1, guidanceScale: 9.0));

        Assert.Contains("Stable Diffusion x4 Upscaler CLIP", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(8)]
    public void UpscaleAVideo_TemporalModuleIsFrameCountIndependentAndIdentityInitialized(int frames)
    {
        var module = new TemporalModule3DLayer<double>(channels: 4, timeEmbeddingDim: 8, spatialSize: 2);
        var input = Filled([1, 4, frames, 2, 2], 0.1, 0.001);
        var time = Filled([1, 8], -0.2, 0.03);

        var output = module.Forward(input, time);

        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.Equal(input.AsSpan().ToArray(), output.AsSpan().ToArray());
        Assert.Equal(5, module.FirstTemporalKernelDepth);
        Assert.Equal(3, module.SecondTemporalKernelDepth);
        Assert.False(module.UsesTransformerAttention);
        Assert.True(module.UsesZeroInitializedOutputProjection);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(8)]
    public void UpscaleAVideo_Transformer3DSupportsArbitraryFramesAndReleasedAttentionPaths(int frames)
    {
        var transformer = new VideoTransformer3DLayer<double>(
            channels: 8,
            contextDimension: 16,
            headCount: 2,
            spatialSize: 2,
            onlyCrossAttention: true);
        var input = Filled([1, 8, frames, 2, 2], 0.05, 0.0007);
        var context = Filled([1, 4, 16], -0.1, 0.002);

        var output = transformer.Forward(input, context);

        Assert.Equal(input.Shape.ToArray(), output.Shape.ToArray());
        Assert.All(output.AsSpan().ToArray(), value =>
            Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        Assert.True(transformer.OnlyCrossAttention);
        Assert.True(transformer.UsesTemporalAttention);
        Assert.True(transformer.TemporalAttentionIsZeroInitialized);
        Assert.True(transformer.ParameterCount > 0);
    }

    [Fact]
    public void UpscaleAVideo_CloneOwnsOptionsSchedulerAndParameterState()
    {
        var callerOptions = new DiffusionModelOptions<double>
        {
            LearningRate = 0.002,
            TrainTimesteps = 1000,
            BetaStart = 0.00085,
            BetaEnd = 0.012,
            BetaSchedule = BetaSchedule.ScaledLinear
        };
        using var model = SmallUpscaleAVideo(callerOptions);
        callerOptions.LearningRate = 99.0;

        using var clone = (UpscaleAVideoModel<double>)model.Clone();

        Assert.Equal(0.002, ((DiffusionModelOptions<double>)model.GetOptions()).LearningRate, 12);
        Assert.Equal(0.002, ((DiffusionModelOptions<double>)clone.GetOptions()).LearningRate, 12);
        Assert.NotSame(model.GetOptions(), clone.GetOptions());
        Assert.NotSame(model.Scheduler, clone.Scheduler);
        Assert.Equal(model.Scheduler.GetType(), clone.Scheduler.GetType());
        Assert.Equal(
            model.NoisePredictor.GetParameters().AsSpan().ToArray(),
            clone.NoisePredictor.GetParameters().AsSpan().ToArray());
        Assert.Equal(
            model.VAE.GetParameters().AsSpan().ToArray(),
            clone.VAE.GetParameters().AsSpan().ToArray());
        Assert.Equal(model.ParameterCount, clone.ParameterCount);
        Assert.Equal(model.GetParameters().AsSpan().ToArray(), clone.GetParameters().AsSpan().ToArray());
    }

    [Fact]
    public void ChannelPinnedConvolutions_MaterializeConsistentParameterSurfaces()
    {
        var conv3D = Conv3DLayer<double>.WithInputChannels(
            inputChannels: 3, outputChannels: 4, kernelSize: 3, padding: 1);
        var deconvolution = DeconvolutionalLayer<double>.WithInputDepth(
            inputDepth: 4, outputDepth: 2, kernelSize: 3, padding: 1);

        Assert.Equal(conv3D.ParameterCount, conv3D.GetParameters().Length);
        Assert.Equal(conv3D.ParameterCount,
            conv3D.GetTrainableParameters().Sum(tensor => tensor.Length));
        Assert.Equal(deconvolution.ParameterCount, deconvolution.GetParameters().Length);
        Assert.Equal(deconvolution.ParameterCount,
            deconvolution.GetTrainableParameters().Sum(tensor => tensor.Length));
    }

    [Fact]
    public void UpscaleAVideo_ParameterEnumerationDoesNotReplayInvalidUnconditionedProbe()
    {
        using var model = SmallUpscaleAVideo(new DiffusionModelOptions<double>());
        var predictor = Assert.IsType<VideoUNetPredictor<double>>(model.NoisePredictor);
        var noisyLatent = Filled([1, 4, 2, 2, 2], 0.01, 0.0001);
        var lowResolutionCondition = Filled([1, 3, 2, 2, 2], -0.02, 0.0002);
        var textCondition = Filled([1, 1, 4], 0.03, 0.001);

        var prediction = predictor.PredictNoiseWithVideoCondition(
            noisyLatent,
            timestep: 5,
            lowResolutionCondition,
            textCondition,
            noiseLevel: 3);
        var parameters = predictor.GetParameters();

        Assert.Equal(noisyLatent.Shape.ToArray(), prediction.Shape.ToArray());
        Assert.NotEmpty(parameters.AsSpan().ToArray());
        Assert.Equal(parameters.Length, predictor.ParameterCount);
    }

    [Fact]
    public void UpscaleAVideo_FlowPropagationWarpUsesEngineNearestSampling()
    {
        using var model = SmallUpscaleAVideo(new DiffusionModelOptions<double>());
        var input = new Tensor<double>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 2, 3]);
        var flow = new Tensor<double>(new double[12], [1, 2, 2, 3]);
        var method = typeof(UpscaleAVideoModel<double>).GetMethod(
            "WarpNearest", BindingFlags.Instance | BindingFlags.NonPublic);

        var output = Assert.IsType<Tensor<double>>(method!.Invoke(model, [input, flow]));

        Assert.Equal(input.AsSpan().ToArray(), output.AsSpan().ToArray());
    }

    [Fact]
    public void StableVideoSrOptions_EnforceNativePaperConstantsAndCopyCollections()
    {
        var options = new StableVideoSROptions { PropagationSteps = [1, 4, 7] };
        options.ValidateNativePaperContract();
        var copy = new StableVideoSROptions(options);
        options.PropagationSteps[0] = 9;

        Assert.Equal([1, 4, 7], copy.PropagationSteps);
        Assert.Equal(256, copy.NumFeatures);
        Assert.Equal(75, copy.NumDenoisingSteps);
        Assert.Equal(9.0, copy.GuidanceScale, 12);
        Assert.Equal(20, copy.NoiseLevel);
        Assert.Equal(8, copy.TemporalWindowSize);
        Assert.Equal(2, copy.TemporalWindowOverlap);
        Assert.Equal(9, copy.NumTemporalModules);
        Assert.Equal("best quality, extremely detailed", copy.Prompt);
        Assert.Equal("blur, worst quality", copy.NegativePrompt);
        Assert.Equal(0.08333, copy.LatentScaleFactor, 12);
        copy.NumFeatures = 320;
        Assert.Throws<ArgumentOutOfRangeException>(copy.ValidateNativePaperContract);
    }

    private static NeuralNetworkArchitecture<double> VideoArchitecture(int height, int width) =>
        new(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: height,
            inputWidth: width,
            inputDepth: 6,
            outputSize: 3);

    private static UpscaleAVideoModel<double> SmallUpscaleAVideo(
        DiffusionModelOptions<double> options)
    {
        var predictor = new VideoUNetPredictor<double>(
            inputChannels: 4,
            outputChannels: 4,
            baseChannels: 8,
            channelMultipliers: [1],
            numResBlocks: 1,
            attentionResolutions: [],
            numTemporalLayers: 1,
            contextDim: 4,
            numHeads: 1,
            inputHeight: 2,
            inputWidth: 2,
            numFrames: 2,
            imageConditionChannels: 3,
            concatenateImageCondition: true,
            numClassEmbeddings: 10,
            seed: 42);
        var vae = new TemporalVAE<double>(
            inputChannels: 3,
            latentChannels: 4,
            baseChannels: 8,
            channelMultipliers: [1],
            numTemporalLayers: 1,
            latentScaleFactor: 0.08333,
            seed: 42);
        return new UpscaleAVideoModel<double>(
            options: options,
            videoUNet: predictor,
            temporalVAE: vae,
            seed: 42);
    }

    private static Tensor<double> Filled(int[] shape, double start, double step)
    {
        int length = shape.Aggregate(1, (product, dimension) => product * dimension);
        var data = new double[length];
        for (int i = 0; i < data.Length; i++) data[i] = start + i * step;
        return new Tensor<double>(data, shape);
    }
}
