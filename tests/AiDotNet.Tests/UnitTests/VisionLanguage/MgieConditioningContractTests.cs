using System.Reflection;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.VisionLanguage.Editing;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class MgieConditioningContractTests
{
    public MgieConditioningContractTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void SourceVae_ModeIsRepeatable_AndScalingIsSeparate()
    {
        using var vae = CreateVae();
        var image = CreateImage();
        var first = vae.Encode(image, sampleMode: false);
        var second = vae.Encode(image, sampleMode: false);
        Assert.Equal(new[] { 1, 4, 4, 4 }, first.Shape.ToArray());
        Assert.Equal(first.ToArray(), second.ToArray());
        var scaled = vae.ScaleLatent(first);
        Assert.Contains(first.ToArray(), value => Math.Abs(value) > 1e-5f);
        for (int i = 0; i < first.Length; i++)
            Assert.InRange(Math.Abs(scaled[i] - first[i] * 0.18215f), 0, 1e-5f);
    }

    [Fact]
    public void RealUnet_EightInputChannelsAndContext_ReturnsFourChannelNoise()
    {
        using var unet = new RecordingUnet();
        var input = new Tensor<float>(new[] { 1, 8, 4, 4 });
        var context = CreateContext();
        var prediction = unet.PredictNoise(input, 10, context);
        Assert.Equal(new[] { 1, 4, 4, 4 }, prediction.Shape.ToArray());
        Assert.All(prediction.ToArray(), value => Assert.False(float.IsNaN(value) || float.IsInfinity(value)));
        Assert.Single(unet.Inputs);
    }

    [Fact]
    public void EditImage_ActualTinyComponents_ReturnsDocumentedImageLayout()
    {
        using var unet = new RecordingUnet();
        using var vae = CreateVae();
        using var model = CreateModel(unet, vae);
        var output = model.EditImage(CreateImage(), "make the image brighter");
        Assert.Equal(new[] { 1, 3, 4, 4 }, output.Shape.ToArray());
        Assert.NotEmpty(unet.Inputs);
    }

    [Fact]
    public void Denoise_ActualUnet_ReceivesUnscaledSourceAndContextAtEveryStep()
    {
        using var unet = new RecordingUnet();
        using var vae = CreateVae();
        using var model = CreateModel(unet, vae);
        var source = vae.Encode(CreateImage(), sampleMode: false);
        var context = CreateContext();
        var denoise = typeof(MGIE<float>).GetMethod("Denoise", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(denoise);
        var nullContext = new Tensor<float>(context.Shape.ToArray());
        var result = denoise.Invoke(model, new object?[] { source, context, nullContext, 397525, null });
        Assert.IsType<Tensor<float>>(result);
        Assert.NotEmpty(unet.Inputs);
        for (int step = 0; step < unet.Inputs.Count; step++)
        {
            Assert.Equal(new[] { 3, 8, 4, 4 }, unet.Shapes[step]);
            Assert.Equal(source.ToArray(), unet.Inputs[step].Skip(source.Length).Take(source.Length).ToArray());
            Assert.Equal(source.ToArray(), unet.Inputs[step].Skip(3 * source.Length).Take(source.Length).ToArray());
            Assert.All(unet.Inputs[step].Skip(5 * source.Length), value => Assert.Equal(0, value));
            var observed = unet.Contexts[step];
            Assert.NotNull(observed);
            Assert.Equal(context.ToArray(), observed.Take(context.Length).ToArray());
            Assert.All(observed.Skip(context.Length), value => Assert.Equal(0, value));
        }
    }

    internal static MGIE<float> CreateModel(RecordingUnet unet, StandardVAE<float> vae) => new(
        options: CreateOptions(),
        unet: unet, vae: vae, seed: 397525);

    internal static MGIEOptions CreateOptions() => new()
        {
            ImageSize = 4,
            VisionPatchSize = 2,
            OutputImageSize = 4,
            VisionDim = 8,
            DecoderDim = 8,
            NumVisionLayers = 1,
            NumDecoderLayers = 1,
            NumHeads = 2,
            VocabSize = 64,
            MaxSequenceLength = 64,
            EditHeadLayers = 1,
            EditHiddenDim = 8,
            EditNumHeads = 2,
            EditTokenCount = 2,
            EditQueryCount = 3,
            MaxGenerationLength = 2,
            EnableExpressiveInstructions = false,
            NumDiffusionSteps = 2,
            DropoutRate = 0
        };

    internal static StandardVAE<float> CreateVae() => new(inputChannels: 3, latentChannels: 4,
        baseChannels: 32, channelMultipliers: new[] { 1 }, numResBlocksPerLevel: 1,
        latentScaleFactor: 0.18215, seed: 397525);

    internal static Tensor<float> CreateImage()
    {
        var image = new Tensor<float>(new[] { 1, 3, 4, 4 });
        for (int i = 0; i < image.Length; i++) image[i] = (i - 19) * 0.017f;
        return image;
    }

    internal static Tensor<float> CreateContext()
    {
        var context = new Tensor<float>(new[] { 1, 3, 768 });
        for (int i = 0; i < context.Length; i++) context[i] = (i % 17 - 8) * 0.03f;
        return context;
    }

    internal sealed class RecordingUnet : UNetNoisePredictor<float>
    {
        internal readonly List<float[]> Inputs = new();
        internal readonly List<int[]> Shapes = new();
        internal readonly List<float[]?> Contexts = new();
        internal readonly List<int> Timesteps = new();
        internal readonly List<float[]> Predictions = new();

        internal RecordingUnet() : base(inputChannels: 8, outputChannels: 4,
            baseChannels: 32, channelMultipliers: new[] { 1 }, numResBlocks: 1,
            attentionResolutions: new[] { 1 }, contextDim: 768, numHeads: 2,
            inputHeight: 4, seed: 397525) { }

        public override Tensor<float> PredictNoise(Tensor<float> noisySample, int timestep,
            Tensor<float>? conditioning = null)
        {
            Shapes.Add(noisySample.Shape.ToArray());
            Inputs.Add(noisySample.ToArray());
            Contexts.Add(conditioning?.ToArray());
            Timesteps.Add(timestep);
            var prediction = base.PredictNoise(noisySample, timestep, conditioning);
            Predictions.Add(prediction.ToArray());
            return prediction;
        }
    }
}
