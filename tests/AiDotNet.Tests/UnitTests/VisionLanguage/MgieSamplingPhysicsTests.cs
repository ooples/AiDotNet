using System.Reflection;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Interfaces;
using AiDotNet.VisionLanguage.Editing;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class MgieSamplingPhysicsTests
{
    public MgieSamplingPhysicsTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void Euler_InputCoordinatesMatchSigmaDefinition_WithoutMutatingSample()
    {
        var scheduler = new EulerDiscreteScheduler<float>(Config());
        var input = JointVisionLanguageStateTests.Values(new[] { 1, 4, 2, 2 });
        var before = input.ToArray();
        Assert.Throws<InvalidOperationException>(() => scheduler.InitialNoiseSigma);
        Assert.Throws<InvalidOperationException>(() => scheduler.ScaleModelInput(input, 0));
        scheduler.SetTimesteps(3);
        double firstSigma = Sigma(scheduler, scheduler.Timesteps[0]);
        Assert.InRange(Math.Abs(scheduler.InitialNoiseSigma - firstSigma), 0, 1e-6);
        foreach (int timestep in scheduler.Timesteps)
        {
            var scaled = scheduler.ScaleModelInput(input, timestep);
            double denominator = Math.Sqrt(Math.Pow(Sigma(scheduler, timestep), 2) + 1);
            for (int i = 0; i < before.Length; i++)
                Assert.InRange(Math.Abs(scaled[i] - before[i] / denominator), 0, 1e-6);
        }
        Assert.Equal(before, input.ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => scheduler.ScaleModelInput(input, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => scheduler.ScaleModelInput(input, scheduler.TrainTimesteps));
    }

    [Fact]
    public void ActualUnet_AllStepsMatchThreeBranchGuidanceAndEulerUpdate()
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        var options = MgieConditioningContractTests.CreateOptions();
        options.GuidanceScale = 2.25;
        options.ImageGuidanceScale = 1.5;
        var scheduler = new EulerDiscreteScheduler<float>(Config());
        using var model = new MGIE<float>(options: options, scheduler: scheduler, unet: unet, vae: vae, seed: 397525);
        Assert.Equal(options.GuidanceScale, model.GuidanceScale);
        var source = vae.Encode(MgieConditioningContractTests.CreateImage(), sampleMode: false);
        var guidance = MgieConditioningContractTests.CreateContext();
        var nullContext = JointVisionLanguageStateTests.Values(guidance.Shape.ToArray());
        var noise = StandardNoise(source.Length);
        var beforeNoise = noise.ToArray();
        var beforeSource = source.ToArray();
        var beforeGuidance = guidance.ToArray();
        var output = Denoise(model, source, guidance, nullContext, noise);
        Assert.Equal(options.NumDiffusionSteps + 1, unet.Inputs.Count); // one outside-arena warmup
        Assert.Equal(beforeNoise, noise.ToArray());
        Assert.Equal(beforeSource, source.ToArray());
        Assert.Equal(beforeGuidance, guidance.ToArray());
        int count = source.Length;
        double sigma = Sigma(scheduler, scheduler.Timesteps[0]);
        double[] sample = beforeNoise.Select(value => value * sigma).ToArray();

        for (int index = 1; index < unet.Inputs.Count; index++)
        {
            int timestep = scheduler.Timesteps[index - 1];
            Assert.Equal(timestep, unet.Timesteps[index]);
            sigma = Sigma(scheduler, timestep);
            double denominator = Math.Sqrt(sigma * sigma + 1);
            var input = unet.Inputs[index];
            var predictions = unet.Predictions[index];
            Assert.Equal(new[] { 3, 8, 4, 4 }, unet.Shapes[index]);
            for (int branch = 0; branch < 3; branch++)
            {
                for (int i = 0; i < count; i++)
                {
                    Assert.InRange(Math.Abs(input[branch * 2 * count + i] - sample[i] / denominator), 0, 2e-5);
                    Assert.Equal(branch < 2 ? beforeSource[i] : 0, input[branch * 2 * count + count + i]);
                }
            }
            var actualContext = unet.Contexts[index];
            Assert.NotNull(actualContext);
            Assert.Equal(beforeGuidance, actualContext.Take(guidance.Length).ToArray());
            Assert.Equal(nullContext.ToArray(), actualContext.Skip(guidance.Length).Take(guidance.Length).ToArray());
            Assert.Equal(nullContext.ToArray(), actualContext.Skip(2 * guidance.Length).ToArray());
            double nextSigma = index == options.NumDiffusionSteps ? 0 : Sigma(scheduler, scheduler.Timesteps[index]);
            for (int i = 0; i < count; i++)
            {
                double textImage = predictions[i];
                double imageOnly = predictions[count + i];
                double unconditional = predictions[2 * count + i];
                double guided = unconditional + options.GuidanceScale * (textImage - imageOnly) +
                    options.ImageGuidanceScale * (imageOnly - unconditional);
                sample[i] += guided * (nextSigma - sigma);
            }
        }
        for (int i = 0; i < count; i++) Assert.InRange(Math.Abs(output[i] - sample[i]), 0, 3e-5);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CallerNoiseAndSourceRemainUnchanged_ForSchedulersWithAndWithoutInputScaling(bool euler)
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        INoiseScheduler<float> scheduler = euler ? new EulerDiscreteScheduler<float>(Config()) : new DDIMScheduler<float>(Config());
        using var model = new MGIE<float>(options: MgieConditioningContractTests.CreateOptions(), scheduler: scheduler,
            unet: unet, vae: vae, seed: 397525);
        var source = vae.Encode(MgieConditioningContractTests.CreateImage(), sampleMode: false);
        var context = MgieConditioningContractTests.CreateContext();
        var nullContext = new Tensor<float>(context.Shape.ToArray());
        var noise = StandardNoise(source.Length);
        var before = noise.ToArray();
        var first = Denoise(model, source, context, nullContext, noise);
        var repeated = Denoise(model, source, context, nullContext, noise);
        Assert.Equal(before, noise.ToArray());
        Assert.Equal(first.ToArray(), repeated.ToArray());
    }

    [Fact]
    public void FixedNoise_ActualDenoisingRespondsToBothSourceAndContext()
    {
        using var unet = new MgieConditioningContractTests.RecordingUnet();
        using var vae = MgieConditioningContractTests.CreateVae();
        using var model = MgieConditioningContractTests.CreateModel(unet, vae);
        var source = vae.Encode(MgieConditioningContractTests.CreateImage(), sampleMode: false);
        var context = MgieConditioningContractTests.CreateContext();
        var nullContext = new Tensor<float>(context.Shape.ToArray());
        var noise = StandardNoise(source.Length);
        var first = Denoise(model, source, context, nullContext, noise);
        var changedSource = source.Clone();
        changedSource[3] += 0.7f;
        var sourceChanged = Denoise(model, changedSource, context, nullContext, noise);
        var changedContext = context.Clone();
        changedContext[9] += 0.9f;
        var contextChanged = Denoise(model, source, changedContext, nullContext, noise);
        AssertDifferent(first, sourceChanged);
        AssertDifferent(first, contextChanged);
    }

    internal static Tensor<float> Denoise(MGIE<float> model, Tensor<float> source, Tensor<float> context,
        Tensor<float> nullContext, Vector<float> noise)
    {
        var method = typeof(MGIE<float>).GetMethod("Denoise", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        return Assert.IsType<Tensor<float>>(method.Invoke(model, new object?[] { source, context, nullContext, 397525, noise }));
    }

    internal static void AssertDifferent(Tensor<float> first, Tensor<float> second)
    {
        Assert.Equal(first.Shape.ToArray(), second.Shape.ToArray());
        double max = first.ToArray().Zip(second.ToArray(), (left, right) => Math.Abs((double)left - right)).Max();
        Assert.True(max > 1e-6, $"The real model did not respond to changed conditioning: maximum difference {max}.");
    }

    internal static Vector<float> StandardNoise(int length)
        => new(Enumerable.Range(0, length).Select(index => (float)Math.Sin(index * 0.73)).ToArray());

    private static SchedulerConfig<float> Config() => new(20, 0.001f, 0.01f, clipSample: false);
    private static double Sigma(INoiseScheduler<float> scheduler, int timestep)
    {
        double alpha = scheduler.GetAlphaCumulativeProduct(timestep);
        return Math.Sqrt((1 - alpha) / alpha);
    }
}
