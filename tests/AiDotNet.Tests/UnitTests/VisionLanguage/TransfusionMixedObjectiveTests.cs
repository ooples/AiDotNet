using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Unified;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Transfusion trains on its own objective, <c>L = L_LM + lambda * L_DDPM</c> (Zhou et al. 2024,
/// arXiv:2408.11039, Eq. 4) -- not on a generic input-to-target regression (#2311).
/// </summary>
/// <remarks>
/// The decisive check is <see cref="TheObjectiveIsTheLanguageTermPlusLambdaTimesTheDiffusionTerm"/>:
/// the objective is affine in <c>lambda</c> with the same two terms at every weight, which is Eq. 4
/// and cannot hold for a regression against <c>expected</c>.
/// </remarks>
public class TransfusionMixedObjectiveTests
{
    private const int ImageSide = 32;
    private const int Vocabulary = 64;

    private static Transfusion<float> CreateModel(double lambda = 5.0)
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: ImageSide,
            inputWidth: ImageSide,
            inputDepth: 3,
            outputSize: 64);

        return new Transfusion<float>(architecture, new TransfusionOptions
        {
            ImageSize = ImageSide,
            OutputImageSize = ImageSide,
            VisionDim = 32,
            DecoderDim = 32,
            NumVisionLayers = 0,
            NumDecoderLayers = 2,
            NumHeads = 4,
            VocabSize = Vocabulary,
            MaxSequenceLength = 16,
            MaxGenerationLength = 8,
            NumVisualTokens = 16,
            DropoutRate = 0.0,
            DiffusionLossWeight = lambda,
            DiffusionTimesteps = 1000,
            Seed = 20261011,
        });
    }

    private static Tensor<float> Image(double scale)
    {
        var image = new Tensor<float>([1, 3, ImageSide, ImageSide]);
        var span = image.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(scale * Math.Sin(i * 0.031));
        return image;
    }

    private static Tensor<float> Caption(params int[] ids)
    {
        var caption = new Tensor<float>([ids.Length]);
        for (int i = 0; i < ids.Length; i++) caption[i] = ids[i];
        return caption;
    }

    [Fact]
    public void TheObjectiveIsTheLanguageTermPlusLambdaTimesTheDiffusionTerm()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var caption = Caption(4, 9, 13, 21, 2);

        double atZero = Measure(model, 0.0);
        double atFive = Measure(model, 5.0);
        double atTen = Measure(model, 10.0);

        // L(lambda) = L_LM + lambda * L_DDPM is affine in lambda with a positive slope, so equal
        // steps in lambda give equal steps in L, and L_DDPM is recoverable from either pair.
        double diffusionFromLowPair = (atFive - atZero) / 5.0;
        double diffusionFromHighPair = (atTen - atFive) / 5.0;
        Assert.Equal(diffusionFromLowPair, diffusionFromHighPair, 4);
        Assert.True(diffusionFromLowPair > 0.0,
            $"The diffusion term must contribute a positive MSE; got {diffusionFromLowPair}.");
        // The language term is what is left at lambda = 0, and cross-entropy over a 64-token
        // vocabulary starts near ln(64) = 4.16 for an untrained head.
        Assert.True(atZero > 0.0, $"The language term must be positive; got {atZero}.");

        double Measure(Transfusion<float> target, double lambda)
        {
            ((TransfusionOptions)target.GetOptions()).DiffusionLossWeight = lambda;
            return target.MixedObjectiveValueForTest(image, caption, timestep: 400);
        }
    }

    [Fact]
    public void TheCaptionChangesTheObjective()
    {
        using var model = CreateModel();
        var image = Image(0.6);

        double first = model.MixedObjectiveValueForTest(image, Caption(4, 9, 13, 21, 2), 400);
        double second = model.MixedObjectiveValueForTest(image, Caption(31, 5, 60, 7, 44), 400);

        Assert.NotEqual(first, second, 6);
    }

    [Fact]
    public void TheImageChangesTheObjective()
    {
        using var model = CreateModel();
        var caption = Caption(4, 9, 13, 21, 2);

        double first = model.MixedObjectiveValueForTest(Image(0.2), caption, 400);
        double second = model.MixedObjectiveValueForTest(Image(0.9), caption, 400);

        Assert.NotEqual(first, second, 6);
    }

    [Fact]
    public void ATargetThatCarriesNoTokensLeavesTheDiffusionTermAlone()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var imageShapedTarget = new Tensor<float>([1, 64]);

        double withoutText = model.MixedObjectiveValueForTest(image, imageShapedTarget, 400);
        ((TransfusionOptions)model.GetOptions()).DiffusionLossWeight = 0.0;
        double languageOnly = model.MixedObjectiveValueForTest(image, imageShapedTarget, 400);

        // No text rows means no cross-entropy terms, so the whole objective is lambda * L_DDPM:
        // at lambda = 0 nothing is left.
        Assert.True(withoutText > 0.0, $"The diffusion term must be positive; got {withoutText}.");
        Assert.Equal(0.0, languageOnly, 6);
    }

    [Fact]
    public void TrainingDrivesTheObjectiveDown()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var caption = Caption(4, 9, 13, 21, 2);

        double before = model.MixedObjectiveValueForTest(image, caption, 400);
        for (int step = 0; step < 40; step++) model.Train(image, caption);
        double after = model.MixedObjectiveValueForTest(image, caption, 400);

        Assert.True(after < before,
            $"40 steps on one pair must lower the objective it minimizes: {before} -> {after}.");
    }

    [Fact]
    public void TrainingUpdatesTheMixedModalHeads()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var caption = Caption(4, 9, 13, 21, 2);

        // One step builds the lazy heads; the snapshot after it is the comparison point, so the
        // difference measured below is a weight UPDATE and not lazy materialization.
        model.Train(image, caption);
        var before = model.GetParameters();
        model.Train(image, caption);
        var after = model.GetParameters();

        Assert.Equal(before.Length, after.Length);
        int changed = 0;
        for (int i = 0; i < before.Length; i++)
            if (Math.Abs(before[i] - after[i]) > 1e-9f) changed++;
        Assert.True(changed > 0, "A training step must move at least one parameter.");
    }

    [Fact]
    public void TheNoiseScheduleIsTheCosineScheduleThePaperAdopts()
    {
        // Nichol and Dhariwal (2021): alphaBar_t = f(t)/f(0), f(t) = cos^2(((t/T)+s)/(1+s) * pi/2),
        // s = 0.008 -- so alphaBar starts at 1, falls monotonically, and is near 1/2 at t = T/2.
        Assert.Equal(1.0, Transfusion<float>.CosineAlphaBar(0, 1000), 6);
        // Closed form at the midpoint: cos^2((0.508/1.008) * pi/2) / cos^2((0.008/1.008) * pi/2)
        // = 0.49384, so the band is the schedule's own value and not a rounded 1/2.
        Assert.Equal(0.49384, Transfusion<float>.CosineAlphaBar(500, 1000), 5);
        Assert.True(Transfusion<float>.CosineAlphaBar(900, 1000)
            < Transfusion<float>.CosineAlphaBar(100, 1000));
    }
}
