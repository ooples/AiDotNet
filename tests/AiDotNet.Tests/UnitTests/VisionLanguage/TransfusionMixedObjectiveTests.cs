using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Unified;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Transfusion trains on its own objective, <c>L = L_LM + lambda * L_DDPM</c> (Zhou et al. 2024,
/// arXiv:2408.11039, Eq. 4) -- not on a generic input-to-target regression (#2311) -- and samples
/// images back through the heads that objective moves.
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

    private static TransfusionOptions Options(double lambda = 5.0, int visualTokens = 16) =>
        new()
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
            NumVisualTokens = visualTokens,
            DropoutRate = 0.0,
            DiffusionLossWeight = lambda,
            DiffusionTimesteps = 1000,
            DiffusionInferenceSteps = 3,
            Seed = 20261011,
        };

    private static Transfusion<float> CreateModel(TransfusionOptions? options = null)
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: ImageSide,
            inputWidth: ImageSide,
            inputDepth: 3,
            outputSize: 64);

        return new Transfusion<float>(architecture, options ?? Options());
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
            return target.EvaluateMixedObjective(image, caption, timestep: 400);
        }
    }

    [Fact]
    public void TheCaptionChangesTheObjective()
    {
        using var model = CreateModel();
        var image = Image(0.6);

        double first = model.EvaluateMixedObjective(image, Caption(4, 9, 13, 21, 2), 400);
        double second = model.EvaluateMixedObjective(image, Caption(31, 5, 60, 7, 44), 400);

        Assert.NotEqual(first, second, 6);
    }

    [Fact]
    public void TheImageChangesTheObjective()
    {
        using var model = CreateModel();
        var caption = Caption(4, 9, 13, 21, 2);

        double first = model.EvaluateMixedObjective(Image(0.2), caption, 400);
        double second = model.EvaluateMixedObjective(Image(0.9), caption, 400);

        Assert.NotEqual(first, second, 6);
    }

    [Fact]
    public void ATargetThatCarriesNoTokensStillScoresTheImageAndItsEndMarker()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var imageShapedTarget = new Tensor<float>([1, 64]);

        double withText = model.EvaluateMixedObjective(image, Caption(4, 9, 13, 21, 2), 400);
        double withoutText = model.EvaluateMixedObjective(image, imageShapedTarget, 400);
        ((TransfusionOptions)model.GetOptions()).DiffusionLossWeight = 0.0;
        double markerOnly = model.EvaluateMixedObjective(image, imageShapedTarget, 400);

        // With no caption the sequence is [BOI, patches, EOI]: the diffusion term is unchanged and
        // the LM term keeps exactly one token to score, the EOI the last patch row predicts --
        // "LM loss is computed per token" with only a BOI input exempted (Sec. 3, footnote 6).
        Assert.True(withoutText > 0.0, $"The diffusion term must be positive; got {withoutText}.");
        Assert.True(markerOnly > 0.0,
            $"The EOI marker is a token and must be scored; got {markerOnly}.");
        // The two objectives differ: one scores six targets (five caption tokens and BOI, then
        // EOI), the other scores EOI alone. Which MEAN cross-entropy is larger is not fixed, so
        // the assertion is inequality rather than an ordering.
        Assert.NotEqual(withText, withoutText, 6);
    }

    [Fact]
    public void TrainingDrivesTheObjectiveDown()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var caption = Caption(4, 9, 13, 21, 2);

        double before = model.EvaluateMixedObjective(image, caption, 400);
        for (int step = 0; step < 40; step++) model.Train(image, caption);
        double after = model.EvaluateMixedObjective(image, caption, 400);

        Assert.True(after < before,
            $"40 steps on one pair must lower the objective it minimizes: {before} -> {after}.");
    }

    [Fact]
    public void TrainingUpdatesBothMixedModalHeads()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var caption = Caption(4, 9, 13, 21, 2);

        // One step builds the lazy heads; the snapshot after it is the comparison point, so the
        // difference measured below is a weight UPDATE and not lazy materialization.
        model.Train(image, caption);
        var (languageBefore, noiseBefore) = model.MixedModalHeadParametersForTest();
        var wholeModelBefore = model.GetParameters();
        model.Train(image, caption);
        var (languageAfter, noiseAfter) = model.MixedModalHeadParametersForTest();
        var wholeModelAfter = model.GetParameters();

        // Each head on its own: the whole-model vector is dominated by the shared trunk, so a
        // move there would pass while a head sat untrained -- which is the failure this guards.
        Assert.True(Moved(languageBefore, languageAfter),
            "The cross-entropy term must update the language head.");
        Assert.True(Moved(noiseBefore, noiseAfter),
            "The diffusion term must update the noise head.");
        Assert.True(Moved(wholeModelBefore, wholeModelAfter),
            "The step must also reach the shared trunk.");

        static bool Moved(Vector<float> before, Vector<float> after)
        {
            Assert.Equal(before.Length, after.Length);
            for (int i = 0; i < before.Length; i++)
                if (Math.Abs(before[i] - after[i]) > 1e-9f) return true;
            return false;
        }
    }
    [Fact]
    public void ThePatchRowsDoNotAttendToTheEndOfImageMarker()
    {
        // The paper's bidirectional attention is bounded "within ... each individual image", so a
        // patch row must not see the EOI row that FOLLOWS its image. The diffusion term is read
        // from that pass and is recoverable on its own as the slope in lambda -- so if the EOI row
        // were visible to the patch rows, changing which token EOI is would change that slope. (It
        // does change the LM term, which taking the slope cancels out.) One model throughout: two
        // instances would differ by their initialization, not by the marker.
        var options = Options();
        using var model = CreateModel(options);
        var image = Image(0.6);
        var caption = Caption(4, 9, 13, 21, 2);

        options.EndOfImageTokenId = 60;
        double firstMarker = DiffusionTerm();
        options.EndOfImageTokenId = 7;
        double secondMarker = DiffusionTerm();

        // Each slope is a difference of two float32 objectives over 5, and the LM term it cancels
        // differs between the markers, so the two slopes carry a few ulps of rounding (1.9e-7 seen
        // on CI). Compare within a tolerance: digit rounding fails when the two values straddle a
        // rounding boundary, which 1.2950645 vs 1.2950644 did.
        Assert.Equal(firstMarker, secondMarker, 1e-5);
        Assert.True(firstMarker > 0.0, $"The diffusion term must be positive; got {firstMarker}.");

        double DiffusionTerm()
        {
            options.DiffusionLossWeight = 0.0;
            double atZero = model.EvaluateMixedObjective(image, caption, 400);
            options.DiffusionLossWeight = 5.0;
            double atFive = model.EvaluateMixedObjective(image, caption, 400);
            return (atFive - atZero) / 5.0;
        }
    }
    [Fact]
    public void TheBorderPixelsReachThePatchesWhenThePatchSideDoesNotDivideTheImage()
    {
        // 9 visual tokens over a 32-pixel image gives a 3x3 grid of 10-pixel patches, which
        // covers 30 of 32 pixels: a floored grid dropped the last two rows and columns, so a
        // change confined to them was invisible to the objective.
        using var model = CreateModel(Options(visualTokens: 9));
        var caption = Caption(4, 9, 13, 21, 2);
        var plain = Image(0.6);
        var withBorder = Image(0.6);
        for (int channel = 0; channel < 3; channel++)
        {
            for (int i = 30; i < ImageSide; i++)
            {
                for (int j = 0; j < ImageSide; j++)
                {
                    withBorder[0, channel, i, j] = 0.95f;
                    withBorder[0, channel, j, i] = 0.95f;
                }
            }
        }

        double untouched = model.EvaluateMixedObjective(plain, caption, 400);
        double bordered = model.EvaluateMixedObjective(withBorder, caption, 400);

        Assert.NotEqual(untouched, bordered, 6);
    }

    [Fact]
    public void AnImageOfTheWrongSizeIsRejected()
    {
        using var model = CreateModel();
        var caption = Caption(4, 9, 13, 21, 2);
        var wrongSize = new Tensor<float>([1, 3, ImageSide, ImageSide / 2]);

        // The objective's patch grid follows the image it is given, while the sampler and the
        // unpatchifier follow the configured square grid: a mismatch would train one patch
        // sequence and generate from another, silently.
        var exception = Assert.Throws<ArgumentException>(
            () => model.Train(wrongSize, caption));
        Assert.Contains($"{ImageSide}x{ImageSide}", exception.Message);
        Assert.Contains($"{ImageSide}x{ImageSide / 2}", exception.Message);
    }

    [Fact]
    public void GenerateImageReadsTheTrainedNoiseHead()
    {
        using var model = CreateModel();
        var image = Image(0.6);
        var caption = Caption(4, 9, 13, 21, 2);

        var before = model.GenerateImage("a small bright square");
        var repeated = model.GenerateImage("a small bright square");
        for (int step = 0; step < 20; step++) model.Train(image, caption);
        var after = model.GenerateImage("a small bright square");

        // The sampler starts from noise seeded by the model's own Seed, so two draws for one
        // prompt are identical and the ONLY thing that can move the pixels is the weights. A
        // sampler that did not read the trained heads would return `before` again (#2311).
        Assert.Equal(ImageSide * ImageSide * 3, before.Length);
        Assert.Equal(ImageSide * ImageSide * 3, after.Length);
        AssertSameImage(before, repeated);
        Assert.True(MaximumDifference(before, after) > 1e-6f,
            "20 training steps must change what the sampler draws for the same prompt.");
        for (int i = 0; i < after.Length; i++)
        {
            Assert.InRange(after[i], 0.0f, 1.0f);
        }
    }

    [Fact]
    public void GenerateImageIsConditionedOnThePrompt()
    {
        using var model = CreateModel();

        var square = model.GenerateImage("a small bright square");
        var mountain = model.GenerateImage("storm clouds over a mountain");

        // Same seed, same noise, same weights: only the caption rows differ, so a difference here
        // is the text conditioning reaching the denoiser.
        Assert.True(MaximumDifference(square, mountain) > 1e-6f,
            "The prompt must change the sampled image.");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void ATimestepCountBelowOneIsRejected(int timesteps)
    {
        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new TransfusionOptions { DiffusionTimesteps = timesteps });
        Assert.Equal(nameof(TransfusionOptions.DiffusionTimesteps), exception.ParamName);
    }

    [Theory]
    [InlineData(-0.5)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void ADiffusionWeightThatIsNegativeOrNonFiniteIsRejected(double lambda)
    {
        // A negative weight would reward noise-prediction error and a non-finite one makes the
        // whole objective non-finite, and both can arrive through GetOptions() after the model
        // is built -- so the setter is where this has to be caught.
        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new TransfusionOptions { DiffusionLossWeight = lambda });
        Assert.Equal(nameof(TransfusionOptions.DiffusionLossWeight), exception.ParamName);
    }

    [Fact]
    public void AnInferenceStepCountBelowOneIsRejected()
    {
        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new TransfusionOptions { DiffusionInferenceSteps = 0 });
        Assert.Equal(nameof(TransfusionOptions.DiffusionInferenceSteps), exception.ParamName);
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

    private static void AssertSameImage(Tensor<float> first, Tensor<float> second)
    {
        Assert.Equal(first.Length, second.Length);
        Assert.True(MaximumDifference(first, second) <= 1e-6f,
            "Two draws for one prompt on one seed must be identical.");
    }

    private static float MaximumDifference(Tensor<float> first, Tensor<float> second)
    {
        float largest = 0.0f;
        for (int i = 0; i < first.Length && i < second.Length; i++)
            largest = Math.Max(largest, Math.Abs(first[i] - second[i]));
        return largest;
    }
}