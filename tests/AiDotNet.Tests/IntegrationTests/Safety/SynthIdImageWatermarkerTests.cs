using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// The SynthID-style watermarker trained once at smoke scale (32 x 32 images, 16 bits, 600 steps) on synthetic smooth images
/// under StegaStamp's distortions. The tests then check every keyed round-trip contract, and the calibrated
/// three-state decision on held-out images: its false positive rate on clean images, its detection rate on
/// watermarked ones, and how often a residual lifted onto another image is called watermarked.
/// </summary>
[Trait("Category", "HeavyTimeout")]
public class SynthIdImageWatermarkerTests : WatermarkRoundTripTestBase<Tensor<float>>, IClassFixture<SynthIdImageWatermarkerTests.TrainedModel>
{
    private const int Size = StegaStampWatermarkerTests.Size;
    private const int Bits = StegaStampWatermarkerTests.Bits;

    /// <summary>Trains and calibrates the networks once for the class.</summary>
    public sealed class TrainedModel
    {
        internal const int Steps = StegaStampWatermarkerTests.TrainedModel.Steps;
        internal const int Batch = StegaStampWatermarkerTests.TrainedModel.Batch;
        internal const int CalibrationImages = 200;

        public SynthIdImageWatermarker<float> Model { get; }

        public TrainedModel()
        {
            var options = new SynthIdImageOptions
            {
                ImageSize = Size, SecretBits = Bits, Seed = 1, FalsePositiveRate = 0.02, SpecificityLossScale = 1.0, SecretLossScale = 3.0,
            };
            StegaStampWatermarkerTests.ApplySchedule(options);
            Model = new SynthIdImageWatermarker<float>(options);
            var rng = RandomHelper.CreateSeededRandom(5);
            for (int step = 0; step < Steps; step++)
            {
                var secrets = new Tensor<float>(new[] { Batch, Bits });
                for (int i = 0; i < secrets.Length; i++) secrets.SetFlat(i, rng.Next(2));
                Model.Train(StegaStampWatermarkerTests.Images(rng, Batch), secrets);
            }

            // Held-out calibration images: seeds disjoint from training (5) and from the evaluation sets (>= 20_000).
            var clean = StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(6), CalibrationImages);
            Model.Calibrate(clean, Watermark(Model, clean, RandomHelper.CreateSeededRandom(7)));
        }
    }

    private readonly TrainedModel _trained;
    private readonly ITestOutputHelper _out;

    public SynthIdImageWatermarkerTests(TrainedModel trained, ITestOutputHelper output)
    {
        _trained = trained;
        _out = output;
    }

    // Each image gets its own random payload, as during training.
    private static Tensor<float> Watermark(SynthIdImageWatermarker<float> model, Tensor<float> images, Random rng)
    {
        var marked = new Tensor<float>(images.Shape.ToArray());
        int per = images.Length / images.Shape[0];
        for (int n = 0; n < images.Shape[0]; n++)
        {
            var secret = Enumerable.Range(0, Bits).Select(_ => rng.Next(2) == 1).ToArray();
            var encoded = model.EncodeSecret(images.Slice(n), secret);
            for (int i = 0; i < per; i++) marked.SetFlat(n * per + i, encoded.GetFlat(i));
        }

        return marked;
    }

    private SynthIdImageWatermarker<float> Watermarker(string key) => _trained.Model.WithKey(key);

    /// <summary>
    /// Smoke-scale bars. A 16-bit payload caps the soft z at sqrt(16) = 4, and 600 steps measured about 3.5 for the own
    /// key, with and without the perturbation. The default 5.7 needs a longer-trained, larger payload.
    /// </summary>
    protected override double OwnKeyMinimumZ => 2.5;

    /// <inheritdoc />
    protected override double PerturbedMinimumZ => 2.5;

    protected override Tensor<float> Sample(int seed) =>
        StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(10_000 + seed), 1).Slice(0);

    protected override Tensor<float> Embed(string key, Tensor<float> content) => Watermarker(key).EmbedWatermark(content);

    protected override WatermarkResult Detect(string key, Tensor<float> content) => Watermarker(key).Detect(content);

    protected override Tensor<float> Perturb(Tensor<float> content, int seed) => StegaStampWatermarkerTests.GaussianNoise(content, seed);

    protected override bool SameContent(Tensor<float> a, Tensor<float> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a.GetFlat(i) != b.GetFlat(i)) return false;
        return true;
    }

    private (int Marked, int Clean, int Abstain) Verdicts(Tensor<float> images)
    {
        int marked = 0, clean = 0, abstain = 0;
        for (int n = 0; n < images.Shape[0]; n++)
        {
            switch (_trained.Model.Decide(images.Slice(n)).Verdict)
            {
                case WatermarkVerdict.Watermarked: marked++; break;
                case WatermarkVerdict.NotWatermarked: clean++; break;
                default: abstain++; break;
            }
        }

        return (marked, clean, abstain);
    }

    [Fact(Timeout = 300000)]
    public async Task CalibratedDecision_OnHeldOutImages_BoundsFalsePositives_AndDetectsWatermarks()
    {
        await Task.Yield();
        const int n = 200;
        var clean = StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(20_000), n);
        var marked = Watermark(_trained.Model, StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(20_001), n),
            RandomHelper.CreateSeededRandom(20_002));
        var onClean = Verdicts(clean);
        var onMarked = Verdicts(marked);
        _out.WriteLine($"clean: watermarked {onClean.Marked}, not {onClean.Clean}, abstain {onClean.Abstain} of {n}");
        _out.WriteLine($"watermarked: watermarked {onMarked.Marked}, not {onMarked.Clean}, abstain {onMarked.Abstain} of {n}");

        // alpha = 0.02. A Binomial(200, 0.02) count exceeds 10 with probability below 0.2%.
        Assert.True(onClean.Marked <= 10, $"{onClean.Marked} of {n} clean images called watermarked at alpha 0.02.");
        Assert.True(onMarked.Clean <= 10, $"{onMarked.Clean} of {n} watermarked images called not watermarked at alpha 0.02.");
        Assert.True(onMarked.Marked >= 0.9 * n, $"Only {onMarked.Marked} of {n} watermarked images called watermarked.");
    }

    [Fact(Timeout = 300000)]
    public async Task LiftedResidual_OnAnotherImage_IsRarelyCalledWatermarked()
    {
        await Task.Yield();
        // The residual the encoder made for one image, added to a different image. The specificity loss trains
        // this composite as unwatermarked. The test reports the measured rate and asserts that it stays well below
        // the detection rate of real marks.
        const int n = 100;
        var sources = StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(30_000), n);
        var targets = StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(30_001), n);
        var marked = Watermark(_trained.Model, sources, RandomHelper.CreateSeededRandom(30_002));
        var lifted = new Tensor<float>(targets.Shape.ToArray());
        for (int i = 0; i < lifted.Length; i++)
            lifted.SetFlat(i, Math.Max(0f, Math.Min(1f, targets.GetFlat(i) + marked.GetFlat(i) - sources.GetFlat(i))));
        var onLifted = Verdicts(lifted);
        var onMarked = Verdicts(marked);
        _out.WriteLine($"lifted: watermarked {onLifted.Marked}, not {onLifted.Clean}, abstain {onLifted.Abstain} of {n}; " +
            $"real marks: watermarked {onMarked.Marked} of {n}");
        Assert.True(onLifted.Marked <= onMarked.Marked / 2,
            $"{onLifted.Marked} of {n} lifted residuals called watermarked, against {onMarked.Marked} of {n} real marks.");
    }

    [Fact(Timeout = 300000)]
    public async Task Payload_DecodesFromWatermarkedImages()
    {
        await Task.Yield();
        var model = _trained.Model;
        var rng = RandomHelper.CreateSeededRandom(40_000);
        double accuracy = 0;
        const int images = 16;
        for (int n = 0; n < images; n++)
        {
            var image = StegaStampWatermarkerTests.Images(rng, 1).Slice(0);
            var secret = Enumerable.Range(0, Bits).Select(_ => rng.Next(2) == 1).ToArray();
            var decoded = model.DecodeSecret(model.EncodeSecret(image, secret));
            accuracy += (double)decoded.Zip(secret, (d, s) => d == s ? 1 : 0).Sum() / Bits / images;
        }

        _out.WriteLine($"mean bit accuracy {accuracy:F3}");
        Assert.True(accuracy >= 0.85, $"Mean bit accuracy {accuracy:F3} over {images} fresh images.");
    }
}
