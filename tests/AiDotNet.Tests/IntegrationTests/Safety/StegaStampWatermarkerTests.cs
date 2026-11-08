using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// The StegaStamp encoder and decoder trained once (smoke scale: 32 x 32 images, 16 bits, no BCH, 600 steps; the pull request records the measured numbers) on synthetic smooth
/// images under the paper's distortions, then every keyed round-trip contract: own key detected, unmarked images and
/// foreign keys calibrated, detection surviving a perturbation, deterministic and key-dependent embedding.
/// </summary>
[Trait("Category", "HeavyTimeout")]
public class StegaStampWatermarkerTests : WatermarkRoundTripTestBase<Tensor<float>>, IClassFixture<StegaStampWatermarkerTests.TrainedModel>
{
    internal const int Size = 32;
    internal const int Bits = 64;

    /// <summary>Trains the networks once for the class; xUnit builds fixtures outside each test's timeout.</summary>
    public sealed class TrainedModel
    {
        internal const int Steps = 2400;
        internal const int Batch = 8;
        internal const double LearningRate = 1e-3;

        public StegaStampWatermarker<float> Model { get; }

        public TrainedModel()
        {
            var options = new StegaStampOptions { ImageSize = Size, SecretBits = Bits, UseErrorCorrection = false, Seed = 1 };
            ApplySchedule(options);
            Model = new StegaStampWatermarker<float>(options);
            var rng = RandomHelper.CreateSeededRandom(5);
            for (int step = 0; step < Steps; step++)
            {
                var secrets = new Tensor<float>(new[] { Batch, Bits });
                for (int i = 0; i < secrets.Length; i++) secrets.SetFlat(i, rng.Next(2));
                Model.Train(Images(rng, Batch), secrets);
            }
        }
    }

    private readonly TrainedModel _trained;

    /// <summary>
    /// The paper's schedule compressed to this run: the secret alone first, then the image and critic losses and the
    /// distortions ramped in.
    /// </summary>
    internal static void ApplySchedule(StegaStampOptions options)
    {
        options.LearningRate = TrainedModel.LearningRate;
        // The reference script's step counts (base.sh), all scaled so its 1500 secret-only steps become a third of
        // this run: the image and critic losses then start at a tenth of their weight and the colour distortions are
        // fully on, as in the reference. The run covers the reference's first 4500 steps.
        options.NoImageLossSteps = TrainedModel.Steps / 3;
        int Scaled(int referenceSteps) => (int)((long)referenceSteps * options.NoImageLossSteps / 1500);
        options.ImageLossRampSteps = Scaled(15000);
        options.CriticLossRampSteps = Scaled(15000);
        options.PerspectiveRampSteps = Scaled(10000);
        options.ColorRampSteps = Scaled(1000);
        options.EdgeLossDelaySteps = Scaled(80000);
        options.EdgeLossRampSteps = Scaled(10000);
        // Ten times the reference learning rate, so the run fits a smoke budget; without a warmup the first full-rate
        // Adam steps saturate the residual for some initialisations before the decoder can read it.
        options.LearningRateWarmupSteps = TrainedModel.Steps / 6;
    }

    /// <summary>Additive Gaussian noise, sd 2% of the [0, 1] range.</summary>
    internal static Tensor<float> GaussianNoise(Tensor<float> content, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(4000 + seed);
        var noisy = new Tensor<float>(content.Shape.ToArray());
        for (int i = 0; i < content.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            noisy.SetFlat(i, (float)Math.Max(0, Math.Min(1, content.GetFlat(i) + 0.02 * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2))));
        }

        return noisy;
    }

    public StegaStampWatermarkerTests(TrainedModel trained) => _trained = trained;

    /// <summary>Smooth random images: a few oriented sinusoids per image plus a little pixel noise, in [0, 1].</summary>
    internal static Tensor<float> Images(Random rng, int count)
    {
        var t = new Tensor<float>(new[] { count, 3, Size, Size });
        for (int n = 0; n < count; n++)
        {
            var waves = Enumerable.Range(0, 12)
                .Select(_ => (Fx: rng.NextDouble() * 4, Fy: rng.NextDouble() * 4, Phase: rng.NextDouble() * 2 * Math.PI, Amp: rng.NextDouble()))
                .ToArray();
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < Size; y++)
                    for (int x = 0; x < Size; x++)
                    {
                        double v = 0.5;
                        foreach (var w in waves) v += 0.08 * w.Amp * Math.Sin(2 * Math.PI * (w.Fx * x + w.Fy * y) / Size + w.Phase + c);
                        v += 0.03 * (rng.NextDouble() - 0.5);
                        t[n, c, y, x] = (float)Math.Max(0, Math.Min(1, v));
                    }
        }

        return t;
    }

    private StegaStampWatermarker<float> Watermarker(string key) => _trained.Model.WithKey(key);

    /// <summary>
    /// Smoke-scale bars, below what this fixture measures (64 bits, 2400 steps on 32 x 32 images): z 5.10-5.19 on the
    /// three marked samples (confidence 0.70-0.73, detected) and 5.06-5.20 after the perturbation. The default 5.7 needs
    /// a longer-trained model.
    /// </summary>
    protected override double OwnKeyMinimumZ => 4.5;

    /// <inheritdoc />
    protected override double PerturbedMinimumZ => 4.5;

    protected override Tensor<float> Sample(int seed) =>
        Images(RandomHelper.CreateSeededRandom(10_000 + seed), 1).Slice(0);

    protected override Tensor<float> Embed(string key, Tensor<float> content) => Watermarker(key).EmbedWatermark(content);

    protected override WatermarkResult Detect(string key, Tensor<float> content) => Watermarker(key).Detect(content);

    protected override Tensor<float> Perturb(Tensor<float> content, int seed) => GaussianNoise(content, seed);

    protected override bool SameContent(Tensor<float> a, Tensor<float> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a.GetFlat(i) != b.GetFlat(i)) return false;
        return true;
    }

    private static double Psnr(Tensor<float> a, Tensor<float> b)
    {
        double mse = 0;
        for (int i = 0; i < a.Length; i++) { double d = a.GetFlat(i) - b.GetFlat(i); mse += d * d; }
        mse /= a.Length;
        return 10 * Math.Log10(1.0 / Math.Max(mse, 1e-12));
    }

    private static double BitAccuracy(bool[] decoded, bool[] secret) =>
        (double)decoded.Zip(secret, (d, s) => d == s ? 1 : 0).Sum() / secret.Length;

    [Fact(Timeout = 120000)]
    public async Task ArbitrarySecrets_DecodeFromTheirEncodedImages()
    {
        await Task.Yield();
        var model = _trained.Model;
        var rng = RandomHelper.CreateSeededRandom(77);
        double accuracy = 0, psnr = 0;
        const int images = 16;
        for (int n = 0; n < images; n++)
        {
            var image = Images(rng, 1).Slice(0);
            var secret = Enumerable.Range(0, Bits).Select(_ => rng.Next(2) == 1).ToArray();
            var encoded = model.EncodeSecret(image, secret);
            accuracy += BitAccuracy(model.DecodeSecret(encoded), secret) / images;
            psnr += Psnr(image, encoded) / images;
        }

        // Smoke scale, measured with the 64-bit payload: bit accuracy 0.785, PSNR 21.9 dB (the 16-bit, 600-step fixture
        // measured 0.934 and 16.7 dB). Chance is 0.5, and over 16 x 64 bits its standard error is 0.016, so 0.75 is far
        // above it. The run is the reference schedule's first 4500 steps, so the image loss has reached 0.6 of its weight.
        Assert.True(accuracy >= 0.75, $"Mean bit accuracy {accuracy:F3} over {images} fresh images (PSNR {psnr:F1} dB).");
        Assert.True(psnr >= 15, $"Encoded images average PSNR {psnr:F1} dB (bit accuracy {accuracy:F3}).");
    }

    [Fact(Timeout = 120000)]
    public async Task Secrets_SurviveRealJpegRounding_AtQuality75()
    {
        await Task.Yield();
        // Training saw the cubic rounding approximation; this applies JPEG's real rounding.
        var model = _trained.Model;
        var rng = RandomHelper.CreateSeededRandom(78);
        double accuracy = 0;
        const int images = 16;
        for (int n = 0; n < images; n++)
        {
            var image = Images(rng, 1);
            var secret = Enumerable.Range(0, Bits).Select(_ => rng.Next(2) == 1).ToArray();
            var encoded = model.EncodeSecret(image, secret);
            var jpeg = model.DifferentiableJpeg(encoded, model.QuantizationTable(1, Size, 75), exactRounding: true);
            accuracy += BitAccuracy(model.DecodeSecret(jpeg), secret) / images;
        }

        // Measured with the 64-bit payload: 0.788 (16 bits: above 0.85).
        Assert.True(accuracy >= 0.75, $"Mean bit accuracy {accuracy:F3} after quality-75 JPEG rounding.");
    }
}

/// <summary>StegaStamp checks that need no training, kept in the pull-request gate.</summary>
public class StegaStampDistortionTests
{
    [Fact(Timeout = 60000)]
    public async Task Jpeg_AtQuality100_IsNearlyTheIdentity()
    {
        await Task.Yield();
        // Guards the colour transforms and DCT: a level-shift error once pushed every pixel outside [0, 1], so the
        // final clip zeroed the encoder's whole gradient.
        var model = new StegaStampWatermarker<float>(new StegaStampOptions { ImageSize = StegaStampWatermarkerTests.Size, SecretBits = 16, UseErrorCorrection = false, Seed = 2 });
        var image = StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(79), 2);
        var table = model.QuantizationTable(2, StegaStampWatermarkerTests.Size, 100);
        foreach (bool exact in new[] { true, false })
        {
            var round = model.DifferentiableJpeg(image, table, exact);
            double worst = 0;
            for (int i = 0; i < image.Length; i++) worst = Math.Max(worst, Math.Abs(round.GetFlat(i) - image.GetFlat(i)));
            Assert.True(worst < 0.02, $"Quality-100 JPEG (exact rounding {exact}) moved a pixel by {worst:F4}.");
        }
    }
}
