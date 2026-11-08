using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Safety.Image;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Watermarking;

/// <summary>
/// Tree-Ring watermarking for diffusion models: a keyed ring pattern written into the Fourier transform of the
/// initial noise, which the sampler then turns into an image, detected by inverting the sampler back to noise.
/// </summary>
/// <remarks>
/// <para>
/// Following Wen et al. (2023): the key is a set of concentric rings in the centred 2D Fourier transform of one
/// channel of the starting noise x_T, a constant value along each ring and the rings' values drawn from a Gaussian.
/// Inside the circular mask of radius <c>r</c> the transform is replaced by the key (Eq. 2); the rest of the noise is
/// untouched, so the sampler sees ordinary noise outside a few low-frequency modes. Generation runs deterministic
/// DDIM from that noise. Detection encodes the image (through the VAE for latent models), runs DDIM inversion with
/// the unconditional noise prediction (the paper's empty prompt) to recover x'_T, and tests y = F(x'_T) on the mask
/// for the key.
/// </para>
/// <para>
/// Three refinements of the published construction:
/// <list type="bullet">
/// <item>The ring values are real. A ring is symmetric under f -> -f, so a real value makes the imprinted spectrum
/// Hermitian and the watermarked noise exactly real. The reference implementation draws complex values and keeps
/// the real part of the inverse transform, which silently discards each ring's imaginary part, so its detector
/// compares against values that were never embedded.</item>
/// <item>Only the independent half of the mask is scored. Under H0 the transform of real noise has y(-f) =
/// conj(y(f)), so the paper's sum over the whole mask counts every mode twice and its chi-squared degrees of freedom
/// are wrong. Scoring one of each conjugate pair, each with two real degrees of freedom (one for the DC term), makes
/// eta exactly noncentral chi-squared under H0.</item>
/// <item>The reported z is the matched filter of the key against y (whitened by the estimated mode variance), not the
/// paper's distance eta = (1/sigma^2) sum_M |k* - y|^2 with its noncentral chi-squared p-value (Eqs. 4-6). The paper
/// estimates sigma^2 from the same modes it scores, which pins |y|^2 / sigma^2 to a constant, so eta's p-value is not
/// uniform when an image carries another key's mark: across 100 foreign keys its z averaged 0.42 instead of 0. The
/// matched filter is N(0, 1) both for unmarked images and, given the image, for any other key, and it is the most
/// powerful test for a known Gaussian signal. The paper's statistic remains available for comparison.</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> a diffusion model turns random noise into a picture. This watermarker chooses that starting
/// noise so that its low-frequency "fingerprint" spells out a secret ring pattern. The finished picture shows nothing,
/// but running the model backwards from the picture recovers the noise, and the rings are still there. Only someone
/// with the same key and the same model can check for them.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust",
    "https://arxiv.org/abs/2305.20030",
    Year = 2023,
    Authors = "Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein")]
// Not an IImageWatermarker: that contract marks a finished image, and Tree-Ring cannot. Its mark exists only in the
// noise a generation starts from (Generate), and detection needs the generating model.
public partial class TreeRingWatermarker<T> : ImageSafetyModuleBase<T>, IKeyedWatermarker
{
    private const int KeySalt = 0x54524E47; // "TRNG"
    private const int NoiseSalt = 0x54524E5A; // "TRNZ"

    private readonly IDiffusionModel<T> _model;
    private readonly WatermarkKey _key;
    private readonly int _radius;
    private readonly int _channel;
    private readonly TreeRingKeyPattern _pattern;
    private readonly int _inferenceSteps;
    private readonly double _reportThreshold;

    /// <inheritdoc />
    public override string ModuleName => "TreeRingWatermarker";

    /// <inheritdoc />
    public bool UsesPublicKey => _key.IsPublic;

    /// <summary>The ring mask radius in Fourier modes (the paper uses 10; its ablation also reports 16).</summary>
    public int Radius => _radius;

    /// <summary>The noise channel that carries the rings.</summary>
    public int Channel => _channel;

    /// <summary>The key pattern written into the mask.</summary>
    public TreeRingKeyPattern Pattern => _pattern;

    /// <summary>DDIM steps used for both generation and inversion (the paper uses 50 for each).</summary>
    public int InferenceSteps => _inferenceSteps;

    /// <summary>
    /// Creates a Tree-Ring watermarker for <paramref name="model"/>.
    /// </summary>
    /// <param name="model">The diffusion model that generates and inverts. A latent diffusion model is sampled in its
    /// latent space and its VAE encodes images for detection.</param>
    /// <param name="key">The secret key; null uses the published key, which proves nothing about who made an image.</param>
    /// <param name="radius">The ring mask radius in Fourier modes.</param>
    /// <param name="channel">The noise channel that carries the rings.</param>
    /// <param name="inferenceSteps">DDIM steps for generation and inversion.</param>
    /// <param name="pattern">The key pattern; see <see cref="TreeRingKeyPattern"/> for what each one proves.</param>
    /// <param name="reportThreshold">Confidence from which <see cref="EvaluateImage"/> reports a finding.</param>
    public TreeRingWatermarker(IDiffusionModel<T> model, string? key = null, int radius = 10, int channel = 0,
        int inferenceSteps = 50, TreeRingKeyPattern pattern = TreeRingKeyPattern.Random, double reportThreshold = 0.3)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        if (radius < 1) throw new ArgumentOutOfRangeException(nameof(radius), radius, "The ring radius must be at least 1.");
        if (channel < 0) throw new ArgumentOutOfRangeException(nameof(channel), channel, "The channel must not be negative.");
        if (inferenceSteps < 1 || inferenceSteps > model.Scheduler.TrainTimesteps)
            throw new ArgumentOutOfRangeException(nameof(inferenceSteps), inferenceSteps,
                $"DDIM needs between 1 and {model.Scheduler.TrainTimesteps} steps.");
        if (!(reportThreshold >= 0 && reportThreshold <= 1))
            throw new ArgumentOutOfRangeException(nameof(reportThreshold), reportThreshold, "The report threshold must lie in [0, 1].");
        var prediction = model.Scheduler.Config.PredictionType;
        if (prediction != DiffusionPredictionType.Epsilon && prediction != DiffusionPredictionType.VPrediction)
            throw new ArgumentException($"DDIM inversion needs an epsilon- or v-predicting model; this one predicts {prediction}.", nameof(model));
        _key = WatermarkKey.FromOptional(key);
        _radius = radius;
        _channel = channel;
        if (!Enum.IsDefined(typeof(TreeRingKeyPattern), pattern))
            throw new ArgumentOutOfRangeException(nameof(pattern), pattern, "Unknown key pattern.");
        _pattern = pattern;
        _inferenceSteps = inferenceSteps;
        _reportThreshold = reportThreshold;
    }

    /// <summary>
    /// Generates a watermarked sample: keyed initial noise, then deterministic DDIM. For a latent diffusion model the
    /// result is decoded to an image.
    /// </summary>
    /// <param name="noiseShape">Shape of the model's sample space, [1, C, H, W] (the latent shape for latent models).</param>
    /// <param name="seed">Seed of the Gaussian noise outside the key, which varies the image.</param>
    /// <param name="noisePredictor">Optional guided noise prediction (for example text conditioning with classifier-free
    /// guidance); null uses the model's unconditional prediction.</param>
    public Tensor<T> Generate(int[] noiseShape, int seed, Func<Tensor<T>, int, Tensor<T>>? noisePredictor = null)
        => Sample(WatermarkedNoise(noiseShape, seed), noisePredictor);

    /// <summary>
    /// Generates the same way as <see cref="Generate"/> but from unmodified Gaussian noise: the unwatermarked
    /// counterpart, for calibrating false-positive rates.
    /// </summary>
    public Tensor<T> GenerateWithoutWatermark(int[] noiseShape, int seed, Func<Tensor<T>, int, Tensor<T>>? noisePredictor = null)
        => Sample(GaussianNoise(noiseShape, seed), noisePredictor);

    /// <summary>The keyed initial noise <see cref="Generate"/> starts from: Gaussian noise with the rings imprinted.</summary>
    public Tensor<T> WatermarkedNoise(int[] noiseShape, int seed)
    {
        var noise = GaussianNoise(noiseShape, seed);
        var (channels, height, width) = SampleDimensions(noise);
        CheckChannelAndRadius(channels, height, width);
        var values = ToDoubles(noise);
        int plane = height * width, offset = _channel * plane;
        var real = new double[plane];
        Array.Copy(values, offset, real, 0, plane);
        var (fr, fi) = Fft2D(real, new double[plane], height, width, inverse: false);
        foreach (var (index, re, im) in KeySpectrum(height, width))
        {
            fr[index] = re;
            fi[index] = im;
        }

        var (xr, _) = Fft2D(fr, fi, height, width, inverse: true);
        Array.Copy(xr, 0, values, offset, plane);
        return FromDoubles(values, noise.Shape.ToArray());
    }

    /// <summary>
    /// Recovers the initial noise of <paramref name="image"/> by DDIM inversion with the unconditional noise
    /// prediction. Inversion does not depend on the key, so one inversion can be tested against many keys with
    /// <see cref="DetectFromNoise"/>.
    /// </summary>
    public Tensor<T> InvertToNoise(Tensor<T> image)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var sample = image.Rank == 3 ? image.Reshape(new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] }) : image;
        if (sample.Rank != 4 || sample.Shape[0] != 1)
            throw new ArgumentException($"Expected one image, [C, H, W] or [1, C, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
        if (_model is ILatentDiffusionModel<T> latentModel) sample = latentModel.EncodeToLatent(sample, sampleMode: false);

        var x = ToDoubles(sample);
        var shape = sample.Shape.ToArray();
        var timesteps = DdimTimesteps();
        double alphaPrevious = 1.0;
        for (int i = timesteps.Length - 1; i >= 0; i--)
        {
            int t = timesteps[i];
            double alpha = NumOps.ToDouble(_model.Scheduler.GetAlphaCumulativeProduct(t));
            var epsilon = Epsilon(_model.PredictNoise, x, shape, t, alpha);
            // x_{t} = sqrt(a_t) * x0_hat + sqrt(1 - a_t) * eps, with x0_hat from the current, less noisy, level.
            double s = Math.Sqrt(alphaPrevious), n = Math.Sqrt(1 - alphaPrevious);
            double sa = Math.Sqrt(alpha), na = Math.Sqrt(1 - alpha);
            for (int k = 0; k < x.Length; k++) x[k] = sa * ((x[k] - n * epsilon[k]) / s) + na * epsilon[k];
            alphaPrevious = alpha;
        }

        return FromDoubles(x, shape);
    }

    /// <summary>
    /// Tests an image for this key's rings: DDIM inversion, then the matched-filter z of <see cref="DetectFromNoise"/>
    /// (not the paper's chi-squared statistic; see the class remarks).
    /// </summary>
    public WatermarkResult Detect(Tensor<T> image) => DetectFromNoise(InvertToNoise(image));

    /// <summary>Tests recovered initial noise (see <see cref="InvertToNoise"/>) for this key's rings.</summary>
    public WatermarkResult DetectFromNoise(Tensor<T> noise)
    {
        var (correlationZ, _, units) = Statistics(noise, withChiSquared: false);
        return Result(correlationZ, units);
    }

    /// <summary>
    /// The paper's own test on recovered noise, as a z: the lower tail of eta = (1/sigma^2) sum |k* - y|^2 under its
    /// noncentral chi-squared null, scored over the independent half of the mask. Kept for comparison with
    /// <see cref="DetectFromNoise"/>; it estimates sigma^2 from the same modes it scores, which fixes |y|^2 / sigma^2
    /// and so is not calibrated against other keys.
    /// </summary>
    internal double PaperChiSquaredZ(Tensor<T> noise) => Statistics(noise, withChiSquared: true).ChiSquaredZ;

    // Both statistics over the independent half of the mask: one mode of each conjugate pair, plus the DC term.
    // Whitened by sigma^2 (per real degree of freedom sigma^2 / 2 off DC, sigma^2 at DC), the matched filter
    // sum w k y / (sigma sqrt(sum w k^2)) is N(0, 1) for unmarked noise (y independent of k) and, given y, for an image
    // marked under another key (k independent of y), so one z is calibrated against both nulls.
    // The chi-squared z is computed only when asked for (PaperChiSquaredZ); detection does not use it.
    private (double CorrelationZ, double ChiSquaredZ, int Units) Statistics(Tensor<T> noise, bool withChiSquared)
    {
        if (noise is null) throw new ArgumentNullException(nameof(noise));
        var (channels, height, width) = SampleDimensions(noise);
        CheckChannelAndRadius(channels, height, width);
        var values = ToDoubles(noise);
        int plane = height * width;
        var real = new double[plane];
        Array.Copy(values, _channel * plane, real, 0, plane);
        bool finite = true;
        foreach (double v in real) finite &= !(double.IsNaN(v) || double.IsInfinity(v));
        if (!finite) return (0.0, 0.0, 0);

        var (yr, yi) = Fft2D(real, new double[plane], height, width, inverse: false);
        var key = new Dictionary<int, (double Re, double Im)>();
        foreach (var (index, re, im) in KeySpectrum(height, width)) key[index] = (re, im);

        // The independent half of the mask: one mode of each conjugate pair, plus the self-conjugate DC term.
        double energy = 0, keyEnergy = 0, distance = 0, correlation = 0;
        int pairs = 0;
        bool hasDc = false;
        foreach (var (index, _, fy, fx) in MaskModesWithFrequency(height, width))
        {
            bool dc = fy == 0 && fx == 0;
            if (!dc && !(fy > 0 || (fy == 0 && fx > 0))) continue;
            var (kr, ki) = key[index];
            double re = yr[index], im = dc ? 0.0 : yi[index];
            energy += re * re + im * im;
            if (dc)
            {
                hasDc = true;
                keyEnergy += kr * kr;
                distance += (kr - re) * (kr - re);
                correlation += kr * re;
            }
            else
            {
                pairs++;
                // Each real and imaginary part has variance sigma^2 / 2 under H0, so they count twice.
                keyEnergy += 2 * (kr * kr + ki * ki);
                distance += 2 * ((kr - re) * (kr - re) + (ki - im) * (ki - im));
                correlation += 2 * (kr * re + ki * im);
            }
        }

        int units = pairs + (hasDc ? 1 : 0);
        if (units == 0) return (0.0, 0.0, 0);
        double sigma2 = energy / units;
        if (double.IsNaN(sigma2) || sigma2 <= 1e-300 || double.IsNaN(keyEnergy) || keyEnergy <= 0) return (0.0, 0.0, units);
        double correlationZ = WatermarkStatistics.CorrelationZ(correlation, keyEnergy * sigma2);
        if (!withChiSquared) return (correlationZ, 0.0, units);
        int degreesOfFreedom = 2 * pairs + (hasDc ? 1 : 0);
        double eta = distance / sigma2, lambda = keyEnergy / sigma2;
        // A small eta means y sits close to the key: the evidence is the lower tail of the noncentral chi-squared.
        double chiSquaredZ = ZFromLogUpperTail(NoncentralChiSquaredLogCdf(eta, degreesOfFreedom, lambda));
        return (correlationZ, chiSquaredZ, units);
    }

    /// <summary>The detection confidence in [0, 1] for <paramref name="image"/> (see <see cref="Detect"/>).</summary>
    public double DetectWatermark(Tensor<T> image) => Detect(image).Confidence;

    /// <inheritdoc />
    public override IReadOnlyList<SafetyFinding> EvaluateImage(Tensor<T> image)
    {
        var findings = new List<SafetyFinding>();
        var result = Detect(image);
        if (result.Confidence > 0 && result.Confidence >= _reportThreshold)
        {
            findings.Add(new SafetyFinding
            {
                Category = SafetyCategory.Watermarked,
                Severity = SafetySeverity.Info,
                Confidence = result.Confidence,
                Description = $"Tree-Ring watermark detected (confidence: {result.Confidence:F3}, z = {result.ZScore:F2}, p = {result.PValue:E1}).",
                RecommendedAction = SafetyAction.Log,
                SourceModule = ModuleName
            });
        }

        return findings;
    }

    private WatermarkResult Result(double z, int units) =>
        WatermarkStatistics.Result(z, units, "image", "Tree-Ring watermark", _reportThreshold);

    private Tensor<T> Sample(Tensor<T> noise, Func<Tensor<T>, int, Tensor<T>>? noisePredictor)
    {
        var predict = noisePredictor ?? _model.PredictNoise;
        var x = ToDoubles(noise);
        var shape = noise.Shape.ToArray();
        var timesteps = DdimTimesteps();
        for (int i = 0; i < timesteps.Length; i++)
        {
            int t = timesteps[i];
            double alpha = NumOps.ToDouble(_model.Scheduler.GetAlphaCumulativeProduct(t));
            double alphaPrevious = i + 1 < timesteps.Length
                ? NumOps.ToDouble(_model.Scheduler.GetAlphaCumulativeProduct(timesteps[i + 1]))
                : 1.0;
            var epsilon = Epsilon(predict, x, shape, t, alpha);
            double sa = Math.Sqrt(alpha), na = Math.Sqrt(1 - alpha);
            double sp = Math.Sqrt(alphaPrevious), np = Math.Sqrt(1 - alphaPrevious);
            for (int k = 0; k < x.Length; k++) x[k] = sp * ((x[k] - na * epsilon[k]) / sa) + np * epsilon[k];
        }

        var sample = FromDoubles(x, shape);
        return _model is ILatentDiffusionModel<T> latentModel ? latentModel.DecodeFromLatent(sample) : sample;
    }

    // The model's noise estimate at (x, t); a v-prediction is converted with eps = sqrt(a) v + sqrt(1 - a) x.
    private double[] Epsilon(Func<Tensor<T>, int, Tensor<T>> predict, double[] x, int[] shape, int t, double alpha)
    {
        var output = ToDoubles(predict(FromDoubles(x, shape), t));
        if (output.Length != x.Length)
            throw new InvalidOperationException($"The noise prediction has {output.Length} values for a {x.Length}-value sample.");
        if (_model.Scheduler.Config.PredictionType == DiffusionPredictionType.VPrediction)
        {
            double sa = Math.Sqrt(alpha), na = Math.Sqrt(1 - alpha);
            for (int k = 0; k < output.Length; k++) output[k] = sa * output[k] + na * x[k];
        }

        return output;
    }

    // DDIM's evenly spaced timesteps, most noisy first: (steps - 1 - i) * (T / steps).
    private int[] DdimTimesteps()
    {
        int train = _model.Scheduler.TrainTimesteps, ratio = train / _inferenceSteps;
        var timesteps = new int[_inferenceSteps];
        for (int i = 0; i < _inferenceSteps; i++) timesteps[i] = (_inferenceSteps - 1 - i) * ratio;
        return timesteps;
    }

    private Tensor<T> GaussianNoise(int[] noiseShape, int seed)
    {
        if (noiseShape is null) throw new ArgumentNullException(nameof(noiseShape));
        if (noiseShape.Length != 4 || noiseShape[0] != 1 || noiseShape.Any(d => d < 1))
            throw new ArgumentException($"Expected a [1, C, H, W] noise shape; got [{string.Join(", ", noiseShape)}].", nameof(noiseShape));
        var rng = RandomHelper.CreateSeededRandom(seed ^ NoiseSalt);
        long count = (long)noiseShape[1] * noiseShape[2] * noiseShape[3];
        var values = new double[count];
        for (long i = 0; i < count; i++) values[i] = Gaussian(rng);
        return FromDoubles(values, noiseShape);
    }

    private static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // The key's value at every mask mode, Hermitian so the keyed noise is exactly real. Values are scaled like a
    // Fourier mode of unit-variance noise: E|F|^2 = H * W.
    // Rings: one real Gaussian per ring; ring 1 also covers the DC term, as in the reference implementation, which
    // assigns radius ceil(|f|) with a minimum of 1. Random: an independent complex Gaussian per conjugate pair (real
    // at DC), so a foreign key's values are independent of this key's and the chi-squared null holds exactly.
    private IEnumerable<(int Index, double Re, double Im)> KeySpectrum(int height, int width)
    {
        ulong seed = _key.Seed(KeySalt, new[] { height, width, _radius, _channel, (int)_pattern });
        double scale = Math.Sqrt((double)height * width);
        double Normal(int item)
        {
            double u1 = 1.0 - WatermarkKey.Uniform(seed, 2 * item), u2 = WatermarkKey.Uniform(seed, 2 * item + 1);
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        foreach (var (index, ring, fy, fx) in MaskModesWithFrequency(height, width))
        {
            if (_pattern == TreeRingKeyPattern.Rings)
            {
                yield return (index, scale * Normal(ring), 0.0);
                continue;
            }

            if (fy == 0 && fx == 0)
            {
                yield return (index, scale * Normal(0), 0.0);
                continue;
            }

            // Both members of a conjugate pair draw from the canonical one's position; the other is its conjugate.
            bool canonical = fy > 0 || (fy == 0 && fx > 0);
            int cy = canonical ? fy : -fy, cx = canonical ? fx : -fx;
            int item = 1 + (cy + _radius) * (2 * _radius + 1) + (cx + _radius);
            double re = scale * Math.Sqrt(0.5) * Normal(2 * item), im = scale * Math.Sqrt(0.5) * Normal(2 * item + 1);
            yield return (index, re, canonical ? im : -im);
        }
    }

    // Modes |f| <= radius of the unshifted transform, with their signed frequencies (numpy fftfreq convention).
    private IEnumerable<(int Index, int Ring, int Fy, int Fx)> MaskModesWithFrequency(int height, int width)
    {
        for (int ky = 0; ky < height; ky++)
        {
            int fy = ky <= (height - 1) / 2 ? ky : ky - height;
            if (Math.Abs(fy) > _radius) continue;
            for (int kx = 0; kx < width; kx++)
            {
                int fx = kx <= (width - 1) / 2 ? kx : kx - width;
                int d2 = fy * fy + fx * fx;
                if (d2 > _radius * _radius) continue;
                int ring = Math.Max(1, (int)Math.Ceiling(Math.Sqrt(d2) - 1e-12));
                yield return (ky * width + kx, ring, fy, fx);
            }
        }
    }

    private void CheckChannelAndRadius(int channels, int height, int width)
    {
        if (_channel >= channels)
            throw new ArgumentException($"The watermark channel {_channel} does not exist in a {channels}-channel sample.");
        // Below the Nyquist mode every ring mode has a distinct conjugate partner, which the exact test relies on.
        if (2 * _radius >= Math.Min(height, width))
            throw new ArgumentException($"A ring radius of {_radius} needs a sample larger than {2 * _radius} in each direction; got {height} x {width}.");
    }

    private static (int Channels, int Height, int Width) SampleDimensions(Tensor<T> sample)
    {
        var shape = sample.Shape.ToArray();
        if (shape.Length == 4 && shape[0] == 1) return (shape[1], shape[2], shape[3]);
        if (shape.Length == 3) return (shape[0], shape[1], shape[2]);
        throw new ArgumentException($"Expected a [1, C, H, W] or [C, H, W] sample; got [{string.Join(", ", shape)}].", nameof(sample));
    }

    private (double[] Real, double[] Imag) Fft2D(double[] real, double[] imag, int height, int width, bool inverse)
    {
        var shape = new[] { height, width };
        var re = FromDoubles(real, shape);
        var im = FromDoubles(imag, shape);
        Tensor<T> outRe, outIm;
        if (inverse) Engine.IFFT2D(re, im, out outRe, out outIm);
        else Engine.FFT2D(re, im, out outRe, out outIm);
        return (ToDoubles(outRe), ToDoubles(outIm));
    }

    private static double[] ToDoubles(Tensor<T> tensor)
    {
        var data = tensor.ToArray();
        var values = new double[data.Length];
        for (int i = 0; i < data.Length; i++) values[i] = NumOps.ToDouble(data[i]);
        return values;
    }

    private static Tensor<T> FromDoubles(double[] values, int[] shape)
    {
        var data = new T[values.Length];
        for (int i = 0; i < values.Length; i++) data[i] = NumOps.FromDouble(values[i]);
        return new Tensor<T>(data, shape);
    }

    // ln Pr(X <= x) for X ~ noncentral chi-squared(k, lambda): the Poisson(lambda / 2) mixture of central
    // chi-squared(k + 2j) CDFs, summed in log space outward from the Poisson mode until the terms stop mattering.
    internal static double NoncentralChiSquaredLogCdf(double x, int k, double lambda)
    {
        if (double.IsNaN(x) || double.IsNaN(lambda) || double.IsInfinity(lambda)) return double.NaN;
        if (double.IsNaN(x) || x <= 0) return double.NegativeInfinity;
        if (double.IsPositiveInfinity(x)) return 0.0;
        if (double.IsNaN(lambda) || lambda <= 0) return LogRegularizedLowerGamma(k / 2.0, x / 2.0);
        // The Poisson mixture is summed outward from its mode, an int; beyond that the series is not evaluated.
        if (lambda / 2.0 >= int.MaxValue - 1024) return double.NaN;
        double halfLambda = lambda / 2.0;
        int mode = (int)Math.Floor(halfLambda);
        double LogTerm(int j) => -halfLambda + j * Math.Log(halfLambda) - LogGamma(j + 1.0)
            + LogRegularizedLowerGamma(k / 2.0 + j, x / 2.0);
        double max = LogTerm(mode);
        if (double.IsNaN(max)) return double.NaN;
        double sum = 1.0;
        for (int j = mode + 1; ; j++)
        {
            double term = LogTerm(j);
            if (double.IsNaN(term)) return double.NaN;
            if (term > max) { sum = sum * Math.Exp(max - term) + 1.0; max = term; continue; }
            sum += Math.Exp(term - max);
            if (term - max < -40 && j > mode + 10) break;
        }

        for (int j = mode - 1; j >= 0; j--)
        {
            double term = LogTerm(j);
            if (term > max) { sum = sum * Math.Exp(max - term) + 1.0; max = term; continue; }
            sum += Math.Exp(term - max);
            if (term - max < -40 && j < mode - 10) break;
        }

        return Math.Min(0.0, max + Math.Log(sum));
    }

    // z with Pr(Z >= z) = exp(logP), accurate where exp(logP) underflows.
    internal static double ZFromLogUpperTail(double logP)
    {
        if (double.IsNaN(logP)) return 0.0;
        if (logP >= 0) return double.NegativeInfinity;
        if (logP > -700) return WatermarkStatistics.ZForUpperTail(Math.Exp(logP));
        // Mills ratio: ln Pr(Z >= z) = -z^2/2 - ln z - ln sqrt(2 pi) + ln(1 - 1/z^2 + ...); solve by Newton steps.
        double z = Math.Sqrt(-2 * logP);
        for (int i = 0; i < 20; i++)
        {
            double f = -0.5 * z * z - Math.Log(z) - 0.5 * Math.Log(2 * Math.PI) + Math.Log(1 - 1 / (z * z) + 3 / (z * z * z * z)) - logP;
            double derivative = -z - 1 / z;
            double step = f / derivative;
            z -= step;
            if (Math.Abs(step) < 1e-12 * z) break;
        }

        return z;
    }

    // ln P(a, x), the regularized lower incomplete gamma: a series below a + 1, else ln(1 - Q) from Lentz's
    // continued fraction for Q (Numerical Recipes 6.2).
    internal static double LogRegularizedLowerGamma(double a, double x)
    {
        if (double.IsNaN(x) || x <= 0) return double.NegativeInfinity;
        double logPrefix = a * Math.Log(x) - x - LogGamma(a);
        if (x < a + 1)
        {
            double term = 1.0 / a, sum = term;
            for (int n = 1; n < 10000; n++)
            {
                term *= x / (a + n);
                sum += term;
                if (term < sum * 1e-16) break;
            }

            return logPrefix + Math.Log(sum);
        }

        const double tiny = 1e-300;
        double b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
        for (int i = 1; i < 10000; i++)
        {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.Abs(d) < tiny) d = tiny;
            c = b + an / c;
            if (Math.Abs(c) < tiny) c = tiny;
            d = 1 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1) < 1e-16) break;
        }

        double q = Math.Exp(logPrefix) * h;
        return Math.Log(Math.Max(0.0, 1 - q));
    }

    // Lanczos approximation (g = 7, n = 9), relative error below 1e-15 for positive arguments.
    internal static double LogGamma(double x)
    {
        double[] c =
        {
            0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
            1.5056327351493116e-7
        };
        if (x < 0.5) return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * x))) - LogGamma(1 - x);
        x -= 1;
        double sum = c[0];
        for (int i = 1; i < 9; i++) sum += c[i] / (x + i);
        double t = x + 7.5;
        return 0.5 * Math.Log(2 * Math.PI) + (x + 0.5) * Math.Log(t) - t + Math.Log(sum);
    }
}

/// <summary>
/// The key written into a <see cref="TreeRingWatermarker{T}"/>'s Fourier mask (Wen et al. 2023, Section 3.2).
/// </summary>
public enum TreeRingKeyPattern
{
    /// <summary>
    /// Tree-Ring_Rand: an independent Gaussian value per Fourier mode (about pi r^2 / 2 independent values). The
    /// detection statistic is exactly calibrated against other keys, so a confident result proves this key made the
    /// image. Not invariant to rotation.
    /// </summary>
    Random = 0,

    /// <summary>
    /// Tree-Ring_Rings, the paper's default: one constant per ring, invariant to rotations and flips. The z is
    /// calibrated against unmarked images, but with only <c>radius</c> independent values another Rings key
    /// correlates with the mark by chance at the scale of a ring's mode count, so its z-scores spread far wider than
    /// N(0, 1). Read a confident Rings result as "carries a Tree-Ring mark", not as proof of this key.
    /// </summary>
    Rings = 1
}
