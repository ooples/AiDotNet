using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Video.Prediction;

/// <summary>
/// MCnet's objective: a pixel loss plus a GRADIENT DIFFERENCE loss in image space, combined with a
/// conditional adversarial term.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Villegas, Yang, Hong, Lin and Lee, "Decomposing Motion and Content for Natural Video Sequence
/// Prediction" (ICLR 2017, arXiv:1706.08033).
/// </para>
/// <code>
///   L      = alpha * L_img + beta * L_GAN
///   L_img  = L_p(x, x_hat) + L_gdl(x, x_hat)
///   L_p    = sum ||y - z||_p^p
///   L_gdl  = sum | |y_i,j - y_i-1,j| - |z_i,j - z_i-1,j| |^lambda  + the j-direction term
///   L_GAN  = -log D([x_1:t, G(x_1:t)])
///   L_disc = -log D([x_1:t, x_t+1:t+T]) - log(1 - D([x_1:t, G(x_1:t)]))
///
///   alpha = 1, beta = 0.02 (KTH) / 0.001 (UCF-101), lambda = 1, p = 2
/// </code>
/// <para>
/// <b>The gradient difference loss is the term a reimplementation drops.</b> It penalises differences
/// between the IMAGE GRADIENTS of prediction and target, not the pixels — which is what keeps
/// predictions sharp. Omitting it leaves a plain L2, and L2 alone produces exactly the blur the paper
/// exists to fight, because averaging over plausible futures minimises squared error.
/// </para>
/// <para>
/// <b>The adversarial form is the NON-saturating one</b>, <c>-log D(G(.))</c>, unlike FIGAN's literal
/// <c>log(1 - D(G(.)))</c>. They are different objectives and should not be harmonised into one shared
/// helper just because both models are adversarial.
/// </para>
/// <para>
/// <b>The discriminator is CONDITIONED</b> on the observed frames: it judges
/// <c>[x_1:t, future]</c>, never the future alone. An unconditional critic cannot tell a plausible
/// continuation of THIS sequence from a plausible video in general, which is a weaker and different
/// requirement.
/// </para>
/// <para><b>For Beginners:</b> Predicting the next video frame by minimising average pixel error gives
/// a blurry compromise between everything that might happen. Also matching how sharply brightness
/// CHANGES across the image — its gradients — forces crisp edges, and a critic network judging whether
/// the continuation looks real given what came before pushes it further.</para>
/// </remarks>
public static class McnetLoss<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The paper's image-loss weight alpha.</summary>
    public const double ImageWeight = 1.0;

    /// <summary>The paper's adversarial weight beta for KTH. UCF-101 uses 0.001.</summary>
    public const double AdversarialWeightKth = 0.02;

    /// <summary>The paper's adversarial weight beta for UCF-101.</summary>
    public const double AdversarialWeightUcf = 0.001;

    /// <summary>The paper's gradient-difference exponent lambda.</summary>
    public const double GradientExponent = 1.0;

    /// <summary>The paper's pixel-loss norm p.</summary>
    public const int PixelNorm = 2;

    /// <summary>
    /// <c>L_p</c>: the pixel loss, <c>sum ||y - z||_p^p</c> with <c>p = 2</c> by default.
    /// </summary>
    /// <remarks>
    /// Averaged over elements so the value does not scale with frame size; the paper's sum form differs
    /// only by that constant factor, which would otherwise make the loss weights resolution-dependent.
    /// </remarks>
    public static double Pixel(Vector<T> predicted, Vector<T> target, int p = PixelNorm)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (predicted.Length != target.Length)
            throw new ArgumentException(
                $"Predicted ({predicted.Length}) and target ({target.Length}) must be the same length.",
                nameof(predicted));
        if (p <= 0) throw new ArgumentOutOfRangeException(nameof(p), p, "The norm must be positive.");

        if (predicted.Length == 0)
        {
            return double.NaN;
        }

        var engine = AiDotNetEngine.Current;
        var d = engine.TensorAbs(engine.TensorSubtract(predicted.AsTensor(), target.AsTensor()));
        return MeanPower(engine, d, p);
    }

    /// <summary>
    /// Mean of <c>|v|^exponent</c> over the non-negative values <paramref name="magnitudes"/>, with the values
    /// divided by their peak before the power so a large exponent cannot overflow <typeparamref name="T"/>.
    /// </summary>
    private static double MeanPower(IEngine engine, Tensor<T> magnitudes, double exponent)
    {
        double peak = Ops.ToDouble(engine.TensorMaxValue(magnitudes));
        if (double.IsNaN(peak))
        {
            return double.NaN;
        }

        if (peak == 0)
        {
            // 0^exponent: 0 for a positive exponent, 1 for exponent 0, as Math.Pow gives
            return exponent == 0 ? 1.0 : 0.0;
        }

        var unit = engine.TensorMultiplyScalar(magnitudes, Ops.FromDouble(1.0 / peak));
        var powered = engine.TensorPow(unit, Ops.FromDouble(exponent));
        return Ops.ToDouble(engine.TensorMean(powered)) * Math.Pow(peak, exponent);
    }

    /// <summary>
    /// <c>L_gdl</c>: the gradient difference loss over a <c>[height, width]</c> single-channel image.
    /// </summary>
    /// <param name="predicted">Predicted frame, row-major.</param>
    /// <param name="target">Target frame, row-major.</param>
    /// <param name="height">Frame height.</param>
    /// <param name="width">Frame width.</param>
    /// <param name="lambda">Exponent; the paper uses 1.</param>
    /// <remarks>
    /// Both directions are included, as the paper's <c>+ ...</c> indicates: the vertical term compares
    /// <c>|y_i,j - y_i-1,j|</c> against the prediction's, and the horizontal term does the same along
    /// j. Using only one direction penalises blur in one axis and lets it through in the other.
    /// </remarks>
    public static double GradientDifference(
        Vector<T> predicted, Vector<T> target, int height, int width, double lambda = GradientExponent)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), height, "Height must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), width, "Width must be positive.");
        if (predicted.Length < height * width || target.Length < height * width)
            throw new ArgumentException(
                $"Both frames must hold at least {height * width} elements for a [{height}, {width}] image.",
                nameof(predicted));

        var engine = AiDotNetEngine.Current;
        var y = engine.Reshape(engine.TensorNarrow(target.AsTensor(), 0, 0, height * width), new[] { height, width });
        var z = engine.Reshape(engine.TensorNarrow(predicted.AsTensor(), 0, 0, height * width), new[] { height, width });

        // |(|y_i,j - y_i-1,j|) - (|z_i,j - z_i-1,j|)| along an axis, the vertical then the horizontal term
        Tensor<T> GradientGap(int axis)
        {
            int extent = axis == 0 ? height : width;
            Tensor<T> Gradient(Tensor<T> image) => engine.TensorAbs(engine.TensorSubtract(
                engine.TensorNarrow(image, axis, 1, extent - 1), engine.TensorNarrow(image, axis, 0, extent - 1)));
            return engine.TensorAbs(engine.TensorSubtract(Gradient(y), Gradient(z)));
        }

        // Horizontal gradients are the paper's second term; omitting it leaves blur in j unpenalised.
        var parts = new List<Tensor<T>>(2);
        if (height > 1)
        {
            parts.Add(engine.Reshape(GradientGap(0), new[] { (height - 1) * width }));
        }

        if (width > 1)
        {
            parts.Add(engine.Reshape(GradientGap(1), new[] { height * (width - 1) }));
        }

        if (parts.Count == 0)
        {
            return 0.0;
        }

        return MeanPower(engine, engine.TensorConcatenate(parts.ToArray(), 0), lambda);
    }

    /// <summary>
    /// <c>L_img = L_p + L_gdl</c>: the image-space loss, pixel plus gradient difference.
    /// </summary>
    public static double Image(Vector<T> predicted, Vector<T> target, int height, int width)
        => Pixel(predicted, target) + GradientDifference(predicted, target, height, width);

    /// <summary>
    /// <c>L_GAN = -log D([x_1:t, G(x_1:t)])</c>, the generator's NON-saturating adversarial term.
    /// </summary>
    /// <param name="discriminatorOnGenerated">
    /// D applied to the observed frames concatenated with the GENERATED continuation.
    /// </param>
    /// <remarks>
    /// Clamped away from zero so a discriminator that is certain the sample is fake yields a large
    /// finite penalty rather than infinity.
    /// </remarks>
    public static double GeneratorAdversarial(double discriminatorOnGenerated)
    {
        double d = Math.Min(Math.Max(discriminatorOnGenerated, 1e-12), 1.0 - 1e-12);
        return -Math.Log(d);
    }

    /// <summary>
    /// <c>L_disc = -log D([x, real]) - log(1 - D([x, fake]))</c>, which the discriminator MINIMISES.
    /// </summary>
    /// <remarks>
    /// Stated as a minimisation, matching the paper's sign convention — the negative log-likelihood of
    /// correct classification. Confident and correct gives a small value.
    /// </remarks>
    public static double DiscriminatorLoss(double discriminatorOnReal, double discriminatorOnGenerated)
    {
        double real = Math.Min(Math.Max(discriminatorOnReal, 1e-12), 1.0 - 1e-12);
        double fake = Math.Min(Math.Max(discriminatorOnGenerated, 1e-12), 1.0 - 1e-12);
        return -Math.Log(real) - Math.Log(1.0 - fake);
    }

    /// <summary>
    /// <c>L = alpha * L_img + beta * L_GAN</c>.
    /// </summary>
    /// <param name="imageLoss">L_img.</param>
    /// <param name="adversarialLoss">L_GAN.</param>
    /// <param name="adversarialWeight">
    /// beta. Defaults to the paper's KTH value of 0.02; UCF-101 uses 0.001.
    /// </param>
    public static double Total(
        double imageLoss, double adversarialLoss, double adversarialWeight = AdversarialWeightKth)
        => (ImageWeight * imageLoss) + (adversarialWeight * adversarialLoss);
}
