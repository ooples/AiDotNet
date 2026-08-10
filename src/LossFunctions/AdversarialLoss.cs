using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Adversarial (GAN) loss driven by a real discriminator, computed on the autodiff tape.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> a GAN trains two networks against each other. The generator produces an image;
/// the discriminator tries to tell generated images from real ones. This loss provides both halves of
/// that game: <see cref="ComputeTapeLoss"/> is what the GENERATOR minimizes (it wants to be judged
/// real), and <see cref="ComputeDiscriminatorTapeLoss"/> is what the DISCRIMINATOR minimizes (it wants
/// to judge correctly). They are trained in alternation.
/// </para>
/// <para>
/// <b>Why this class exists.</b> A GAN term is only real if the discriminator's forward pass is part of
/// the differentiated graph — the generator learns exclusively through the discriminator's gradient.
/// A "GAN loss" that scores a discriminator output with plain arithmetic outside the tape produces a
/// plausible-looking number while sending the generator no signal at all, which is indistinguishable
/// from having no adversarial term. Every operation here goes through <c>Engine</c>, so gradients flow
/// from the verdict back into whatever produced the image.
/// </para>
/// <para>
/// <b>Formulation.</b> The non-saturating form from Goodfellow et al. (2014), which pix2pix (Isola et
/// al., 2017) describes as training "G to maximize log D(x, G(x, z))" rather than minimizing
/// <c>log(1 - D)</c> — the latter's gradient vanishes exactly when the generator is losing, which is
/// when it most needs signal. Concretely, with discriminator logits <c>d</c>:
/// </para>
/// <para>
/// generator: <c>mean(softplus(-d_fake))</c>; discriminator:
/// <c>mean(softplus(-d_real)) + mean(softplus(d_fake))</c>, where <c>softplus(x) = log(1 + e^x)</c>.
/// This is binary cross-entropy with the sigmoid folded in, evaluated through the numerically stable
/// identity <c>softplus(x) = max(x, 0) + log(1 + e^-|x|)</c> so no term ever overflows or takes the
/// logarithm of zero.
/// </para>
/// <para>
/// <b>Discriminator output convention.</b> Prefer a discriminator that emits LOGITS
/// (<c>new PatchGANDiscriminator&lt;T&gt;(applySigmoid: false)</c>). That is the same objective the paper
/// describes — its sigmoid is folded into this loss — but computed stably. If the discriminator already
/// applies a sigmoid, construct this loss with <c>discriminatorOutputsProbabilities: true</c> and the
/// probabilities are clamped away from 0 and 1 before the logarithm.
/// </para>
/// <para>
/// <b>Patch averaging.</b> A PatchGAN emits a GRID of verdicts. Averaging that grid is exactly the
/// paper's "run convolutionally across the image, averaging all responses", so no separate handling is
/// needed: the reduction here is the averaging step.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// var d = new PatchGANDiscriminator&lt;float&gt;(applySigmoid: false);
/// var adv = new AdversarialLoss&lt;float&gt;(d);
///
/// // Generator step: include adv.ComputeTapeLoss(fake, unused) in the generator objective.
/// // Discriminator step: minimize adv.ComputeDiscriminatorTapeLoss(fake, real).
/// </code>
/// </para>
/// </remarks>
public class AdversarialLoss<T> : LossFunctionBase<T>
{
    #region Constants

    /// <summary>
    /// Clamp applied to discriminator PROBABILITIES before taking a logarithm, keeping them inside
    /// (eps, 1 - eps). Only used when <c>discriminatorOutputsProbabilities</c> is true; the logit path
    /// needs no clamping because softplus is stable everywhere.
    /// </summary>
    private const double ProbabilityEpsilon = 1e-7;

    #endregion

    #region Fields

    private readonly ILayer<T> _discriminator;
    private readonly bool _outputsProbabilities;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates the adversarial loss.
    /// </summary>
    /// <param name="discriminator">The discriminator network. Any layer works;
    /// <see cref="PatchGANDiscriminator{T}"/> is the one specified by Stream-DiffVSR and pix2pix.</param>
    /// <param name="discriminatorOutputsProbabilities">True when the discriminator already applies a
    /// sigmoid. Default false, meaning it emits logits — the numerically stable arrangement.</param>
    public AdversarialLoss(ILayer<T> discriminator, bool discriminatorOutputsProbabilities = false)
    {
        Guard.NotNull(discriminator);
        _discriminator = discriminator;
        _outputsProbabilities = discriminatorOutputsProbabilities;
    }

    #endregion

    #region Properties

    /// <summary>
    /// Gets the discriminator, so a training loop can update it on its own alternating step and put it
    /// in the right training/evaluation mode.
    /// </summary>
    public ILayer<T> Discriminator => _discriminator;

    #endregion

    #region Tape Loss

    /// <inheritdoc/>
    /// <param name="predicted">The generated image or clip being judged.</param>
    /// <param name="target">Unused. The generator's adversarial term depends only on how its OWN output
    /// is judged; the ground truth enters the objective through the reconstruction and perceptual terms
    /// instead. The parameter exists because it is part of the shared loss contract.</param>
    /// <returns>The generator's adversarial loss: low when the discriminator judges the input real.</returns>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));

        var verdict = _discriminator.Forward(predicted);

        // The generator wants "real". With logits, that is softplus(-d); with probabilities, -log(d).
        return _outputsProbabilities
            ? MeanNegativeLog(verdict)
            : MeanSoftplus(Engine.TensorNegate(verdict));
    }

    /// <summary>
    /// Computes the DISCRIMINATOR's loss for one alternating step.
    /// </summary>
    /// <param name="generated">The generator's output. Detached, because this step trains only the
    /// discriminator — leaving it attached would push the generator to make its own output easier to
    /// detect, which is the opposite of the adversarial game.</param>
    /// <param name="real">A real sample.</param>
    /// <returns>The discriminator's loss: low when it judges real as real and generated as fake.</returns>
    public Tensor<T> ComputeDiscriminatorTapeLoss(Tensor<T> generated, Tensor<T> real)
    {
        if (generated is null) throw new ArgumentNullException(nameof(generated));
        if (real is null) throw new ArgumentNullException(nameof(real));

        var realVerdict = _discriminator.Forward(real);
        var fakeVerdict = _discriminator.Forward(Engine.StopGradient(generated));

        if (_outputsProbabilities)
        {
            // -log(D(real)) - log(1 - D(fake))
            var realTerm = MeanNegativeLog(realVerdict);
            var oneMinusFake = Engine.TensorAddScalar(
                Engine.TensorNegate(fakeVerdict), NumOps.One);
            var fakeTerm = MeanNegativeLog(oneMinusFake);
            return Engine.TensorAdd(realTerm, fakeTerm);
        }

        // softplus(-d_real) + softplus(d_fake)
        var realLoss = MeanSoftplus(Engine.TensorNegate(realVerdict));
        var fakeLoss = MeanSoftplus(fakeVerdict);
        return Engine.TensorAdd(realLoss, fakeLoss);
    }

    #endregion

    #region Numerics

    /// <summary>
    /// Mean of <c>softplus(x) = log(1 + e^x)</c>, evaluated as
    /// <c>max(x, 0) + log(1 + e^-|x|)</c>.
    /// </summary>
    /// <remarks>
    /// The naive form overflows for large positive x and underflows to <c>log(1) = 0</c> for large
    /// negative x, losing the gradient. The shifted identity is exact and bounded: the exponent is
    /// never positive, and <c>max(x, 0)</c> is computed as <c>(x + |x|) / 2</c> so it stays a
    /// tape-recorded expression rather than an elementwise comparison written outside the graph.
    /// </remarks>
    private Tensor<T> MeanSoftplus(Tensor<T> x)
    {
        var abs = Engine.TensorAbs(x);
        var maxZero = Engine.TensorMultiplyScalar(
            Engine.TensorAdd(x, abs), NumOps.FromDouble(0.5));
        var expNegAbs = Engine.TensorExp(Engine.TensorNegate(abs));
        var log1p = Engine.TensorLog(Engine.TensorAddScalar(expNegAbs, NumOps.One));
        return MeanAll(Engine.TensorAdd(maxZero, log1p));
    }

    /// <summary>
    /// Mean of <c>-log(clamp(p))</c>, for a discriminator that already emits probabilities.
    /// </summary>
    private Tensor<T> MeanNegativeLog(Tensor<T> probabilities)
    {
        var clamped = Engine.TensorClamp(
            probabilities,
            NumOps.FromDouble(ProbabilityEpsilon),
            NumOps.FromDouble(1.0 - ProbabilityEpsilon));
        return MeanAll(Engine.TensorNegate(Engine.TensorLog(clamped)));
    }

    /// <summary>Reduces every axis to a scalar — the PatchGAN's "averaging all responses".</summary>
    private Tensor<T> MeanAll(Tensor<T> x)
    {
        var axes = Enumerable.Range(0, x.Shape.Length).ToArray();
        return Engine.ReduceMean(x, axes, keepDims: false);
    }

    #endregion

    #region Unsupported Flat-Vector API

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">
    /// Always thrown: the discriminator is convolutional and needs the image's channel and spatial
    /// structure, which a flat vector does not carry. Use <see cref="ComputeTapeLoss"/>.
    /// </exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        throw new NotSupportedException(
            "AdversarialLoss must run a convolutional discriminator and cannot recover image shape " +
            "from a flat vector. Use ComputeTapeLoss(Tensor, Tensor).");
    }


    #endregion
}
