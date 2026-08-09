using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.LossFunctions;

/// <summary>
/// FactorVAE's training objective: return reconstruction plus the KL divergence between the
/// posterior and prior factor distributions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> this model learns hidden "factors" that explain how a group of stocks move.
/// During training it is allowed to peek at the future returns to work out what the factors must have
/// been (the <i>posterior</i>). At prediction time the future is unknown, so it has to guess the factors
/// from observable data alone (the <i>prior</i>). The KL term is what forces those two to agree, so the
/// guess made without the future is close to the answer derived with it. Without it, the model would
/// learn factors it can only recover by cheating, and would be useless at prediction time.
/// </para>
/// <para>
/// <b>Definition</b> (Duan et al., AAAI 2022, section on prior-posterior learning):
/// <c>L = L_recon(y_hat, y) + gamma * KL(q(z | y, x) || p(z | x))</c>, where <c>q</c> is the posterior
/// factor distribution inferred with future returns and <c>p</c> the prior inferred without them. For
/// diagonal Gaussians the KL has the closed form
/// <c>sum over factors of [ log(s_p/s_q) + (s_q^2 + (m_q - m_p)^2) / (2 s_p^2) - 1/2 ]</c>.
/// </para>
/// <para>
/// The KL is supplied by the model through a callback rather than recomputed here, because only the
/// model knows its own prior/posterior heads. The callback returns a tape-connected scalar so the KL
/// gradient reaches both heads; returning a detached value would leave the prior untrained and silently
/// reduce this to plain reconstruction.
/// </para>
/// </remarks>
public class FactorVAEObjective<T> : LossFunctionBase<T>
{
    private readonly LossFunctionBase<T> _reconstruction;
    private readonly Func<Tensor<T>?> _klProvider;
    private readonly double _klWeight;

    /// <summary>
    /// Creates the objective.
    /// </summary>
    /// <param name="reconstruction">Reconstruction loss on predicted vs. realized returns.</param>
    /// <param name="klProvider">Returns the current step's KL divergence as a tape-connected scalar, or
    /// null when no posterior pass ran (e.g. during inference).</param>
    /// <param name="klWeight">Weight on the KL term (the paper's gamma).</param>
    public FactorVAEObjective(
        LossFunctionBase<T> reconstruction,
        Func<Tensor<T>?> klProvider,
        double klWeight)
    {
        Guard.NotNull(reconstruction);
        Guard.NotNull(klProvider);

        _reconstruction = reconstruction;
        _klProvider = klProvider;
        _klWeight = klWeight;
    }

    /// <inheritdoc/>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var recon = _reconstruction.ComputeTapeLoss(predicted, target);

        var kl = _klProvider();
        if (kl is null || _klWeight == 0.0)
        {
            // REDUCED ON THIS PATH TOO, so the returned rank does not depend on runtime state. The
            // branch is chosen by _klProvider(), which returns null during inference and on any step
            // with no posterior pass, so an unnormalized return here made the loss rank flip between
            // steps of one run: a backward seed built for a scalar met whatever rank the
            // reconstruction loss happened to produce, and a fused compiled path that traces the
            // graph once and replays it had its traced plan invalidated.
            return ToRank0(recon);
        }

        var weighted = Engine.TensorMultiplyScalar(ToRank0(kl), NumOps.FromDouble(_klWeight));
        return Engine.TensorAdd(ToRank0(recon), weighted);
    }

    /// <summary>
    /// Fully reduces a term to the rank-0 scalar the <c>ComputeTapeLoss</c> contract requires.
    /// </summary>
    /// <remarks>
    /// This previously reshaped a fully-reduced term UP to rank-1 <c>[1]</c>, which is the opposite
    /// of the contract: a <c>[1]</c> tape root leaves the backward pass with no scalar to seed from.
    /// The KL term arrives from a caller-supplied provider rather than from a sibling loss, so the
    /// reduction itself is still needed -- only its target rank changes.
    /// </remarks>
    private Tensor<T> ToRank0(Tensor<T> value)
    {
        if (value.Shape.Length == 0) return value;

        var axes = Enumerable.Range(0, value.Shape.Length).ToArray();
        return Engine.ReduceSum(value, axes, keepDims: false);
    }

    /// <summary>
    /// Not supported. FactorVAE's objective cannot be computed on the flat-vector surface.
    /// </summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    /// <remarks>
    /// DELEGATING TO THE RECONSTRUCTION LOSS WOULD DROP THE KL TERM AND THE gamma WEIGHT, silently.
    /// That is precisely the failure this class's own remarks warn about: a detached KL value
    /// "would leave the prior untrained and silently reduce this to plain reconstruction". A caller
    /// on this surface would receive a plausible loss number while the objective they configured was
    /// not the objective being computed, with no exception and no diagnostic.
    ///
    /// FlowLoss and AdversarialLoss handle their own unsupported paths this way. If the
    /// reconstruction-only value is genuinely wanted, call the reconstruction loss directly, where
    /// the omission is visible at the call site.
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        => throw new NotSupportedException(
            $"{nameof(FactorVAEObjective<T>)} cannot be evaluated on the flat-vector surface: the KL "
            + "divergence needs the posterior tensors, which a vector does not carry. Train through "
            + "ComputeTapeLoss (TrainWithTape), or call the reconstruction loss directly if only that "
            + "term is wanted.");

    /// <summary>
    /// Not supported, for the same reason as <see cref="CalculateLoss"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
        => throw new NotSupportedException(
            $"{nameof(FactorVAEObjective<T>)} cannot produce a derivative on the flat-vector surface: "
            + "the KL term's gradient reaches the encoder through the tape. Train through "
            + "ComputeTapeLoss (TrainWithTape).");
}
