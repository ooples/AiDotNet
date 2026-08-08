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
            return recon;
        }

        var weighted = Engine.TensorMultiplyScalar(AsScalar(kl), NumOps.FromDouble(_klWeight));
        return Engine.TensorAdd(AsScalar(recon), weighted);
    }

    /// <summary>
    /// Reduces a tensor to a rank-1 single-element scalar so the two terms are addable regardless of
    /// how each was reduced.
    /// </summary>
    private Tensor<T> AsScalar(Tensor<T> value)
    {
        if (value.Shape.Length == 1 && value.Shape[0] == 1) return value;

        var axes = Enumerable.Range(0, value.Shape.Length).ToArray();
        var reduced = Engine.ReduceSum(value, axes, keepDims: false);
        return reduced.Shape.Length == 0 ? Engine.Reshape(reduced, [1]) : reduced;
    }

    /// <inheritdoc/>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
        => _reconstruction.CalculateLoss(predicted, actual);

    /// <inheritdoc/>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
        => _reconstruction.CalculateDerivative(predicted, actual);
}
