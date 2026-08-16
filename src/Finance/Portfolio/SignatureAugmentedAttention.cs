using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Portfolio;

/// <summary>
/// SIT's signature-augmented attention bias (Hwang and Zohren, arXiv:2510.03129):
/// <c>Logits = QK^T / sqrt(d_k) + gamma * B</c>, where B carries pairwise lead-lag evidence.
/// </summary>
/// <remarks>
/// <para>
/// The distinguishing choice is WHERE the signature information enters. Prior signature work feeds
/// signatures in as input features or uses signature kernels OUTSIDE the attention mechanism; SIT
/// injects cross-asset signature evidence as a dynamic, query-conditioned bias INSIDE attention, so
/// pairwise signed-area evidence modulates which assets attend to which at each decision point.
/// </para>
/// <para>
/// The bias is query-conditioned rather than a fixed lookup:
/// <c>b_{h,j,l} = &lt;(q^dyn_j)_h, (beta_{j,l})_h&gt;</c>, with <c>q^dyn_j = MLP_q(x_j)</c> and
/// <c>beta_{j,l} = MLP_beta(c_{j,l})</c>. A static bias built from the cross-signature alone could not
/// let the current market state decide how much a given lead-lag relationship matters right now.
/// </para>
/// <para>
/// <b>The bias is not symmetrized.</b> Lead-lag is directional and the underlying signed area is
/// antisymmetric, so <c>b_{j,l}</c> and <c>b_{l,j}</c> are meant to differ — symmetrizing would
/// reduce the signal to something a correlation could already express and discard the asymmetry the
/// method is built on.
/// </para>
/// <para><b>For Beginners:</b> Attention decides how much each asset should "look at" each other
/// asset. This nudges those decisions using evidence about which asset tends to move first.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class SignatureAugmentedAttention<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private double _gammaLogit;

    /// <summary>
    /// Gets the effective bias gate <c>gamma = softplus(gammaHat)</c>.
    /// </summary>
    /// <remarks>
    /// Softplus, not a raw parameter: gamma scales how much the signature evidence counts, and a
    /// NEGATIVE gate would invert every lead-lag relationship, making the model attend away from
    /// assets the evidence says it should attend to. Softplus keeps it strictly positive while still
    /// allowing it to approach zero, so the model can learn to ignore the bias but never to flip it.
    /// </remarks>
    public double Gamma => Softplus(_gammaLogit);

    /// <summary>Gets or sets the raw pre-softplus gate parameter.</summary>
    public double GammaLogit
    {
        get => _gammaLogit;
        set
        {
            // Non-finite, not just NaN. Softplus(PositiveInfinity) takes the x > 30 branch and
            // returns infinity, so ApplyBias then computes gamma * bias -- infinity for a non-zero
            // bias, NaN for a zero one. The gate is documented as strictly positive and finite in
            // effect, and infinity satisfies neither.
            if (double.IsNaN(value) || double.IsInfinity(value))
                throw new ArgumentOutOfRangeException(nameof(value), value,
                    "gammaHat must be a finite value.");
            _gammaLogit = value;
        }
    }

    /// <summary>Creates the bias module.</summary>
    /// <param name="gammaLogit">
    /// Initial pre-softplus gate. Default 0, giving <c>gamma = softplus(0) = ln 2 ≈ 0.693</c> — the
    /// signature evidence starts contributing at a moderate strength rather than switched off, so the
    /// gradient can move it in either direction from the first step.
    /// </param>
    public SignatureAugmentedAttention(double gammaLogit = 0.0)
    {
        GammaLogit = gammaLogit;
    }

    /// <summary>Numerically stable softplus, positive for every finite input.</summary>
    /// <remarks>
    /// <para>
    /// Three branches, each guarding a different failure of the naive
    /// <c>log(1 + exp(x))</c>:
    /// </para>
    /// <list type="bullet">
    /// <item>Large x: <c>exp(x)</c> overflows to infinity, while softplus is asymptotically x.</item>
    /// <item>Very negative x: <c>1 + exp(x)</c> rounds to exactly 1.0, so the log returns exactly
    /// ZERO and the supposedly-positive gate reaches zero. Since <c>log(1 + y) ≈ y</c> for tiny y,
    /// returning <c>exp(x)</c> keeps it strictly positive and representable.</item>
    /// <item>Otherwise the direct form is accurate.</item>
    /// </list>
    /// <para>
    /// The middle case matters here specifically because the gate must never reach zero-or-below: at
    /// exactly zero the bias is switched off with no gradient path back, so a gate that underflowed
    /// could never recover.
    /// </para>
    /// </remarks>
    public static double Softplus(double x)
    {
        if (x > 30.0) return x;
        if (x < -30.0) return Math.Exp(x);
        return Math.Log(1.0 + Math.Exp(x));
    }

    /// <summary>
    /// Computes the per-head bias matrix
    /// <c>b_{h,j,l} = &lt;(q^dyn_j)_h, (beta_{j,l})_h&gt;</c>.
    /// </summary>
    /// <param name="dynamicQueries">
    /// <c>MLP_q</c> output, shaped <c>[assets, heads, dBeta]</c>.
    /// </param>
    /// <param name="relationalEmbeddings">
    /// <c>MLP_beta</c> output over cross-signature features, shaped
    /// <c>[assets, assets, heads, dBeta]</c>.
    /// </param>
    /// <returns>The bias, shaped <c>[heads, assets, assets]</c>.</returns>
    public Tensor<T> ComputeBias(Tensor<T> dynamicQueries, Tensor<T> relationalEmbeddings)
    {
        if (dynamicQueries == null) throw new ArgumentNullException(nameof(dynamicQueries));
        if (relationalEmbeddings == null) throw new ArgumentNullException(nameof(relationalEmbeddings));
        if (dynamicQueries.Shape.Length != 3)
            throw new ArgumentException(
                $"dynamicQueries must be [assets, heads, dBeta]; got rank {dynamicQueries.Shape.Length}.",
                nameof(dynamicQueries));
        if (relationalEmbeddings.Shape.Length != 4)
            throw new ArgumentException(
                $"relationalEmbeddings must be [assets, assets, heads, dBeta]; got rank {relationalEmbeddings.Shape.Length}.",
                nameof(relationalEmbeddings));

        int assets = dynamicQueries.Shape[0];
        int heads = dynamicQueries.Shape[1];
        int dBeta = dynamicQueries.Shape[2];

        if (relationalEmbeddings.Shape[0] != assets || relationalEmbeddings.Shape[1] != assets
            || relationalEmbeddings.Shape[2] != heads || relationalEmbeddings.Shape[3] != dBeta)
            throw new ArgumentException(
                $"relationalEmbeddings must be [{assets}, {assets}, {heads}, {dBeta}] to match dynamicQueries.",
                nameof(relationalEmbeddings));

        var bias = new Tensor<T>(new[] { heads, assets, assets });

        for (int h = 0; h < heads; h++)
        {
            for (int j = 0; j < assets; j++)
            {
                int qOffset = ((j * heads) + h) * dBeta;
                for (int l = 0; l < assets; l++)
                {
                    int bOffset = ((((j * assets) + l) * heads) + h) * dBeta;
                    double dot = 0.0;
                    for (int d = 0; d < dBeta; d++)
                        dot += NumOps.ToDouble(dynamicQueries[qOffset + d]) *
                               NumOps.ToDouble(relationalEmbeddings[bOffset + d]);

                    bias[((h * assets) + j) * assets + l] = NumOps.FromDouble(dot);
                }
            }
        }

        return bias;
    }

    /// <summary>
    /// Adds the gated bias to attention logits: <c>logits + gamma * bias</c>.
    /// </summary>
    /// <param name="logits">Scaled dot-product logits, shaped <c>[heads, assets, assets]</c>.</param>
    /// <param name="bias">Bias from <see cref="ComputeBias"/>, same shape.</param>
    /// <remarks>
    /// Applied BEFORE the softmax, which is what makes it a bias on the attention distribution rather
    /// than a rescaling of its output. Adding it after the softmax would break the normalization and
    /// no longer express a preference over keys.
    /// </remarks>
    public Tensor<T> ApplyBias(Tensor<T> logits, Tensor<T> bias)
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (bias == null) throw new ArgumentNullException(nameof(bias));
        if (logits.Length != bias.Length)
            throw new ArgumentException(
                $"logits and bias must have the same element count; got {logits.Length} and {bias.Length}.",
                nameof(bias));

        double gamma = Gamma;
        var result = new Tensor<T>(logits.Shape.ToArray());
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.FromDouble(NumOps.ToDouble(logits[i]) + (gamma * NumOps.ToDouble(bias[i])));

        return result;
    }

    /// <summary>
    /// Scaled dot-product logits <c>QK^T / sqrt(d_k)</c> for one head.
    /// </summary>
    /// <param name="queries">Queries, shaped <c>[assets, dK]</c>.</param>
    /// <param name="keys">Keys, shaped <c>[assets, dK]</c>.</param>
    /// <remarks>
    /// The <c>1/sqrt(d_k)</c> scaling is not decorative: without it the dot products grow with dK and
    /// push the softmax into saturation, where gradients vanish.
    /// </remarks>
    public Tensor<T> ScaledDotProductLogits(Tensor<T> queries, Tensor<T> keys)
    {
        if (queries == null) throw new ArgumentNullException(nameof(queries));
        if (keys == null) throw new ArgumentNullException(nameof(keys));
        if (queries.Shape.Length != 2 || keys.Shape.Length != 2)
            throw new ArgumentException("queries and keys must be [assets, dK].");
        if (queries.Shape[1] != keys.Shape[1])
            throw new ArgumentException(
                $"Key dimension mismatch: {queries.Shape[1]} vs {keys.Shape[1]}.", nameof(keys));

        int nq = queries.Shape[0];
        int nk = keys.Shape[0];
        int dk = queries.Shape[1];

        // A zero key dimension passes the checks above -- both ranks are 2 and the dimensions match --
        // but then 1 / sqrt(0) is infinity and every logit becomes 0 * infinity = NaN.
        if (dk <= 0)
            throw new ArgumentException(
                $"Key dimension must be positive; got {dk}.", nameof(queries));

        double scale = 1.0 / Math.Sqrt(dk);

        var logits = new Tensor<T>(new[] { nq, nk });
        for (int i = 0; i < nq; i++)
        {
            for (int j = 0; j < nk; j++)
            {
                double dot = 0.0;
                for (int d = 0; d < dk; d++)
                    dot += NumOps.ToDouble(queries[(i * dk) + d]) * NumOps.ToDouble(keys[(j * dk) + d]);
                logits[(i * nk) + j] = NumOps.FromDouble(dot * scale);
            }
        }

        return logits;
    }
}
