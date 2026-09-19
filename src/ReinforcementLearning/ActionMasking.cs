using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning;

/// <summary>
/// Applies a legal-action mask to the values a policy selects from.
/// </summary>
/// <remarks>
/// <para>Shared so the three distinct selection paths mask CORRECTLY and identically, rather than each agent
/// improvising. The failure modes below are the ones real implementations ship:</para>
///
/// <list type="bullet">
/// <item><b>Masking the argmax but not the exploration draw.</b> An epsilon-greedy agent that filters its
/// greedy choice while still sampling uniformly over ALL actions selects illegal actions at rate epsilon —
/// most of the time, early in training, which is exactly when the damage is done.</item>
/// <item><b>Zeroing probabilities after the softmax.</b> The remaining probabilities no longer sum to one, so
/// a cumulative-sum sampler is biased toward later indices and any cached log-probability is computed against
/// a distribution that does not exist. For PPO that log-probability is the denominator of the importance
/// ratio, so the update is silently wrong rather than merely suboptimal.</item>
/// <item><b>Masking with zero instead of negative infinity.</b> Zero is a perfectly ordinary logit; an action
/// masked to 0.0 can still win an argmax against negative logits, and still receives
/// <c>exp(0) = 1</c> weight through a softmax.</item>
/// </list>
/// </remarks>
public static class ActionMasking
{
    /// <summary>
    /// The key under which an environment publishes its legal-action mask in the info dictionary returned by
    /// <see cref="AiDotNet.Interfaces.IEnvironment{T}.Step"/>.
    /// </summary>
    /// <remarks>
    /// <para>The value matches the convention the surrounding ecosystem settled on — PettingZoo, Shimmy and
    /// OpenSpiel all name it <c>action_mask</c> — so a mask crossing into or out of this library needs no
    /// renaming.</para>
    ///
    /// <para>It lives here rather than on <see cref="AiDotNet.Interfaces.IMaskedActionEnvironment{T}"/>, where
    /// it would read more naturally, for a hard reason: this library targets <c>net471</c>, and constants in an
    /// interface compile only against a runtime providing the default-interface-members feature, which .NET
    /// Framework does not. That is a target constraint, not a style choice — moving it back breaks the
    /// <c>net471</c> build and nothing else.</para>
    /// </remarks>
    public const string ActionMaskKey = "action_mask";

    /// <summary>
    /// Validates a mask against an action-space size, returning <see langword="null"/> when it does not apply.
    /// </summary>
    /// <remarks>
    /// A null mask means "no restriction" and is the normal case. A WRONG-LENGTH mask is not: it means the
    /// caller and the environment disagree about the action space, and silently ignoring it would mask the
    /// wrong actions. That throws.
    /// </remarks>
    /// <exception cref="ArgumentException">The mask length does not match <paramref name="actionSpaceSize"/>.</exception>
    /// <exception cref="InvalidOperationException">Every action is masked out.</exception>
    public static bool[]? Validate(bool[]? mask, int actionSpaceSize)
    {
        if (mask is null)
        {
            return null;
        }

        if (mask.Length != actionSpaceSize)
        {
            throw new ArgumentException(
                $"Legal-action mask has {mask.Length} entries but the action space has {actionSpaceSize}. "
                + "A mask of the wrong length would restrict the wrong actions.",
                nameof(mask));
        }

        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                return mask;
            }
        }

        throw new InvalidOperationException(
            "Every action is masked out, so there is nothing legal to select. An environment that can reach "
            + "this state should model it as terminal or keep an explicit no-op action legal.");
    }

    /// <summary>
    /// The index of the highest-valued LEGAL entry — the masked argmax.
    /// </summary>
    /// <remarks>
    /// Skips illegal indices outright rather than lowering their value, so the result cannot depend on how
    /// negative "very negative" happens to be for the numeric type in play.
    /// </remarks>
    public static int ArgMaxLegal<T>(Vector<T> values, bool[]? mask, INumericOperations<T> ops)
    {
        int best = -1;
        for (int i = 0; i < values.Length; i++)
        {
            if (mask is not null && !mask[i])
            {
                continue;
            }

            if (best < 0 || ops.GreaterThan(values[i], values[best]))
            {
                best = i;
            }
        }

        // Only reachable when the mask is all-false, which Validate refuses; guard anyway rather than return
        // -1 into an indexer.
        return best < 0 ? 0 : best;
    }

    /// <summary>
    /// Copies <paramref name="logits"/>, driving every illegal entry to negative infinity.
    /// </summary>
    /// <remarks>
    /// <para>Apply this BEFORE the softmax. <c>exp(-inf) = 0</c>, so the illegal actions contribute nothing to
    /// the normalising sum and the resulting distribution is a true distribution over the legal set — which is
    /// what makes a sampled log-probability meaningful.</para>
    ///
    /// <para>Negative infinity rather than a large negative constant: a constant is a tuning parameter that
    /// silently stops working when logits grow, and "how negative is negative enough" is not a question a
    /// correctness property should depend on. The softmax here subtracts the max logit before exponentiating,
    /// which keeps <c>-inf</c> well-behaved.</para>
    /// </remarks>
    public static Vector<T> MaskLogits<T>(Vector<T> logits, bool[]? mask, INumericOperations<T> ops)
    {
        if (mask is null)
        {
            return logits;
        }

        var masked = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            masked[i] = mask[i] ? logits[i] : ops.FromDouble(double.NegativeInfinity);
        }

        return masked;
    }

    /// <summary>
    /// Copies <paramref name="probabilities"/>, zeroing illegal entries and RENORMALISING the remainder.
    /// </summary>
    /// <remarks>
    /// <para>For a policy whose network already emits probabilities, there is no logit stage to mask, so the
    /// mask has to be applied here — and the renormalisation is the whole point. Without it the entries no
    /// longer sum to one, and a cumulative-sum sampler drawing <c>r</c> from <c>[0, 1)</c> falls through its
    /// loop whenever <c>r</c> exceeds the reduced total, returning the LAST index regardless of whether that
    /// action is legal.</para>
    ///
    /// <para>If the legal entries sum to zero — a degenerate policy output — the legal set is made uniform
    /// rather than left at zero, because a zero-sum distribution has the same fall-through failure.</para>
    /// </remarks>
    public static Vector<T> MaskProbabilities<T>(Vector<T> probabilities, bool[]? mask, INumericOperations<T> ops)
    {
        if (mask is null)
        {
            return probabilities;
        }

        var masked = new Vector<T>(probabilities.Length);
        double total = 0;
        int legalCount = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (!mask[i])
            {
                continue;
            }

            double p = ops.ToDouble(probabilities[i]);
            if (p > 0 && !double.IsNaN(p) && !double.IsInfinity(p))
            {
                total += p;
            }

            legalCount++;
        }

        for (int i = 0; i < probabilities.Length; i++)
        {
            if (!mask[i])
            {
                masked[i] = ops.Zero;
                continue;
            }

            if (total > 0)
            {
                double p = ops.ToDouble(probabilities[i]);
                double safe = p > 0 && !double.IsNaN(p) && !double.IsInfinity(p) ? p : 0;
                masked[i] = ops.FromDouble(safe / total);
            }
            else
            {
                masked[i] = ops.FromDouble(1.0 / legalCount);
            }
        }

        return masked;
    }

    /// <summary>
    /// A uniformly-random LEGAL action index — the exploration counterpart to <see cref="ArgMaxLegal"/>.
    /// </summary>
    /// <remarks>
    /// The site most often missed. An epsilon-greedy agent that masks only its greedy branch still explores
    /// into illegal actions at rate epsilon, and early in training epsilon is near one — so almost every
    /// action taken is one the environment cannot honour.
    /// </remarks>
    public static int RandomLegal(Random random, bool[]? mask, int actionSpaceSize)
    {
        if (mask is null)
        {
            return random.Next(actionSpaceSize);
        }

        int legalCount = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                legalCount++;
            }
        }

        if (legalCount == 0)
        {
            return 0;
        }

        int target = random.Next(legalCount);
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] && target-- == 0)
            {
                return i;
            }
        }

        return 0;
    }
}
