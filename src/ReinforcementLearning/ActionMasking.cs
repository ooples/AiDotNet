using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

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
internal static class ActionMasking
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
    /// <para>Skips illegal indices outright rather than lowering their value, so the result cannot depend on
    /// how negative "very negative" happens to be for the numeric type in play.</para>
    ///
    /// <para>Validates on entry. This is a public entry point, so it cannot assume a caller reached it through
    /// <see cref="Validate"/> first: an all-false or wrong-length mask has to be refused HERE, or the only
    /// thing standing between an external caller and an illegal action is a call-order convention.</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The mask length does not match <paramref name="values"/>.</exception>
    /// <exception cref="InvalidOperationException">Every action is masked out.</exception>
    public static int ArgMaxLegal<T>(Vector<T> values, bool[]? mask, INumericOperations<T> ops)
    {
        mask = Validate(mask, values.Length);

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

        // Validate guarantees at least one in-range legal index, so the loop always assigned.
        return best;
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
    ///
    /// <para>Validates on entry, for the reason given on <see cref="ArgMaxLegal"/>. An all-false mask here
    /// would drive EVERY logit to negative infinity, and the softmax downstream would divide zero by zero.</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The mask length does not match <paramref name="logits"/>.</exception>
    /// <exception cref="InvalidOperationException">Every action is masked out.</exception>
    public static Vector<T> MaskLogits<T>(Vector<T> logits, bool[]? mask, INumericOperations<T> ops)
    {
        mask = Validate(mask, logits.Length);
        if (mask is null)
        {
            return logits;
        }

        var blocked = NegativeInfinity(ops);
        var masked = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            masked[i] = mask[i] ? logits[i] : blocked;
        }

        return masked;
    }

    /// <summary>Refuses finite-only numeric types before a masked policy can be sampled or trained.</summary>
    internal static T NegativeInfinity<T>(INumericOperations<T> ops)
    {
        try
        {
            var value = ops.FromDouble(double.NegativeInfinity);
            if (double.IsNegativeInfinity(ops.ToDouble(value))) return value;
        }
        catch (OverflowException ex)
        {
            throw new NotSupportedException($"Masked policy logits require a numeric type supporting infinity; {typeof(T).Name} does not.", ex);
        }
        throw new NotSupportedException($"Masked policy logits require a numeric type supporting infinity; {typeof(T).Name} does not.");
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
    ///
    /// <para>Validates on entry, for the reason given on <see cref="ArgMaxLegal"/>. That is also what makes the
    /// uniform fallback below safe: it divides by the legal count, which an all-false mask would leave at
    /// zero.</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The mask length does not match <paramref name="probabilities"/>.</exception>
    /// <exception cref="InvalidOperationException">Every action is masked out.</exception>
    public static Vector<T> MaskProbabilities<T>(Vector<T> probabilities, bool[]? mask, INumericOperations<T> ops)
    {
        mask = Validate(mask, probabilities.Length);
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
    /// <para>The site most often missed. An epsilon-greedy agent that masks only its greedy branch still
    /// explores into illegal actions at rate epsilon, and early in training epsilon is near one — so almost
    /// every action taken is one the environment cannot honour.</para>
    ///
    /// <para>Validates on entry, for the reason given on <see cref="ArgMaxLegal"/>. An all-false mask used to
    /// return index <c>0</c> here — a silent illegal action, which is the exact failure this class exists to
    /// prevent, and worse than an exception because nothing downstream can tell it apart from a real choice.</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The mask length does not match <paramref name="actionSpaceSize"/>.</exception>
    /// <exception cref="InvalidOperationException">Every action is masked out.</exception>
    public static int RandomLegal(Random random, bool[]? mask, int actionSpaceSize)
    {
        mask = Validate(mask, actionSpaceSize);
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

        int target = random.Next(legalCount);
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] && target-- == 0)
            {
                return i;
            }
        }

        throw new InvalidOperationException(
            "Unreachable: the draw is bounded by the legal count counted from the same mask.");
    }

    /// <summary>
    /// <see cref="MaskProbabilities{T}(Vector{T}, bool[], INumericOperations{T})"/> for a policy that has
    /// already reduced its distribution to <see cref="double"/>.
    /// </summary>
    /// <remarks>
    /// <para>An actor that emits LOGITS should be masked with <see cref="MaskLogits{T}"/> before its softmax,
    /// which is both cheaper and exact. This overload is for the case where it cannot be: a softmax that
    /// refuses a non-finite logit as evidence of a diverged actor would reject the negative infinities
    /// <see cref="MaskLogits{T}"/> introduces, so the mask has to be applied to the distribution instead.
    /// The two are equivalent — renormalising a softmax over the legal set gives the same numbers as a
    /// softmax over the legal logits — so nothing is lost by masking on this side of it.</para>
    ///
    /// <para>Delegates to the generic implementation rather than restating it, so the zeroing, the
    /// renormalisation and the uniform fallback for a zero-sum legal set cannot drift between the two.</para>
    /// </remarks>
    /// <exception cref="ArgumentException">The mask length does not match <paramref name="probabilities"/>.</exception>
    /// <exception cref="InvalidOperationException">Every action is masked out.</exception>
    public static double[] MaskProbabilities(double[] probabilities, bool[]? mask)
    {
        Guard.NotNull(probabilities);
        return MaskProbabilities(
            new Vector<double>(probabilities), mask, MathHelper.GetNumericOperations<double>()).ToArray();
    }

    /// <summary>
    /// <see cref="ArgMaxLegal{T}(Vector{T}, bool[], INumericOperations{T})"/> for values already reduced to
    /// <see cref="double"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The mask length does not match <paramref name="values"/>.</exception>
    /// <exception cref="InvalidOperationException">Every action is masked out.</exception>
    public static int ArgMaxLegal(double[] values, bool[]? mask)
    {
        Guard.NotNull(values);
        return ArgMaxLegal(new Vector<double>(values), mask, MathHelper.GetNumericOperations<double>());
    }
}
