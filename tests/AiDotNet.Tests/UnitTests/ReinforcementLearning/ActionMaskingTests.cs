using System;
using System.Linq;
using AiDotNet.ReinforcementLearning;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// <see cref="ActionMasking"/> — the shared primitives that keep a legal-action mask CORRECT across the three
/// structurally different selection paths.
///
/// <para>Each test below targets a failure mode that real masking implementations ship. They are written as
/// falsifications: remove the guard and the named test goes red, which is the only way a correctness helper
/// earns trust.</para>
/// </summary>
public class ActionMaskingTests
{
    /// <summary>
    /// Obtained the way the library itself does. There is no concrete <c>DoubleOperations</c> type to
    /// construct — <c>MathHelper.GetNumericOperations&lt;T&gt;()</c> is the accessor every consumer uses.
    /// </summary>
    private static readonly INumericOperations<double> Ops = MathHelper.GetNumericOperations<double>();

    private static Vector<double> Vec(params double[] values)
    {
        var v = new Vector<double>(values.Length);
        for (var i = 0; i < values.Length; i++)
        {
            v[i] = values[i];
        }

        return v;
    }

    // ---- Validate ------------------------------------------------------------------------------

    [Fact]
    [Trait("category", "unit")]
    public void A_null_mask_means_no_restriction()
    {
        Assert.Null(ActionMasking.Validate(null, 3));
    }

    /// <summary>
    /// A wrong-length mask means the caller and the environment disagree about the action space. Ignoring it
    /// would mask the WRONG actions — silently trading something the account is not cleared for — so it throws.
    /// </summary>
    [Theory]
    [Trait("category", "unit")]
    [InlineData(2)]
    [InlineData(4)]
    public void A_mask_of_the_wrong_length_is_refused(int maskLength)
    {
        var mask = Enumerable.Repeat(true, maskLength).ToArray();

        var ex = Assert.Throws<ArgumentException>(() => ActionMasking.Validate(mask, 3));

        Assert.Contains("wrong length", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// THE LOAD-BEARING GUARD. An all-false mask is not merely empty — it is numerically fatal downstream.
    /// A softmax that subtracts its max logit computes <c>exp(-inf - -inf)</c> for every entry, which is
    /// <c>NaN</c>, so the policy distribution becomes NaN rather than "no legal action". Measured directly:
    /// <c>Math.Exp(double.NegativeInfinity - double.NegativeInfinity) == NaN</c>.
    ///
    /// <para>Refusing here converts a silent NaN-poisoned update into a loud, locatable failure.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void An_all_masked_state_is_refused_rather_than_producing_NaN()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => ActionMasking.Validate([false, false, false], 3));

        Assert.Contains("masked out", ex.Message, StringComparison.OrdinalIgnoreCase);

        // The reason it must throw, demonstrated rather than asserted by assertion alone.
        Assert.True(double.IsNaN(Math.Exp(double.NegativeInfinity - double.NegativeInfinity)));
    }

    // ---- ArgMaxLegal ---------------------------------------------------------------------------

    /// <summary>The highest-valued entry wins when it is legal — the ordinary case.</summary>
    [Fact]
    [Trait("category", "unit")]
    public void The_masked_argmax_picks_the_best_legal_action()
    {
        Assert.Equal(1, ActionMasking.ArgMaxLegal(Vec(0.1, 0.9, 0.5), [true, true, true], Ops));
    }

    /// <summary>
    /// THE DEFECT MASKING EXISTS TO PREVENT: the globally-best action is illegal, so the best LEGAL one must
    /// win. An unmasked argmax returns index 1 here.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void The_masked_argmax_skips_an_illegal_maximum()
    {
        Assert.Equal(2, ActionMasking.ArgMaxLegal(Vec(0.1, 0.9, 0.5), [false, false, true], Ops));
    }

    /// <summary>
    /// Masking by LOWERING a value rather than skipping it is the subtler version of the same bug: zero is an
    /// ordinary score, and an illegal action scored 0 still beats legal ones scored negative. Skipping indices
    /// makes the result independent of how negative "masked" happens to be.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void An_illegal_action_cannot_win_against_negative_legal_values()
    {
        Assert.Equal(2, ActionMasking.ArgMaxLegal(Vec(5.0, 3.0, -7.0), [false, false, true], Ops));
    }

    // ---- MaskLogits ----------------------------------------------------------------------------

    /// <summary>
    /// Illegal logits go to negative infinity so that <c>exp</c> drives them to exactly zero, leaving a TRUE
    /// distribution over the legal set. This is what makes a sampled log-probability meaningful — and for PPO
    /// that log-probability is the denominator of the importance ratio, so an unnormalised distribution makes
    /// the update wrong rather than merely worse.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Masked_logits_softmax_to_zero_and_the_legal_set_still_sums_to_one()
    {
        var masked = ActionMasking.MaskLogits(Vec(3.0, 2.0, 1.0, 0.5), [false, true, false, true], Ops);

        Assert.True(double.IsNegativeInfinity(masked[0]));
        Assert.True(double.IsNegativeInfinity(masked[2]));

        var probs = Softmax(masked);
        Assert.Equal(0.0, probs[0], 12);
        Assert.Equal(0.0, probs[2], 12);
        Assert.Equal(1.0, probs[1] + probs[3], 12);
        Assert.All(probs, p => Assert.False(double.IsNaN(p)));
    }

    /// <summary>A null mask leaves the logits untouched, so the unmasked path costs nothing.</summary>
    [Fact]
    [Trait("category", "unit")]
    public void Logits_are_unchanged_when_there_is_no_mask()
    {
        var original = Vec(3.0, 2.0, 1.0);

        Assert.Same(original, ActionMasking.MaskLogits(original, null, Ops));
    }

    // ---- MaskProbabilities ---------------------------------------------------------------------

    /// <summary>
    /// THE RENORMALISATION, which is the whole point for a network that emits probabilities directly and has
    /// no logit stage to mask. Zeroing without renormalising leaves the entries summing to less than one, and
    /// a cumulative-sum sampler drawing r in [0,1) then falls through its loop and returns the LAST index —
    /// legal or not.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Masked_probabilities_are_renormalised_to_sum_to_one()
    {
        var masked = ActionMasking.MaskProbabilities(Vec(0.5, 0.3, 0.2), [false, true, true], Ops);

        Assert.Equal(0.0, masked[0], 12);
        Assert.Equal(0.6, masked[1], 12);   // 0.3 / 0.5
        Assert.Equal(0.4, masked[2], 12);   // 0.2 / 0.5
        Assert.Equal(1.0, masked[0] + masked[1] + masked[2], 12);
    }

    /// <summary>
    /// The fall-through this prevents, demonstrated against the sampler shape the agents actually use: with
    /// UNRENORMALISED masked probabilities, a draw above the reduced total walks off the end and returns the
    /// last index — which here is illegal.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Renormalising_prevents_the_cumulative_sampler_falling_through_to_an_illegal_action()
    {
        double[] naive = [0.0, 0.3, 0.0];      // zeroed but NOT renormalised; total 0.3
        Assert.Equal(2, CumulativeSample(naive, draw: 0.9));   // falls through to the last (illegal) index

        var correct = ActionMasking.MaskProbabilities(Vec(0.5, 0.3, 0.2), [false, true, false], Ops);
        Assert.Equal(1, CumulativeSample([correct[0], correct[1], correct[2]], draw: 0.9));
    }

    /// <summary>
    /// A degenerate policy output — every legal entry zero — must still yield a usable distribution. Left at
    /// zero it has the same fall-through failure, so the legal set is made uniform.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void A_zero_sum_legal_set_becomes_uniform_rather_than_unusable()
    {
        var masked = ActionMasking.MaskProbabilities(Vec(0.0, 0.0, 0.7), [true, true, false], Ops);

        Assert.Equal(0.5, masked[0], 12);
        Assert.Equal(0.5, masked[1], 12);
        Assert.Equal(0.0, masked[2], 12);
    }

    // ---- RandomLegal ---------------------------------------------------------------------------

    /// <summary>
    /// THE SITE MOST OFTEN MISSED. An epsilon-greedy agent that masks only its greedy branch still explores
    /// into illegal actions at rate epsilon — and early in training epsilon is near one, so nearly every action
    /// taken is one the environment cannot honour.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Random_exploration_never_selects_an_illegal_action()
    {
        var random = new Random(17);
        bool[] mask = [false, true, false, false, true];

        for (var i = 0; i < 2000; i++)
        {
            var picked = ActionMasking.RandomLegal(random, mask, mask.Length);
            Assert.True(mask[picked], $"exploration selected illegal action {picked}");
        }
    }

    /// <summary>
    /// And it must reach EVERY legal action — a "masked" sampler that always returned the first legal index
    /// would pass the test above while destroying exploration entirely.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Random_exploration_still_covers_the_whole_legal_set()
    {
        var random = new Random(17);
        bool[] mask = [false, true, false, false, true];
        var seen = new bool[mask.Length];

        for (var i = 0; i < 2000; i++)
        {
            seen[ActionMasking.RandomLegal(random, mask, mask.Length)] = true;
        }

        Assert.True(seen[1] && seen[4], "exploration did not reach every legal action");
    }

    /// <summary>With no mask, exploration is the ordinary uniform draw over the whole space.</summary>
    [Fact]
    [Trait("category", "unit")]
    public void Unmasked_exploration_covers_the_whole_action_space()
    {
        var random = new Random(17);
        var seen = new bool[4];

        for (var i = 0; i < 2000; i++)
        {
            seen[ActionMasking.RandomLegal(random, null, 4)] = true;
        }

        Assert.All(seen, s => Assert.True(s));
    }

    // ---- validation at every entry point -------------------------------------------------------

    /// <summary>
    /// The production agents happen to call <see cref="ActionMasking.Validate"/> before reaching a helper, but
    /// all five methods are public, so that ordering is a CONVENTION and nothing enforces it on an external
    /// caller. Each helper therefore validates for itself.
    ///
    /// <para>An all-false mask is the dangerous shape: before this, <c>ArgMaxLegal</c> and <c>RandomLegal</c>
    /// both fell back to index <c>0</c> — a silently ILLEGAL action, indistinguishable downstream from a
    /// deliberate one, which is precisely the failure this class exists to prevent.</para>
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Every_helper_refuses_an_all_masked_state()
    {
        var none = new[] { false, false, false };

        Assert.Throws<InvalidOperationException>(() => ActionMasking.ArgMaxLegal(Vec(1.0, 2.0, 3.0), none, Ops));
        Assert.Throws<InvalidOperationException>(() => ActionMasking.MaskLogits(Vec(1.0, 2.0, 3.0), none, Ops));
        Assert.Throws<InvalidOperationException>(() => ActionMasking.MaskProbabilities(Vec(0.2, 0.3, 0.5), none, Ops));
        Assert.Throws<InvalidOperationException>(() => ActionMasking.RandomLegal(new Random(1), none, 3));
    }

    /// <summary>
    /// A mask shorter than the action space used to throw <see cref="IndexOutOfRangeException"/> from deep
    /// inside a loop, or — for <c>RandomLegal</c>, which iterates the MASK rather than the space — not
    /// throw at all while quietly making the trailing actions unreachable. Both are now the same explicit
    /// refusal the agents already got: the caller and the environment disagree about the action space.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Every_helper_refuses_a_wrong_length_mask()
    {
        var tooShort = new[] { true, true };

        Assert.Throws<ArgumentException>(() => ActionMasking.ArgMaxLegal(Vec(1.0, 2.0, 3.0), tooShort, Ops));
        Assert.Throws<ArgumentException>(() => ActionMasking.MaskLogits(Vec(1.0, 2.0, 3.0), tooShort, Ops));
        Assert.Throws<ArgumentException>(() => ActionMasking.MaskProbabilities(Vec(0.2, 0.3, 0.5), tooShort, Ops));
        Assert.Throws<ArgumentException>(() => ActionMasking.RandomLegal(new Random(1), tooShort, 3));
    }

    /// <summary>
    /// Validating on entry must not disturb the null path, which is the ordinary unmasked case every existing
    /// caller in the library takes.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Validation_on_entry_leaves_the_unmasked_path_alone()
    {
        var logits = Vec(1.0, 5.0, 3.0);

        Assert.Equal(1, ActionMasking.ArgMaxLegal(logits, null, Ops));
        Assert.Same(logits, ActionMasking.MaskLogits(logits, null, Ops));
        Assert.Same(logits, ActionMasking.MaskProbabilities(logits, null, Ops));
        Assert.InRange(ActionMasking.RandomLegal(new Random(3), null, 3), 0, 2);
    }

    // ---- helpers -------------------------------------------------------------------------------

    /// <summary>The max-subtracting softmax the agents use, so the -inf behaviour is tested as it is shipped.</summary>
    private static double[] Softmax(Vector<double> logits)
    {
        var max = double.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        var exps = new double[logits.Length];
        double sum = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            exps[i] = Math.Exp(logits[i] - max);
            sum += exps[i];
        }

        for (var i = 0; i < exps.Length; i++)
        {
            exps[i] /= sum;
        }

        return exps;
    }

    /// <summary>The cumulative-sum sampler shape both FinancialPPOAgent and FinancialA2CAgent use.</summary>
    private static int CumulativeSample(double[] probabilities, double draw)
    {
        double cumulative = 0;
        for (var i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (draw < cumulative)
            {
                return i;
            }
        }

        return probabilities.Length - 1;
    }
}
