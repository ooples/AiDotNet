using System.Collections.Generic;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// The environment is the authority on which actions are legal, and it publishes that in two places: the
/// <see cref="IMaskedActionEnvironment{T}.LegalActionMask"/> property, and a mirror in the info dictionary
/// returned by <c>Step</c>.
///
/// <para><b>Why the mirror needs its own coverage.</b> The property is what an agent holding the environment
/// reads; the info entry is what a consumer holding only a step result reads — a replay buffer, an offline
/// dataset writer, a logged trajectory. Those are exactly the consumers that cannot call back into the
/// environment, so if the mirror silently stops being written, masking survives in online training and
/// disappears everywhere the trajectory is replayed. Nothing else in this suite touches that path.</para>
/// </summary>
public class MaskedEnvironmentInfoTests
{
    private const int ActionCount = 3;

    /// <summary>A mask an incorrect implementation cannot reproduce by accident: middle action forbidden.</summary>
    private static bool[] Mask => [true, false, true];

    [Fact]
    [Trait("category", "unit")]
    public void Step_mirrors_the_legal_action_mask_into_the_info_dictionary()
    {
        var mask = Mask;
        var environment = new MaskPublishingEnvironment(mask);
        environment.Reset();

        var info = environment.Step(new Vector<double>(ActionCount)).Info;

        Assert.True(
            info.ContainsKey(ActionMasking.ActionMaskKey),
            "an environment that restricts actions must mirror the mask into the step info dictionary");
        Assert.Equal(mask, Assert.IsType<bool[]>(info[ActionMasking.ActionMaskKey]));
    }

    /// <summary>
    /// Pins the literal key. The value of a convention is that it is the same word everywhere — PettingZoo,
    /// Shimmy and OpenSpiel all spell it <c>action_mask</c> — so renaming the constant would silently break
    /// interoperability with every one of them while leaving the mirror test above green.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void The_info_key_is_the_conventional_action_mask_spelling()
    {
        Assert.Equal("action_mask", ActionMasking.ActionMaskKey);
    }

    /// <summary>
    /// The entry is ABSENT, not present-and-null, when the environment places no restriction. A consumer
    /// testing <c>ContainsKey</c> would otherwise read a null as "every action illegal".
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Step_omits_the_key_when_the_environment_restricts_nothing()
    {
        var environment = new MaskPublishingEnvironment(legalActionMask: null);
        environment.Reset();

        var info = environment.Step(new Vector<double>(ActionCount)).Info;

        Assert.False(info.ContainsKey(ActionMasking.ActionMaskKey));
    }

    /// <summary>
    /// The property is the authority and the info entry is a mirror, so the two must agree. They are read at
    /// different moments, and an implementation that recomputed the mask between them could diverge.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void The_mirrored_mask_agrees_with_the_property()
    {
        var environment = new MaskPublishingEnvironment(Mask);
        environment.Reset();

        var info = environment.Step(new Vector<double>(ActionCount)).Info;

        Assert.Equal(
            ((IMaskedActionEnvironment<double>)environment).LegalActionMask,
            (bool[])info[ActionMasking.ActionMaskKey]);
    }

    /// <summary>
    /// A minimal discrete trading environment that publishes a fixed mask. Deliberately the smallest thing that
    /// exercises the base class's <c>Step</c>: the mirror lives in <see cref="TradingEnvironment{T}"/>, so any
    /// concrete subclass would only add noise between the test and the behaviour under test.
    /// </summary>
    private sealed class MaskPublishingEnvironment : TradingEnvironment<double>
    {
        private readonly bool[]? _legalActionMask;

        public MaskPublishingEnvironment(bool[]? legalActionMask)
            : base(RisingPrices(steps: 8), windowSize: 2, initialCapital: 10_000.0)
        {
            _legalActionMask = legalActionMask;
        }

        public override int ActionSpaceSize => ActionCount;

        public override bool IsContinuousActionSpace => false;

        public override bool[]? LegalActionMask => _legalActionMask;

        /// <summary>No-op: this environment exists to exercise the info dictionary, not the book-keeping.</summary>
        protected override void ApplyAction(Vector<double> action, Vector<double> prices)
        {
        }

        /// <summary>One asset, strictly positive and rising, shaped [time, assets] as the base class requires.</summary>
        private static Tensor<double> RisingPrices(int steps)
        {
            var prices = new double[steps];
            for (int i = 0; i < steps; i++)
            {
                prices[i] = 100.0 + i;
            }

            return new Tensor<double>(prices, [steps, 1]);
        }
    }
}
