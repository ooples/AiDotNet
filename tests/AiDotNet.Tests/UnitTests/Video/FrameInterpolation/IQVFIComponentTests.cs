using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.FrameInterpolation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.FrameInterpolation;

/// <summary>
/// Verifies IQ-VFI's two defining mechanisms against the paper (Mengshun Hu, Kui Jiang, Zhihang Zhong,
/// Zheng Wang and Yinqiang Zheng, "IQ-VFI: Implicit Quadratic Motion Estimation for Video Frame
/// Interpolation", CVPR 2024): acceleration-modulated quadratic motion, and selectively masked
/// knowledge distillation.
/// </summary>
/// <remarks>
/// Each test targets a way the method collapses back into what it improves on: dropping the
/// acceleration term (linear VFI), getting its sign wrong (bending the trajectory outward),
/// distilling everywhere instead of where the student is worse (letting the teacher's privileged view
/// of the answer frame leak in), or distilling only one flow direction.
/// </remarks>
public class IQVFIComponentTests
{
    private static Tensor<double> Field(int n, Func<int, double> value)
    {
        var t = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) t[i] = value(i);
        return t;
    }

    // ------------------------------------------------------------ quadratic motion

    [Fact]
    public void AccelerationWeightVanishesAtBothEndpoints()
    {
        // (t^2 - t)/2 is 0 at t = 0 and t = 1, so the quadratic term bends only the INTERIOR of the
        // trajectory and never contradicts the measured endpoint flows. A weight that did not vanish
        // there would make f_0t disagree with f_01 at t = 1.
        Assert.Equal(0.0, ImplicitQuadraticMotion<double>.ForwardAccelerationWeight(0.0), 12);
        Assert.Equal(0.0, ImplicitQuadraticMotion<double>.ForwardAccelerationWeight(1.0), 12);
        Assert.Equal(0.0, ImplicitQuadraticMotion<double>.BackwardAccelerationWeight(0.0), 12);
        Assert.Equal(0.0, ImplicitQuadraticMotion<double>.BackwardAccelerationWeight(1.0), 12);
    }

    [Fact]
    public void AccelerationWeightIsNegativeInsideTheIntervalAndExtremeAtTheMidpoint()
    {
        // Always negative on (0,1), reaching -1/8 at t = 0.5. A sign error would push the trajectory
        // outward and curve it the wrong way, which is exactly the failure a "quadratic" model that
        // still looks plausible would exhibit.
        Assert.Equal(-0.125, ImplicitQuadraticMotion<double>.ForwardAccelerationWeight(0.5), 12);

        for (double t = 0.05; t < 1.0; t += 0.05)
            Assert.True(ImplicitQuadraticMotion<double>.ForwardAccelerationWeight(t) < 0.0,
                $"weight at t={t} should be negative.");

        // The midpoint is the extreme; nothing is more negative.
        double mid = ImplicitQuadraticMotion<double>.ForwardAccelerationWeight(0.5);
        for (double t = 0.05; t < 1.0; t += 0.05)
            Assert.True(ImplicitQuadraticMotion<double>.ForwardAccelerationWeight(t) >= mid - 1e-12,
                $"weight at t={t} dipped below the midpoint extreme.");
    }

    [Fact]
    public void ForwardAndBackwardWeightsAreMirrorImagesNotNegations()
    {
        // The backward weight is the forward weight at (1 - t) — the SAME sign, because acceleration is
        // a property of the trajectory, not of the direction it is traversed. Negating it instead would
        // bend the two warps in opposite directions and tear the synthesized frame.
        for (double t = 0.0; t <= 1.0; t += 0.1)
        {
            Assert.Equal(
                ImplicitQuadraticMotion<double>.ForwardAccelerationWeight(1.0 - t),
                ImplicitQuadraticMotion<double>.BackwardAccelerationWeight(t), 12);
            Assert.True(ImplicitQuadraticMotion<double>.BackwardAccelerationWeight(t) <= 1e-12,
                "the backward weight must not be positive inside [0,1].");
        }
    }

    [Fact]
    public void ZeroAccelerationReducesExactlyToLinearInterpolation()
    {
        // Acceleration is a strict GENERALIZATION of the linear model: with a = 0 the quadratic model
        // must reproduce t * f_01 exactly, so it can never do worse than the method it extends.
        var motion = new ImplicitQuadraticMotion<double>();
        var flow = Field(4, i => (i + 1) * 2.0);
        var zeroAcceleration = Field(4, _ => 0.0);

        var forward = motion.ModulateForward(flow, zeroAcceleration, 0.5);
        for (int i = 0; i < 4; i++) Assert.Equal(0.5 * flow[i], forward[i], 10);

        var backward = motion.ModulateBackward(flow, zeroAcceleration, 0.25);
        for (int i = 0; i < 4; i++) Assert.Equal(0.75 * flow[i], backward[i], 10);
    }

    [Fact]
    public void QuadraticFlowMatchesThePapersClosedForm()
    {
        // f_0t = t * f_01 + (a/2)(t^2 - t), evaluated by hand at t = 0.5 with f_01 = 10 and a = 8:
        //   0.5 * 10 + (8/2) * (0.25 - 0.5) = 5 + 4 * (-0.25) = 4.
        var motion = new ImplicitQuadraticMotion<double>();
        var flow = Field(1, _ => 10.0);
        var acceleration = Field(1, _ => 8.0);

        Assert.Equal(4.0, motion.ModulateForward(flow, acceleration, 0.5)[0], 10);

        // f_1t = (1-t) * f_10 + (a/2)((1-t)^2 - (1-t)); at t = 0.5 this is the same by symmetry.
        Assert.Equal(4.0, motion.ModulateBackward(flow, acceleration, 0.5)[0], 10);
    }

    [Fact]
    public void AtTheEndpointsTheQuadraticFlowEqualsTheMeasuredFlow()
    {
        // The consistency property that follows from the vanishing weight: at t = 1 the forward flow
        // must be exactly f_01 no matter how large the acceleration.
        var motion = new ImplicitQuadraticMotion<double>();
        var flow = Field(3, i => i + 1.0);
        var hugeAcceleration = Field(3, _ => 1000.0);

        var atOne = motion.ModulateForward(flow, hugeAcceleration, 1.0);
        for (int i = 0; i < 3; i++) Assert.Equal(flow[i], atOne[i], 8);

        var atZero = motion.ModulateForward(flow, hugeAcceleration, 0.0);
        for (int i = 0; i < 3; i++) Assert.Equal(0.0, atZero[i], 8);
    }

    [Fact]
    public void PyramidModulationAppliesAtEveryLevel()
    {
        // Coarse-to-fine: acceleration must reach every level. A model that modulated only the finest
        // level would leave the coarse trajectory linear, which is where large curvature lives.
        var motion = new ImplicitQuadraticMotion<double>();
        var flows = new[] { Field(2, _ => 4.0), Field(4, _ => 8.0) };
        var accelerations = new[] { Field(2, _ => 2.0), Field(4, _ => 2.0) };

        var modulated = motion.ModulatePyramid(flows, accelerations, 0.5, forward: true);

        Assert.Equal(2, modulated.Length);
        // level 0: 0.5*4 + 1*(-0.25) = 1.75 ; level 1: 0.5*8 + 1*(-0.25) = 3.75
        Assert.Equal(1.75, modulated[0][0], 10);
        Assert.Equal(3.75, modulated[1][0], 10);
    }

    [Fact]
    public void MotionRejectsMismatchedFieldsAndOutOfRangeTime()
    {
        var motion = new ImplicitQuadraticMotion<double>();
        Assert.Throws<ArgumentException>(() =>
            motion.ModulateForward(Field(4, _ => 1.0), Field(3, _ => 1.0), 0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImplicitQuadraticMotion<double>.ValidateTime(1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => ImplicitQuadraticMotion<double>.ValidateTime(-0.1));
        Assert.Throws<ArgumentException>(() => motion.ModulatePyramid(
            new[] { Field(2, _ => 1.0) }, new[] { Field(2, _ => 1.0), Field(2, _ => 1.0) }, 0.5, true));
    }

    // ------------------------------------------------------------ distillation

    [Fact]
    public void MaskIsSetOnlyWhereTheStudentIsWorseThanTheTeacher()
    {
        // THE distinguishing idea. Distilling everywhere lets the teacher's privileged view of the
        // answer frame leak in as pressure to imitate it even where the student is already right, which
        // is the overfitting the paper set out to avoid.
        var d = new IQVFIDistillation<double>();
        var truth = Field(4, _ => 10.0);
        var student = Field(4, i => i switch { 0 => 10.0, 1 => 13.0, 2 => 10.5, _ => 10.0 });
        var teacher = Field(4, i => i switch { 0 => 12.0, 1 => 10.5, 2 => 10.5, _ => 10.0 });

        var mask = d.SelectiveMask(student, teacher, truth);

        Assert.Equal(0.0, mask[0], 12);   // student better -> no distillation
        Assert.Equal(1.0, mask[1], 12);   // student worse  -> distil
        Assert.Equal(0.0, mask[2], 12);   // tie -> strict inequality leaves it off
        Assert.Equal(0.0, mask[3], 12);   // both perfect -> nothing to learn
    }

    [Fact]
    public void APerfectStudentReceivesNoDistillationPressure()
    {
        // Follows from the strict inequality, and matters: a student that already matches the truth
        // should not be dragged toward a teacher that is merely different.
        var d = new IQVFIDistillation<double>();
        var truth = Field(5, i => i * 3.0);
        var teacher = Field(5, i => (i * 3.0) + 0.7);

        var mask = d.SelectiveMask(truth, teacher, truth);
        for (int i = 0; i < 5; i++) Assert.Equal(0.0, mask[i], 12);

        // And with an all-zero mask the motion loss is exactly zero however different the flows are.
        var sFwd = Field(5, _ => 0.0);
        var tFwd = Field(5, _ => 99.0);
        Assert.Equal(0.0, d.MotionDistillationLoss(sFwd, tFwd, sFwd, tFwd, mask), 12);
    }

    [Fact]
    public void MotionLossDistilsBothFlowDirections()
    {
        // The frame is synthesized by warping from both sides, so supervising only f_0t would leave
        // f_1t unconstrained. Each direction must contribute independently.
        var d = new IQVFIDistillation<double>();
        var mask = Field(2, _ => 1.0);

        var sFwd = Field(2, _ => 0.0);
        var tFwd = Field(2, _ => 1.0);
        var sBwd = Field(2, _ => 0.0);
        var tBwd = Field(2, _ => 0.0);

        double forwardOnly = d.MotionDistillationLoss(sFwd, tFwd, sBwd, tBwd, mask);
        double bothDiffer = d.MotionDistillationLoss(sFwd, tFwd, sFwd, tFwd, mask);

        Assert.True(forwardOnly > 0.0, "A forward-flow discrepancy must be penalized.");
        Assert.True(bothDiffer > forwardOnly,
            "A discrepancy in BOTH directions must cost more than one; the backward flow is not ignored.");
    }

    [Fact]
    public void AccelerationLossIsPlainL1AndUnmasked()
    {
        // The acceleration prior is a latent field with no per-pixel reconstruction error to compare
        // against, so there is no basis on which to gate it — unlike the motion loss.
        var d = new IQVFIDistillation<double>();
        var s = Field(4, _ => 1.0);
        var t = Field(4, _ => 3.0);

        Assert.Equal(2.0, d.AccelerationDistillationLoss(s, t), 10);
        Assert.Equal(0.0, d.AccelerationDistillationLoss(s, s), 12);
    }

    [Fact]
    public void PyramidLossWeightsCoarseLevelsMoreHeavily()
    {
        // Weights double as the pyramid coarsens, so large-scale structure — what a wrong trajectory
        // ruins — outweighs fine texture. Equal weighting would let detail dominate the gradient.
        var d = new IQVFIDistillation<double>();
        var zero = Field(4, _ => 0.0);
        var one = Field(4, _ => 1.0);

        // Error at the coarser level (index 1, weight 2) must cost more than the same error at level 0.
        double fineError = d.PyramidReconstructionLoss(new[] { one, zero }, new[] { zero, zero });
        double coarseError = d.PyramidReconstructionLoss(new[] { zero, one }, new[] { zero, zero });

        Assert.Equal(1.0, fineError, 10);
        Assert.Equal(2.0, coarseError, 10);
        Assert.True(coarseError > fineError);
        Assert.Equal(5, IQVFIDistillation<double>.PaperPyramidLevels);
    }

    [Fact]
    public void TotalLossCombinesTheThreeTermsAndRejectsNegativeWeights()
    {
        var d = new IQVFIDistillation<double>();
        Assert.Equal(6.0, d.TotalLoss(1.0, 2.0, 3.0), 10);
        Assert.Equal(1.0 + 0.2 + 0.9, d.TotalLoss(1.0, 2.0, 3.0, 1.0, 0.1, 0.3), 10);

        // A negative weight would reward the very error it is meant to penalize.
        Assert.Throws<ArgumentOutOfRangeException>(() => d.TotalLoss(1.0, 1.0, 1.0, -1.0, 1.0, 1.0));
    }

    [Fact]
    public void DistillationRejectsMisalignedInputs()
    {
        var d = new IQVFIDistillation<double>();
        var four = Field(4, _ => 1.0);
        var three = Field(3, _ => 1.0);

        Assert.Throws<ArgumentException>(() => d.SelectiveMask(four, three, four));
        Assert.Throws<ArgumentException>(() => d.AccelerationDistillationLoss(four, three));
        Assert.Throws<ArgumentException>(() => d.MotionDistillationLoss(four, three, four, four, four));
        // A mask that does not tile the flows cannot gate them.
        Assert.Throws<ArgumentException>(() => d.MotionDistillationLoss(four, four, four, four, three));
    }
}
