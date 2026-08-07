using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// IQ-VFI's implicit quadratic motion model (Hu, Jiang, Zhong, Wang and Zheng, CVPR 2024): modulates
/// linear intermediate flows into quadratic ones using a latent acceleration prior.
/// </summary>
/// <remarks>
/// <para>
/// Linear VFI approximates the intermediate flow as a straight-line interpolation of the
/// frame-to-frame flow, which cannot follow curvilinear motion. IQ-VFI adds the acceleration term of
/// a quadratic motion model:
/// </para>
/// <code>
/// f_0t = t * f_01       + (a / 2) * (t^2 - t)
/// f_1t = (1 - t) * f_10 + (a / 2) * ((1 - t)^2 - (1 - t))
/// </code>
/// <para>
/// The acceleration weight <c>(t^2 - t) / 2</c> has two properties that make the decomposition
/// well-formed, and both are asserted in the tests rather than assumed:
/// </para>
/// <list type="bullet">
/// <item>It VANISHES at <c>t = 0</c> and <c>t = 1</c>, so the quadratic term bends the trajectory only
/// in the interior and never contradicts the measured endpoint flows.</item>
/// <item>It is NEGATIVE throughout <c>(0, 1)</c>, reaching its extreme <c>-1/8</c> at the midpoint. The
/// correction therefore always pulls against the linear estimate; a sign error would push the
/// trajectory outward and bend it the wrong way.</item>
/// </list>
/// <para>
/// With <c>a = 0</c> the model degenerates exactly to the linear one, which is what makes acceleration
/// a strict generalization rather than a competing estimate.
/// </para>
/// <para>
/// Acceleration is assumed CONSTANT over the interval — the paper says so explicitly. That is what
/// allows a single acceleration field to serve both directions, and it is why only TWO input frames
/// are needed where earlier quadratic methods required four.
/// </para>
/// <para><b>For Beginners:</b> Halfway between two video frames, an object moving in a curve is not
/// halfway along the straight line between its start and end. This works out how much to bend that
/// straight-line guess.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class ImplicitQuadraticMotion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The acceleration weight <c>(t^2 - t) / 2</c> applied to the forward flow at time
    /// <paramref name="t"/>.
    /// </summary>
    public static double ForwardAccelerationWeight(double t) => (((t * t) - t)) / 2.0;

    /// <summary>
    /// The acceleration weight for the backward flow, <c>((1-t)^2 - (1-t)) / 2</c>.
    /// </summary>
    /// <remarks>
    /// Note this is <see cref="ForwardAccelerationWeight"/> evaluated at <c>1 - t</c>, not its negation:
    /// both directions receive the SAME sign of correction, because the acceleration is a property of
    /// the trajectory rather than of the direction it is traversed in.
    /// </remarks>
    public static double BackwardAccelerationWeight(double t) => ForwardAccelerationWeight(1.0 - t);

    /// <summary>
    /// Modulates a linear forward flow into a quadratic one:
    /// <c>f_0t = t * f_01 + (a/2) * (t^2 - t)</c>.
    /// </summary>
    /// <param name="flowForward">The measured flow <c>f_01</c>, any shape.</param>
    /// <param name="acceleration">The latent acceleration prior, same shape.</param>
    /// <param name="t">Target time in [0, 1].</param>
    public Tensor<T> ModulateForward(Tensor<T> flowForward, Tensor<T> acceleration, double t)
        => Modulate(flowForward, acceleration, t, ForwardAccelerationWeight(t), nameof(flowForward));

    /// <summary>
    /// Modulates a linear backward flow into a quadratic one:
    /// <c>f_1t = (1-t) * f_10 + (a/2) * ((1-t)^2 - (1-t))</c>.
    /// </summary>
    public Tensor<T> ModulateBackward(Tensor<T> flowBackward, Tensor<T> acceleration, double t)
        => Modulate(flowBackward, acceleration, 1.0 - t, BackwardAccelerationWeight(t), nameof(flowBackward));

    private Tensor<T> Modulate(
        Tensor<T> flow, Tensor<T> acceleration, double linearScale, double accelerationWeight, string flowName)
    {
        if (flow == null) throw new ArgumentNullException(flowName);
        if (acceleration == null) throw new ArgumentNullException(nameof(acceleration));
        if (flow.Length != acceleration.Length)
            throw new ArgumentException(
                $"Flow has {flow.Length} elements but acceleration has {acceleration.Length}; " +
                "the acceleration prior is a per-pixel field and must match the flow.",
                nameof(acceleration));

        var result = new Tensor<T>(flow.Shape.ToArray());
        for (int i = 0; i < result.Length; i++)
        {
            double v = (linearScale * NumOps.ToDouble(flow[i]))
                       + (accelerationWeight * NumOps.ToDouble(acceleration[i]));
            result[i] = NumOps.FromDouble(v);
        }
        return result;
    }

    /// <summary>
    /// Validates a target time and throws if it lies outside [0, 1].
    /// </summary>
    /// <remarks>
    /// Outside that range the quadratic weight changes sign and the "intermediate" frame is no longer
    /// between the inputs, so extrapolation would silently produce a differently-shaped trajectory.
    /// </remarks>
    public static void ValidateTime(double t)
    {
        if (t is < 0.0 or > 1.0 || double.IsNaN(t))
            throw new ArgumentOutOfRangeException(nameof(t), t, "t must be in [0, 1].");
    }

    /// <summary>
    /// Progressively modulates flows across a coarse-to-fine pyramid, refining the acceleration
    /// contribution level by level.
    /// </summary>
    /// <param name="flowsPerLevel">Linear flows, coarsest level first.</param>
    /// <param name="accelerationPerLevel">Acceleration priors, one per level, same shapes.</param>
    /// <param name="t">Target time in [0, 1].</param>
    /// <param name="forward">True for <c>f_0t</c>, false for <c>f_1t</c>.</param>
    /// <returns>The modulated flow at each level.</returns>
    /// <remarks>
    /// Coarse-to-fine rather than single-shot: acceleration at a coarse level captures the broad
    /// trajectory curvature while finer levels correct local detail. Applying one acceleration field at
    /// full resolution would have to represent both scales at once.
    /// </remarks>
    public Tensor<T>[] ModulatePyramid(
        Tensor<T>[] flowsPerLevel, Tensor<T>[] accelerationPerLevel, double t, bool forward)
    {
        if (flowsPerLevel == null) throw new ArgumentNullException(nameof(flowsPerLevel));
        if (accelerationPerLevel == null) throw new ArgumentNullException(nameof(accelerationPerLevel));
        ValidateTime(t);

        if (flowsPerLevel.Length == 0)
            throw new ArgumentException("At least one pyramid level is required.", nameof(flowsPerLevel));
        if (flowsPerLevel.Length != accelerationPerLevel.Length)
            throw new ArgumentException(
                $"Got {flowsPerLevel.Length} flow levels but {accelerationPerLevel.Length} acceleration " +
                "levels; the pyramid must be modulated at every level.", nameof(accelerationPerLevel));

        var result = new Tensor<T>[flowsPerLevel.Length];
        for (int level = 0; level < flowsPerLevel.Length; level++)
        {
            result[level] = forward
                ? ModulateForward(flowsPerLevel[level], accelerationPerLevel[level], t)
                : ModulateBackward(flowsPerLevel[level], accelerationPerLevel[level], t);
        }

        return result;
    }
}
