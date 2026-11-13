using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// IQ-VFI's knowledge-distillation objective (Hu, Jiang, Zhong, Wang and Zheng, CVPR 2024):
/// a pyramid reconstruction loss plus acceleration and motion distillation from a privileged teacher.
/// </summary>
/// <remarks>
/// <para>
/// Ground-truth intermediate MOTION does not exist, so the flows cannot be supervised directly. IQ-VFI
/// instead trains a teacher that sees three consecutive frames — including the ground-truth
/// intermediate frame <c>I_t</c> — and distils its acceleration prior and intermediate flows into a
/// student that sees only the two input frames.
/// </para>
/// <para>
/// <b>The mask is the distinguishing idea.</b> Motion distillation is applied only where
/// <c>||I_t^S - I_t||_1 &gt; ||I_t^T - I_t||_1</c>, i.e. only at pixels the student reconstructs WORSE
/// than the teacher. Prior schemes distil everywhere, which lets the teacher's privileged access to
/// <c>I_t</c> leak in as pressure to imitate it even where the student is already correct — the
/// overfitting the paper sets out to avoid. Gating on relative error means the teacher is copied only
/// where it demonstrably knows better.
/// </para>
/// <para>
/// <b>For Beginners:</b> A "teacher" model is allowed to peek at the answer frame; the "student" is
/// not. The student is corrected toward the teacher only in the places where it is actually doing
/// worse.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class IQVFIDistillation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Number of image pyramid levels in the reconstruction loss. Paper: 5.</summary>
    public const int PaperPyramidLevels = 5;

    /// <summary>
    /// The binary mask <c>M</c>: 1 where the student's reconstruction error exceeds the teacher's,
    /// 0 elsewhere.
    /// </summary>
    /// <param name="studentFrame">The student's interpolated frame.</param>
    /// <param name="teacherFrame">The teacher's interpolated frame.</param>
    /// <param name="groundTruth">The true intermediate frame.</param>
    /// <remarks>
    /// A STRICT inequality, so a tie leaves the mask at 0: where the two models are equally good there
    /// is nothing to distil, and distilling anyway would push the student toward a teacher that holds no
    /// advantage. In particular a perfect student receives no distillation pressure at all.
    /// </remarks>
    public Tensor<T> SelectiveMask(Tensor<T> studentFrame, Tensor<T> teacherFrame, Tensor<T> groundTruth)
    {
        if (studentFrame == null) throw new ArgumentNullException(nameof(studentFrame));
        if (teacherFrame == null) throw new ArgumentNullException(nameof(teacherFrame));
        if (groundTruth == null) throw new ArgumentNullException(nameof(groundTruth));
        if (studentFrame.Length != teacherFrame.Length || studentFrame.Length != groundTruth.Length)
            throw new ArgumentException(
                $"Frames must align: student {studentFrame.Length}, teacher {teacherFrame.Length}, " +
                $"ground truth {groundTruth.Length}.", nameof(teacherFrame));

        // 1 where the student is further from the ground truth than the teacher
        var engine = AiDotNetEngine.Current;
        var truth = Flat(groundTruth);
        var studentError = engine.TensorAbs(engine.TensorSubtract(Flat(studentFrame), truth));
        var teacherError = engine.TensorAbs(engine.TensorSubtract(Flat(teacherFrame), truth));
        return engine.Reshape(engine.TensorGreaterThan(studentError, teacherError), studentFrame.Shape.ToArray());
    }

    /// <summary>The tensor's elements as a vector-shaped tensor.</summary>
    private static Tensor<T> Flat(Tensor<T> tensor) => AiDotNetEngine.Current.Reshape(tensor, new[] { tensor.Length });

    /// <summary>Mean absolute difference of two equally long tensors, 0 when they are empty.</summary>
    private static double MeanAbsoluteDifference(Tensor<T> a, Tensor<T> b)
    {
        if (a.Length == 0)
        {
            return 0.0;
        }

        var engine = AiDotNetEngine.Current;
        return NumOps.ToDouble(engine.TensorMean(engine.TensorAbs(engine.TensorSubtract(Flat(a), Flat(b)))));
    }

    /// <summary>
    /// Implicit acceleration distillation loss <c>L_IA = ||P^S - P^T||_1</c>.
    /// </summary>
    /// <remarks>
    /// Unmasked, unlike the motion loss: the acceleration prior is a latent field with no per-pixel
    /// reconstruction error to compare against, so there is no basis on which to gate it.
    /// </remarks>
    public double AccelerationDistillationLoss(Tensor<T> studentPrior, Tensor<T> teacherPrior)
    {
        if (studentPrior == null) throw new ArgumentNullException(nameof(studentPrior));
        if (teacherPrior == null) throw new ArgumentNullException(nameof(teacherPrior));
        if (studentPrior.Length != teacherPrior.Length)
            throw new ArgumentException(
                $"Priors must align; got {studentPrior.Length} and {teacherPrior.Length}.",
                nameof(teacherPrior));

        return MeanAbsoluteDifference(studentPrior, teacherPrior);
    }

    /// <summary>
    /// Implicit motion distillation loss
    /// <c>L_IM = M * ||f_0t^S - f_0t^T||_1 + M * ||f_1t^S - f_1t^T||_1</c>.
    /// </summary>
    /// <param name="studentForward">Student <c>f_0t</c>.</param>
    /// <param name="teacherForward">Teacher <c>f_0t</c>.</param>
    /// <param name="studentBackward">Student <c>f_1t</c>.</param>
    /// <param name="teacherBackward">Teacher <c>f_1t</c>.</param>
    /// <param name="mask">The mask from <see cref="SelectiveMask"/>.</param>
    /// <remarks>
    /// BOTH directions are distilled. Supervising only the forward flow would leave the backward flow
    /// unconstrained, and the intermediate frame is synthesized by warping from both sides.
    /// </remarks>
    public double MotionDistillationLoss(
        Tensor<T> studentForward, Tensor<T> teacherForward,
        Tensor<T> studentBackward, Tensor<T> teacherBackward,
        Tensor<T> mask)
    {
        if (studentForward == null) throw new ArgumentNullException(nameof(studentForward));
        if (teacherForward == null) throw new ArgumentNullException(nameof(teacherForward));
        if (studentBackward == null) throw new ArgumentNullException(nameof(studentBackward));
        if (teacherBackward == null) throw new ArgumentNullException(nameof(teacherBackward));
        if (mask == null) throw new ArgumentNullException(nameof(mask));

        int n = studentForward.Length;
        if (teacherForward.Length != n || studentBackward.Length != n || teacherBackward.Length != n)
            throw new ArgumentException("All four flow fields must have the same element count.");

        // Empty fields carry no error; return before any tensor work on empty intermediates.
        if (n == 0) return 0.0;

        // The mask is derived from a FRAME while the flows are 2-channel fields, so it may be smaller.
        // Expanding it keeps a single mask governing both flow channels, which is the intent: the gate is
        // a per-pixel decision about reconstruction quality, not a per-channel one.
        var engine = AiDotNetEngine.Current;
        var m = ExpandMaskToFlow(mask, studentForward);
        var error = engine.TensorAdd(
            engine.TensorAbs(engine.TensorSubtract(Flat(studentForward), Flat(teacherForward))),
            engine.TensorAbs(engine.TensorSubtract(Flat(studentBackward), Flat(teacherBackward))));

        // Masked-out pixels contribute exactly nothing, even where an error is not finite
        var gated = engine.TensorWhere(
            engine.TensorEquals(m, NumOps.Zero), engine.TensorZerosLike(error), engine.TensorMultiply(m, error));
        return NumOps.ToDouble(engine.TensorSum(gated)) / n;
    }

    /// <summary>
    /// The mask laid over every element of <paramref name="flow"/>, flattened. Which flow elements share a mask
    /// pixel depends on where the flow keeps its channel axis, so the expansion follows the shapes: a mask of the
    /// flow's own size is used as is; one of equal rank broadcasts along its size-1 axes ([B,1,H,W] over channel-first
    /// [B,2,H,W], [B,H,W,1] over channel-last [B,H,W,2]); a lower-rank mask is aligned with the flow's trailing axes
    /// (channel-first, [H,W] over [2,H,W]) or, failing that, its leading axes (channel-last, [H,W] over [H,W,2]).
    /// A flat mask shorter than a flat flow names no layout and is rejected.
    /// </summary>
    private static Tensor<T> ExpandMaskToFlow(Tensor<T> mask, Tensor<T> flow)
    {
        int n = flow.Length;
        if (mask.Length == n)
            return Flat(mask);

        int[] maskShape = mask.Shape.ToArray();
        int[] flowShape = flow.Shape.ToArray();
        int[]? aligned = null;
        if (mask.Length > 0 && maskShape.Length <= flowShape.Length)
        {
            int extra = flowShape.Length - maskShape.Length;
            if (BroadcastsOver(maskShape, flowShape, extra))
                aligned = Prepend(maskShape, extra);
            else if (extra > 0 && BroadcastsOver(maskShape, flowShape, 0))
                aligned = Append(maskShape, extra);
        }

        if (aligned is null)
            throw new ArgumentException(
                $"A mask shaped [{string.Join(", ", maskShape)}] does not broadcast over flows shaped " +
                $"[{string.Join(", ", flowShape)}].", nameof(mask));

        var repeats = new int[flowShape.Length];
        for (int axis = 0; axis < repeats.Length; axis++)
            repeats[axis] = flowShape[axis] / aligned[axis];
        var engine = AiDotNetEngine.Current;
        return Flat(engine.TensorTile(engine.Reshape(mask, aligned), repeats));
    }

    /// <summary>True when every axis of <paramref name="mask"/>, placed at <paramref name="offset"/> in
    /// <paramref name="flow"/>, equals the flow's axis or is 1.</summary>
    private static bool BroadcastsOver(int[] mask, int[] flow, int offset)
    {
        for (int axis = 0; axis < mask.Length; axis++)
        {
            if (mask[axis] != flow[axis + offset] && mask[axis] != 1)
                return false;
        }

        return true;
    }

    private static int[] Prepend(int[] shape, int count)
    {
        var result = new int[shape.Length + count];
        for (int i = 0; i < count; i++) result[i] = 1;
        Array.Copy(shape, 0, result, count, shape.Length);
        return result;
    }

    private static int[] Append(int[] shape, int count)
    {
        var result = new int[shape.Length + count];
        Array.Copy(shape, result, shape.Length);
        for (int i = shape.Length; i < result.Length; i++) result[i] = 1;
        return result;
    }

    /// <summary>
    /// Pyramid reconstruction loss <c>L_R = sum_i w_i * ||L_i(student) - L_i(truth)||_1</c> over image
    /// pyramid levels.
    /// </summary>
    /// <param name="studentLevels">Student pyramid, finest level first.</param>
    /// <param name="truthLevels">Ground-truth pyramid, same shapes.</param>
    /// <remarks>
    /// Level weights double as the pyramid coarsens (<c>2^i</c>), so coarse levels — which carry the
    /// large-scale structure a wrong trajectory ruins — outweigh fine detail. Weighting every level
    /// equally would let fine-grained texture dominate the gradient.
    /// </remarks>
    public double PyramidReconstructionLoss(Tensor<T>[] studentLevels, Tensor<T>[] truthLevels)
    {
        if (studentLevels == null) throw new ArgumentNullException(nameof(studentLevels));
        if (truthLevels == null) throw new ArgumentNullException(nameof(truthLevels));
        if (studentLevels.Length == 0)
            throw new ArgumentException("At least one pyramid level is required.", nameof(studentLevels));
        if (studentLevels.Length != truthLevels.Length)
            throw new ArgumentException(
                $"Got {studentLevels.Length} student levels and {truthLevels.Length} truth levels.",
                nameof(truthLevels));

        double total = 0.0;
        for (int level = 0; level < studentLevels.Length; level++)
        {
            var s = studentLevels[level];
            var g = truthLevels[level];
            if (s.Length != g.Length)
                throw new ArgumentException(
                    $"Level {level} mismatch: {s.Length} vs {g.Length}.", nameof(truthLevels));

            double weight = Math.Pow(2.0, level);
            total += weight * MeanAbsoluteDifference(s, g);
        }

        return total;
    }

    /// <summary>
    /// The overall objective <c>L_total = w_R * L_R + w_IA * L_IA + w_IM * L_IM</c>.
    /// </summary>
    /// <remarks>
    /// The paper writes the three balancing parameters symbolically and does not state their values, so
    /// callers supply them; the defaults here weight all three equally and are ours, not the paper's.
    /// </remarks>
    public double TotalLoss(
        double reconstructionLoss, double accelerationLoss, double motionLoss,
        double reconstructionWeight = 1.0, double accelerationWeight = 1.0, double motionWeight = 1.0)
    {
        if (reconstructionWeight < 0.0 || accelerationWeight < 0.0 || motionWeight < 0.0)
            throw new ArgumentOutOfRangeException(nameof(reconstructionWeight),
                "Loss weights cannot be negative; a negative weight would reward the error it penalizes.");

        return (reconstructionWeight * reconstructionLoss)
               + (accelerationWeight * accelerationLoss)
               + (motionWeight * motionLoss);
    }
}
