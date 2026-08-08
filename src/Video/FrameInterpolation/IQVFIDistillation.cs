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

        var mask = new Tensor<T>(studentFrame.Shape.ToArray());
        for (int i = 0; i < mask.Length; i++)
        {
            double gt = NumOps.ToDouble(groundTruth[i]);
            double studentError = Math.Abs(NumOps.ToDouble(studentFrame[i]) - gt);
            double teacherError = Math.Abs(NumOps.ToDouble(teacherFrame[i]) - gt);
            mask[i] = studentError > teacherError ? NumOps.One : NumOps.Zero;
        }

        return mask;
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

        double sum = 0.0;
        for (int i = 0; i < studentPrior.Length; i++)
            sum += Math.Abs(NumOps.ToDouble(studentPrior[i]) - NumOps.ToDouble(teacherPrior[i]));

        return sum / Math.Max(1, studentPrior.Length);
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

        // The mask is derived from a FRAME while the flows are 2-channel fields, so it may be smaller.
        // Tiling it keeps a single mask governing both flow channels, which is the intent: the gate is
        // a per-pixel decision about reconstruction quality, not a per-channel one.
        if (mask.Length == 0 || n % mask.Length != 0)
            throw new ArgumentException(
                $"A mask of {mask.Length} elements does not tile flows of {n} elements.", nameof(mask));

        int repeat = n / mask.Length;
        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            double m = NumOps.ToDouble(mask[i / repeat]);
            if (m == 0.0) continue;

            sum += m * Math.Abs(NumOps.ToDouble(studentForward[i]) - NumOps.ToDouble(teacherForward[i]));
            sum += m * Math.Abs(NumOps.ToDouble(studentBackward[i]) - NumOps.ToDouble(teacherBackward[i]));
        }

        return sum / Math.Max(1, n);
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

            double sum = 0.0;
            for (int i = 0; i < s.Length; i++)
                sum += Math.Abs(NumOps.ToDouble(s[i]) - NumOps.ToDouble(g[i]));

            double weight = Math.Pow(2.0, level);
            total += weight * (sum / Math.Max(1, s.Length));
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
