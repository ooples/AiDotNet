using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Concerto's 3D intra-modal self-distillation objective (arXiv:2510.23607), built on the
/// Sonata / DINOv2 online-clustering formulation.
/// </summary>
/// <remarks>
/// <para>A student encoder is optimized to match a momentum-updated teacher across two augmented
/// views of the same point cloud, under a clustering-based cross-entropy. The teacher is never
/// back-propagated into; it only tracks an exponential moving average of the student, which is
/// what keeps the objective from collapsing to a constant.</para>
/// <para>Two stabilizers from the DINO line are essential rather than decorative:</para>
/// <list type="bullet">
/// <item><b>Centering</b> subtracts a running mean from the teacher logits, preventing one
/// cluster from dominating every assignment.</item>
/// <item><b>Sharpening</b> gives the teacher a lower temperature than the student, so the target
/// distribution is more confident than the prediction. Equal temperatures let the pair satisfy
/// the loss by both going uniform, which is the classic collapse.</item>
/// </list>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public static class ConcertoIntraModalObjective<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Cross-entropy between the sharpened, centered teacher assignment and the student's.
    /// </summary>
    /// <param name="studentLogits">Student cluster logits, shape <c>[numPoints, numPrototypes]</c>.</param>
    /// <param name="teacherLogits">Teacher cluster logits, same shape.</param>
    /// <param name="center">
    /// Running center over teacher logits, shape <c>[numPrototypes]</c>. Pass null to skip
    /// centering, though the paper's formulation includes it.
    /// </param>
    /// <param name="studentTemperature">Student softmax temperature. Defaults to 0.1.</param>
    /// <param name="teacherTemperature">
    /// Teacher softmax temperature; must be LOWER than the student's to sharpen the target.
    /// Defaults to 0.04.
    /// </param>
    /// <returns>Mean cross-entropy over points, so lower is better.</returns>
    public static T ComputeLoss(
        Tensor<T> studentLogits,
        Tensor<T> teacherLogits,
        Tensor<T>? center = null,
        double studentTemperature = 0.1,
        double teacherTemperature = 0.04)
    {
        if (studentLogits is null) throw new ArgumentNullException(nameof(studentLogits));
        if (teacherLogits is null) throw new ArgumentNullException(nameof(teacherLogits));
        if (studentTemperature <= 0) throw new ArgumentOutOfRangeException(nameof(studentTemperature));
        if (teacherTemperature <= 0) throw new ArgumentOutOfRangeException(nameof(teacherTemperature));

        int numPoints = studentLogits.Shape[0];
        int numPrototypes = studentLogits.Shape[^1];

        double total = 0;

        for (int p = 0; p < numPoints; p++)
        {
            // Teacher: center, sharpen, softmax. This is the TARGET, so it is treated as a
            // constant — the teacher receives no gradient by construction.
            var teacherProbabilities = new double[numPrototypes];
            double teacherMax = double.NegativeInfinity;
            for (int k = 0; k < numPrototypes; k++)
            {
                double logit = NumOps.ToDouble(teacherLogits[p, k]);
                if (center is not null) logit -= NumOps.ToDouble(center[k]);
                logit /= teacherTemperature;
                teacherProbabilities[k] = logit;
                if (logit > teacherMax) teacherMax = logit;
            }

            double teacherSum = 0;
            for (int k = 0; k < numPrototypes; k++)
            {
                teacherProbabilities[k] = Math.Exp(teacherProbabilities[k] - teacherMax);
                teacherSum += teacherProbabilities[k];
            }

            // Student: log-softmax at its own, higher temperature.
            double studentMax = double.NegativeInfinity;
            var studentScaled = new double[numPrototypes];
            for (int k = 0; k < numPrototypes; k++)
            {
                double logit = NumOps.ToDouble(studentLogits[p, k]) / studentTemperature;
                studentScaled[k] = logit;
                if (logit > studentMax) studentMax = logit;
            }

            double studentSum = 0;
            for (int k = 0; k < numPrototypes; k++)
                studentSum += Math.Exp(studentScaled[k] - studentMax);

            double logStudentSum = Math.Log(studentSum) + studentMax;

            // H(teacher, student) = -sum_k teacher_k * log student_k
            for (int k = 0; k < numPrototypes; k++)
            {
                double teacherProbability = teacherProbabilities[k] / teacherSum;
                if (teacherProbability <= 0) continue;
                total -= teacherProbability * (studentScaled[k] - logStudentSum);
            }
        }

        return numPoints > 0 ? NumOps.FromDouble(total / numPoints) : NumOps.Zero;
    }

    /// <summary>
    /// Advances the teacher weights one exponential-moving-average step toward the student.
    /// </summary>
    /// <param name="teacher">Teacher parameters, updated in place.</param>
    /// <param name="student">Student parameters.</param>
    /// <param name="momentum">
    /// EMA momentum <c>m</c>: <c>teacher = m * teacher + (1 - m) * student</c>. The paper's
    /// default is 0.996.
    /// </param>
    /// <remarks>
    /// The teacher is updated ONLY here — never by the optimizer. Letting gradients reach it
    /// collapses the objective, because the pair can then trivially agree by both moving to the
    /// same point.
    /// </remarks>
    public static void UpdateTeacher(Vector<T> teacher, Vector<T> student, double momentum)
    {
        if (teacher is null) throw new ArgumentNullException(nameof(teacher));
        if (student is null) throw new ArgumentNullException(nameof(student));
        if (teacher.Length != student.Length)
            throw new ArgumentException($"Teacher has {teacher.Length} parameters but student has {student.Length}.", nameof(student));
        if (momentum is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(momentum), momentum, "Momentum must be in [0, 1].");

        for (int i = 0; i < teacher.Length; i++)
        {
            double t = NumOps.ToDouble(teacher[i]);
            double s = NumOps.ToDouble(student[i]);
            teacher[i] = NumOps.FromDouble((momentum * t) + ((1.0 - momentum) * s));
        }
    }

    /// <summary>
    /// Advances the running center used to de-bias the teacher's assignments.
    /// </summary>
    /// <param name="center">Current center, shape <c>[numPrototypes]</c>, updated in place.</param>
    /// <param name="teacherLogits">This batch's teacher logits.</param>
    /// <param name="momentum">EMA momentum for the center. Defaults to 0.9.</param>
    public static void UpdateCenter(Tensor<T> center, Tensor<T> teacherLogits, double momentum = 0.9)
    {
        if (center is null) throw new ArgumentNullException(nameof(center));
        if (teacherLogits is null) throw new ArgumentNullException(nameof(teacherLogits));

        int numPoints = teacherLogits.Shape[0];
        int numPrototypes = teacherLogits.Shape[^1];
        if (numPoints == 0) return;

        for (int k = 0; k < numPrototypes; k++)
        {
            double batchMean = 0;
            for (int p = 0; p < numPoints; p++)
                batchMean += NumOps.ToDouble(teacherLogits[p, k]);
            batchMean /= numPoints;

            double current = NumOps.ToDouble(center[k]);
            center[k] = NumOps.FromDouble((momentum * current) + ((1.0 - momentum) * batchMean));
        }
    }
}
