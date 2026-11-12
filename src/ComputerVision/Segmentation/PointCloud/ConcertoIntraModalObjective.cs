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

    private static AiDotNet.Tensors.Engines.IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    /// <summary>
    /// The same cross-entropy as <see cref="ComputeLoss"/>, expressed in engine operations so a
    /// <c>GradientTape</c> can differentiate it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="ComputeLoss"/> evaluates in raw <c>double</c> and returns a bare scalar, so it
    /// carries no gradient graph: it can report a loss but cannot train anything. That is why this
    /// overload exists rather than the pretraining loop calling the scalar one -- a loop built on
    /// the scalar version would run, print a falling-looking number, and leave every weight
    /// untouched.
    /// </para>
    /// <para>
    /// The teacher side is wrapped in <c>StopGradient</c>. That is not an optimisation: the teacher
    /// is an exponential moving average of the student, and back-propagating into it is what
    /// collapses this objective to a constant.
    /// </para>
    /// </remarks>
    /// <param name="studentLogits">Student cluster logits, shape [points, prototypes].</param>
    /// <param name="teacherLogits">Teacher cluster logits, same shape.</param>
    /// <param name="center">Running centering term over prototypes, or null to skip centering.</param>
    /// <param name="studentTemperature">Student softmax temperature.</param>
    /// <param name="teacherTemperature">Teacher softmax temperature; lower than the student's.</param>
    /// <returns>A scalar tensor carrying the gradient graph.</returns>
    public static Tensor<T> ComputeTapeLoss(
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

        var engine = Engine;

        var teacherCentered = center is null
            ? teacherLogits
            : engine.TensorSubtract(teacherLogits, center);
        var teacherScaled = engine.TensorDivideScalar(
            teacherCentered, NumOps.FromDouble(teacherTemperature));
        var teacherProbabilities = engine.StopGradient(engine.TensorSoftmax(teacherScaled, axis: -1));

        var studentScaled = engine.TensorDivideScalar(
            studentLogits, NumOps.FromDouble(studentTemperature));
        var logStudent = engine.TensorLogSoftmax(studentScaled, axis: -1);

        // H(teacher, student) per point, then averaged -- matching ComputeLoss's reduction.
        var product = engine.TensorMultiply(teacherProbabilities, logStudent);
        var perPoint = engine.ReduceSum(product, new[] { studentLogits.Shape.Length - 1 }, keepDims: false);
        var mean = engine.ReduceMean(perPoint, new[] { 0 }, keepDims: false);
        return engine.TensorNegate(mean);
    }

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
        if (numPoints == 0) return NumOps.Zero;

        // A reported value, not a training signal: evaluated off the gradient tape.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var engine = Engine;

        // Teacher: center, sharpen, softmax. This is the TARGET, so it is treated as a constant.
        var teacherCentered = center is null ? teacherLogits : engine.TensorSubtract(teacherLogits, center);
        var teacherProbabilities = engine.TensorSoftmax(
            engine.TensorDivideScalar(teacherCentered, NumOps.FromDouble(teacherTemperature)), axis: -1);

        // Student: log-softmax at its own, higher temperature.
        var logStudent = engine.TensorLogSoftmax(
            engine.TensorDivideScalar(studentLogits, NumOps.FromDouble(studentTemperature)), axis: -1);

        // H(teacher, student) = -sum_k teacher_k * log student_k; prototypes the teacher gives zero
        // probability contribute nothing (so 0 * -inf never turns the loss into NaN).
        var terms = engine.TensorWhere(
            engine.TensorGreaterThan(teacherProbabilities, NumOps.Zero),
            engine.TensorMultiply(teacherProbabilities, logStudent),
            engine.TensorFullLike(teacherProbabilities, NumOps.Zero));
        return NumOps.Negate(NumOps.Divide(engine.TensorSum(terms), NumOps.FromDouble(numPoints)));
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

        // teacher = m * teacher + (1 - m) * student = student + m * (teacher - student), written back in place.
        var updated = Engine.Lerp(student, teacher, NumOps.FromDouble(momentum));
        updated.AsSpan().CopyTo(teacher.AsWritableSpan());
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
        if (numPoints == 0) return;

        // center = m * center + (1 - m) * mean over points of the teacher logits, written back in place.
        var batchMean = Engine.ReduceMean(teacherLogits, new[] { 0 }, false);
        var updated = Engine.TensorLerp(batchMean, center, NumOps.FromDouble(momentum));
        updated.AsSpan().CopyTo(center.AsWritableSpan());
    }
}
