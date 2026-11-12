using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Implements FedKD — communication-efficient federated learning via adaptive mutual knowledge
/// distillation and dynamic SVD gradient compression.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Every client keeps a large private "mentor" (teacher) model and trains it
/// together with a small shared "mentee" (student) model; only the mentee's updates are sent to the server.
/// The two models teach each other (mutual distillation), and the mentee's gradients are compressed with a
/// truncated SVD before upload and download.</para>
///
/// <para>Adaptive mutual distillation (per client, for labelled data):</para>
/// <code>
/// L_t^t, L_t^s = cross-entropy task losses of the teacher and the student
/// scale        = 1 / (L_t^t + L_t^s)                       // adaptive: trust distillation more when both are good
/// L_d^t = KL(y^s || y^t) * scale,  L_d^s = KL(y^t || y^s) * scale
/// L_h   = (MSE(H^t, H^s W_h) + MSE(A^t, A^s)) * scale       // hidden states and attention heatmaps
/// L^t = L_t^t + L_d^t + L_h,  L^s = L_t^s + L_d^s + L_h
/// </code>
///
/// <para>Dynamic gradient approximation: a gradient matrix g ≈ U Σ V keeps the K largest singular values,
/// the fewest whose energy sum_{j &lt;= K} σ_j² / sum_j σ_j² reaches T(t), where the threshold grows
/// linearly from T_start to T_end over the rounds (paper defaults 0.95 and 0.98).</para>
///
/// <para>Reference: Wu, C., Wu, F., Lyu, L., Huang, Y., and Xie, X. (2022). "Communication-efficient
/// federated learning via knowledge distillation." Nature Communications 13, 2032.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedKDCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _temperature;
    private readonly double _startEnergyThreshold;
    private readonly double _endEnergyThreshold;

    /// <summary>
    /// Creates a new FedKD compressor.
    /// </summary>
    /// <param name="temperature">Softmax temperature for the distillation terms. The paper distills plain
    /// probabilities, i.e. temperature 1; with T &gt; 1 the KL terms are scaled by T² (Hinton et al.). Default: 1.0.</param>
    /// <param name="startEnergyThreshold">SVD energy threshold T_start at the first round. Default: 0.95.</param>
    /// <param name="endEnergyThreshold">SVD energy threshold T_end at the last round. Default: 0.98.</param>
    public FedKDCompressor(double temperature = 1.0, double startEnergyThreshold = 0.95, double endEnergyThreshold = 0.98)
    {
        if (temperature <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive.");
        }

        if (startEnergyThreshold <= 0 || startEnergyThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(startEnergyThreshold), "Energy threshold must be in (0, 1].");
        }

        if (endEnergyThreshold < startEnergyThreshold || endEnergyThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(endEnergyThreshold),
                "End energy threshold must be in [startEnergyThreshold, 1].");
        }

        _temperature = temperature;
        _startEnergyThreshold = startEnergyThreshold;
        _endEnergyThreshold = endEnergyThreshold;
    }

    /// <summary>
    /// Computes the adaptive mutual distillation losses of a teacher (mentor) and student (mentee) pair.
    /// </summary>
    /// <param name="teacherLogits">Teacher logits [samples, classes].</param>
    /// <param name="studentLogits">Student logits [samples, classes].</param>
    /// <param name="labels">Class label of each sample.</param>
    /// <param name="teacherHidden">Teacher hidden states [samples, teacherDim].</param>
    /// <param name="studentHidden">Student hidden states [samples, studentDim].</param>
    /// <param name="hiddenProjection">Learnable projection W_h [studentDim, teacherDim] that maps student hidden
    /// states into the teacher's space; may be null when both dimensions are equal (identity).</param>
    /// <param name="teacherAttention">Optional teacher attention heatmaps (any shape).</param>
    /// <param name="studentAttention">Optional student attention heatmaps (same shape as the teacher's).</param>
    /// <returns>The task, distillation and hidden losses and the two totals, each averaged over samples.</returns>
    public FedKDLosses<T> ComputeMutualDistillationLosses(
        Tensor<T> teacherLogits,
        Tensor<T> studentLogits,
        int[] labels,
        Tensor<T> teacherHidden,
        Tensor<T> studentHidden,
        Tensor<T>? hiddenProjection = null,
        Tensor<T>? teacherAttention = null,
        Tensor<T>? studentAttention = null)
    {
        Guard.NotNull(teacherLogits);
        Guard.NotNull(studentLogits);
        Guard.NotNull(labels);
        Guard.NotNull(teacherHidden);
        Guard.NotNull(studentHidden);
        if (teacherLogits.Rank != 2 || !teacherLogits.Shape.Equals(studentLogits.Shape))
        {
            throw new ArgumentException("Teacher and student logits must both be [samples, classes].", nameof(studentLogits));
        }

        int samples = teacherLogits.Shape[0];
        int classes = teacherLogits.Shape[1];
        if (samples == 0 || labels.Length != samples)
        {
            throw new ArgumentException($"Expected {samples} labels for a non-empty batch.", nameof(labels));
        }

        // One-hot labels via a single index-copy.
        var flatLabelIndex = new int[samples];
        for (int i = 0; i < samples; i++)
        {
            int label = labels[i];
            if (label < 0 || label >= classes)
            {
                throw new ArgumentOutOfRangeException(nameof(labels), $"Label {label} of sample {i} is outside [0, {classes}).");
            }

            flatLabelIndex[i] = i * classes + label;
        }

        var oneHot = Engine.TensorIndexCopy(
            new Tensor<T>(new[] { samples * classes }), 0,
            new Tensor<int>(flatLabelIndex, new[] { samples }),
            Engine.TensorOnesLike(new Tensor<T>(new[] { samples })))
            .Reshape(new[] { samples, classes });

        var invSamples = NumOps.FromDouble(1.0 / samples);

        // Task losses: cross-entropy with the labels.
        var teacherLogProbs = Engine.TensorLogSoftmax(teacherLogits, 1);
        var studentLogProbs = Engine.TensorLogSoftmax(studentLogits, 1);
        T teacherTask = NumOps.Negate(NumOps.Multiply(Engine.TensorSum(Engine.TensorMultiply(oneHot, teacherLogProbs)), invSamples));
        T studentTask = NumOps.Negate(NumOps.Multiply(Engine.TensorSum(Engine.TensorMultiply(oneHot, studentLogProbs)), invSamples));
        T adaptiveScale = NumOps.Divide(NumOps.One, NumOps.Add(teacherTask, studentTask));

        // Distillation: KL(y^t || y^s) teaches the student, KL(y^s || y^t) teaches the teacher.
        var temperatureT = NumOps.FromDouble(_temperature);
        var teacherSoftLog = Engine.TensorLogSoftmax(Engine.TensorDivideScalar(teacherLogits, temperatureT), 1);
        var studentSoftLog = Engine.TensorLogSoftmax(Engine.TensorDivideScalar(studentLogits, temperatureT), 1);
        var logRatio = Engine.TensorSubtract(teacherSoftLog, studentSoftLog);
        var kdScale = NumOps.Multiply(invSamples, NumOps.FromDouble(_temperature * _temperature));
        T klTeacherToStudent = NumOps.Multiply(
            Engine.TensorSum(Engine.TensorMultiply(Engine.TensorExp(teacherSoftLog), logRatio)), kdScale);
        T klStudentToTeacher = NumOps.Multiply(
            NumOps.Negate(Engine.TensorSum(Engine.TensorMultiply(Engine.TensorExp(studentSoftLog), logRatio))), kdScale);

        // Hidden loss: MSE(H^t, H^s W_h) + MSE(A^t, A^s).
        Tensor<T> projectedStudent;
        if (hiddenProjection is not null)
        {
            projectedStudent = Engine.TensorMatMul(studentHidden, hiddenProjection);
        }
        else if (studentHidden.Shape.Equals(teacherHidden.Shape))
        {
            projectedStudent = studentHidden;
        }
        else
        {
            throw new ArgumentException(
                "A hidden projection is required when teacher and student hidden sizes differ.", nameof(hiddenProjection));
        }

        T hiddenMse = Engine.TensorMean(Engine.TensorSquare(Engine.TensorSubtract(teacherHidden, projectedStudent)));
        if (teacherAttention is not null || studentAttention is not null)
        {
            if (teacherAttention is null || studentAttention is null || !teacherAttention.Shape.Equals(studentAttention.Shape))
            {
                throw new ArgumentException("Teacher and student attention heatmaps must both be given with the same shape.",
                    nameof(studentAttention));
            }

            hiddenMse = NumOps.Add(hiddenMse, Engine.TensorMean(Engine.TensorSquare(Engine.TensorSubtract(teacherAttention, studentAttention))));
        }

        return new FedKDLosses<T>(
            teacherTask,
            studentTask,
            NumOps.Multiply(klStudentToTeacher, adaptiveScale),
            NumOps.Multiply(klTeacherToStudent, adaptiveScale),
            NumOps.Multiply(hiddenMse, adaptiveScale));
    }

    /// <summary>
    /// Gets the dynamic SVD energy threshold T(t) = T_start + (T_end - T_start) * t / totalRounds.
    /// </summary>
    /// <param name="round">The current round t (0-based).</param>
    /// <param name="totalRounds">The total number of rounds.</param>
    /// <returns>The energy threshold for this round.</returns>
    public double GetEnergyThreshold(int round, int totalRounds)
    {
        if (totalRounds <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalRounds), "Total rounds must be positive.");
        }

        double progress = Math.Min(Math.Max((double)round / totalRounds, 0.0), 1.0);
        return _startEnergyThreshold + (_endEnergyThreshold - _startEnergyThreshold) * progress;
    }

    /// <summary>
    /// Compresses a gradient matrix g ≈ U Σ V by keeping the fewest singular values whose energy reaches
    /// the threshold.
    /// </summary>
    /// <param name="gradient">The gradient matrix [P x Q].</param>
    /// <param name="energyThreshold">Energy fraction to keep, e.g. from <see cref="GetEnergyThreshold"/>.</param>
    /// <returns>The truncated factors.</returns>
    public FedKDCompressedGradient<T> CompressGradient(Matrix<T> gradient, double energyThreshold)
    {
        Guard.NotNull(gradient);
        if (energyThreshold <= 0 || energyThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(energyThreshold), "Energy threshold must be in (0, 1].");
        }

        if (gradient.Rows == 0 || gradient.Columns == 0)
        {
            throw new ArgumentException("Gradient matrix must be non-empty.", nameof(gradient));
        }

        var (u, s, vh) = LinalgDispatch<T>.Svd(Tensor<T>.FromMatrix(gradient), fullMatrices: false);
        int available = s.Length;

        // Energy ratios of the leading singular values, normalized by sigma_1 first so the squares stay in
        // range for narrow T. K = 1 + the number of prefixes still below the threshold.
        int keep = 1;
        T largest = s[0];
        if (NumOps.GreaterThan(largest, NumOps.Zero))
        {
            var energy = Engine.TensorSquare(Engine.TensorDivideScalar(s, largest));
            var cumulative = Engine.TensorCumSum(energy, 0);
            var ratios = Engine.TensorDivideScalar(cumulative, cumulative[available - 1]);
            keep = Math.Min(available, 1 + Engine.TensorCountNonzero(Engine.TensorLessThan(ratios, NumOps.FromDouble(energyThreshold))));
        }

        int rows = gradient.Rows;
        int cols = gradient.Columns;
        var uK = Engine.TensorSlice(u, new[] { 0, 0 }, new[] { rows, keep });
        var sK = Engine.TensorSlice(s, new[] { 0 }, new[] { keep });
        var vK = Engine.TensorSlice(vh, new[] { 0, 0 }, new[] { keep, cols });
        return new FedKDCompressedGradient<T>(uK.ToMatrix(), sK.ToVector(), vK.ToMatrix());
    }

    /// <summary>
    /// Reconstructs a gradient matrix U Σ V from its compressed factors.
    /// </summary>
    /// <param name="compressed">The compressed gradient.</param>
    /// <returns>The approximated gradient matrix.</returns>
    public Matrix<T> DecompressGradient(FedKDCompressedGradient<T> compressed)
    {
        Guard.NotNull(compressed);
        var u = Tensor<T>.FromMatrix(compressed.U);
        var scaled = Engine.TensorMultiply(u, compressed.SingularValues.AsTensor().Reshape(new[] { 1, compressed.Rank }));
        return Engine.TensorMatMul(scaled, Tensor<T>.FromMatrix(compressed.V)).ToMatrix();
    }

    /// <summary>
    /// Server step: reconstructs every client's compressed gradient and returns their weighted average,
    /// which the server compresses again (with <see cref="CompressGradient"/>) before distributing it.
    /// </summary>
    /// <param name="clientGradients">Compressed gradients by client.</param>
    /// <param name="clientWeights">Optional per-client weights (default 1).</param>
    /// <returns>The aggregated gradient matrix.</returns>
    public Matrix<T> AggregateCompressedGradients(
        Dictionary<int, FedKDCompressedGradient<T>> clientGradients,
        Dictionary<int, double>? clientWeights = null)
    {
        Guard.NotNull(clientGradients);
        if (clientGradients.Count == 0)
        {
            throw new ArgumentException("No client gradients provided.", nameof(clientGradients));
        }

        double totalWeight = 0;
        foreach (var clientId in clientGradients.Keys)
        {
            totalWeight += clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
        }

        if (totalWeight <= 0)
        {
            throw new ArgumentException("Total client weight must be positive.", nameof(clientWeights));
        }

        Tensor<T>? sum = null;
        foreach (var (clientId, compressed) in clientGradients)
        {
            var p = NumOps.FromDouble((clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0) / totalWeight);
            var weighted = Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(DecompressGradient(compressed)), p);
            if (sum is null)
            {
                sum = weighted;
            }
            else if (!sum.Shape.Equals(weighted.Shape))
            {
                throw new ArgumentException($"Client {clientId} gradient shape differs from the others.", nameof(clientGradients));
            }
            else
            {
                sum = Engine.TensorAdd(sum, weighted);
            }
        }

        return (sum ?? new Tensor<T>(new[] { 0, 0 })).ToMatrix();
    }

    /// <summary>Gets the distillation temperature.</summary>
    public double Temperature => _temperature;

    /// <summary>Gets the first-round SVD energy threshold.</summary>
    public double StartEnergyThreshold => _startEnergyThreshold;

    /// <summary>Gets the last-round SVD energy threshold.</summary>
    public double EndEnergyThreshold => _endEnergyThreshold;
}

/// <summary>
/// The losses of one FedKD adaptive mutual distillation step, averaged over the batch.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class FedKDLosses<T>
{
    private static readonly AiDotNet.Tensors.Interfaces.INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates the loss breakdown.</summary>
    /// <param name="teacherTaskLoss">Teacher cross-entropy L_t^t.</param>
    /// <param name="studentTaskLoss">Student cross-entropy L_t^s.</param>
    /// <param name="teacherDistillationLoss">KL(y^s || y^t) / (L_t^t + L_t^s).</param>
    /// <param name="studentDistillationLoss">KL(y^t || y^s) / (L_t^t + L_t^s).</param>
    /// <param name="hiddenLoss">(MSE(H^t, H^s W_h) + MSE(A^t, A^s)) / (L_t^t + L_t^s).</param>
    public FedKDLosses(T teacherTaskLoss, T studentTaskLoss, T teacherDistillationLoss, T studentDistillationLoss, T hiddenLoss)
    {
        TeacherTaskLoss = teacherTaskLoss;
        StudentTaskLoss = studentTaskLoss;
        TeacherDistillationLoss = teacherDistillationLoss;
        StudentDistillationLoss = studentDistillationLoss;
        HiddenLoss = hiddenLoss;
    }

    /// <summary>Teacher (mentor) cross-entropy task loss.</summary>
    public T TeacherTaskLoss { get; }

    /// <summary>Student (mentee) cross-entropy task loss.</summary>
    public T StudentTaskLoss { get; }

    /// <summary>Adaptive distillation loss of the teacher, KL(y^s || y^t) / (L_t^t + L_t^s).</summary>
    public T TeacherDistillationLoss { get; }

    /// <summary>Adaptive distillation loss of the student, KL(y^t || y^s) / (L_t^t + L_t^s).</summary>
    public T StudentDistillationLoss { get; }

    /// <summary>Adaptive hidden-state and attention loss shared by both models.</summary>
    public T HiddenLoss { get; }

    /// <summary>Total teacher loss L^t = L_t^t + L_d^t + L_h.</summary>
    public T TeacherTotalLoss => Ops.Add(Ops.Add(TeacherTaskLoss, TeacherDistillationLoss), HiddenLoss);

    /// <summary>Total student loss L^s = L_t^s + L_d^s + L_h.</summary>
    public T StudentTotalLoss => Ops.Add(Ops.Add(StudentTaskLoss, StudentDistillationLoss), HiddenLoss);
}

/// <summary>
/// A gradient matrix compressed by FedKD's truncated SVD: g ≈ U diag(S) V.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class FedKDCompressedGradient<T>
{
    /// <summary>Creates a compressed gradient.</summary>
    /// <param name="u">Left singular vectors [P x K].</param>
    /// <param name="singularValues">The K kept singular values.</param>
    /// <param name="v">Right singular vectors [K x Q].</param>
    public FedKDCompressedGradient(Matrix<T> u, Vector<T> singularValues, Matrix<T> v)
    {
        Guard.NotNull(u);
        Guard.NotNull(singularValues);
        Guard.NotNull(v);
        if (u.Columns != singularValues.Length || v.Rows != singularValues.Length)
        {
            throw new ArgumentException("U columns, singular value count and V rows must all equal the rank.");
        }

        U = u;
        SingularValues = singularValues;
        V = v;
    }

    /// <summary>Left singular vectors [P x K].</summary>
    public Matrix<T> U { get; }

    /// <summary>The kept singular values.</summary>
    public Vector<T> SingularValues { get; }

    /// <summary>Right singular vectors [K x Q].</summary>
    public Matrix<T> V { get; }

    /// <summary>The kept rank K.</summary>
    public int Rank => SingularValues.Length;

    /// <summary>Number of values transmitted: K * (P + Q + 1).</summary>
    public int TransmittedValues => Rank * (U.Rows + V.Columns + 1);
}
