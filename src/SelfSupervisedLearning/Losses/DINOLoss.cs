using AiDotNet.Interfaces;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning.Losses;

/// <summary>
/// DINO (Self-Distillation with No Labels) Loss for self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DINO loss is a cross-entropy loss between student and teacher
/// outputs, where the teacher is an EMA of the student. It uses centering and sharpening
/// to prevent collapse.</para>
///
/// <para><b>Key components:</b></para>
/// <list type="bullet">
/// <item><b>Sharpening:</b> Lower temperature for teacher outputs (default: 0.04)</item>
/// <item><b>Student temperature:</b> Higher temperature for student (default: 0.1)</item>
/// <item><b>Centering:</b> Subtract running mean from teacher outputs to prevent collapse</item>
/// </list>
///
/// <para><b>Loss formula:</b></para>
/// <code>
/// L = -Σ_crops Σ_k P_t(k) * log(P_s(k))
/// where P_t = softmax((z_t - c) / τ_t) and P_s = softmax(z_s / τ_s)
/// </code>
///
/// <para><b>Reference:</b> Caron et al., "Emerging Properties in Self-Supervised Vision
/// Transformers" (ICCV 2021)</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Emerging Properties in Self-Supervised Vision Transformers", "https://arxiv.org/abs/2104.14294", Year = 2021, Authors = "Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin")]
public class DINOLoss<T> : IContrastiveLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly double _studentTemperature;
    private readonly double _teacherTemperature;
    private readonly int _outputDim;
    private T[] _center;
    private readonly double _centerMomentum;

    /// <summary>
    /// Gets the student temperature parameter.
    /// </summary>
    public double StudentTemperature => _studentTemperature;

    /// <summary>
    /// Gets the teacher temperature parameter.
    /// </summary>
    public double TeacherTemperature => _teacherTemperature;

    /// <summary>
    /// Initializes a new instance of the DINOLoss class.
    /// </summary>
    /// <param name="outputDim">Output dimension of the network.</param>
    /// <param name="studentTemperature">Temperature for student outputs (default: 0.1).</param>
    /// <param name="teacherTemperature">Temperature for teacher outputs (default: 0.04).</param>
    /// <param name="centerMomentum">Momentum for center update (default: 0.9).</param>
    public DINOLoss(
        int outputDim,
        double studentTemperature = 0.1,
        double teacherTemperature = 0.04,
        double centerMomentum = 0.9)
    {
        if (studentTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(studentTemperature), "Temperature must be positive");
        if (teacherTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(teacherTemperature), "Temperature must be positive");

        _studentTemperature = studentTemperature;
        _teacherTemperature = teacherTemperature;
        _outputDim = outputDim;
        _centerMomentum = centerMomentum;

        // Initialize center to zeros
        _center = new T[outputDim];
        for (int i = 0; i < outputDim; i++)
        {
            _center[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Computes the DINO loss between student and teacher outputs.
    /// </summary>
    /// <param name="studentOutput">Student network output [batch_size, dim].</param>
    /// <param name="teacherOutput">Teacher network output [batch_size, dim].</param>
    /// <param name="updateCenter">Whether to update the center (default: true).</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeLoss(Tensor<T> studentOutput, Tensor<T> teacherOutput, bool updateCenter = true)
    {
        if (studentOutput is null) throw new ArgumentNullException(nameof(studentOutput));
        if (teacherOutput is null) throw new ArgumentNullException(nameof(teacherOutput));

        var batchSize = studentOutput.Shape[0];

        // Apply centering to teacher output
        var centeredTeacher = ApplyCenter(teacherOutput);

        // Compute softmax with different temperatures
        var studentProbs = Softmax(studentOutput, _studentTemperature);
        var teacherProbs = Softmax(centeredTeacher, _teacherTemperature);

        // Compute cross-entropy loss
        T totalLoss = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _outputDim; d++)
            {
                // L = -Σ P_t * log(P_s)
                var logPs = NumOps.Log(NumOps.Add(studentProbs[b, d], NumOps.FromDouble(1e-8)));
                totalLoss = NumOps.Subtract(totalLoss, NumOps.Multiply(teacherProbs[b, d], logPs));
            }
        }

        // Update center with EMA
        if (updateCenter)
        {
            UpdateCenter(teacherOutput);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes DINO loss for multiple student crops against global teacher views.
    /// </summary>
    /// <param name="studentOutputs">List of student outputs (local + global crops).</param>
    /// <param name="teacherOutputs">List of teacher outputs (global crops only).</param>
    /// <returns>The computed loss value.</returns>
    public T ComputeMultiCropLoss(
        IList<Tensor<T>> studentOutputs,
        IList<Tensor<T>> teacherOutputs)
    {
        if (studentOutputs.Count == 0)
            throw new ArgumentException("Must provide at least one student output", nameof(studentOutputs));
        if (teacherOutputs.Count == 0)
            throw new ArgumentException("Must provide at least one teacher output", nameof(teacherOutputs));

        T totalLoss = NumOps.Zero;
        int pairCount = 0;

        // For each teacher output (global views)
        foreach (var teacherOut in teacherOutputs)
        {
            var centeredTeacher = ApplyCenter(teacherOut);
            var teacherProbs = Softmax(centeredTeacher, _teacherTemperature);

            // For each student output
            foreach (var studentOut in studentOutputs)
            {
                // Skip if same view (avoid trivial solution)
                if (ReferenceEquals(studentOut, teacherOut))
                    continue;

                var studentProbs = Softmax(studentOut, _studentTemperature);
                var batchSize = studentProbs.Shape[0];

                for (int b = 0; b < batchSize; b++)
                {
                    for (int d = 0; d < _outputDim; d++)
                    {
                        var logPs = NumOps.Log(NumOps.Add(studentProbs[b, d], NumOps.FromDouble(1e-8)));
                        totalLoss = NumOps.Subtract(totalLoss, NumOps.Multiply(teacherProbs[b, d], logPs));
                    }
                }

                pairCount += batchSize;
            }
        }

        // Update center with mean of all teacher outputs
        UpdateCenterFromMultiple(teacherOutputs);

        return pairCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(pairCount)) : NumOps.Zero;
    }

    /// <summary>
    /// Computes DINO loss with gradients for backpropagation.
    /// </summary>
    public (T loss, Tensor<T> gradStudent) ComputeLossWithGradients(
        Tensor<T> studentOutput, Tensor<T> teacherOutput)
    {
        var batchSize = studentOutput.Shape[0];

        var centeredTeacher = ApplyCenter(teacherOutput);
        var studentProbs = Softmax(studentOutput, _studentTemperature);
        var teacherProbs = Softmax(centeredTeacher, _teacherTemperature);

        var gradStudent = new T[batchSize * _outputDim];
        T totalLoss = NumOps.Zero;

        var invStudentTemp = NumOps.FromDouble(1.0 / _studentTemperature);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _outputDim; d++)
            {
                // Cross-entropy loss
                var logPs = NumOps.Log(NumOps.Add(studentProbs[b, d], NumOps.FromDouble(1e-8)));
                totalLoss = NumOps.Subtract(totalLoss, NumOps.Multiply(teacherProbs[b, d], logPs));

                // Gradient: (P_s - P_t) / τ_s
                var grad = NumOps.Multiply(
                    NumOps.Subtract(studentProbs[b, d], teacherProbs[b, d]),
                    invStudentTemp);

                gradStudent[b * _outputDim + d] = grad;
            }
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
        var scale = NumOps.FromDouble(1.0 / batchSize);

        for (int i = 0; i < gradStudent.Length; i++)
        {
            gradStudent[i] = NumOps.Multiply(gradStudent[i], scale);
        }

        UpdateCenter(teacherOutput);

        return (avgLoss, new Tensor<T>(gradStudent, [batchSize, _outputDim]));
    }

    /// <summary>
    /// Gets the current center values.
    /// </summary>
    public T[] GetCenter() => (T[])_center.Clone();

    /// <summary>
    /// Resets the center to zeros.
    /// </summary>
    public void ResetCenter()
    {
        for (int i = 0; i < _outputDim; i++)
        {
            _center[i] = NumOps.Zero;
        }
    }

    private Tensor<T> ApplyCenter(Tensor<T> output)
    {
        var batchSize = output.Shape[0];
        var result = new T[batchSize * _outputDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < _outputDim; d++)
            {
                result[b * _outputDim + d] = NumOps.Subtract(output[b, d], _center[d]);
            }
        }

        return new Tensor<T>(result, [batchSize, _outputDim]);
    }

    private void UpdateCenter(Tensor<T> teacherOutput)
    {
        var batchSize = teacherOutput.Shape[0];
        var momentum = NumOps.FromDouble(_centerMomentum);
        var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);
        var invBatch = NumOps.FromDouble(1.0 / batchSize);

        for (int d = 0; d < _outputDim; d++)
        {
            // Compute batch mean for this dimension
            T batchMean = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                batchMean = NumOps.Add(batchMean, teacherOutput[b, d]);
            }
            batchMean = NumOps.Multiply(batchMean, invBatch);

            // EMA update: center = momentum * center + (1 - momentum) * batch_mean
            _center[d] = NumOps.Add(
                NumOps.Multiply(momentum, _center[d]),
                NumOps.Multiply(oneMinusMomentum, batchMean));
        }
    }

    private void UpdateCenterFromMultiple(IList<Tensor<T>> teacherOutputs)
    {
        var momentum = NumOps.FromDouble(_centerMomentum);
        var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);

        // Compute mean across all teacher outputs
        var meanValues = new T[_outputDim];
        int totalSamples = 0;

        foreach (var output in teacherOutputs)
        {
            var batchSize = output.Shape[0];
            totalSamples += batchSize;

            for (int d = 0; d < _outputDim; d++)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    meanValues[d] = NumOps.Add(meanValues[d], output[b, d]);
                }
            }
        }

        var invTotal = NumOps.FromDouble(1.0 / totalSamples);

        for (int d = 0; d < _outputDim; d++)
        {
            var batchMean = NumOps.Multiply(meanValues[d], invTotal);

            _center[d] = NumOps.Add(
                NumOps.Multiply(momentum, _center[d]),
                NumOps.Multiply(oneMinusMomentum, batchMean));
        }
    }

    private Tensor<T> Softmax(Tensor<T> input, double temperature)
    {
        var batchSize = input.Shape[0];
        var dim = input.Shape[1];
        var result = new T[batchSize * dim];
        var invTemp = NumOps.FromDouble(1.0 / temperature);

        for (int b = 0; b < batchSize; b++)
        {
            // Find max for numerical stability
            T maxVal = input[b, 0];
            for (int d = 1; d < dim; d++)
            {
                if (NumOps.GreaterThan(input[b, d], maxVal))
                    maxVal = input[b, d];
            }

            // Compute exp and sum
            T sumExp = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                var scaled = NumOps.Multiply(NumOps.Subtract(input[b, d], maxVal), invTemp);
                result[b * dim + d] = NumOps.Exp(scaled);
                sumExp = NumOps.Add(sumExp, result[b * dim + d]);
            }

            // Normalize
            for (int d = 0; d < dim; d++)
            {
                result[b * dim + d] = NumOps.Divide(result[b * dim + d], sumExp);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }
    /// <summary>
    /// The differentiable DINO objective, built entirely from <c>IEngine</c> operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SEPARATE FROM <see cref="ComputeLoss(Tensor{T}, Tensor{T}, bool)"/> BECAUSE THAT ONE CANNOT
    /// TRAIN. It assembles its result from host loops over tensor indexers, which severs the gradient
    /// tape -- a correct loss VALUE carrying no history for an optimizer to backpropagate.
    /// </para>
    /// <para>
    /// The teacher branch is deliberately NOT differentiated. DINO's teacher is an
    /// exponential-moving-average copy updated outside the optimizer, and its centre is a running
    /// statistic; both are constants as far as this step's gradient is concerned. The teacher
    /// distribution is therefore built from the centred, sharpened logits as DATA, and only the
    /// student's log-softmax carries the tape.
    /// </para>
    /// <para>
    /// The centre is NOT updated here. This overload has no <c>updateCenter</c> switch, and mutating
    /// a running statistic from inside a loss evaluation would make the objective depend on how many
    /// times it had been measured. Call <see cref="ComputeLoss(Tensor{T}, Tensor{T}, bool)"/> when
    /// the centre should advance.
    /// </para>
    /// </remarks>
    Tensor<T> IContrastiveLoss<T>.ComputeLoss(Tensor<T> view1, Tensor<T> view2)
    {
        ContrastiveTapeOps<T>.RequireMatchingRank2(
            view1, view2, "DINO", nameof(view1), nameof(view2));

        // The output width is checked against the CONFIGURED one too, because CenterRow() returns
        // [1, _outputDim]: a mismatch would otherwise surface as an engine broadcast error that
        // never mentions the DINO output dimension the caller actually got wrong.
        if (view2.Shape[1] != _outputDim)
        {
            throw new ArgumentException(
                $"DINO was configured for an output dimension of {_outputDim}, but the logits are "
                + $"{view2.Shape[1]} wide.", nameof(view2));
        }

        // Student: sharpened log-softmax, on the tape.
        var studentLogProbabilities = Engine.TensorLogSoftmax(
            Engine.TensorMultiplyScalar(view1, NumOps.FromDouble(1.0 / _studentTemperature)), axis: 1);

        // Teacher: centred and sharpened softmax, as constant data.
        // Teacher: centred and sharpened softmax, DETACHED. The teacher is an EMA copy updated
        // outside the optimizer and its centre is a running statistic, so neither is differentiated
        // -- but saying so is not enough on its own. TensorSubtract, TensorMultiplyScalar and
        // TensorSoftmax all record, so any tape history view2 arrived with would have leaked a
        // gradient path into the teacher branch. StopGradient blocks it at the boundary.
        var teacherLogits = Engine.StopGradient(view2);
        var teacherProbabilities = Engine.TensorSoftmax(
            Engine.TensorMultiplyScalar(
                Engine.TensorSubtract(teacherLogits, CenterRow()),
                NumOps.FromDouble(1.0 / _teacherTemperature)),
            axis: 1);

        var crossEntropy = Engine.ReduceSum(
            Engine.TensorMultiply(teacherProbabilities, studentLogProbabilities),
            new[] { 1 }, keepDims: false);

        return Engine.TensorNegate(Engine.ReduceMean(crossEntropy, new[] { 0 }, keepDims: false));
    }

    /// <summary>The running centre as a broadcastable <c>[1, outputDim]</c> row of constant data.</summary>
    private Tensor<T> CenterRow()
    {
        var row = new Tensor<T>(new[] { 1, _outputDim });
        for (int c = 0; c < _outputDim; c++) row[c] = _center[c];
        return row;
    }
}
