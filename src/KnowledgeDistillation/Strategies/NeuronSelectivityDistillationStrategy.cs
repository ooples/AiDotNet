using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Neuron selectivity distillation that transfers the activation patterns and selectivity of individual neurons.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Production Use:</b> This strategy focuses on matching how individual neurons respond to inputs.
/// Some neurons are highly selective (activate strongly for specific patterns), while others are more general.
/// Transferring this selectivity helps the student learn meaningful feature representations.</para>
///
/// <para><b>Key Concept:</b> Neuron selectivity measures how discriminative each neuron is. A highly selective
/// neuron activates strongly for certain inputs and weakly for others. The distribution of selectivity across
/// neurons is important for model performance.</para>
///
/// <para><b>Implementation:</b> We measure selectivity using:
/// 1. Activation variance (how much neuron output varies across samples)
/// 2. Sparsity (what percentage of time the neuron is active)
/// 3. Peak-to-average ratio (how peaked the activation distribution is)</para>
///
/// <para><b>Usage Pattern:</b> This strategy implements both standard output-based distillation and
/// intermediate activation-based selectivity matching. Use as follows:</para>
///
/// <para><b>Standard Usage (via IDistillationStrategy):</b>
/// <code>
/// T outputLoss = strategy.ComputeLoss(studentOutput, teacherOutput, labels);
/// Matrix&lt;T&gt; outputGrad = strategy.ComputeGradient(studentOutput, teacherOutput, labels);
/// </code></para>
///
/// <para><b>With Intermediate Activations (via IIntermediateActivationStrategy):</b>
/// <code>
/// // Collect activations during forward pass
/// var studentActivations = new IntermediateActivations&lt;T&gt;();
/// studentActivations.Add("layer3", studentLayer3Output);
///
/// var teacherActivations = new IntermediateActivations&lt;T&gt;();
/// teacherActivations.Add("layer3", teacherLayer3Output);
///
/// // Compute combined loss
/// T outputLoss = strategy.ComputeLoss(studentOutput, teacherOutput, labels);
/// T selectivityLoss = strategy.ComputeIntermediateLoss(studentActivations, teacherActivations);
/// T totalLoss = outputLoss + selectivityLoss; // selectivityLoss is already weighted
/// </code></para>
///
/// <para>The selectivityWeight and metric parameters control the intermediate activation loss component.</para>
/// </remarks>
public class NeuronSelectivityDistillationStrategy<T> : DistillationStrategyBase<T>, IIntermediateActivationStrategy<T>
{
    private readonly double _selectivityWeight;
    private readonly SelectivityMetric _metric;
    private readonly string _targetLayerName;

    public NeuronSelectivityDistillationStrategy(
        string targetLayerName = "default",
        double selectivityWeight = 0.5,
        SelectivityMetric metric = SelectivityMetric.Variance,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        if (string.IsNullOrWhiteSpace(targetLayerName))
            throw new ArgumentException("Target layer name cannot be null or whitespace", nameof(targetLayerName));
        if (selectivityWeight < 0 || selectivityWeight > 1)
            throw new ArgumentException("Selectivity weight must be between 0 and 1", nameof(selectivityWeight));

        _targetLayerName = targetLayerName;
        _selectivityWeight = selectivityWeight;
        _metric = metric;
    }

    /// <summary>
    /// Computes the base distillation loss on final outputs.
    /// </summary>
    /// <remarks>
    /// This method implements standard distillation loss (soft + hard loss) on final outputs.
    /// It does NOT include the selectivity component, which requires intermediate activations.
    /// Use ComputeSelectivityLoss separately and combine the losses manually. See class remarks for usage pattern.
    /// </remarks>
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        T totalLoss = NumOps.Zero;

        // Standard distillation loss on final outputs
        // Selectivity loss requires intermediate activations - use ComputeSelectivityLoss separately
        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentOutput = studentBatchOutput.GetRow(r);
            Vector<T> teacherOutput = teacherBatchOutput.GetRow(r);
            Vector<T>? trueLabels = trueLabelsBatch?.GetRow(r);

            // Standard distillation loss
            var studentSoft = DistillationHelper<T>.Softmax(studentOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, Temperature);
            var softLoss = DistillationHelper<T>.KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

            T sampleLoss;
            if (trueLabels != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentOutput, 1.0);
                var hardLoss = DistillationHelper<T>.CrossEntropy(studentProbs, trueLabels);
                sampleLoss = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
            }
            else
            {
                sampleLoss = softLoss;
            }

            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes the gradient of the base distillation loss on final outputs.
    /// </summary>
    /// <remarks>
    /// This method implements standard distillation gradient (soft + hard) on final outputs.
    /// It does NOT include the selectivity gradient, which requires intermediate activations.
    /// Selectivity gradients must be computed separately and backpropagated through the network.
    /// See class remarks for usage pattern.
    /// </remarks>
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        int outputDim = studentBatchOutput.Columns;
        var gradientBatch = new Matrix<T>(batchSize, outputDim);

        // Standard distillation gradient on final outputs
        // Selectivity gradients require intermediate activations and must be backpropagated separately
        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentOutput = studentBatchOutput.GetRow(r);
            Vector<T> teacherOutput = teacherBatchOutput.GetRow(r);
            Vector<T>? trueLabels = trueLabelsBatch?.GetRow(r);

            var gradient = new Vector<T>(outputDim);

            var studentSoft = DistillationHelper<T>.Softmax(studentOutput, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherOutput, Temperature);

            if (trueLabels != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentOutput, 1.0);

                for (int i = 0; i < outputDim; i++)
                {
                    // Soft gradient (temperature-scaled)
                    var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                    softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                    // Hard gradient
                    var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);

                    // Combined gradient: Alpha * hardGrad + (1 - Alpha) * softGrad
                    var combined = NumOps.Add(
                        NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                        NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softGrad));

                    gradient[i] = combined;
                }
            }
            else
            {
                for (int i = 0; i < outputDim; i++)
                {
                    // Soft gradient (temperature-scaled)
                    var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                    softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                    gradient[i] = softGrad;
                }
            }

            gradientBatch.SetRow(r, gradient);
        }

        return gradientBatch;
    }

    /// <summary>
    /// Computes neuron selectivity loss by comparing activation patterns across a batch.
    /// </summary>
    /// <param name="studentActivations">Student neuron activations for a batch [batchSize x numNeurons].</param>
    /// <param name="teacherActivations">Teacher neuron activations for a batch [batchSize x numNeurons].</param>
    /// <returns>Selectivity matching loss.</returns>
    /// <remarks>
    /// <para>This should be called with intermediate layer activations, not final outputs.
    /// Collect activations for an entire batch, then call this method.</para>
    /// </remarks>
    public T ComputeSelectivityLoss(Vector<T>[] studentActivations, Vector<T>[] teacherActivations)
    {
        if (studentActivations.Length != teacherActivations.Length)
            throw new ArgumentException("Student and teacher must have same batch size");

        if (studentActivations.Length == 0)
            return NumOps.Zero;

        int batchSize = studentActivations.Length;
        int numNeurons = studentActivations[0].Length;

        if (numNeurons == 0)
            throw new ArgumentException(
                "Activation vectors cannot be empty. Each vector must have at least one neuron.",
                nameof(studentActivations));

        // Validate all activation vectors have consistent dimensions
        for (int i = 0; i < batchSize; i++)
        {
            if (studentActivations[i].Length != numNeurons)
                throw new ArgumentException(
                    $"Student activation vector at index {i} has dimension {studentActivations[i].Length}, " +
                    $"but expected {numNeurons} (from activation 0). All activation vectors must have the same dimension.",
                    nameof(studentActivations));

            if (teacherActivations[i].Length != numNeurons)
                throw new ArgumentException(
                    $"Teacher activation vector at index {i} has dimension {teacherActivations[i].Length}, " +
                    $"but expected {numNeurons} (from activation 0). All activation vectors must have the same dimension.",
                    nameof(teacherActivations));
        }

        // Compute selectivity for each neuron
        var studentSelectivity = ComputeSelectivityScores(studentActivations, numNeurons);
        var teacherSelectivity = ComputeSelectivityScores(teacherActivations, numNeurons);

        // MSE between selectivity scores
        T loss = NumOps.Zero;
        for (int i = 0; i < numNeurons; i++)
        {
            var diff = studentSelectivity[i] - teacherSelectivity[i];
            loss = NumOps.Add(loss, NumOps.FromDouble(diff * diff));
        }

        loss = NumOps.Divide(loss, NumOps.FromDouble(numNeurons));
        return NumOps.Multiply(loss, NumOps.FromDouble(_selectivityWeight));
    }

    private double[] ComputeSelectivityScores(Vector<T>[] activations, int numNeurons)
    {
        int batchSize = activations.Length;
        var scores = new double[numNeurons];

        for (int neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++)
        {
            // Collect activations for this neuron across all samples
            var neuronActivations = new double[batchSize];
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                neuronActivations[sampleIdx] = Convert.ToDouble(activations[sampleIdx][neuronIdx]);
            }

            scores[neuronIdx] = _metric switch
            {
                SelectivityMetric.Variance => ComputeVariance(neuronActivations),
                SelectivityMetric.Sparsity => ComputeSparsity(neuronActivations),
                SelectivityMetric.PeakToAverage => ComputePeakToAverage(neuronActivations),
                _ => throw new NotImplementedException($"Metric {_metric} not implemented")
            };
        }

        return scores;
    }

    private double ComputeVariance(double[] values)
    {
        double mean = values.Average();
        double sumSquaredDiff = values.Sum(v => (v - mean) * (v - mean));
        return sumSquaredDiff / values.Length;
    }

    private double ComputeSparsity(double[] values)
    {
        // Percentage of near-zero activations (< 0.01)
        int nearZeroCount = values.Count(v => Math.Abs(v) < 0.01);
        return (double)nearZeroCount / values.Length;
    }

    private double ComputePeakToAverage(double[] values)
    {
        double avg = values.Average();
        double max = values.Max();
        return max / (avg + 1e-10);
    }

    /// <summary>
    /// Computes intermediate activation loss by matching neuron selectivity between teacher and student.
    /// </summary>
    /// <param name="studentIntermediateActivations">Student's intermediate layer activations.</param>
    /// <param name="teacherIntermediateActivations">Teacher's intermediate layer activations.</param>
    /// <returns>The selectivity matching loss (already weighted by selectivityWeight).</returns>
    /// <remarks>
    /// <para>This implements the IIntermediateActivationStrategy interface to properly integrate
    /// selectivity matching into the training loop. The loss is computed from the activations of
    /// the layer specified by targetLayerName in the constructor.</para>
    ///
    /// <para>If the target layer is not found in the activation dictionaries, returns zero loss.</para>
    /// </remarks>
    public T ComputeIntermediateLoss(
        IntermediateActivations<T> studentIntermediateActivations,
        IntermediateActivations<T> teacherIntermediateActivations)
    {
        if (studentIntermediateActivations == null)
            throw new ArgumentNullException(nameof(studentIntermediateActivations));
        if (teacherIntermediateActivations == null)
            throw new ArgumentNullException(nameof(teacherIntermediateActivations));

        // Get activations for the target layer
        var studentMatrix = studentIntermediateActivations.Get(_targetLayerName);
        var teacherMatrix = teacherIntermediateActivations.Get(_targetLayerName);

        // If layer not found, return zero loss
        if (studentMatrix == null || teacherMatrix == null)
            return NumOps.Zero;

        // Validate dimensions match
        if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
        {
            throw new ArgumentException(
                $"Student and teacher activation dimensions must match for layer '{_targetLayerName}'. " +
                $"Student: [{studentMatrix.Rows} x {studentMatrix.Columns}], " +
                $"Teacher: [{teacherMatrix.Rows} x {teacherMatrix.Columns}]");
        }

        int batchSize = studentMatrix.Rows;
        if (batchSize == 0)
            return NumOps.Zero;

        // Convert Matrix<T> rows to Vector<T>[] for ComputeSelectivityLoss
        var studentActivations = new Vector<T>[batchSize];
        var teacherActivations = new Vector<T>[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            studentActivations[i] = studentMatrix.GetRow(i);
            teacherActivations[i] = teacherMatrix.GetRow(i);
        }

        // Compute selectivity loss (already weighted by _selectivityWeight)
        return ComputeSelectivityLoss(studentActivations, teacherActivations);
    }

    /// <summary>
    /// Computes gradients of intermediate activation loss with respect to student activations.
    /// </summary>
    /// <param name="studentIntermediateActivations">Student's intermediate layer activations.</param>
    /// <param name="teacherIntermediateActivations">Teacher's intermediate layer activations.</param>
    /// <returns>Gradients for the target layer (already weighted by selectivityWeight).</returns>
    /// <remarks>
    /// <para>Computes analytical gradients for selectivity loss based on the chosen metric.
    /// For Variance metric, uses ∂var/∂a = (2/B) * (a - mean).
    /// For Sparsity and PeakToAverage, uses numerical approximation due to non-differentiable operations.</para>
    /// </remarks>
    public IntermediateActivations<T> ComputeIntermediateGradient(
        IntermediateActivations<T> studentIntermediateActivations,
        IntermediateActivations<T> teacherIntermediateActivations)
    {
        if (studentIntermediateActivations == null)
            throw new ArgumentNullException(nameof(studentIntermediateActivations));
        if (teacherIntermediateActivations == null)
            throw new ArgumentNullException(nameof(teacherIntermediateActivations));

        var gradients = new IntermediateActivations<T>();

        var studentMatrix = studentIntermediateActivations.Get(_targetLayerName);
        var teacherMatrix = teacherIntermediateActivations.Get(_targetLayerName);

        // If layer not found, return empty gradients
        if (studentMatrix == null || teacherMatrix == null)
            return gradients;

        // Validate dimensions match
        if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
        {
            throw new ArgumentException(
                $"Student and teacher activation dimensions must match for layer '{_targetLayerName}'. " +
                $"Student: [{studentMatrix.Rows} x {studentMatrix.Columns}], " +
                $"Teacher: [{teacherMatrix.Rows} x {teacherMatrix.Columns}]");
        }

        int batchSize = studentMatrix.Rows;
        if (batchSize == 0)
            return gradients;

        int numNeurons = studentMatrix.Columns;
        if (numNeurons == 0)
            return gradients;

        // Convert matrices to vector arrays
        var studentActivations = new Vector<T>[batchSize];
        var teacherActivations = new Vector<T>[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            studentActivations[i] = studentMatrix.GetRow(i);
            teacherActivations[i] = teacherMatrix.GetRow(i);
        }

        // Compute selectivity scores
        var studentSelectivity = ComputeSelectivityScores(studentActivations, numNeurons);
        var teacherSelectivity = ComputeSelectivityScores(teacherActivations, numNeurons);

        // Compute gradients based on metric
        var gradientMatrix = new Matrix<T>(batchSize, numNeurons);

        switch (_metric)
        {
            case SelectivityMetric.Variance:
                ComputeVarianceGradient(studentActivations, studentSelectivity, teacherSelectivity, gradientMatrix);
                break;

            case SelectivityMetric.Sparsity:
                ComputeSparsityGradient(studentActivations, studentSelectivity, teacherSelectivity, gradientMatrix);
                break;

            case SelectivityMetric.PeakToAverage:
                ComputePeakToAverageGradient(studentActivations, studentSelectivity, teacherSelectivity, gradientMatrix);
                break;

            default:
                throw new NotImplementedException($"Gradient for metric {_metric} not implemented");
        }

        // Scale by selectivity weight
        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < numNeurons; c++)
            {
                gradientMatrix[r, c] = NumOps.Multiply(gradientMatrix[r, c], NumOps.FromDouble(_selectivityWeight));
            }
        }

        gradients.Add(_targetLayerName, gradientMatrix);
        return gradients;
    }

    /// <summary>
    /// Computes gradient for variance-based selectivity loss.
    /// </summary>
    /// <remarks>
    /// For variance metric: var_i = (1/B) * Σ_j (a_ji - μ_i)²
    /// Gradient: ∂L/∂a_ji = (2/N) * (studentVar_i - teacherVar_i) * (2/B) * (a_ji - μ_i)
    /// where N = numNeurons, B = batchSize
    /// </remarks>
    private void ComputeVarianceGradient(
        Vector<T>[] studentActivations,
        double[] studentSelectivity,
        double[] teacherSelectivity,
        Matrix<T> gradientMatrix)
    {
        int batchSize = studentActivations.Length;
        int numNeurons = studentActivations[0].Length;

        // For each neuron, compute mean and gradient contribution
        for (int neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++)
        {
            // Compute mean activation for this neuron
            double mean = 0;
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                mean += Convert.ToDouble(studentActivations[sampleIdx][neuronIdx]);
            }
            mean /= batchSize;

            // Selectivity difference for this neuron
            double selectivityDiff = studentSelectivity[neuronIdx] - teacherSelectivity[neuronIdx];

            // Gradient scale: (2/N) * selectivityDiff * (2/B) = (4 / (N*B)) * selectivityDiff
            double gradScale = (4.0 / (numNeurons * batchSize)) * selectivityDiff;

            // For each sample, compute gradient contribution
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                double activation = Convert.ToDouble(studentActivations[sampleIdx][neuronIdx]);
                double grad = gradScale * (activation - mean);
                gradientMatrix[sampleIdx, neuronIdx] = NumOps.FromDouble(grad);
            }
        }
    }

    /// <summary>
    /// Computes gradient for sparsity-based selectivity loss.
    /// </summary>
    /// <remarks>
    /// Sparsity is based on percentage of near-zero activations, which is non-differentiable.
    /// Uses smooth approximation: sigmoid(-k * (|a| - threshold)) where k controls steepness.
    /// This makes activations near threshold contribute more to sparsity gradient.
    /// </remarks>
    private void ComputeSparsityGradient(
        Vector<T>[] studentActivations,
        double[] studentSelectivity,
        double[] teacherSelectivity,
        Matrix<T> gradientMatrix)
    {
        int batchSize = studentActivations.Length;
        int numNeurons = studentActivations[0].Length;
        double threshold = 0.01;
        double k = 100.0; // Steepness of sigmoid approximation

        for (int neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++)
        {
            double selectivityDiff = studentSelectivity[neuronIdx] - teacherSelectivity[neuronIdx];
            double gradScale = (2.0 / numNeurons) * selectivityDiff;

            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                double activation = Convert.ToDouble(studentActivations[sampleIdx][neuronIdx]);
                double absActivation = Math.Abs(activation);

                // Gradient of sigmoid approximation
                double sigmoidArg = -k * (absActivation - threshold);
                double sigmoidVal = 1.0 / (1.0 + Math.Exp(-sigmoidArg));
                double sigmoidGrad = k * sigmoidVal * (1.0 - sigmoidVal);

                // Gradient w.r.t. activation (chain rule through abs)
                double grad = gradScale * sigmoidGrad * (activation >= 0 ? 1.0 : -1.0) / batchSize;
                gradientMatrix[sampleIdx, neuronIdx] = NumOps.FromDouble(grad);
            }
        }
    }

    /// <summary>
    /// Computes gradient for peak-to-average ratio selectivity loss.
    /// </summary>
    /// <remarks>
    /// PeakToAverage: r_i = max_j(a_ji) / (avg_j(a_ji) + ε)
    /// For the max element: ∂r_i/∂a_ji = 1 / (avg + ε)
    /// For non-max elements: ∂r_i/∂a_ji = -max / ((avg + ε)² * B)
    /// </remarks>
    private void ComputePeakToAverageGradient(
        Vector<T>[] studentActivations,
        double[] studentSelectivity,
        double[] teacherSelectivity,
        Matrix<T> gradientMatrix)
    {
        int batchSize = studentActivations.Length;
        int numNeurons = studentActivations[0].Length;
        double epsilon = 1e-10;

        for (int neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++)
        {
            // Find max and average for this neuron
            double maxVal = double.MinValue;
            int maxIdx = 0;
            double sum = 0;

            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                double activation = Convert.ToDouble(studentActivations[sampleIdx][neuronIdx]);
                sum += activation;
                if (activation > maxVal)
                {
                    maxVal = activation;
                    maxIdx = sampleIdx;
                }
            }

            double avg = sum / batchSize;
            double avgPlusEps = avg + epsilon;

            double selectivityDiff = studentSelectivity[neuronIdx] - teacherSelectivity[neuronIdx];
            double gradScale = (2.0 / numNeurons) * selectivityDiff;

            // Gradient for each sample
            for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
            {
                double grad;
                if (sampleIdx == maxIdx)
                {
                    // Gradient for max element: ∂(max/avg)/∂max = 1/avg
                    grad = gradScale * (1.0 / avgPlusEps);
                }
                else
                {
                    // Gradient for non-max: ∂(max/avg)/∂a = -max / (avg² * B)
                    grad = gradScale * (-maxVal / (avgPlusEps * avgPlusEps * batchSize));
                }

                gradientMatrix[sampleIdx, neuronIdx] = NumOps.FromDouble(grad);
            }
        }
    }
}

public enum SelectivityMetric
{
    /// <summary>
    /// Variance of activations (how much the neuron's output varies).
    /// </summary>
    Variance,

    /// <summary>
    /// Sparsity (percentage of near-zero activations).
    /// </summary>
    Sparsity,

    /// <summary>
    /// Peak-to-average ratio (how peaked the activation distribution is).
    /// </summary>
    PeakToAverage
}


