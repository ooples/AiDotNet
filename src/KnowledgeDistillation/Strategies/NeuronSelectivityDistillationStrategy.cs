
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

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
/// var NumOps = MathHelper.GetNumericOperations&lt;double&gt;();
/// T outputLoss = strategy.ComputeLoss(studentOutput, teacherOutput, labels);
/// Matrix&lt;T&gt; outputGrad = strategy.ComputeGradient(studentOutput, teacherOutput, labels);
/// </code></para>
///
/// <para><b>With Intermediate Activations (via IIntermediateActivationStrategy):</b>
/// <code>
/// var NumOps = MathHelper.GetNumericOperations&lt;double&gt;();
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Like What You Like: Knowledge Distill via Neuron Selectivity Transfer",
    "https://arxiv.org/abs/1707.01219",
    Year = 2017,
    Authors = "Zehao Huang, Naiyan Wang")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
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

        // Standard distillation loss on final outputs
        // Selectivity loss requires intermediate activations - use ComputeSelectivityLoss separately
        T totalLoss = Engine.Sum(DistillationHelper<T>.StandardSampleLosses(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha));

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

        // Standard distillation gradient on final outputs:
        // Alpha * hardGrad + (1 - Alpha) * softGrad with labels, the temperature-scaled soft gradient without
        // Selectivity gradients require intermediate activations and must be backpropagated separately
        var gradientBatch = DistillationHelper<T>.StandardSampleGradients(
            studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha);

        // Average gradients over the batch
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        gradientBatch = Engine.MatrixMultiplyScalar(gradientBatch, oneOverBatch);

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
        var diff = Engine.Subtract(studentSelectivity, teacherSelectivity);
        T loss = Engine.DotProduct(diff, diff);

        loss = NumOps.Divide(loss, NumOps.FromDouble(numNeurons));
        return NumOps.Multiply(loss, NumOps.FromDouble(_selectivityWeight));
    }

    private Vector<T> ComputeSelectivityScores(Vector<T>[] activations, int numNeurons)
    {
        // Activations for each neuron across all samples are the columns of a [batchSize, numNeurons] tensor
        var values = DistillationHelper<T>.StackRows(activations);

        var scores = _metric switch
        {
            SelectivityMetric.Variance => ComputeVariance(values),
            SelectivityMetric.Sparsity => ComputeSparsity(values),
            SelectivityMetric.PeakToAverage => ComputePeakToAverage(values),
            _ => throw new ArgumentOutOfRangeException(nameof(_metric), _metric, $"Unrecognised SelectivityMetric '{_metric}'. Valid metrics: Variance, Sparsity, PeakToAverage.")
        };

        return Engine.Reshape(scores, [numNeurons]).ToVector();
    }

    /// <summary>
    /// Per-neuron (population) variance over the batch: [batch, neurons] -> [1, neurons].
    /// </summary>
    private Tensor<T> ComputeVariance(Tensor<T> values)
    {
        var mean = Engine.ReduceMean(values, [0], keepDims: true);
        var centered = Engine.TensorSubtract(values, Engine.TensorBroadcastTo(mean, values.Shape.ToArray()));
        return Engine.ReduceMean(Engine.TensorMultiply(centered, centered), [0], keepDims: true);
    }

    /// <summary>
    /// Per-neuron fraction of near-zero activations (|a| &lt; 0.01): [batch, neurons] -> [1, neurons].
    /// </summary>
    private Tensor<T> ComputeSparsity(Tensor<T> values)
    {
        var nearZero = Engine.TensorLessThan(Engine.TensorAbs(values), NumOps.FromDouble(0.01));
        return Engine.ReduceMean(nearZero, [0], keepDims: true);
    }

    /// <summary>
    /// Per-neuron peak-to-average ratio max / (mean + 1e-10): [batch, neurons] -> [1, neurons].
    /// </summary>
    private Tensor<T> ComputePeakToAverage(Tensor<T> values)
    {
        var avg = Engine.ReduceMean(values, [0], keepDims: true);
        var max = Engine.ReduceMax(values, [0], keepDims: true);
        return Engine.TensorDivide(max, Engine.TensorAddScalar(avg, NumOps.FromDouble(1e-10)));
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
        var activations = DistillationHelper<T>.StackRows(studentActivations);
        var selectivityDiff = Engine.Reshape(
            Tensor<T>.FromVector(Engine.Subtract(studentSelectivity, teacherSelectivity)), [1, numNeurons]);

        Tensor<T> gradientMatrix = _metric switch
        {
            SelectivityMetric.Variance => ComputeVarianceGradient(activations, selectivityDiff),
            SelectivityMetric.Sparsity => ComputeSparsityGradient(activations, selectivityDiff),
            SelectivityMetric.PeakToAverage => ComputePeakToAverageGradient(activations, selectivityDiff),
            _ => throw new ArgumentOutOfRangeException(nameof(_metric), _metric, $"Gradient is not defined for SelectivityMetric '{_metric}'. Valid metrics: Variance, Sparsity, PeakToAverage.")
        };

        // Scale by selectivity weight
        gradientMatrix = Engine.TensorMultiplyScalar(gradientMatrix, NumOps.FromDouble(_selectivityWeight));

        gradients.Add(_targetLayerName, gradientMatrix.ToMatrix());
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
    private Tensor<T> ComputeVarianceGradient(Tensor<T> activations, Tensor<T> selectivityDiff)
    {
        int batchSize = activations.Shape[0];
        int numNeurons = activations.Shape[1];
        int[] shape = activations.Shape.ToArray();

        // Mean activation for each neuron
        var mean = Engine.ReduceMean(activations, [0], keepDims: true);

        // Gradient scale: (2/N) * selectivityDiff * (2/B) = (4 / (N*B)) * selectivityDiff
        var gradScale = Engine.TensorMultiplyScalar(selectivityDiff, NumOps.FromDouble(4.0 / (numNeurons * batchSize)));

        // grad[j, i] = gradScale[i] * (a[j, i] - mean[i])
        return Engine.TensorMultiply(
            Engine.TensorBroadcastTo(gradScale, shape),
            Engine.TensorSubtract(activations, Engine.TensorBroadcastTo(mean, shape)));
    }

    /// <summary>
    /// Computes gradient for sparsity-based selectivity loss.
    /// </summary>
    /// <remarks>
    /// Sparsity is based on percentage of near-zero activations, which is non-differentiable.
    /// Uses smooth approximation: sigmoid(-k * (|a| - threshold)) where k controls steepness.
    /// This makes activations near threshold contribute more to sparsity gradient.
    /// </remarks>
    private Tensor<T> ComputeSparsityGradient(Tensor<T> activations, Tensor<T> selectivityDiff)
    {
        int batchSize = activations.Shape[0];
        int numNeurons = activations.Shape[1];
        int[] shape = activations.Shape.ToArray();
        double threshold = 0.01;
        double k = 100.0; // Steepness of sigmoid approximation

        var gradScale = Engine.TensorMultiplyScalar(selectivityDiff, NumOps.FromDouble(2.0 / numNeurons));

        // Gradient of sigmoid approximation: k * σ * (1 - σ) with σ = sigmoid(-k * (|a| - threshold))
        var sigmoidArg = Engine.TensorMultiplyScalar(
            Engine.TensorSubtractScalar(Engine.TensorAbs(activations), NumOps.FromDouble(threshold)), NumOps.FromDouble(-k));
        var sigmoidVal = Engine.TensorSigmoid(sigmoidArg);
        var sigmoidGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(sigmoidVal, Engine.TensorSubtract(Tensor<T>.CreateDefault(shape, NumOps.One), sigmoidVal)),
            NumOps.FromDouble(k));

        // Gradient w.r.t. activation (chain rule through abs; a >= 0 counts as +1)
        var negative = Engine.TensorLessThan(activations, NumOps.Zero);
        var sign = Engine.TensorSubtract(
            Tensor<T>.CreateDefault(shape, NumOps.One), Engine.TensorMultiplyScalar(negative, NumOps.FromDouble(2.0)));

        return Engine.TensorDivideScalar(
            Engine.TensorMultiply(Engine.TensorMultiply(Engine.TensorBroadcastTo(gradScale, shape), sigmoidGrad), sign),
            NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes gradient for peak-to-average ratio selectivity loss.
    /// </summary>
    /// <remarks>
    /// PeakToAverage: r_i = max_j(a_ji) / (avg_j(a_ji) + ε)
    /// Every element moves the average: ∂r_i/∂a_ji ⊇ -max / ((avg + ε)² * B)
    /// The max element (first index on ties) also moves the peak: + 1 / (avg + ε)
    /// </remarks>
    private Tensor<T> ComputePeakToAverageGradient(Tensor<T> activations, Tensor<T> selectivityDiff)
    {
        int batchSize = activations.Shape[0];
        int numNeurons = activations.Shape[1];
        int[] shape = activations.Shape.ToArray();
        double epsilon = 1e-10;

        // Max and average for each neuron
        var maxVal = Engine.ReduceMax(activations, [0], keepDims: true);
        var avgPlusEps = Engine.TensorAddScalar(Engine.ReduceMean(activations, [0], keepDims: true), NumOps.FromDouble(epsilon));

        // One-hot of the first sample attaining each neuron's max
        var positions = Engine.TensorBroadcastTo(
            Engine.Reshape(Engine.TensorArange<T>(0, batchSize, 1), [batchSize, 1]), shape);
        var candidate = Engine.TensorWhere(
            Engine.TensorEquals(activations, Engine.TensorBroadcastTo(maxVal, shape)),
            positions,
            Tensor<T>.CreateDefault(shape, NumOps.FromDouble(batchSize)));
        var firstMax = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(candidate), [0], keepDims: true));
        var isMax = Engine.TensorEquals(positions, Engine.TensorBroadcastTo(firstMax, shape));

        var gradScale = Engine.TensorMultiplyScalar(selectivityDiff, NumOps.FromDouble(2.0 / numNeurons));

        // ∂(max/avg)/∂max = 1/avg for the peak; every element (peak included) also gets -max / (avg² * B)
        var peakTerm = Engine.TensorDivide(isMax, Engine.TensorBroadcastTo(avgPlusEps, shape));
        var averageTerm = Engine.TensorDivideScalar(
            Engine.TensorDivide(maxVal, Engine.TensorMultiply(avgPlusEps, avgPlusEps)), NumOps.FromDouble(batchSize));
        var ratioGrad = Engine.TensorSubtract(peakTerm, Engine.TensorBroadcastTo(averageTerm, shape));

        return Engine.TensorMultiply(Engine.TensorBroadcastTo(gradScale, shape), ratioGrad);
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


