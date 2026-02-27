using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.ContinualLearning;

/// <summary>
/// Implements Data-Free Federated Continual Learning — prevents forgetting without storing real data.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Most continual learning methods require storing old training data
/// (replay buffers) to remember previous tasks. This is a problem in federated learning because
/// clients may not be allowed to store data (privacy constraints, storage limits). Data-Free FCL
/// instead uses the global model itself to generate synthetic "pseudo-samples" that capture
/// knowledge of previous tasks. These synthetic samples are used during training on new tasks
/// to prevent forgetting, without ever storing or sharing real client data.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// 1. After each task, freeze global model as "teacher"
/// 2. Generate synthetic data by optimizing noise inputs to maximize teacher confidence
/// 3. Train new model on real data + knowledge distillation from teacher on synthetic data
/// 4. KD loss: L_kd = KL(teacher(x_syn) || student(x_syn))
/// </code>
///
/// <para>Reference: Data-Free Federated Continual Learning (2024). Extends
/// Luo et al., "Data-Free Knowledge Distillation for Heterogeneous FL," NeurIPS 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class DataFreeFCL<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedContinualLearningStrategy<T>
{
    private readonly double _distillationTemperature;
    private readonly double _distillationWeight;
    private readonly int _syntheticSamplesPerClass;
    private readonly int _generationSteps;
    private Vector<T>? _accumulatedImportance;

    /// <summary>
    /// Creates a new Data-Free FCL strategy.
    /// </summary>
    /// <param name="distillationTemperature">Temperature for KD softmax. Higher = softer distributions. Default: 3.0.</param>
    /// <param name="distillationWeight">Weight of distillation loss vs new-task loss. Default: 0.5.</param>
    /// <param name="syntheticSamplesPerClass">Number of synthetic samples to generate per class. Default: 10.</param>
    /// <param name="generationSteps">Optimization steps for synthetic data generation. Default: 100.</param>
    public DataFreeFCL(
        double distillationTemperature = 3.0,
        double distillationWeight = 0.5,
        int syntheticSamplesPerClass = 10,
        int generationSteps = 100)
    {
        if (distillationTemperature <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(distillationTemperature), "Temperature must be positive.");
        }

        if (distillationWeight < 0 || distillationWeight > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(distillationWeight), "Distillation weight must be in [0, 1].");
        }

        if (syntheticSamplesPerClass <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(syntheticSamplesPerClass), "Must generate at least 1 sample per class.");
        }

        if (generationSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(generationSteps), "Must have at least 1 generation step.");
        }

        _distillationTemperature = distillationTemperature;
        _distillationWeight = distillationWeight;
        _syntheticSamplesPerClass = syntheticSamplesPerClass;
        _generationSteps = generationSteps;
    }

    /// <inheritdoc/>
    public Vector<T> ComputeImportance(Vector<T> modelParameters, Matrix<T> taskData)
    {
        // Data-free approach: importance is estimated from the model's output sensitivity
        // rather than from stored data. We use gradient magnitude as a proxy for importance.
        int paramCount = modelParameters.Length;
        var importance = new T[paramCount];

        // Compute per-parameter sensitivity: parameters with larger magnitude relative
        // to layer statistics are considered more important (proxy for Fisher information).
        double sumSquared = 0;
        for (int i = 0; i < paramCount; i++)
        {
            double v = NumOps.ToDouble(modelParameters[i]);
            sumSquared += v * v;
        }

        double rms = Math.Sqrt(sumSquared / Math.Max(paramCount, 1));

        for (int i = 0; i < paramCount; i++)
        {
            double v = Math.Abs(NumOps.ToDouble(modelParameters[i]));
            // Importance proportional to how far parameter is from zero relative to RMS.
            double imp = rms > 0 ? v / rms : 1.0;
            importance[i] = NumOps.FromDouble(Math.Min(imp, 10.0)); // cap to avoid outliers
        }

        var importanceVector = new Vector<T>(importance);

        // Accumulate across tasks.
        if (_accumulatedImportance == null)
        {
            _accumulatedImportance = importanceVector;
        }
        else
        {
            var acc = new T[paramCount];
            int len = Math.Min(paramCount, _accumulatedImportance.Length);
            for (int i = 0; i < len; i++)
            {
                acc[i] = NumOps.Add(_accumulatedImportance[i], importance[i]);
            }

            for (int i = len; i < paramCount; i++)
            {
                acc[i] = importance[i];
            }

            _accumulatedImportance = new Vector<T>(acc);
        }

        return importanceVector;
    }

    /// <inheritdoc/>
    public T ComputeRegularizationPenalty(Vector<T> currentParameters, Vector<T> referenceParameters,
        Vector<T> importanceWeights, double regularizationStrength)
    {
        // Knowledge distillation regularization: penalize deviation from teacher model
        // weighted by importance (analogous to EWC but motivated by KD).
        double penalty = 0;
        int len = Math.Min(currentParameters.Length,
            Math.Min(referenceParameters.Length, importanceWeights.Length));

        for (int i = 0; i < len; i++)
        {
            double diff = NumOps.ToDouble(currentParameters[i]) - NumOps.ToDouble(referenceParameters[i]);
            double importance = NumOps.ToDouble(importanceWeights[i]);
            penalty += importance * diff * diff;
        }

        // Scale by distillation weight and regularization strength.
        return NumOps.FromDouble(_distillationWeight * regularizationStrength * penalty / Math.Max(len, 1));
    }

    /// <inheritdoc/>
    public Vector<T> ProjectGradient(Vector<T> gradient, Vector<T> importanceWeights)
    {
        // Soft projection: scale down gradient components proportionally to their importance
        // for previous tasks. This is gentler than hard orthogonal projection, suited to
        // the data-free setting where importance estimates are noisier.
        int len = Math.Min(gradient.Length, importanceWeights.Length);
        var projected = new T[gradient.Length];

        double maxImportance = 0;
        for (int i = 0; i < len; i++)
        {
            double imp = NumOps.ToDouble(importanceWeights[i]);
            if (imp > maxImportance)
            {
                maxImportance = imp;
            }
        }

        for (int i = 0; i < len; i++)
        {
            double g = NumOps.ToDouble(gradient[i]);
            double imp = NumOps.ToDouble(importanceWeights[i]);
            // Scale: high-importance → small gradient; low-importance → full gradient.
            double scale = maxImportance > 0 ? 1.0 - (imp / maxImportance) * _distillationWeight : 1.0;
            projected[i] = NumOps.FromDouble(g * Math.Max(scale, 0.01)); // floor to avoid zero gradients
        }

        // Parameters beyond importance length pass through unchanged.
        for (int i = len; i < gradient.Length; i++)
        {
            projected[i] = gradient[i];
        }

        return new Vector<T>(projected);
    }

    /// <inheritdoc/>
    public Vector<T> AggregateImportance(Dictionary<int, Vector<T>> clientImportances, Dictionary<int, double>? clientWeights)
    {
        if (clientImportances.Count == 0)
        {
            throw new ArgumentException("Client importances cannot be empty.", nameof(clientImportances));
        }

        int maxLen = 0;
        foreach (var imp in clientImportances.Values)
        {
            if (imp.Length > maxLen)
            {
                maxLen = imp.Length;
            }
        }

        double totalWeight = 0;
        var aggregated = new double[maxLen];

        foreach (var (clientId, importance) in clientImportances)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            for (int i = 0; i < importance.Length; i++)
            {
                aggregated[i] += w * NumOps.ToDouble(importance[i]);
            }
        }

        var result = new T[maxLen];
        if (totalWeight > 0)
        {
            for (int i = 0; i < maxLen; i++)
            {
                result[i] = NumOps.FromDouble(aggregated[i] / totalWeight);
            }
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Computes the knowledge distillation loss between teacher and student soft predictions.
    /// </summary>
    /// <param name="teacherLogits">Teacher model logits on synthetic data.</param>
    /// <param name="studentLogits">Student model logits on synthetic data.</param>
    /// <returns>KL divergence loss scaled by temperature squared.</returns>
    public double ComputeDistillationLoss(double[] teacherLogits, double[] studentLogits)
    {
        if (teacherLogits.Length != studentLogits.Length)
        {
            throw new ArgumentException("Teacher and student logits must have the same length.");
        }

        int n = teacherLogits.Length;
        if (n == 0)
        {
            return 0;
        }

        // Softmax with temperature.
        double teacherMax = teacherLogits.Max();
        double studentMax = studentLogits.Max();
        double teacherSum = 0, studentSum = 0;

        var teacherProbs = new double[n];
        var studentProbs = new double[n];

        for (int i = 0; i < n; i++)
        {
            teacherProbs[i] = Math.Exp((teacherLogits[i] - teacherMax) / _distillationTemperature);
            studentProbs[i] = Math.Exp((studentLogits[i] - studentMax) / _distillationTemperature);
            teacherSum += teacherProbs[i];
            studentSum += studentProbs[i];
        }

        // KL(teacher || student) = sum(teacher * log(teacher / student))
        double kl = 0;
        for (int i = 0; i < n; i++)
        {
            double p = teacherProbs[i] / teacherSum;
            double q = studentProbs[i] / studentSum;
            if (p > 1e-10)
            {
                kl += p * Math.Log(p / Math.Max(q, 1e-10));
            }
        }

        // Scale by T² as per standard knowledge distillation.
        return kl * _distillationTemperature * _distillationTemperature;
    }

    /// <summary>
    /// Generates synthetic pseudo-samples that activate the teacher model's knowledge for a target class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of storing real data (which may violate privacy), we
    /// create fake inputs that the teacher model is confident about. Starting from random noise,
    /// we iteratively adjust the input so the teacher model's output becomes more and more confident
    /// for a specific class. The resulting synthetic input captures what the teacher "thinks" that
    /// class looks like, preserving its knowledge without using any real data.</para>
    ///
    /// <para>Optimization objective per sample:</para>
    /// <code>
    /// x* = argmin_x { -log(softmax(teacher(x))[c]) + lambda * ||x||^2 }
    /// </code>
    /// <para>where c is the target class and lambda is a regularization term.</para>
    /// </remarks>
    /// <param name="teacherLogitsFn">Function that produces teacher model logits given a flattened input vector.
    /// The caller is responsible for running the teacher model's forward pass.</param>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="inputDim">Dimension of the input space.</param>
    /// <param name="seed">Random seed for initialization. Default: 42.</param>
    /// <returns>Dictionary of class index to list of synthetic input vectors for that class.</returns>
    public Dictionary<int, List<double[]>> GenerateSyntheticData(
        Func<double[], double[]> teacherLogitsFn,
        int numClasses,
        int inputDim,
        int seed = 42)
    {
        if (numClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be positive.");
        }

        if (inputDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputDim), "Input dimension must be positive.");
        }

        var rng = new Random(seed);
        var syntheticData = new Dictionary<int, List<double[]>>();
        double optimLr = 0.01; // learning rate for input optimization
        double regularization = 0.001; // L2 regularization on input norm

        for (int c = 0; c < numClasses; c++)
        {
            var classSamples = new List<double[]>();

            for (int s = 0; s < _syntheticSamplesPerClass; s++)
            {
                // Initialize with random noise.
                var x = new double[inputDim];
                for (int i = 0; i < inputDim; i++)
                {
                    double u1 = 1.0 - rng.NextDouble();
                    double u2 = 1.0 - rng.NextDouble();
                    x[i] = 0.1 * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                }

                // Optimize x to maximize teacher's confidence for class c.
                for (int step = 0; step < _generationSteps; step++)
                {
                    var logits = teacherLogitsFn(x);
                    if (logits.Length != numClasses)
                    {
                        throw new InvalidOperationException(
                            $"Teacher returned {logits.Length} logits but expected {numClasses}.");
                    }

                    // Compute softmax probabilities.
                    double maxLogit = double.NegativeInfinity;
                    for (int i = 0; i < numClasses; i++)
                    {
                        if (logits[i] > maxLogit) maxLogit = logits[i];
                    }

                    double expSum = 0;
                    var probs = new double[numClasses];
                    for (int i = 0; i < numClasses; i++)
                    {
                        probs[i] = Math.Exp(logits[i] - maxLogit);
                        expSum += probs[i];
                    }

                    for (int i = 0; i < numClasses; i++)
                    {
                        probs[i] /= expSum;
                    }

                    // Numerical gradient of -log(p_c) + lambda*||x||² via finite differences.
                    // For efficiency, use the analytic gradient approximation:
                    // d(-log(p_c))/dx ≈ -d(logit_c)/dx + sum_j p_j * d(logit_j)/dx
                    // Since we don't have analytic gradients, use SPSA-style estimation.
                    double epsilon = 0.001;
                    var grad = new double[inputDim];

                    // Use coordinate-wise finite differences (more stable than SPSA for small dims).
                    // For large dims, subsample coordinates.
                    int coordsToUpdate = Math.Min(inputDim, 50); // update subset each step for efficiency
                    for (int coordIdx = 0; coordIdx < coordsToUpdate; coordIdx++)
                    {
                        int i = rng.Next(inputDim);

                        x[i] += epsilon;
                        var logitsPlus = teacherLogitsFn(x);
                        x[i] -= 2 * epsilon;
                        var logitsMinus = teacherLogitsFn(x);
                        x[i] += epsilon; // restore

                        // Compute p_c for both perturbations.
                        double pPlus = ComputeSoftmaxProb(logitsPlus, c);
                        double pMinus = ComputeSoftmaxProb(logitsMinus, c);

                        // Gradient of -log(p_c): -(p_plus - p_minus) / (2*epsilon*p_c)
                        double pc = Math.Max(probs[c], 1e-10);
                        grad[i] = -(pPlus - pMinus) / (2 * epsilon * pc) + 2 * regularization * x[i];
                    }

                    // Update x.
                    for (int i = 0; i < inputDim; i++)
                    {
                        x[i] -= optimLr * grad[i];
                    }
                }

                classSamples.Add(x);
            }

            syntheticData[c] = classSamples;
        }

        return syntheticData;
    }

    private static double ComputeSoftmaxProb(double[] logits, int classIdx)
    {
        double maxLogit = double.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }

        double expSum = 0;
        double expTarget = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double e = Math.Exp(logits[i] - maxLogit);
            expSum += e;
            if (i == classIdx) expTarget = e;
        }

        return expTarget / expSum;
    }

    /// <summary>Gets the distillation temperature.</summary>
    public double DistillationTemperature => _distillationTemperature;

    /// <summary>Gets the distillation weight.</summary>
    public double DistillationWeight => _distillationWeight;

    /// <summary>Gets the number of synthetic samples per class.</summary>
    public int SyntheticSamplesPerClass => _syntheticSamplesPerClass;

    /// <summary>Gets the number of generation optimization steps.</summary>
    public int GenerationSteps => _generationSteps;
}
