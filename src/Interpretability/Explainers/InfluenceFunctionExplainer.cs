using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Influence Function explainer for training data attribution.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Influence Functions answer the question: "Which training examples
/// were most responsible for this prediction?"
///
/// This is different from feature attribution (which features matter) - instead it tells
/// you which TRAINING DATA points matter. This is incredibly useful for:
///
/// <b>Use Cases:</b>
/// - <b>Debugging:</b> Finding mislabeled training data
/// - <b>Data cleaning:</b> Identifying harmful training examples
/// - <b>Understanding:</b> Seeing which examples the model learned from
/// - <b>Fairness:</b> Finding training data that causes biased predictions
///
/// <b>How it works:</b>
/// Influence Functions use calculus to efficiently approximate: "What would happen to
/// this test prediction if we removed a specific training example and retrained?"
///
/// Instead of actually retraining (which is expensive), we use the Hessian (second
/// derivatives of the loss) to estimate the effect mathematically.
///
/// <b>The math (simplified):</b>
/// influence(training_point) = (gradient_test) * (inverse_Hessian) * (gradient_train)
///
/// - gradient_test: How the test loss changes with parameters
/// - inverse_Hessian: How parameter changes propagate through the model
/// - gradient_train: How this training point affected the parameters
///
/// <b>Interpretation:</b>
/// - Positive influence: Removing this training point would HURT test performance
/// - Negative influence: Removing this training point would HELP test performance
/// - Large magnitude: This training point had a big effect
/// </para>
/// </remarks>
public class InfluenceFunctionExplainer<T> : IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly INeuralNetwork<T>? _network;
    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>, T> _lossFunction;
    private readonly Func<Vector<T>, Vector<T>, Vector<T>>? _gradientFunction;
    private readonly Matrix<T> _trainingData;
    private readonly Vector<T> _trainingLabels;
    private readonly InverseHessianMethod _method;
    private readonly double _damping;
    private readonly int _maxIterations;
    private readonly int _recursionDepth;
    private readonly double _scale;
    private readonly int? _randomState;
    private GPUExplainerHelper<T>? _gpuHelper;

    // Cached training gradients for efficiency
    private Matrix<T>? _cachedTrainingGradients;

    /// <summary>
    /// Gets the method name.
    /// </summary>
    public string MethodName => "InfluenceFunctions";

    /// <summary>
    /// Gets whether this explainer supports local explanations.
    /// </summary>
    public bool SupportsLocalExplanations => true;

    /// <summary>
    /// Gets whether this explainer supports global explanations.
    /// </summary>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new Influence Function explainer.
    /// </summary>
    /// <param name="network">The neural network model.</param>
    /// <param name="lossFunction">Function that computes loss given prediction and target.</param>
    /// <param name="trainingData">The training data matrix (rows = samples).</param>
    /// <param name="trainingLabels">The training labels.</param>
    /// <param name="method">Method for computing inverse Hessian-vector products.</param>
    /// <param name="damping">Damping factor for Hessian (regularization for numerical stability).</param>
    /// <param name="maxIterations">Maximum iterations for iterative methods.</param>
    /// <param name="recursionDepth">Recursion depth for LiSSA.</param>
    /// <param name="scale">Scale factor for LiSSA updates.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>network:</b> The trained model you want to explain
    /// - <b>lossFunction:</b> How to measure prediction error (e.g., MSE, cross-entropy)
    /// - <b>trainingData:</b> The data used to train the model
    /// - <b>method:</b> Algorithm for computing inverse Hessian (LiSSA is recommended)
    /// - <b>damping:</b> Higher = more stable but less accurate (try 0.01)
    /// </para>
    /// </remarks>
    public InfluenceFunctionExplainer(
        INeuralNetwork<T> network,
        Func<Vector<T>, Vector<T>, T> lossFunction,
        Matrix<T> trainingData,
        Vector<T> trainingLabels,
        InverseHessianMethod method = InverseHessianMethod.LiSSA,
        double damping = 0.01,
        int maxIterations = 100,
        int recursionDepth = 5000,
        double scale = 10.0,
        int? randomState = null)
    {
        _network = network ?? throw new ArgumentNullException(nameof(network));
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _trainingData = trainingData ?? throw new ArgumentNullException(nameof(trainingData));
        _trainingLabels = trainingLabels ?? throw new ArgumentNullException(nameof(trainingLabels));
        _method = method;
        _damping = damping;
        _maxIterations = maxIterations;
        _recursionDepth = recursionDepth;
        _scale = scale;
        _randomState = randomState;

        _predictFunction = input =>
        {
            var tensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));
            return network.Predict(tensor).ToVector();
        };
    }

    /// <summary>
    /// Initializes a new Influence Function explainer with custom gradient function.
    /// </summary>
    /// <param name="predictFunction">Model prediction function.</param>
    /// <param name="lossFunction">Loss computation function.</param>
    /// <param name="gradientFunction">Function that computes gradient of loss w.r.t. parameters.</param>
    /// <param name="trainingData">The training data matrix.</param>
    /// <param name="trainingLabels">The training labels.</param>
    /// <param name="method">Method for computing inverse Hessian-vector products.</param>
    /// <param name="damping">Damping factor for Hessian.</param>
    /// <param name="maxIterations">Maximum iterations for iterative methods.</param>
    /// <param name="recursionDepth">Recursion depth for LiSSA.</param>
    /// <param name="scale">Scale factor for LiSSA updates.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have custom gradient computation.
    /// The gradientFunction should return the gradient of the loss with respect to all
    /// model parameters (weights and biases flattened into a single vector).
    /// </para>
    /// </remarks>
    public InfluenceFunctionExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>, T> lossFunction,
        Func<Vector<T>, Vector<T>, Vector<T>> gradientFunction,
        Matrix<T> trainingData,
        Vector<T> trainingLabels,
        InverseHessianMethod method = InverseHessianMethod.LiSSA,
        double damping = 0.01,
        int maxIterations = 100,
        int recursionDepth = 5000,
        double scale = 10.0,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _gradientFunction = gradientFunction ?? throw new ArgumentNullException(nameof(gradientFunction));
        _trainingData = trainingData ?? throw new ArgumentNullException(nameof(trainingData));
        _trainingLabels = trainingLabels ?? throw new ArgumentNullException(nameof(trainingLabels));
        _method = method;
        _damping = damping;
        _maxIterations = maxIterations;
        _recursionDepth = recursionDepth;
        _scale = scale;
        _randomState = randomState;
    }

    /// <summary>
    /// Computes the influence of all training samples on a test sample.
    /// </summary>
    /// <param name="testInput">The test input to explain.</param>
    /// <param name="testLabel">The true label for the test input.</param>
    /// <returns>Influence scores for each training sample.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you which training examples were most influential
    /// in determining the model's prediction on this test input.
    ///
    /// <b>Interpretation:</b>
    /// - High positive influence: These training examples "taught" the model to make
    ///   this prediction. Removing them would hurt performance on this test.
    /// - High negative influence: These training examples are "fighting against" this
    ///   prediction. Removing them would actually help performance on this test.
    ///
    /// <b>Use this to:</b>
    /// - Find similar training examples (high positive influence)
    /// - Find conflicting training examples (high negative influence)
    /// - Debug wrong predictions by finding harmful training data
    /// </para>
    /// </remarks>
    public InfluenceFunctionResult<T> ComputeInfluence(Vector<T> testInput, T testLabel)
    {
        // Step 1: Compute gradient of test loss w.r.t. parameters
        var testGradient = ComputeGradient(testInput, new Vector<T>(new[] { testLabel }));

        // Step 2: Compute inverse Hessian-vector product: H^(-1) * test_gradient
        var ihvp = ComputeInverseHessianVectorProduct(testGradient);

        // Step 3: Compute training gradients (cached if available)
        EnsureTrainingGradientsComputed();

        // Step 4: Compute influence scores: influence[i] = -train_gradient[i] dot ihvp
        var influences = new T[_trainingData.Rows];
        for (int i = 0; i < _trainingData.Rows; i++)
        {
            double dot = 0;
            for (int j = 0; j < ihvp.Length && j < _cachedTrainingGradients!.Columns; j++)
            {
                dot += NumOps.ToDouble(_cachedTrainingGradients[i, j]) * NumOps.ToDouble(ihvp[j]);
            }
            influences[i] = NumOps.FromDouble(-dot);
        }

        // Get prediction and loss
        var prediction = _predictFunction(testInput);
        var loss = _lossFunction(prediction, new Vector<T>(new[] { testLabel }));

        return new InfluenceFunctionResult<T>(
            testInput: testInput,
            testLabel: testLabel,
            influences: new Vector<T>(influences),
            prediction: prediction,
            loss: loss);
    }

    /// <summary>
    /// Computes the self-influence of each training sample.
    /// </summary>
    /// <returns>Self-influence scores for each training sample.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Self-influence measures how much a training sample
    /// influences its own prediction. This is incredibly useful for DATA CLEANING.
    ///
    /// <b>Key insight:</b> Training samples that are:
    /// - Hard to learn (high loss even after training)
    /// - Very different from other samples
    /// - Potentially mislabeled
    ///
    /// ...will have HIGH self-influence scores.
    ///
    /// <b>Data cleaning workflow:</b>
    /// 1. Compute self-influence for all training samples
    /// 2. Examine samples with highest self-influence
    /// 3. Many of these will be mislabeled or corrupted
    /// 4. Clean/correct these samples
    /// 5. Retrain with cleaner data
    ///
    /// This can significantly improve model quality!
    /// </para>
    /// </remarks>
    public SelfInfluenceResult<T> ComputeSelfInfluence()
    {
        EnsureTrainingGradientsComputed();

        var selfInfluences = new T[_trainingData.Rows];

        for (int i = 0; i < _trainingData.Rows; i++)
        {
            // Get gradient for this training sample
            var trainGradient = new Vector<T>(_cachedTrainingGradients!.Columns);
            for (int j = 0; j < trainGradient.Length; j++)
            {
                trainGradient[j] = _cachedTrainingGradients[i, j];
            }

            // Compute inverse Hessian-vector product
            var ihvp = ComputeInverseHessianVectorProduct(trainGradient);

            // Self-influence = -gradient dot ihvp
            double selfInfluence = 0;
            for (int j = 0; j < ihvp.Length && j < trainGradient.Length; j++)
            {
                selfInfluence += NumOps.ToDouble(trainGradient[j]) * NumOps.ToDouble(ihvp[j]);
            }
            selfInfluences[i] = NumOps.FromDouble(-selfInfluence);
        }

        return new SelfInfluenceResult<T>(
            selfInfluences: new Vector<T>(selfInfluences),
            trainingData: _trainingData,
            trainingLabels: _trainingLabels);
    }

    /// <summary>
    /// Computes TracIn-style influence using gradient checkpoints.
    /// </summary>
    /// <param name="testInput">The test input.</param>
    /// <param name="testLabel">The test label.</param>
    /// <param name="checkpointGradients">Gradients at different training checkpoints.</param>
    /// <returns>TracIn influence scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TracIn (Tracing Influence) is a simpler alternative to
    /// full influence functions. Instead of computing the inverse Hessian (expensive),
    /// it just sums up gradient dot products from different training checkpoints.
    ///
    /// <b>Intuition:</b> If the gradient of a training sample and test sample point
    /// in the same direction at many checkpoints, they're similar and the training
    /// sample influenced the test prediction.
    ///
    /// <b>TracIn vs Influence Functions:</b>
    /// - TracIn is faster (no Hessian computation)
    /// - Influence Functions are more theoretically grounded
    /// - Both give similar rankings in practice
    ///
    /// <b>Requirements:</b> You need to save gradient checkpoints during training.
    /// </para>
    /// </remarks>
    public TracInResult<T> ComputeTracIn(
        Vector<T> testInput,
        T testLabel,
        List<Matrix<T>> checkpointGradients)
    {
        if (checkpointGradients == null || checkpointGradients.Count == 0)
            throw new ArgumentException("At least one checkpoint is required.", nameof(checkpointGradients));

        int numTrainingSamples = _trainingData.Rows;
        var tracInScores = new T[numTrainingSamples];

        // Compute test gradient at current parameters
        var testGradient = ComputeGradient(testInput, new Vector<T>(new[] { testLabel }));

        // For each checkpoint, compute dot product contribution
        foreach (var checkpointGrad in checkpointGradients)
        {
            if (checkpointGrad.Rows != numTrainingSamples)
                throw new ArgumentException("Checkpoint gradients must have same number of rows as training data.");

            for (int i = 0; i < numTrainingSamples; i++)
            {
                double dot = 0;
                int gradLen = Math.Min(testGradient.Length, checkpointGrad.Columns);
                for (int j = 0; j < gradLen; j++)
                {
                    dot += NumOps.ToDouble(testGradient[j]) * NumOps.ToDouble(checkpointGrad[i, j]);
                }
                tracInScores[i] = NumOps.Add(tracInScores[i], NumOps.FromDouble(dot));
            }
        }

        return new TracInResult<T>(
            testInput: testInput,
            testLabel: testLabel,
            tracInScores: new Vector<T>(tracInScores),
            numCheckpoints: checkpointGradients.Count);
    }

    /// <summary>
    /// Computes gradient of loss with respect to model parameters.
    /// </summary>
    private Vector<T> ComputeGradient(Vector<T> input, Vector<T> target)
    {
        if (_gradientFunction != null)
        {
            return _gradientFunction(input, target);
        }

        if (_network != null)
        {
            // Use neural network's backpropagation
            var inputTensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));

            // Forward pass with memory
            _network.SetTrainingMode(true);
            var output = _network.ForwardWithMemory(inputTensor);

            // Compute loss gradient
            var prediction = output.ToVector();
            var lossValue = _lossFunction(prediction, target);

            // Create output gradient (derivative of loss w.r.t. output)
            // For MSE: gradient = 2 * (pred - target)
            var outputGrad = new T[prediction.Length];
            for (int i = 0; i < prediction.Length && i < target.Length; i++)
            {
                outputGrad[i] = NumOps.Multiply(
                    NumOps.FromDouble(2.0),
                    NumOps.Subtract(prediction[i], target[i]));
            }

            var outputGradTensor = new Tensor<T>(new[] { 1, outputGrad.Length });
            for (int i = 0; i < outputGrad.Length; i++)
            {
                outputGradTensor[0, i] = outputGrad[i];
            }

            // Backpropagate
            _network.Backpropagate(outputGradTensor);
            var parameterGradients = _network.GetParameterGradients();

            _network.SetTrainingMode(false);
            return parameterGradients;
        }

        // Numerical gradients as fallback
        return ComputeNumericalGradients(input, target);
    }

    /// <summary>
    /// Computes numerical gradients using finite differences.
    /// </summary>
    /// <remarks>
    /// <para><b>WARNING:</b> This fallback computes INPUT gradients (∂L/∂x) instead of the
    /// mathematically required PARAMETER gradients (∂L/∂θ). Influence functions fundamentally
    /// require gradients with respect to model parameters. This fallback produces approximate
    /// results that may not accurately reflect training point influence.</para>
    /// <para>For correct influence function computation, the model must implement
    /// <see cref="IInterpretableModel{T}"/> with proper parameter gradient support.</para>
    /// </remarks>
    private Vector<T> ComputeNumericalGradients(Vector<T> input, Vector<T> target)
    {
        // WARNING: This computes input gradients instead of parameter gradients.
        // Influence functions require ∂L/∂θ, but we're computing ∂L/∂x as a proxy.
        // Results are approximate and may be mathematically incorrect.
        double epsilon = 1e-5;
        var gradients = new T[input.Length];

        var prediction = _predictFunction(input);
        double baseLoss = NumOps.ToDouble(_lossFunction(prediction, target));

        for (int i = 0; i < input.Length; i++)
        {
            var perturbedInput = input.Clone();
            perturbedInput[i] = NumOps.Add(perturbedInput[i], NumOps.FromDouble(epsilon));

            var perturbedPrediction = _predictFunction(perturbedInput);
            double perturbedLoss = NumOps.ToDouble(_lossFunction(perturbedPrediction, target));

            gradients[i] = NumOps.FromDouble((perturbedLoss - baseLoss) / epsilon);
        }

        return new Vector<T>(gradients);
    }

    /// <summary>
    /// Computes inverse Hessian-vector product using the selected method.
    /// </summary>
    private Vector<T> ComputeInverseHessianVectorProduct(Vector<T> vector)
    {
        return _method switch
        {
            InverseHessianMethod.LiSSA => ComputeIHVP_LiSSA(vector),
            InverseHessianMethod.ConjugateGradient => ComputeIHVP_ConjugateGradient(vector),
            InverseHessianMethod.Direct => ComputeIHVP_Direct(vector),
            _ => ComputeIHVP_LiSSA(vector)
        };
    }

    /// <summary>
    /// Computes inverse Hessian-vector product using LiSSA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> LiSSA (Linear-time Stochastic Second-order Algorithm) is
    /// an efficient way to approximate H^(-1) * v without computing the full Hessian.
    ///
    /// The Hessian H can be huge (millions of parameters × millions of parameters),
    /// but we only need H^(-1) * v for a specific vector v.
    ///
    /// LiSSA works by:
    /// 1. Starting with v as the initial estimate
    /// 2. Iteratively refining: estimate = v + (I - H/scale) * estimate
    /// 3. This converges to H^(-1) * v
    ///
    /// The trick is computing (I - H/scale) * estimate efficiently using
    /// gradients of gradients (Hessian-vector products).
    /// </para>
    /// </remarks>
    private Vector<T> ComputeIHVP_LiSSA(Vector<T> vector)
    {
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int n = vector.Length;
        var estimate = vector.Clone();

        for (int iter = 0; iter < _recursionDepth; iter++)
        {
            // Sample a random training point
            int idx = rand.Next(_trainingData.Rows);
            var input = _trainingData.GetRow(idx);
            var label = _trainingLabels[idx];

            // Compute Hessian-vector product for this sample
            var hvp = ComputeHessianVectorProduct(input, new Vector<T>(new[] { label }), estimate);

            // Update: estimate = vector + (1 - damping) * estimate - hvp / scale
            for (int i = 0; i < n && i < hvp.Length; i++)
            {
                double v = NumOps.ToDouble(vector[i]);
                double e = NumOps.ToDouble(estimate[i]);
                double h = NumOps.ToDouble(hvp[i]);

                estimate[i] = NumOps.FromDouble(v + (1 - _damping) * e - h / _scale);
            }
        }

        return estimate;
    }

    /// <summary>
    /// Computes inverse Hessian-vector product using conjugate gradient.
    /// </summary>
    private Vector<T> ComputeIHVP_ConjugateGradient(Vector<T> vector)
    {
        int n = vector.Length;
        var x = new Vector<T>(n); // Initial guess
        var r = vector.Clone(); // Residual
        var p = r.Clone(); // Search direction

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Compute average Hessian-vector product over training data
            var Ap = ComputeAverageHessianVectorProduct(p);

            // Add damping: Ap = Ap + damping * p
            for (int i = 0; i < n && i < Ap.Length; i++)
            {
                Ap[i] = NumOps.Add(Ap[i], NumOps.Multiply(NumOps.FromDouble(_damping), p[i]));
            }

            // Compute step size
            double rDotR = DotProduct(r, r);
            double pDotAp = DotProduct(p, Ap);

            if (Math.Abs(pDotAp) < 1e-10) break;

            double alpha = rDotR / pDotAp;

            // Update solution: x = x + alpha * p
            for (int i = 0; i < n; i++)
            {
                x[i] = NumOps.Add(x[i], NumOps.Multiply(NumOps.FromDouble(alpha), p[i]));
            }

            // Update residual: r = r - alpha * Ap
            var rNew = new Vector<T>(n);
            for (int i = 0; i < n && i < Ap.Length; i++)
            {
                rNew[i] = NumOps.Subtract(r[i], NumOps.Multiply(NumOps.FromDouble(alpha), Ap[i]));
            }

            double rNewDotRNew = DotProduct(rNew, rNew);
            if (rNewDotRNew < 1e-10) break;

            // Update search direction
            double beta = rNewDotRNew / rDotR;
            for (int i = 0; i < n; i++)
            {
                p[i] = NumOps.Add(rNew[i], NumOps.Multiply(NumOps.FromDouble(beta), p[i]));
            }

            r = rNew;
        }

        return x;
    }

    /// <summary>
    /// Computes inverse Hessian-vector product directly (only for small models).
    /// </summary>
    private Vector<T> ComputeIHVP_Direct(Vector<T> vector)
    {
        // Compute full Hessian (expensive!)
        int n = vector.Length;

        // For very small models only
        if (n > 100)
        {
            // Fall back to LiSSA for larger models
            return ComputeIHVP_LiSSA(vector);
        }

        var hessian = new Matrix<T>(n, n);

        // Compute Hessian columns via Hessian-vector products with unit vectors
        for (int j = 0; j < n; j++)
        {
            // Use Hessian-vector product with unit vector
            var unitVector = new Vector<T>(n);
            unitVector[j] = NumOps.One;

            var hvp = ComputeAverageHessianVectorProduct(unitVector);

            for (int i = 0; i < n && i < hvp.Length; i++)
            {
                hessian[i, j] = hvp[i];
            }

            // Add damping
            hessian[j, j] = NumOps.Add(hessian[j, j], NumOps.FromDouble(_damping));
        }

        // Solve (H + damping*I) * result = vector using Cholesky
        return SolvePositiveDefinite(hessian, vector);
    }

    /// <summary>
    /// Computes Hessian-vector product for a single sample.
    /// </summary>
    private Vector<T> ComputeHessianVectorProduct(Vector<T> input, Vector<T> target, Vector<T> vector)
    {
        // Use finite differences: H*v ≈ (gradient(params + epsilon*v) - gradient(params)) / epsilon
        // Since we don't have direct parameter access, we approximate via input perturbations
        double epsilon = 1e-5;

        var baseGrad = ComputeGradient(input, target);
        int n = baseGrad.Length;
        var hvp = new Vector<T>(n);

        // Approximate Hessian-vector product using second-order finite differences
        for (int i = 0; i < n && i < input.Length; i++)
        {
            double vi = NumOps.ToDouble(vector[i % vector.Length]);
            if (Math.Abs(vi) < 1e-10) continue;

            var perturbedInput = input.Clone();
            perturbedInput[i] = NumOps.Add(perturbedInput[i], NumOps.FromDouble(epsilon * vi));

            var perturbedGrad = ComputeGradient(perturbedInput, target);

            for (int j = 0; j < n && j < perturbedGrad.Length; j++)
            {
                double diff = (NumOps.ToDouble(perturbedGrad[j]) - NumOps.ToDouble(baseGrad[j])) / (epsilon * vi);
                hvp[j] = NumOps.Add(hvp[j], NumOps.FromDouble(diff * vi));
            }
        }

        return hvp;
    }

    /// <summary>
    /// Computes average Hessian-vector product over training data.
    /// </summary>
    private Vector<T> ComputeAverageHessianVectorProduct(Vector<T> vector)
    {
        int n = vector.Length;
        var avgHvp = new Vector<T>(n);
        int numSamples = Math.Min(50, _trainingData.Rows); // Subsample for efficiency

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value + 1000)
            : RandomHelper.CreateSecureRandom();

        for (int s = 0; s < numSamples; s++)
        {
            int idx = rand.Next(_trainingData.Rows);
            var input = _trainingData.GetRow(idx);
            var label = _trainingLabels[idx];

            var hvp = ComputeHessianVectorProduct(input, new Vector<T>(new[] { label }), vector);

            for (int i = 0; i < n && i < hvp.Length; i++)
            {
                avgHvp[i] = NumOps.Add(avgHvp[i], hvp[i]);
            }
        }

        // Average
        for (int i = 0; i < n; i++)
        {
            avgHvp[i] = NumOps.Divide(avgHvp[i], NumOps.FromDouble(numSamples));
        }

        return avgHvp;
    }

    /// <summary>
    /// Computes average gradient over training data.
    /// </summary>
    private Vector<T> ComputeAverageGradient()
    {
        EnsureTrainingGradientsComputed();

        int n = _cachedTrainingGradients!.Columns;
        var avg = new Vector<T>(n);

        for (int i = 0; i < _trainingData.Rows; i++)
        {
            for (int j = 0; j < n; j++)
            {
                avg[j] = NumOps.Add(avg[j], _cachedTrainingGradients[i, j]);
            }
        }

        for (int j = 0; j < n; j++)
        {
            avg[j] = NumOps.Divide(avg[j], NumOps.FromDouble(_trainingData.Rows));
        }

        return avg;
    }

    /// <summary>
    /// Ensures training gradients are computed and cached.
    /// </summary>
    private void EnsureTrainingGradientsComputed()
    {
        if (_cachedTrainingGradients != null) return;

        int numSamples = _trainingData.Rows;
        int numParams = 0;

        // Compute first gradient to determine size
        var firstGrad = ComputeGradient(_trainingData.GetRow(0),
            new Vector<T>(new[] { _trainingLabels[0] }));
        numParams = firstGrad.Length;

        _cachedTrainingGradients = new Matrix<T>(numSamples, numParams);

        for (int j = 0; j < numParams; j++)
        {
            _cachedTrainingGradients[0, j] = firstGrad[j];
        }

        for (int i = 1; i < numSamples; i++)
        {
            var grad = ComputeGradient(_trainingData.GetRow(i),
                new Vector<T>(new[] { _trainingLabels[i] }));

            for (int j = 0; j < numParams && j < grad.Length; j++)
            {
                _cachedTrainingGradients[i, j] = grad[j];
            }
        }
    }

    /// <summary>
    /// Computes dot product of two vectors.
    /// </summary>
    private double DotProduct(Vector<T> a, Vector<T> b)
    {
        double sum = 0;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            sum += NumOps.ToDouble(a[i]) * NumOps.ToDouble(b[i]);
        }
        return sum;
    }

    /// <summary>
    /// Solves a positive definite system using Cholesky decomposition.
    /// </summary>
    private Vector<T> SolvePositiveDefinite(Matrix<T> A, Vector<T> b)
    {
        int n = A.Rows;
        var L = new Matrix<T>(n, n);

        // Cholesky decomposition
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = NumOps.ToDouble(A[i, j]);
                for (int k = 0; k < j; k++)
                {
                    sum -= NumOps.ToDouble(L[i, k]) * NumOps.ToDouble(L[j, k]);
                }

                if (i == j)
                {
                    L[i, j] = NumOps.FromDouble(Math.Sqrt(Math.Max(sum, 1e-10)));
                }
                else
                {
                    double ljj = NumOps.ToDouble(L[j, j]);
                    L[i, j] = NumOps.FromDouble(ljj > 1e-10 ? sum / ljj : 0);
                }
            }
        }

        // Solve L * y = b
        var y = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            double sum = NumOps.ToDouble(b[i]);
            for (int j = 0; j < i; j++)
            {
                sum -= NumOps.ToDouble(L[i, j]) * NumOps.ToDouble(y[j]);
            }
            double lii = NumOps.ToDouble(L[i, i]);
            y[i] = NumOps.FromDouble(lii > 1e-10 ? sum / lii : 0);
        }

        // Solve L^T * x = y
        var x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = NumOps.ToDouble(y[i]);
            for (int j = i + 1; j < n; j++)
            {
                sum -= NumOps.ToDouble(L[j, i]) * NumOps.ToDouble(x[j]);
            }
            double lii = NumOps.ToDouble(L[i, i]);
            x[i] = NumOps.FromDouble(lii > 1e-10 ? sum / lii : 0);
        }

        return x;
    }
}

/// <summary>
/// Methods for computing inverse Hessian-vector products.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Hessian is a huge matrix of second derivatives.
/// We need its inverse times a vector, but computing the inverse directly is too slow.
/// These methods approximate H^(-1) * v efficiently.
/// </para>
/// </remarks>
public enum InverseHessianMethod
{
    /// <summary>
    /// LiSSA (Linear-time Stochastic Second-order Algorithm).
    /// Fast and memory-efficient, recommended for large models.
    /// </summary>
    LiSSA,

    /// <summary>
    /// Conjugate Gradient method.
    /// More accurate but slower than LiSSA.
    /// </summary>
    ConjugateGradient,

    /// <summary>
    /// Direct matrix inversion.
    /// Only feasible for very small models (less than 100 parameters).
    /// </summary>
    Direct
}

/// <summary>
/// Result of influence function computation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class InfluenceFunctionResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the test input that was explained.
    /// </summary>
    public Vector<T> TestInput { get; }

    /// <summary>
    /// Gets the test label.
    /// </summary>
    public T TestLabel { get; }

    /// <summary>
    /// Gets influence scores for each training sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Positive = helpful training sample, negative = harmful.
    /// </para>
    /// </remarks>
    public Vector<T> Influences { get; }

    /// <summary>
    /// Gets the model's prediction on the test input.
    /// </summary>
    public Vector<T> Prediction { get; }

    /// <summary>
    /// Gets the loss on the test input.
    /// </summary>
    public T Loss { get; }

    /// <summary>
    /// Gets the number of training samples.
    /// </summary>
    public int NumTrainingSamples => Influences.Length;

    /// <summary>
    /// Initializes a new influence function result.
    /// </summary>
    public InfluenceFunctionResult(
        Vector<T> testInput,
        T testLabel,
        Vector<T> influences,
        Vector<T> prediction,
        T loss)
    {
        TestInput = testInput;
        TestLabel = testLabel;
        Influences = influences;
        Prediction = prediction;
        Loss = loss;
    }

    /// <summary>
    /// Gets the top K most influential training samples.
    /// </summary>
    /// <param name="k">Number of samples to return.</param>
    /// <returns>Indices and influence scores of top influential samples.</returns>
    public IEnumerable<(int Index, T Influence)> GetTopInfluential(int k = 10)
    {
        return Enumerable.Range(0, Influences.Length)
            .Select(i => (Index: i, Influence: Influences[i]))
            .OrderByDescending(x => Math.Abs(NumOps.ToDouble(x.Influence)))
            .Take(k);
    }

    /// <summary>
    /// Gets the most helpful training samples (highest positive influence).
    /// </summary>
    public IEnumerable<(int Index, T Influence)> GetMostHelpful(int k = 10)
    {
        return Enumerable.Range(0, Influences.Length)
            .Select(i => (Index: i, Influence: Influences[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.Influence))
            .Take(k);
    }

    /// <summary>
    /// Gets the most harmful training samples (lowest/most negative influence).
    /// </summary>
    public IEnumerable<(int Index, T Influence)> GetMostHarmful(int k = 10)
    {
        return Enumerable.Range(0, Influences.Length)
            .Select(i => (Index: i, Influence: Influences[i]))
            .OrderBy(x => NumOps.ToDouble(x.Influence))
            .Take(k);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var topHelpful = GetMostHelpful(3).ToList();
        var topHarmful = GetMostHarmful(3).ToList();

        return $"Influence Function Result:\n" +
               $"  Test loss: {Loss}\n" +
               $"  Top helpful: {string.Join(", ", topHelpful.Select(x => $"#{x.Index}({x.Influence:F4})"))}\n" +
               $"  Top harmful: {string.Join(", ", topHarmful.Select(x => $"#{x.Index}({x.Influence:F4})"))}";
    }
}

/// <summary>
/// Result of self-influence computation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SelfInfluenceResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the self-influence score for each training sample.
    /// </summary>
    public Vector<T> SelfInfluences { get; }

    /// <summary>
    /// Gets the training data.
    /// </summary>
    public Matrix<T> TrainingData { get; }

    /// <summary>
    /// Gets the training labels.
    /// </summary>
    public Vector<T> TrainingLabels { get; }

    /// <summary>
    /// Initializes a new self-influence result.
    /// </summary>
    public SelfInfluenceResult(Vector<T> selfInfluences, Matrix<T> trainingData, Vector<T> trainingLabels)
    {
        SelfInfluences = selfInfluences;
        TrainingData = trainingData;
        TrainingLabels = trainingLabels;
    }

    /// <summary>
    /// Gets samples most likely to be mislabeled or problematic.
    /// </summary>
    /// <param name="k">Number of samples to return.</param>
    /// <returns>Indices and self-influence scores of potentially problematic samples.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> High self-influence often indicates:
    /// - Mislabeled data
    /// - Outliers
    /// - Hard examples
    ///
    /// These are good candidates for manual review in a data cleaning workflow.
    /// </para>
    /// </remarks>
    public IEnumerable<(int Index, T SelfInfluence, Vector<T> Data, T Label)> GetPotentiallyProblematic(int k = 10)
    {
        return Enumerable.Range(0, SelfInfluences.Length)
            .Select(i => (Index: i, SelfInfluence: SelfInfluences[i], Data: TrainingData.GetRow(i), Label: TrainingLabels[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.SelfInfluence))
            .Take(k);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var topProblematic = GetPotentiallyProblematic(5).ToList();

        double mean = 0, max = double.MinValue;
        for (int i = 0; i < SelfInfluences.Length; i++)
        {
            double val = NumOps.ToDouble(SelfInfluences[i]);
            mean += val;
            if (val > max) max = val;
        }
        mean /= SelfInfluences.Length;

        return $"Self-Influence Analysis:\n" +
               $"  Mean self-influence: {mean:F4}\n" +
               $"  Max self-influence: {max:F4}\n" +
               $"  Top problematic samples: {string.Join(", ", topProblematic.Select(x => $"#{x.Index}({NumOps.ToDouble(x.SelfInfluence):F4})"))}";
    }
}

/// <summary>
/// Result of TracIn computation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TracInResult<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the test input.
    /// </summary>
    public Vector<T> TestInput { get; }

    /// <summary>
    /// Gets the test label.
    /// </summary>
    public T TestLabel { get; }

    /// <summary>
    /// Gets TracIn scores for each training sample.
    /// </summary>
    public Vector<T> TracInScores { get; }

    /// <summary>
    /// Gets the number of checkpoints used.
    /// </summary>
    public int NumCheckpoints { get; }

    /// <summary>
    /// Initializes a new TracIn result.
    /// </summary>
    public TracInResult(Vector<T> testInput, T testLabel, Vector<T> tracInScores, int numCheckpoints)
    {
        TestInput = testInput;
        TestLabel = testLabel;
        TracInScores = tracInScores;
        NumCheckpoints = numCheckpoints;
    }

    /// <summary>
    /// Gets the most influential training samples according to TracIn.
    /// </summary>
    public IEnumerable<(int Index, T Score)> GetTopInfluential(int k = 10)
    {
        return Enumerable.Range(0, TracInScores.Length)
            .Select(i => (Index: i, Score: TracInScores[i]))
            .OrderByDescending(x => NumOps.ToDouble(x.Score))
            .Take(k);
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var top = GetTopInfluential(5).ToList();
        return $"TracIn Result (using {NumCheckpoints} checkpoints):\n" +
               $"  Top influential: {string.Join(", ", top.Select(x => $"#{x.Index}({NumOps.ToDouble(x.Score):F4})"))}";
    }
}
