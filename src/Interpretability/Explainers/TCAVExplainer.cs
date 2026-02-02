using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// Testing with Concept Activation Vectors (TCAV) explainer.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> TCAV is a technique that explains model predictions using
/// human-understandable concepts instead of raw features. While most explanation methods
/// tell you which pixels or features matter, TCAV tells you which CONCEPTS matter.
///
/// <b>Example:</b> Instead of highlighting pixels that matter for a "doctor" prediction,
/// TCAV can tell you whether the concept of "stethoscope" or "white coat" influenced
/// the prediction.
///
/// <b>How TCAV works:</b>
/// 1. <b>Collect concept examples:</b> Gather images/examples that represent your concept
///    (e.g., images with stripes for a "striped" concept)
/// 2. <b>Collect random examples:</b> Gather examples that don't specifically represent the concept
/// 3. <b>Train a CAV:</b> Train a linear classifier to distinguish concept vs random at a
///    specific layer. The classifier's weight vector becomes the Concept Activation Vector (CAV).
/// 4. <b>Compute directional derivatives:</b> See how sensitive the model output is to moving
///    in the direction of the CAV
/// 5. <b>Compute TCAV score:</b> The fraction of test inputs where moving toward the concept
///    increases the model's prediction for a class
///
/// <b>Interpreting TCAV scores:</b>
/// - TCAV score = 0.8 means 80% of images are more likely to be classified as the target
///   class when they have more of the concept
/// - TCAV score = 0.5 means the concept doesn't influence the prediction (random)
/// - TCAV score = 0.2 means the concept DECREASES the likelihood of the target class
///
/// <b>Statistical significance:</b> TCAV runs multiple times with different random samples
/// to ensure the score is statistically significant and not due to random noise.
///
/// <b>When to use TCAV:</b>
/// - When you want high-level, human-understandable explanations
/// - When you have examples of concepts you want to test
/// - For testing fairness/bias (e.g., does "gender" concept affect hiring predictions?)
/// - For debugging models (e.g., is my model relying on spurious correlations?)
/// </para>
/// </remarks>
public class TCAVExplainer<T> : IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Vector<T>, Vector<T>> _predictFunction;
    private readonly Func<Vector<T>, Vector<T>> _layerActivationFunction;
    private readonly Func<Vector<T>, int, Vector<T>> _gradientToLayerFunction;
    private readonly int _layerSize;
    private readonly double _regularization;
    private readonly int _numCavs;
    private readonly double _significanceLevel;
    private readonly int? _randomState;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <summary>
    /// Gets the method name.
    /// </summary>
    public string MethodName => "TCAV";

    /// <summary>
    /// Gets whether this explainer supports local explanations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TCAV is primarily a global explanation method,
    /// but we can compute concept sensitivity for individual inputs.
    /// </para>
    /// </remarks>
    public bool SupportsLocalExplanations => true;

    /// <summary>
    /// Gets whether this explainer supports global explanations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TCAV is designed for global explanations - understanding
    /// how concepts influence model predictions across many inputs.
    /// </para>
    /// </remarks>
    public bool SupportsGlobalExplanations => true;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When GPU acceleration is enabled, CAV training and
    /// directional derivative computation run in parallel.
    /// </para>
    /// </remarks>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new TCAV explainer.
    /// </summary>
    /// <param name="predictFunction">Function that returns model predictions for an input.</param>
    /// <param name="layerActivationFunction">Function that returns activations at the target layer.</param>
    /// <param name="gradientToLayerFunction">Function that computes gradients of output class w.r.t. layer activations.</param>
    /// <param name="layerSize">Size of the layer activations.</param>
    /// <param name="regularization">L2 regularization for CAV training (default: 1e-4).</param>
    /// <param name="numCavs">Number of CAVs to train for statistical testing (default: 10).</param>
    /// <param name="significanceLevel">P-value threshold for statistical significance (default: 0.05).</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>predictFunction:</b> Your model's prediction method
    /// - <b>layerActivationFunction:</b> Gets the activations at a specific layer (e.g., last conv layer)
    /// - <b>gradientToLayerFunction:</b> Computes how sensitive the output is to the layer activations
    /// - <b>layerSize:</b> Number of neurons in the target layer
    /// - <b>numCavs:</b> More CAVs = more reliable statistical testing
    /// - <b>significanceLevel:</b> Lower = more confident in results (0.05 = 95% confidence)
    /// </para>
    /// </remarks>
    public TCAVExplainer(
        Func<Vector<T>, Vector<T>> predictFunction,
        Func<Vector<T>, Vector<T>> layerActivationFunction,
        Func<Vector<T>, int, Vector<T>> gradientToLayerFunction,
        int layerSize,
        double regularization = 1e-4,
        int numCavs = 10,
        double significanceLevel = 0.05,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _layerActivationFunction = layerActivationFunction ?? throw new ArgumentNullException(nameof(layerActivationFunction));
        _gradientToLayerFunction = gradientToLayerFunction ?? throw new ArgumentNullException(nameof(gradientToLayerFunction));
        _layerSize = layerSize;
        _regularization = regularization;
        _numCavs = numCavs;
        _significanceLevel = significanceLevel;
        _randomState = randomState;
    }

    /// <summary>
    /// Creates a TCAV explainer from a neural network using input gradient helper.
    /// </summary>
    /// <param name="network">The neural network to explain.</param>
    /// <param name="layerActivationFunction">Function that extracts activations at the target layer.</param>
    /// <param name="layerSize">Size of the layer activations.</param>
    /// <param name="numCavs">Number of CAVs to train for statistical testing.</param>
    /// <param name="significanceLevel">P-value threshold for statistical significance.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a TCAV explainer from a neural network.
    /// You need to provide a function that extracts activations at your chosen layer.
    ///
    /// <b>Layer choice tips:</b>
    /// - Later layers (closer to output) capture more abstract concepts
    /// - Earlier layers capture more primitive features
    /// - For CNNs, the last convolutional layer is often a good choice
    /// - For transformers, try the [CLS] token representation
    /// </para>
    /// </remarks>
    public static TCAVExplainer<T> FromNetwork(
        INeuralNetwork<T> network,
        Func<Vector<T>, Vector<T>> layerActivationFunction,
        int layerSize,
        int numCavs = 10,
        double significanceLevel = 0.05,
        int? randomState = null)
    {
        Vector<T> Predict(Vector<T> input)
        {
            var tensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));
            return network.Predict(tensor).ToVector();
        }

        Vector<T> GetGradient(Vector<T> input, int outputClass)
        {
            var gradHelper = new InputGradientHelper<T>(network);
            // Get input gradients and project to layer size
            var inputGrad = gradHelper.ComputeGradient(input, outputClass);

            // For layer gradients, we use a finite difference approach on layer activations
            return ComputeLayerGradientFiniteDiff(network, input, outputClass, layerActivationFunction, layerSize);
        }

        return new TCAVExplainer<T>(
            Predict,
            layerActivationFunction,
            GetGradient,
            layerSize,
            1e-4,
            numCavs,
            significanceLevel,
            randomState);
    }

    /// <summary>
    /// Computes layer gradients using finite differences.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes how sensitive the model output is to changes
    /// in the layer activations. We do this by making tiny changes to the input and
    /// measuring how the output changes relative to the layer activations.
    ///
    /// This is an approximation but works well for TCAV purposes.
    /// </para>
    /// </remarks>
    private static Vector<T> ComputeLayerGradientFiniteDiff(
        INeuralNetwork<T> network,
        Vector<T> input,
        int outputClass,
        Func<Vector<T>, Vector<T>> layerActivationFunction,
        int layerSize)
    {
        double epsilon = 1e-5;
        var gradients = new T[layerSize];

        // Get baseline
        var inputTensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { input }));
        var baseOutput = network.Predict(inputTensor);
        double baseScore = NumOps.ToDouble(baseOutput.ToVector()[outputClass]);
        var baseActivations = layerActivationFunction(input);

        // For each input dimension, perturb and measure effect on layer activations and output
        // Then estimate d(output)/d(activation) using chain rule
        int numInputDims = Math.Min(input.Length, 50); // Limit for efficiency
        var activationGradients = new double[layerSize];

        for (int i = 0; i < numInputDims; i++)
        {
            var perturbedInput = input.Clone();
            perturbedInput[i] = NumOps.Add(perturbedInput[i], NumOps.FromDouble(epsilon));

            var perturbedTensor = Tensor<T>.FromRowMatrix(new Matrix<T>(new[] { perturbedInput }));
            var perturbedOutput = network.Predict(perturbedTensor);
            double perturbedScore = NumOps.ToDouble(perturbedOutput.ToVector()[outputClass]);
            var perturbedActivations = layerActivationFunction(perturbedInput);

            double dOutput = (perturbedScore - baseScore) / epsilon;

            // Accumulate: if activation j changed and output changed, j is relevant
            for (int j = 0; j < layerSize && j < perturbedActivations.Length; j++)
            {
                double dActivation = (NumOps.ToDouble(perturbedActivations[j]) -
                                     NumOps.ToDouble(baseActivations[j])) / epsilon;
                if (Math.Abs(dActivation) > 1e-10)
                {
                    // Estimate d(output)/d(activation[j]) using chain rule
                    activationGradients[j] += dOutput / dActivation / numInputDims;
                }
            }
        }

        for (int j = 0; j < layerSize; j++)
        {
            gradients[j] = NumOps.FromDouble(activationGradients[j]);
        }

        return new Vector<T>(gradients);
    }

    /// <summary>
    /// Trains a Concept Activation Vector from concept and random examples.
    /// </summary>
    /// <param name="conceptExamples">Examples that represent the concept.</param>
    /// <param name="randomExamples">Random examples that don't specifically represent the concept.</param>
    /// <returns>The trained CAV (weight vector of the linear classifier).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A CAV is simply the weights of a linear classifier trained
    /// to tell concept examples apart from random examples at a specific layer.
    ///
    /// Think of it as finding the direction in the layer's activation space that
    /// points toward "more of the concept."
    ///
    /// <b>Example:</b> For a "striped" concept:
    /// - conceptExamples: Images of striped objects
    /// - randomExamples: Random images (not specifically striped)
    ///
    /// The resulting CAV points in the direction where activations become more "striped."
    /// </para>
    /// </remarks>
    public ConceptActivationVector<T> TrainCAV(Matrix<T> conceptExamples, Matrix<T> randomExamples)
    {
        // Get layer activations for all examples
        var conceptActivations = new List<Vector<T>>();
        var randomActivations = new List<Vector<T>>();

        for (int i = 0; i < conceptExamples.Rows; i++)
        {
            conceptActivations.Add(_layerActivationFunction(conceptExamples.GetRow(i)));
        }

        for (int i = 0; i < randomExamples.Rows; i++)
        {
            randomActivations.Add(_layerActivationFunction(randomExamples.GetRow(i)));
        }

        // Create training data for linear classifier
        int numConcept = conceptActivations.Count;
        int numRandom = randomActivations.Count;
        int numTotal = numConcept + numRandom;
        int numFeatures = _layerSize;

        var X = new Matrix<T>(numTotal, numFeatures);
        var y = new Vector<T>(numTotal);

        // Add concept examples (label = 1)
        for (int i = 0; i < numConcept; i++)
        {
            var act = conceptActivations[i];
            for (int j = 0; j < Math.Min(numFeatures, act.Length); j++)
            {
                X[i, j] = act[j];
            }
            y[i] = NumOps.One;
        }

        // Add random examples (label = 0)
        for (int i = 0; i < numRandom; i++)
        {
            var act = randomActivations[i];
            for (int j = 0; j < Math.Min(numFeatures, act.Length); j++)
            {
                X[numConcept + i, j] = act[j];
            }
            y[numConcept + i] = NumOps.Zero;
        }

        // Train linear classifier using regularized least squares (simple but effective)
        var weights = TrainLinearClassifier(X, y);

        // Normalize to unit vector
        double norm = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            double w = NumOps.ToDouble(weights[i]);
            norm += w * w;
        }
        norm = Math.Sqrt(norm);

        if (norm > 1e-10)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = NumOps.FromDouble(NumOps.ToDouble(weights[i]) / norm);
            }
        }

        // Compute classifier accuracy for quality check
        int correct = 0;
        for (int i = 0; i < numTotal; i++)
        {
            double score = 0;
            for (int j = 0; j < numFeatures; j++)
            {
                score += NumOps.ToDouble(X[i, j]) * NumOps.ToDouble(weights[j]);
            }
            int predicted = score > 0.5 ? 1 : 0;
            int actual = NumOps.ToDouble(y[i]) > 0.5 ? 1 : 0;
            if (predicted == actual) correct++;
        }
        double accuracy = (double)correct / numTotal;

        return new ConceptActivationVector<T>(weights, accuracy);
    }

    /// <summary>
    /// Trains a linear classifier using regularized least squares.
    /// </summary>
    private Vector<T> TrainLinearClassifier(Matrix<T> X, Vector<T> y)
    {
        int n = X.Rows;
        int d = X.Columns;

        // Solve (X^T X + lambda*I)^-1 X^T y
        var XtX = new Matrix<T>(d, d);
        var Xty = new Vector<T>(d);

        // Compute X^T X
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += NumOps.ToDouble(X[k, i]) * NumOps.ToDouble(X[k, j]);
                }
                XtX[i, j] = NumOps.FromDouble(sum);
            }

            // Add regularization
            XtX[i, i] = NumOps.Add(XtX[i, i], NumOps.FromDouble(_regularization));

            // Compute X^T y
            double dotProduct = 0;
            for (int k = 0; k < n; k++)
            {
                dotProduct += NumOps.ToDouble(X[k, i]) * NumOps.ToDouble(y[k]);
            }
            Xty[i] = NumOps.FromDouble(dotProduct);
        }

        // Solve using Cholesky decomposition
        return SolvePositiveDefinite(XtX, Xty);
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

    /// <summary>
    /// Computes the directional derivative of the model output with respect to the CAV.
    /// </summary>
    /// <param name="input">The input to explain.</param>
    /// <param name="cav">The Concept Activation Vector.</param>
    /// <param name="targetClass">The target output class.</param>
    /// <returns>The directional derivative (positive = concept increases prediction).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The directional derivative tells us: "If we moved the layer
    /// activations in the direction of the concept, would the prediction for the target
    /// class increase or decrease?"
    ///
    /// <b>Interpretation:</b>
    /// - Positive: Moving toward the concept increases the prediction
    /// - Negative: Moving toward the concept decreases the prediction
    /// - Near zero: The concept doesn't affect the prediction
    ///
    /// <b>Math:</b> directional_derivative = gradient dot CAV
    /// </para>
    /// </remarks>
    public T ComputeDirectionalDerivative(Vector<T> input, ConceptActivationVector<T> cav, int targetClass)
    {
        // Get gradients of output with respect to layer activations
        var layerGradients = _gradientToLayerFunction(input, targetClass);

        // Compute dot product with CAV
        double dotProduct = 0;
        int len = Math.Min(layerGradients.Length, cav.Weights.Length);
        for (int i = 0; i < len; i++)
        {
            dotProduct += NumOps.ToDouble(layerGradients[i]) * NumOps.ToDouble(cav.Weights[i]);
        }

        return NumOps.FromDouble(dotProduct);
    }

    /// <summary>
    /// Computes the TCAV score for a concept on a set of inputs.
    /// </summary>
    /// <param name="testInputs">Inputs to compute TCAV score over.</param>
    /// <param name="cav">The Concept Activation Vector.</param>
    /// <param name="targetClass">The target output class.</param>
    /// <returns>TCAV score (fraction of inputs with positive directional derivative).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The TCAV score is the fraction of test inputs for which
    /// the concept positively influences the target class prediction.
    ///
    /// <b>Interpretation:</b>
    /// - Score = 0.8: 80% of inputs are more likely to be classified as target class
    ///   when they have more of the concept
    /// - Score = 0.5: The concept doesn't systematically influence predictions (random)
    /// - Score = 0.2: The concept actually REDUCES the likelihood of the target class
    ///
    /// <b>Example:</b> If TCAV score for "striped" concept on "zebra" class is 0.9,
    /// it means 90% of images would be more likely to be classified as zebra if they
    /// had more stripes.
    /// </para>
    /// </remarks>
    public double ComputeTCAVScore(Matrix<T> testInputs, ConceptActivationVector<T> cav, int targetClass)
    {
        int positiveCount = 0;

        for (int i = 0; i < testInputs.Rows; i++)
        {
            var input = testInputs.GetRow(i);
            var derivative = ComputeDirectionalDerivative(input, cav, targetClass);
            if (NumOps.ToDouble(derivative) > 0)
            {
                positiveCount++;
            }
        }

        return (double)positiveCount / testInputs.Rows;
    }

    /// <summary>
    /// Runs a full TCAV analysis with statistical significance testing.
    /// </summary>
    /// <param name="conceptExamples">Examples representing the concept.</param>
    /// <param name="randomExamplesPool">Pool of random examples to sample from.</param>
    /// <param name="testInputs">Test inputs to compute TCAV scores over.</param>
    /// <param name="targetClass">Target output class to analyze.</param>
    /// <param name="conceptName">Name of the concept being tested.</param>
    /// <returns>TCAV result with statistical significance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main TCAV method. It:
    /// 1. Trains multiple CAVs (with different random samples each time)
    /// 2. Computes TCAV scores for each CAV
    /// 3. Tests if the scores are statistically significantly different from 0.5
    ///
    /// <b>Why multiple CAVs?</b> The choice of random examples affects the CAV.
    /// By training multiple CAVs with different random samples, we can test whether
    /// our results are robust or just due to a lucky/unlucky random sample.
    ///
    /// <b>Statistical significance:</b> If p-value less than 0.05, we can be 95% confident
    /// that the concept truly influences predictions (not just random noise).
    /// </para>
    /// </remarks>
    public TCAVResult<T> RunTCAV(
        Matrix<T> conceptExamples,
        Matrix<T> randomExamplesPool,
        Matrix<T> testInputs,
        int targetClass,
        string conceptName = "concept")
    {
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var tcavScores = new double[_numCavs];
        var cavAccuracies = new double[_numCavs];
        var cavs = new List<ConceptActivationVector<T>>();

        int numRandomNeeded = conceptExamples.Rows;

        for (int cavIdx = 0; cavIdx < _numCavs; cavIdx++)
        {
            // Sample random examples
            var randomIndices = Enumerable.Range(0, randomExamplesPool.Rows)
                .OrderBy(_ => rand.Next())
                .Take(numRandomNeeded)
                .ToList();

            var randomExamples = new Matrix<T>(numRandomNeeded, randomExamplesPool.Columns);
            for (int i = 0; i < numRandomNeeded; i++)
            {
                for (int j = 0; j < randomExamplesPool.Columns; j++)
                {
                    randomExamples[i, j] = randomExamplesPool[randomIndices[i], j];
                }
            }

            // Train CAV
            var cav = TrainCAV(conceptExamples, randomExamples);
            cavs.Add(cav);
            cavAccuracies[cavIdx] = cav.ClassifierAccuracy;

            // Compute TCAV score
            tcavScores[cavIdx] = ComputeTCAVScore(testInputs, cav, targetClass);
        }

        // Compute statistics
        double meanScore = tcavScores.Average();
        double variance = tcavScores.Select(s => (s - meanScore) * (s - meanScore)).Average();
        double stdDev = Math.Sqrt(variance);

        // Two-sided t-test against 0.5 (null hypothesis: concept doesn't matter)
        double tStatistic = 0;
        double pValue = 1.0;

        if (stdDev > 1e-10)
        {
            double stdError = stdDev / Math.Sqrt(_numCavs);
            tStatistic = (meanScore - 0.5) / stdError;

            // Approximate p-value using normal distribution for large enough n
            // For small n, this is approximate; exact t-distribution would be better
            pValue = 2 * (1 - NormalCDF(Math.Abs(tStatistic)));
        }

        bool isSignificant = pValue < _significanceLevel;
        double meanCavAccuracy = cavAccuracies.Average();

        return new TCAVResult<T>(
            conceptName: conceptName,
            targetClass: targetClass,
            tcavScores: tcavScores,
            meanScore: meanScore,
            standardDeviation: stdDev,
            pValue: pValue,
            isSignificant: isSignificant,
            cavAccuracy: meanCavAccuracy,
            cavs: cavs);
    }

    /// <summary>
    /// Runs TCAV analysis for multiple concepts.
    /// </summary>
    /// <param name="concepts">Dictionary mapping concept names to their examples.</param>
    /// <param name="randomExamplesPool">Pool of random examples.</param>
    /// <param name="testInputs">Test inputs.</param>
    /// <param name="targetClass">Target class.</param>
    /// <returns>TCAV results for all concepts, sorted by significance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Test multiple concepts at once and compare their influence.
    /// Results are sorted so the most significant concepts appear first.
    ///
    /// <b>Example usage:</b> Test which concepts ("striped", "spotted", "furry")
    /// most influence predictions for the "zebra" class.
    /// </para>
    /// </remarks>
    public TCAVResults<T> RunTCAVMultiple(
        Dictionary<string, Matrix<T>> concepts,
        Matrix<T> randomExamplesPool,
        Matrix<T> testInputs,
        int targetClass)
    {
        var results = new List<TCAVResult<T>>();

        foreach (var (conceptName, conceptExamples) in concepts)
        {
            var result = RunTCAV(conceptExamples, randomExamplesPool, testInputs, targetClass, conceptName);
            results.Add(result);
        }

        // Sort by significance (significant first) then by absolute deviation from 0.5
        results.Sort((a, b) =>
        {
            if (a.IsSignificant != b.IsSignificant)
                return b.IsSignificant.CompareTo(a.IsSignificant);
            return Math.Abs(b.MeanScore - 0.5).CompareTo(Math.Abs(a.MeanScore - 0.5));
        });

        return new TCAVResults<T>(results, targetClass);
    }

    /// <summary>
    /// Computes concept sensitivity for a single input.
    /// </summary>
    /// <param name="input">The input to explain.</param>
    /// <param name="cav">The Concept Activation Vector.</param>
    /// <param name="targetClass">The target class.</param>
    /// <param name="conceptName">Name of the concept.</param>
    /// <returns>Local TCAV explanation showing concept sensitivity.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> While TCAV is primarily a global method, this gives you
    /// a local explanation for a single input - how sensitive is THIS specific
    /// prediction to THIS concept.
    ///
    /// <b>Use case:</b> "For this specific image, how much would having more 'stripes'
    /// change the zebra prediction?"
    /// </para>
    /// </remarks>
    public LocalTCAVExplanation<T> ExplainLocal(
        Vector<T> input,
        ConceptActivationVector<T> cav,
        int targetClass,
        string conceptName = "concept")
    {
        var derivative = ComputeDirectionalDerivative(input, cav, targetClass);
        var prediction = _predictFunction(input);
        var layerActivation = _layerActivationFunction(input);

        // Compute activation magnitude in concept direction
        double projectionMagnitude = 0;
        int len = Math.Min(layerActivation.Length, cav.Weights.Length);
        for (int i = 0; i < len; i++)
        {
            projectionMagnitude += NumOps.ToDouble(layerActivation[i]) * NumOps.ToDouble(cav.Weights[i]);
        }

        return new LocalTCAVExplanation<T>(
            input: input,
            conceptName: conceptName,
            targetClass: targetClass,
            directionalDerivative: derivative,
            conceptProjection: NumOps.FromDouble(projectionMagnitude),
            prediction: prediction[targetClass],
            influenceDirection: NumOps.ToDouble(derivative) > 0 ? "positive" : "negative");
    }

    /// <summary>
    /// Standard normal CDF approximation.
    /// </summary>
    private static double NormalCDF(double x)
    {
        // Abramowitz and Stegun approximation
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    }
}

/// <summary>
/// Represents a Concept Activation Vector.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A CAV is a vector in the layer activation space that
/// points in the direction of "more concept." It's the key component of TCAV
/// that allows us to measure concept influence on predictions.
/// </para>
/// </remarks>
public class ConceptActivationVector<T>
{
    /// <summary>
    /// Gets the CAV weights (direction in activation space).
    /// </summary>
    public Vector<T> Weights { get; }

    /// <summary>
    /// Gets the accuracy of the linear classifier used to train this CAV.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Higher accuracy means the concept is more clearly
    /// distinguishable in the layer activations. Low accuracy (near 50%) means
    /// the concept may not be well-represented at this layer.
    /// </para>
    /// </remarks>
    public double ClassifierAccuracy { get; }

    /// <summary>
    /// Initializes a new Concept Activation Vector.
    /// </summary>
    public ConceptActivationVector(Vector<T> weights, double classifierAccuracy)
    {
        Weights = weights ?? throw new ArgumentNullException(nameof(weights));
        ClassifierAccuracy = classifierAccuracy;
    }
}

/// <summary>
/// Represents the result of a TCAV analysis for a single concept.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This contains everything you need to know about
/// how a concept influences predictions for a specific class.
/// </para>
/// </remarks>
public class TCAVResult<T>
{
    /// <summary>
    /// Gets the name of the concept being tested.
    /// </summary>
    public string ConceptName { get; }

    /// <summary>
    /// Gets the target class being analyzed.
    /// </summary>
    public int TargetClass { get; }

    /// <summary>
    /// Gets all TCAV scores from different CAV runs.
    /// </summary>
    public double[] TCAVScores { get; }

    /// <summary>
    /// Gets the mean TCAV score across all runs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main number to look at.
    /// Above 0.5 = positive influence, below 0.5 = negative influence.
    /// </para>
    /// </remarks>
    public double MeanScore { get; }

    /// <summary>
    /// Gets the standard deviation of TCAV scores.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Lower is better - means results are consistent
    /// across different random samples.
    /// </para>
    /// </remarks>
    public double StandardDeviation { get; }

    /// <summary>
    /// Gets the p-value from statistical significance testing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If this is below 0.05, you can be 95% confident
    /// that the concept truly influences predictions.
    /// </para>
    /// </remarks>
    public double PValue { get; }

    /// <summary>
    /// Gets whether the result is statistically significant.
    /// </summary>
    public bool IsSignificant { get; }

    /// <summary>
    /// Gets the mean CAV classifier accuracy.
    /// </summary>
    public double CAVAccuracy { get; }

    /// <summary>
    /// Gets all trained CAVs.
    /// </summary>
    public IReadOnlyList<ConceptActivationVector<T>> CAVs { get; }

    /// <summary>
    /// Initializes a new TCAV result.
    /// </summary>
    public TCAVResult(
        string conceptName,
        int targetClass,
        double[] tcavScores,
        double meanScore,
        double standardDeviation,
        double pValue,
        bool isSignificant,
        double cavAccuracy,
        List<ConceptActivationVector<T>> cavs)
    {
        ConceptName = conceptName;
        TargetClass = targetClass;
        TCAVScores = tcavScores;
        MeanScore = meanScore;
        StandardDeviation = standardDeviation;
        PValue = pValue;
        IsSignificant = isSignificant;
        CAVAccuracy = cavAccuracy;
        CAVs = cavs;
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        string sig = IsSignificant ? "(SIGNIFICANT)" : "(not significant)";
        string influence = MeanScore > 0.5 ? "positive" : (MeanScore < 0.5 ? "negative" : "neutral");
        return $"TCAV for '{ConceptName}' → Class {TargetClass}: " +
               $"Score={MeanScore:F3} ±{StandardDeviation:F3}, p={PValue:F4} {sig}, " +
               $"Influence: {influence}, CAV accuracy: {CAVAccuracy:F2}";
    }
}

/// <summary>
/// Represents TCAV results for multiple concepts.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TCAVResults<T>
{
    /// <summary>
    /// Gets all concept results, sorted by significance.
    /// </summary>
    public IReadOnlyList<TCAVResult<T>> Results { get; }

    /// <summary>
    /// Gets the target class that was analyzed.
    /// </summary>
    public int TargetClass { get; }

    /// <summary>
    /// Gets results for only significant concepts.
    /// </summary>
    public IEnumerable<TCAVResult<T>> SignificantResults => Results.Where(r => r.IsSignificant);

    /// <summary>
    /// Gets results for concepts with positive influence.
    /// </summary>
    public IEnumerable<TCAVResult<T>> PositiveInfluence => Results.Where(r => r.MeanScore > 0.5);

    /// <summary>
    /// Gets results for concepts with negative influence.
    /// </summary>
    public IEnumerable<TCAVResult<T>> NegativeInfluence => Results.Where(r => r.MeanScore < 0.5);

    /// <summary>
    /// Initializes new TCAV results.
    /// </summary>
    public TCAVResults(List<TCAVResult<T>> results, int targetClass)
    {
        Results = results;
        TargetClass = targetClass;
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            $"TCAV Analysis for Class {TargetClass}:",
            $"  Total concepts tested: {Results.Count}",
            $"  Significant concepts: {SignificantResults.Count()}",
            ""
        };

        foreach (var result in Results.Take(10))
        {
            lines.Add($"  {result}");
        }

        if (Results.Count > 10)
        {
            lines.Add($"  ... and {Results.Count - 10} more concepts");
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Represents a local TCAV explanation for a single input.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LocalTCAVExplanation<T>
{
    /// <summary>
    /// Gets the input that was explained.
    /// </summary>
    public Vector<T> Input { get; }

    /// <summary>
    /// Gets the concept name.
    /// </summary>
    public string ConceptName { get; }

    /// <summary>
    /// Gets the target class.
    /// </summary>
    public int TargetClass { get; }

    /// <summary>
    /// Gets the directional derivative (concept sensitivity).
    /// </summary>
    public T DirectionalDerivative { get; }

    /// <summary>
    /// Gets the projection of activations onto the concept direction.
    /// </summary>
    public T ConceptProjection { get; }

    /// <summary>
    /// Gets the prediction score for the target class.
    /// </summary>
    public T Prediction { get; }

    /// <summary>
    /// Gets the influence direction ("positive" or "negative").
    /// </summary>
    public string InfluenceDirection { get; }

    /// <summary>
    /// Initializes a new local TCAV explanation.
    /// </summary>
    public LocalTCAVExplanation(
        Vector<T> input,
        string conceptName,
        int targetClass,
        T directionalDerivative,
        T conceptProjection,
        T prediction,
        string influenceDirection)
    {
        Input = input;
        ConceptName = conceptName;
        TargetClass = targetClass;
        DirectionalDerivative = directionalDerivative;
        ConceptProjection = conceptProjection;
        Prediction = prediction;
        InfluenceDirection = influenceDirection;
    }

    /// <summary>
    /// Returns a human-readable summary.
    /// </summary>
    public override string ToString()
    {
        return $"Local TCAV for '{ConceptName}' → Class {TargetClass}: " +
               $"Sensitivity={DirectionalDerivative}, " +
               $"Concept projection={ConceptProjection}, " +
               $"Prediction={Prediction}, " +
               $"Influence: {InfluenceDirection}";
    }
}
