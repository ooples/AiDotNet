using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.SemiSupervised;

/// <summary>
/// Implements the Label Spreading algorithm for semi-supervised classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Label Spreading is a variant of Label Propagation that uses symmetric normalization
/// and a clamping factor (alpha) to balance between original labels and propagated information.
/// </para>
/// <para>
/// <b>For Beginners:</b> Label Spreading improves on Label Propagation in two ways:
///
/// 1. <b>Symmetric Normalization:</b> Instead of just normalizing rows, it normalizes both
///    rows and columns. This makes the algorithm more stable when you have clusters of
///    very different sizes. Think of it like making sure influence flows equally in both
///    directions between connected points.
///
/// 2. <b>Alpha Parameter:</b> This controls how much the original labels are preserved.
///    With alpha = 0.2, each iteration keeps 20% of the original label and takes 80% from
///    neighbors. This prevents the algorithm from "forgetting" the original labels while
///    still allowing information to spread.
///
/// The result is often more robust predictions, especially on noisy data or when clusters
/// have unequal sizes.
/// </para>
/// </remarks>
public class LabelSpreading<T> : SemiSupervisedClassifierBase<T>
{
    #region Fields

    /// <summary>
    /// The kernel function used to compute similarity between samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The kernel function measures how similar two data points are.
    /// A common choice is the RBF (Radial Basis Function) kernel, which gives high similarity
    /// to nearby points and low similarity to distant points.
    /// </para>
    /// </remarks>
    private readonly IKernelFunction<T> _kernel;

    /// <summary>
    /// Maximum number of iterations for label spreading.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance for stopping the spreading early.
    /// </summary>
    private readonly T _tolerance;

    /// <summary>
    /// The clamping factor (alpha) that balances original labels vs. propagated information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Alpha controls how much the original labels "persist" through iterations.
    /// - alpha = 0: Completely ignores neighbors, only uses original labels (essentially memorization)
    /// - alpha = 1: Completely ignores original labels, only uses neighbors (pure propagation)
    /// - alpha = 0.2 (default): Keeps 20% of original labels, takes 80% from neighbors
    ///
    /// Lower alpha values are more conservative and stay closer to the original labeled examples.
    /// Higher values allow more influence from the graph structure.
    /// </para>
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// The symmetrically normalized affinity matrix (Laplacian-style normalization).
    /// </summary>
    private Matrix<T>? _normalizedAffinity;

    /// <summary>
    /// The combined feature matrix (labeled + unlabeled) after training.
    /// </summary>
    private Matrix<T>? _allFeatures;

    /// <summary>
    /// The label distribution matrix where each row is a sample and each column is a class.
    /// </summary>
    private Matrix<T>? _labelDistributions;

    /// <summary>
    /// The initial label distributions (before propagation).
    /// </summary>
    private Matrix<T>? _initialDistributions;

    /// <summary>
    /// Number of labeled samples stored during training.
    /// </summary>
    private int _numLabeled;

    /// <summary>
    /// Random number generator for tie-breaking.
    /// </summary>
    private readonly Random _random;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the LabelSpreading class with default settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a Label Spreading classifier with sensible defaults:
    /// - RBF kernel for measuring similarity
    /// - Up to 30 iterations of spreading
    /// - Alpha = 0.2 (keeps 20% original labels, 80% from neighbors)
    /// </para>
    /// </remarks>
    public LabelSpreading()
    {
        _kernel = CreateDefaultKernel();
        _maxIterations = 30;
        _tolerance = NumOps.FromDouble(1e-3);
        _alpha = NumOps.FromDouble(0.2);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the LabelSpreading class with specified parameters.
    /// </summary>
    /// <param name="kernel">The kernel function for computing sample similarities. If null, uses RBF kernel.</param>
    /// <param name="maxIterations">Maximum number of spreading iterations. Default is 30.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-3.</param>
    /// <param name="alpha">Clamping factor (0 to 1). Default is 0.2.</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses cryptographically secure random.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you customize how the algorithm works:
    /// - kernel: How to measure similarity between data points
    /// - maxIterations: How many times to spread labels before stopping
    /// - tolerance: How small the changes need to be to stop early
    /// - alpha: How much to trust neighbors vs. original labels (higher = trust neighbors more)
    /// - seed: A number for reproducible results
    /// </para>
    /// </remarks>
    public LabelSpreading(
        IKernelFunction<T>? kernel,
        int maxIterations,
        T tolerance,
        T alpha,
        int? seed)
    {
        _kernel = kernel ?? CreateDefaultKernel();
        _maxIterations = maxIterations;

        // Accept provided tolerance as-is - zero is valid (means run until maxIterations)
        _tolerance = tolerance;

        // Accept provided alpha as-is - zero is valid (means keep original labels, no spreading)
        _alpha = alpha;

        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Validate alpha is in [0, 1]
        if (NumOps.Compare(_alpha, NumOps.Zero) < 0 || NumOps.Compare(_alpha, NumOps.One) > 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be between 0 and 1.");
        }
    }

    #endregion

    #region Semi-Supervised Training

    /// <summary>
    /// Core implementation of semi-supervised training using label spreading.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method does the actual work of learning from both labeled
    /// and unlabeled data. It:
    /// 1. Combines all data into one matrix
    /// 2. Builds a similarity graph with symmetric normalization
    /// 3. Initializes labels (known for labeled samples, uniform for unlabeled)
    /// 4. Repeatedly spreads labels while keeping some of the original information
    /// </para>
    /// </remarks>
    protected override void TrainSemiSupervisedCore(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        _numLabeled = labeledX.Rows;
        int totalSamples = labeledX.Rows + unlabeledX.Rows;

        // Combine labeled and unlabeled data
        _allFeatures = CombineData(labeledX, unlabeledX);

        // Build and normalize affinity matrix using symmetric normalization
        var affinity = BuildAffinityMatrix(_allFeatures);
        _normalizedAffinity = SymmetricNormalize(affinity);

        // Initialize label distributions
        _initialDistributions = InitializeLabelDistributions(labeledY, totalSamples);
        _labelDistributions = CloneMatrix(_initialDistributions);

        // Spread labels
        SpreadLabels();

        // Extract pseudo-labels for unlabeled data
        ExtractPseudoLabels();
    }

    /// <summary>
    /// Core implementation of supervised training (using only labeled data).
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The class labels.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you only have labeled data, Label Spreading stores the
    /// training data for later prediction using similarity-weighted voting.
    /// </para>
    /// </remarks>
    protected override void TrainSupervisedCore(Matrix<T> x, Vector<T> y)
    {
        _numLabeled = x.Rows;
        _allFeatures = x;
        var affinity = BuildAffinityMatrix(x);
        _normalizedAffinity = SymmetricNormalize(affinity);
        _initialDistributions = InitializeLabelDistributions(y, x.Rows);
        _labelDistributions = CloneMatrix(_initialDistributions);
    }

    #endregion

    #region Graph Construction

    /// <summary>
    /// Combines labeled and unlabeled data into a single feature matrix.
    /// </summary>
    /// <param name="labeledX">The labeled feature matrix.</param>
    /// <param name="unlabeledX">The unlabeled feature matrix.</param>
    /// <returns>A combined matrix with labeled samples first, then unlabeled.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This puts all your data together into one big table.
    /// The labeled examples come first, followed by the unlabeled examples.
    /// </para>
    /// </remarks>
    private Matrix<T> CombineData(Matrix<T> labeledX, Matrix<T> unlabeledX)
    {
        int totalRows = labeledX.Rows + unlabeledX.Rows;
        var combined = new Matrix<T>(totalRows, labeledX.Columns);

        for (int i = 0; i < labeledX.Rows; i++)
        {
            for (int j = 0; j < labeledX.Columns; j++)
            {
                combined[i, j] = labeledX[i, j];
            }
        }

        for (int i = 0; i < unlabeledX.Rows; i++)
        {
            for (int j = 0; j < unlabeledX.Columns; j++)
            {
                combined[labeledX.Rows + i, j] = unlabeledX[i, j];
            }
        }

        return combined;
    }

    /// <summary>
    /// Builds the affinity (similarity) matrix using the kernel function.
    /// </summary>
    /// <param name="features">The combined feature matrix.</param>
    /// <returns>A symmetric matrix of pairwise similarities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The affinity matrix is like a table showing how similar
    /// each pair of samples is. High values mean the samples are similar.
    /// </para>
    /// </remarks>
    private Matrix<T> BuildAffinityMatrix(Matrix<T> features)
    {
        int n = features.Rows;
        var affinity = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            var xi = features.GetRow(i);
            for (int j = i; j < n; j++)
            {
                if (i == j)
                {
                    affinity[i, j] = NumOps.Zero;
                }
                else
                {
                    var xj = features.GetRow(j);
                    var similarity = _kernel.Calculate(xi, xj);
                    affinity[i, j] = similarity;
                    affinity[j, i] = similarity;
                }
            }
        }

        return affinity;
    }

    /// <summary>
    /// Applies symmetric normalization to the affinity matrix.
    /// </summary>
    /// <param name="affinity">The affinity matrix.</param>
    /// <returns>The symmetrically normalized affinity matrix D^(-1/2) * W * D^(-1/2).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Symmetric normalization is a key difference from Label Propagation.
    /// Instead of just making rows sum to 1, it normalizes both rows and columns equally.
    ///
    /// The formula is: S = D^(-1/2) * W * D^(-1/2)
    /// where D is the degree matrix (sum of each row) and W is the affinity matrix.
    ///
    /// This makes the influence between two points equal in both directions, which is
    /// more mathematically principled and leads to more stable results.
    /// </para>
    /// </remarks>
    private Matrix<T> SymmetricNormalize(Matrix<T> affinity)
    {
        int n = affinity.Rows;
        var normalized = new Matrix<T>(n, n);

        // Compute degree (row sums)
        var degrees = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                sum = NumOps.Add(sum, affinity[i, j]);
            }
            degrees[i] = sum;
        }

        // Compute D^(-1/2)
        var invSqrtDegrees = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            if (NumOps.Compare(degrees[i], NumOps.Zero) > 0)
            {
                invSqrtDegrees[i] = NumOps.Divide(NumOps.One, NumOps.Sqrt(degrees[i]));
            }
            else
            {
                invSqrtDegrees[i] = NumOps.Zero;
            }
        }

        // Compute D^(-1/2) * W * D^(-1/2)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                normalized[i, j] = NumOps.Multiply(
                    NumOps.Multiply(invSqrtDegrees[i], affinity[i, j]),
                    invSqrtDegrees[j]);
            }
        }

        return normalized;
    }

    #endregion

    #region Label Spreading

    /// <summary>
    /// Initializes the label distribution matrix.
    /// </summary>
    /// <param name="labeledY">The known labels for labeled samples.</param>
    /// <param name="totalSamples">Total number of samples.</param>
    /// <returns>A matrix where each row is a sample and each column is a class probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates the starting point for label spreading.
    /// Labeled samples get probability 1 for their known class and 0 for others.
    /// Unlabeled samples start with equal probability for all classes.
    /// </para>
    /// </remarks>
    private Matrix<T> InitializeLabelDistributions(Vector<T> labeledY, int totalSamples)
    {
        var distributions = new Matrix<T>(totalSamples, NumClasses);
        T uniformProb = NumOps.Divide(NumOps.One, NumOps.FromDouble(NumClasses));

        for (int i = 0; i < labeledY.Length; i++)
        {
            int classIndex = GetClassIndex(labeledY[i]);
            for (int c = 0; c < NumClasses; c++)
            {
                distributions[i, c] = (c == classIndex) ? NumOps.One : NumOps.Zero;
            }
        }

        for (int i = labeledY.Length; i < totalSamples; i++)
        {
            for (int c = 0; c < NumClasses; c++)
            {
                distributions[i, c] = uniformProb;
            }
        }

        return distributions;
    }

    /// <summary>
    /// Spreads labels through the graph iteratively with clamping.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the heart of Label Spreading. In each iteration:
    /// 1. Spread labels through the graph (like Label Propagation)
    /// 2. Mix the result with the original labels using alpha: new = alpha * original + (1-alpha) * spread
    /// 3. Normalize so probabilities sum to 1
    /// 4. Check if changes are small enough to stop
    ///
    /// The alpha mixing ensures that labeled samples don't completely forget their original
    /// labels, while still allowing information to flow through the graph.
    /// </para>
    /// </remarks>
    private void SpreadLabels()
    {
        int n = _labelDistributions!.Rows;
        T oneMinusAlpha = NumOps.Subtract(NumOps.One, _alpha);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Store previous for convergence check
            var prevDistributions = CloneMatrix(_labelDistributions);

            // Spread: Y = S @ Y
            var spread = MultiplyMatrices(_normalizedAffinity!, _labelDistributions);

            // Clamp: Y = alpha * Y_0 + (1 - alpha) * spread
            for (int i = 0; i < n; i++)
            {
                for (int c = 0; c < NumClasses; c++)
                {
                    _labelDistributions[i, c] = NumOps.Add(
                        NumOps.Multiply(_alpha, _initialDistributions![i, c]),
                        NumOps.Multiply(oneMinusAlpha, spread[i, c]));
                }
            }

            // Normalize rows to sum to 1
            NormalizeRows(_labelDistributions);

            // Check convergence
            T maxChange = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                for (int c = 0; c < NumClasses; c++)
                {
                    T change = NumOps.Abs(NumOps.Subtract(_labelDistributions[i, c], prevDistributions[i, c]));
                    if (NumOps.Compare(change, maxChange) > 0)
                    {
                        maxChange = change;
                    }
                }
            }

            if (NumOps.Compare(maxChange, _tolerance) < 0)
            {
                break;
            }
        }
    }

    /// <summary>
    /// Normalizes each row of a matrix to sum to 1.
    /// </summary>
    /// <param name="matrix">The matrix to normalize in place.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After spreading, the values in each row might not add up to 1.
    /// Since we want probabilities (which must sum to 1), we divide each value by the row sum.
    /// </para>
    /// </remarks>
    private void NormalizeRows(Matrix<T> matrix)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            T rowSum = NumOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                rowSum = NumOps.Add(rowSum, matrix[i, j]);
            }

            if (NumOps.Compare(rowSum, NumOps.Zero) > 0)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = NumOps.Divide(matrix[i, j], rowSum);
                }
            }
        }
    }

    /// <summary>
    /// Creates a deep copy of a matrix.
    /// </summary>
    /// <param name="source">The matrix to clone.</param>
    /// <returns>A new matrix with the same values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates an independent copy of a matrix so we can
    /// modify one without affecting the other.
    /// </para>
    /// </remarks>
    private Matrix<T> CloneMatrix(Matrix<T> source)
    {
        var clone = new Matrix<T>(source.Rows, source.Columns);
        for (int i = 0; i < source.Rows; i++)
        {
            for (int j = 0; j < source.Columns; j++)
            {
                clone[i, j] = source[i, j];
            }
        }
        return clone;
    }

    /// <summary>
    /// Multiplies two matrices.
    /// </summary>
    /// <param name="a">First matrix.</param>
    /// <param name="b">Second matrix.</param>
    /// <returns>The product matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matrix multiplication is used to compute how labels spread
    /// through the graph in one step.
    /// </para>
    /// </remarks>
    private Matrix<T> MultiplyMatrices(Matrix<T> a, Matrix<T> b)
    {
        int m = a.Rows;
        int n = a.Columns;
        int p = b.Columns;
        var result = new Matrix<T>(m, p);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(a[i, k], b[k, j]));
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Extracts pseudo-labels from the converged label distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After spreading finishes, each unlabeled sample has a probability
    /// distribution. This method picks the most likely class as the pseudo-label and records
    /// the confidence.
    /// </para>
    /// </remarks>
    private void ExtractPseudoLabels()
    {
        int numUnlabeled = _labelDistributions!.Rows - _numLabeled;
        if (numUnlabeled <= 0) return;

        PseudoLabels = new Vector<T>(numUnlabeled);
        PseudoLabelConfidences = new Vector<T>(numUnlabeled);

        for (int i = 0; i < numUnlabeled; i++)
        {
            int sampleIndex = _numLabeled + i;
            int bestClass = 0;
            T bestProb = _labelDistributions[sampleIndex, 0];

            for (int c = 1; c < NumClasses; c++)
            {
                if (NumOps.Compare(_labelDistributions[sampleIndex, c], bestProb) > 0)
                {
                    bestProb = _labelDistributions[sampleIndex, c];
                    bestClass = c;
                }
            }

            PseudoLabels[i] = ClassLabels![bestClass];
            PseudoLabelConfidences[i] = bestProb;
        }
    }

    #endregion

    #region Prediction

    /// <summary>
    /// Predicts class labels for new samples.
    /// </summary>
    /// <param name="input">The feature matrix of samples to classify.</param>
    /// <returns>A vector of predicted class labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For new samples, we compute their similarity to training samples
    /// and use weighted voting to predict the class.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_allFeatures is null || _labelDistributions is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var sample = input.GetRow(i);
            predictions[i] = PredictSingle(sample);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts the class label for a single sample.
    /// </summary>
    /// <param name="sample">The feature vector of the sample.</param>
    /// <returns>The predicted class label.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> To predict a single sample, we compute its similarity to all
    /// training samples, weight their label distributions by similarity, and pick the
    /// most likely class.
    /// </para>
    /// </remarks>
    private T PredictSingle(Vector<T> sample)
    {
        int n = _allFeatures!.Rows;
        var similarities = new Vector<T>(n);
        T sumSim = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            var trainSample = _allFeatures.GetRow(i);
            similarities[i] = _kernel.Calculate(sample, trainSample);
            sumSim = NumOps.Add(sumSim, similarities[i]);
        }

        if (NumOps.Compare(sumSim, NumOps.Zero) > 0)
        {
            for (int i = 0; i < n; i++)
            {
                similarities[i] = NumOps.Divide(similarities[i], sumSim);
            }
        }

        var distribution = new Vector<T>(NumClasses);
        for (int c = 0; c < NumClasses; c++)
        {
            T weightedSum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                weightedSum = NumOps.Add(weightedSum,
                    NumOps.Multiply(similarities[i], _labelDistributions![i, c]));
            }
            distribution[c] = weightedSum;
        }

        int bestClass = 0;
        T bestProb = distribution[0];
        for (int c = 1; c < NumClasses; c++)
        {
            if (NumOps.Compare(distribution[c], bestProb) > 0)
            {
                bestProb = distribution[c];
                bestClass = c;
            }
        }

        return ClassLabels![bestClass];
    }

    /// <summary>
    /// Predicts class probabilities for each sample.
    /// </summary>
    /// <param name="input">The feature matrix of samples.</param>
    /// <returns>A matrix where each row is a sample and each column is a class probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the full probability distribution over classes
    /// for each sample, useful when you need confidence information.
    /// </para>
    /// </remarks>
    public Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_allFeatures is null || _labelDistributions is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            var sample = input.GetRow(i);
            var probs = PredictProbabilitiesSingle(sample);
            for (int c = 0; c < NumClasses; c++)
            {
                probabilities[i, c] = probs[c];
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Predicts class probabilities for a single sample.
    /// </summary>
    /// <param name="sample">The feature vector of the sample.</param>
    /// <returns>A vector of class probabilities.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes a probability distribution over classes
    /// for one sample using similarity-weighted averaging.
    /// </para>
    /// </remarks>
    private Vector<T> PredictProbabilitiesSingle(Vector<T> sample)
    {
        int n = _allFeatures!.Rows;
        var similarities = new Vector<T>(n);
        T sumSim = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            var trainSample = _allFeatures.GetRow(i);
            similarities[i] = _kernel.Calculate(sample, trainSample);
            sumSim = NumOps.Add(sumSim, similarities[i]);
        }

        if (NumOps.Compare(sumSim, NumOps.Zero) > 0)
        {
            for (int i = 0; i < n; i++)
            {
                similarities[i] = NumOps.Divide(similarities[i], sumSim);
            }
        }

        var distribution = new Vector<T>(NumClasses);
        for (int c = 0; c < NumClasses; c++)
        {
            T weightedSum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                weightedSum = NumOps.Add(weightedSum,
                    NumOps.Multiply(similarities[i], _labelDistributions![i, c]));
            }
            distribution[c] = weightedSum;
        }

        return distribution;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets the index of a class label in the ClassLabels array.
    /// </summary>
    /// <param name="label">The class label to find.</param>
    /// <returns>The zero-based index of the class.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts a label value to its position in our list of classes.
    /// </para>
    /// </remarks>
    private int GetClassIndex(T label)
    {
        for (int i = 0; i < ClassLabels!.Length; i++)
        {
            if (NumOps.Compare(ClassLabels[i], label) == 0)
            {
                return i;
            }
        }
        throw new ArgumentException($"Unknown class label: {label}");
    }

    /// <summary>
    /// Creates the default kernel (RBF/Gaussian).
    /// </summary>
    /// <returns>An RBF kernel with default parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The RBF kernel is a good default choice that works well
    /// for most datasets.
    /// </para>
    /// </remarks>
    private IKernelFunction<T> CreateDefaultKernel()
    {
        return new Kernels.GaussianKernel<T>(1.0);
    }

    #endregion

    #region ICloneable Implementation

    /// <summary>
    /// Creates a deep copy of this classifier.
    /// </summary>
    /// <returns>A new instance with the same parameters and state.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cloning creates an independent copy that can be modified
    /// without affecting the original.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new LabelSpreading<T>(
            _kernel,
            _maxIterations,
            _tolerance,
            _alpha,
            _random.Next());

        if (_allFeatures is not null)
        {
            clone._allFeatures = CloneMatrix(_allFeatures);
        }

        if (_labelDistributions is not null)
        {
            clone._labelDistributions = CloneMatrix(_labelDistributions);
        }

        if (_initialDistributions is not null)
        {
            clone._initialDistributions = CloneMatrix(_initialDistributions);
        }

        if (_normalizedAffinity is not null)
        {
            clone._normalizedAffinity = CloneMatrix(_normalizedAffinity);
        }

        clone._numLabeled = _numLabeled;

        return clone;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enum value for Label Spreading.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This identifies what kind of model this is within the
    /// AiDotNet library's type system.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.LabelSpreading;
    }

    /// <summary>
    /// Gets all learnable parameters of the model as a single vector.
    /// </summary>
    /// <returns>An empty vector as Label Spreading is non-parametric.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike neural networks that have weights to learn, Label Spreading
    /// is a non-parametric method. It stores all training data and computes similarities at
    /// prediction time, so there are no "learned" parameters in the traditional sense.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Label Spreading is non-parametric - it stores training data, not learned weights
        return new Vector<T>(0);
    }

    /// <summary>
    /// Creates a new instance of the model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use (ignored for non-parametric models).</param>
    /// <returns>A new instance of the classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Spreading doesn't have learnable parameters,
    /// this just creates a new instance with the same configuration.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return new LabelSpreading<T>(_kernel, _maxIterations, _tolerance, _alpha, _random.Next());
    }

    /// <summary>
    /// Sets the parameters of this model.
    /// </summary>
    /// <param name="parameters">The parameters to set (ignored for non-parametric models).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Spreading doesn't have learnable parameters,
    /// this method does nothing.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // Non-parametric model - no parameters to set
    }

    /// <summary>
    /// Computes gradients for gradient-based optimization.
    /// </summary>
    /// <param name="input">The input features.</param>
    /// <param name="target">The target labels.</param>
    /// <param name="lossFunction">The loss function (optional).</param>
    /// <returns>An empty gradient vector as Label Spreading doesn't use gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Label Spreading is not trained with gradient descent like neural
    /// networks. Instead, it uses graph-based iterative label spreading with clamping. Therefore,
    /// there are no gradients to compute.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Label Spreading is not gradient-based
        return new Vector<T>(0);
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">The gradients to apply (ignored).</param>
    /// <param name="learningRate">The learning rate (ignored).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Spreading doesn't use gradient-based learning,
    /// this method does nothing.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Non-parametric model - no gradients to apply
    }

    /// <summary>
    /// Creates a new instance of this classifier with default configuration.
    /// </summary>
    /// <returns>A new LabelSpreading instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used internally for operations like cloning or serialization
    /// that need to create a fresh instance of the same type.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new LabelSpreading<T>(_kernel, _maxIterations, _tolerance, _alpha, _random.Next());
    }

    #endregion
}
