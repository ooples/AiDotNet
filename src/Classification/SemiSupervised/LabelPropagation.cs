using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.SemiSupervised;

/// <summary>
/// Implements the Label Propagation algorithm for semi-supervised classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Label Propagation is a graph-based semi-supervised learning algorithm that propagates labels
/// from labeled samples to unlabeled samples through a similarity graph.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you have a few people in a social network whose political views
/// you know (labeled), and many others whose views you don't know (unlabeled). Label Propagation
/// assumes that connected people (friends) tend to have similar views.
///
/// The algorithm builds a graph where people are connected based on similarity, then lets the
/// known labels "spread" through the connections. After many rounds of spreading, even people
/// far from the labeled ones get assigned labels based on how the information flowed through
/// the network.
///
/// The key insight is that similar data points should have similar labels, and this similarity
/// can be captured through a graph structure.
/// </para>
/// </remarks>
public class LabelPropagation<T> : SemiSupervisedClassifierBase<T>
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
    /// Maximum number of iterations for label propagation.
    /// </summary>
    private readonly int _maxIterations;

    /// <summary>
    /// Convergence tolerance for stopping the propagation early.
    /// </summary>
    private readonly T _tolerance;

    /// <summary>
    /// The affinity matrix representing pairwise similarities between all samples.
    /// </summary>
    private Matrix<T>? _affinityMatrix;

    /// <summary>
    /// The combined feature matrix (labeled + unlabeled) after training.
    /// </summary>
    private Matrix<T>? _allFeatures;

    /// <summary>
    /// The label distribution matrix where each row is a sample and each column is a class.
    /// </summary>
    private Matrix<T>? _labelDistributions;

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
    /// Initializes a new instance of the LabelPropagation class with default settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a Label Propagation classifier with sensible defaults.
    /// It will automatically measure similarity using a Gaussian/RBF kernel and run up to
    /// 1000 iterations to spread labels through the graph.
    /// </para>
    /// </remarks>
    public LabelPropagation()
    {
        _kernel = CreateDefaultKernel();
        _maxIterations = 1000;
        _tolerance = NumOps.FromDouble(1e-3);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the LabelPropagation class with specified parameters.
    /// </summary>
    /// <param name="kernel">The kernel function for computing sample similarities. If null, uses RBF kernel.</param>
    /// <param name="maxIterations">Maximum number of propagation iterations. Default is 1000.</param>
    /// <param name="tolerance">Convergence tolerance. Default is 1e-3.</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses cryptographically secure random.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you customize how the algorithm works:
    /// - kernel: How to measure similarity between data points
    /// - maxIterations: How many times to spread labels before stopping
    /// - tolerance: How small the changes need to be to consider the algorithm "converged"
    /// - seed: A number for reproducible results (same seed = same results)
    /// </para>
    /// </remarks>
    public LabelPropagation(
        IKernelFunction<T>? kernel,
        int maxIterations,
        T tolerance,
        int? seed)
    {
        _kernel = kernel ?? CreateDefaultKernel();
        _maxIterations = maxIterations;

        // Use provided tolerance if it's not the default zero value, otherwise use 1e-3
        if (tolerance is null || NumOps.Compare(tolerance, NumOps.Zero) == 0)
        {
            _tolerance = NumOps.FromDouble(1e-3);
        }
        else
        {
            _tolerance = tolerance;
        }

        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    #endregion

    #region Semi-Supervised Training

    /// <summary>
    /// Core implementation of semi-supervised training using label propagation.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method does the actual work of learning from both labeled
    /// and unlabeled data. It:
    /// 1. Combines all data into one big matrix
    /// 2. Builds a similarity graph connecting similar samples
    /// 3. Initializes labels (known for labeled samples, unknown for unlabeled)
    /// 4. Repeatedly spreads labels through the graph until they stabilize
    /// </para>
    /// </remarks>
    protected override void TrainSemiSupervisedCore(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        _numLabeled = labeledX.Rows;
        int totalSamples = labeledX.Rows + unlabeledX.Rows;

        // Combine labeled and unlabeled data
        _allFeatures = CombineData(labeledX, unlabeledX);

        // Build affinity matrix
        _affinityMatrix = BuildAffinityMatrix(_allFeatures);

        // Normalize the affinity matrix to create transition probabilities
        var transitionMatrix = NormalizeAffinity(_affinityMatrix);

        // Initialize label distributions
        _labelDistributions = InitializeLabelDistributions(labeledY, totalSamples);

        // Store the initial labeled distributions (they won't change)
        var fixedLabels = new Matrix<T>(_numLabeled, NumClasses);
        for (int i = 0; i < _numLabeled; i++)
        {
            for (int j = 0; j < NumClasses; j++)
            {
                fixedLabels[i, j] = _labelDistributions[i, j];
            }
        }

        // Propagate labels
        PropagateLabels(transitionMatrix, fixedLabels);

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
    /// <b>For Beginners:</b> When you only have labeled data and no unlabeled data,
    /// Label Propagation essentially becomes a nearest-neighbor classifier. This method
    /// just stores the training data for later prediction.
    /// </para>
    /// </remarks>
    protected override void TrainSupervisedCore(Matrix<T> x, Vector<T> y)
    {
        _numLabeled = x.Rows;
        _allFeatures = x;
        _affinityMatrix = BuildAffinityMatrix(x);
        _labelDistributions = InitializeLabelDistributions(y, x.Rows);
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
    /// The labeled examples come first, followed by the unlabeled examples. This
    /// combined view is needed to build the similarity graph.
    /// </para>
    /// </remarks>
    private Matrix<T> CombineData(Matrix<T> labeledX, Matrix<T> unlabeledX)
    {
        int totalRows = labeledX.Rows + unlabeledX.Rows;
        var combined = new Matrix<T>(totalRows, labeledX.Columns);

        // Copy labeled data
        for (int i = 0; i < labeledX.Rows; i++)
        {
            for (int j = 0; j < labeledX.Columns; j++)
            {
                combined[i, j] = labeledX[i, j];
            }
        }

        // Copy unlabeled data
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
    /// each pair of samples is. If sample A and sample B are very similar, their
    /// entry in this table will be high. If they're very different, it will be low.
    ///
    /// This matrix forms the foundation of the graph - high similarities become
    /// strong connections, allowing labels to flow more easily between similar samples.
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
                    // Self-similarity is 0 (no self-loops in the graph)
                    affinity[i, j] = NumOps.Zero;
                }
                else
                {
                    var xj = features.GetRow(j);
                    var similarity = _kernel.Calculate(xi, xj);
                    affinity[i, j] = similarity;
                    affinity[j, i] = similarity; // Symmetric
                }
            }
        }

        return affinity;
    }

    /// <summary>
    /// Normalizes the affinity matrix to create transition probabilities.
    /// </summary>
    /// <param name="affinity">The affinity matrix.</param>
    /// <returns>A row-normalized transition matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Raw similarity values need to be normalized so they work
    /// like probabilities. This method divides each similarity by the sum of all
    /// similarities for that sample. After normalization, each row sums to 1, and
    /// the values represent how likely a label is to spread from one sample to another.
    /// </para>
    /// </remarks>
    private Matrix<T> NormalizeAffinity(Matrix<T> affinity)
    {
        int n = affinity.Rows;
        var transition = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            T rowSum = NumOps.Zero;
            for (int j = 0; j < n; j++)
            {
                rowSum = NumOps.Add(rowSum, affinity[i, j]);
            }

            if (NumOps.Compare(rowSum, NumOps.Zero) > 0)
            {
                for (int j = 0; j < n; j++)
                {
                    transition[i, j] = NumOps.Divide(affinity[i, j], rowSum);
                }
            }
        }

        return transition;
    }

    #endregion

    #region Label Propagation

    /// <summary>
    /// Initializes the label distribution matrix.
    /// </summary>
    /// <param name="labeledY">The known labels for labeled samples.</param>
    /// <param name="totalSamples">Total number of samples (labeled + unlabeled).</param>
    /// <returns>A matrix where each row is a sample and each column is a class probability.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates the starting point for label propagation.
    /// For labeled samples, we set the probability of their known class to 1 and all
    /// other classes to 0. For unlabeled samples, we start with uniform probability
    /// across all classes (complete uncertainty).
    /// </para>
    /// </remarks>
    private Matrix<T> InitializeLabelDistributions(Vector<T> labeledY, int totalSamples)
    {
        var distributions = new Matrix<T>(totalSamples, NumClasses);
        T uniformProb = NumOps.Divide(NumOps.One, NumOps.FromDouble(NumClasses));

        // Initialize labeled samples with one-hot encoding
        for (int i = 0; i < labeledY.Length; i++)
        {
            int classIndex = GetClassIndex(labeledY[i]);
            for (int c = 0; c < NumClasses; c++)
            {
                distributions[i, c] = (c == classIndex) ? NumOps.One : NumOps.Zero;
            }
        }

        // Initialize unlabeled samples with uniform distribution
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
    /// Propagates labels through the graph iteratively.
    /// </summary>
    /// <param name="transitionMatrix">The normalized transition probability matrix.</param>
    /// <param name="fixedLabels">The label distributions for labeled samples (these remain fixed).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the heart of the algorithm. In each iteration:
    /// 1. Each sample's label distribution is updated based on its neighbors' distributions
    /// 2. The labeled samples' distributions are reset to their original values
    /// 3. We check if the changes are small enough to stop early
    ///
    /// Over time, labels "spread" from labeled samples to nearby unlabeled samples,
    /// like ink spreading through water.
    /// </para>
    /// </remarks>
    private void PropagateLabels(Matrix<T> transitionMatrix, Matrix<T> fixedLabels)
    {
        int n = _labelDistributions!.Rows;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Store previous distributions for convergence check
            var prevDistributions = new Matrix<T>(n, NumClasses);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < NumClasses; j++)
                {
                    prevDistributions[i, j] = _labelDistributions[i, j];
                }
            }

            // Propagate: new_dist = transition_matrix @ old_dist
            var newDistributions = MultiplyMatrices(transitionMatrix, _labelDistributions);

            // Clamp the labeled samples back to their original distributions
            for (int i = 0; i < _numLabeled; i++)
            {
                for (int c = 0; c < NumClasses; c++)
                {
                    newDistributions[i, c] = fixedLabels[i, c];
                }
            }

            _labelDistributions = newDistributions;

            // Check convergence
            T maxChange = NumOps.Zero;
            for (int i = _numLabeled; i < n; i++)
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
    /// Multiplies two matrices.
    /// </summary>
    /// <param name="a">First matrix (m x n).</param>
    /// <param name="b">Second matrix (n x p).</param>
    /// <returns>Result matrix (m x p).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matrix multiplication combines two tables of numbers in a specific way.
    /// Here, it's used to compute how label information flows through the graph - the transition
    /// matrix says "how connected" samples are, and the label matrix says "what labels they have."
    /// Multiplying them together tells us what new labels each sample should have after one step of propagation.
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
    /// <b>For Beginners:</b> After propagation finishes, each unlabeled sample has a probability
    /// distribution over all classes. This method picks the most likely class as the pseudo-label
    /// and records the confidence (how sure the algorithm is about that choice).
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
    /// <b>For Beginners:</b> For new samples not seen during training, we compute their
    /// similarity to all training samples, then use those similarities as weights to
    /// combine the training samples' label distributions. The class with the highest
    /// weighted probability becomes the prediction.
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
    /// <b>For Beginners:</b> To predict a single sample, we:
    /// 1. Compute how similar it is to each training sample
    /// 2. Use those similarities as weights to average the training samples' label distributions
    /// 3. Pick the class with the highest average probability
    /// </para>
    /// </remarks>
    private T PredictSingle(Vector<T> sample)
    {
        // Compute similarities to all training samples
        int n = _allFeatures!.Rows;
        var similarities = new Vector<T>(n);
        T sumSim = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            var trainSample = _allFeatures.GetRow(i);
            similarities[i] = _kernel.Calculate(sample, trainSample);
            sumSim = NumOps.Add(sumSim, similarities[i]);
        }

        // Normalize similarities
        if (NumOps.Compare(sumSim, NumOps.Zero) > 0)
        {
            for (int i = 0; i < n; i++)
            {
                similarities[i] = NumOps.Divide(similarities[i], sumSim);
            }
        }

        // Compute weighted label distribution
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

        // Find best class
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
    /// <b>For Beginners:</b> Instead of just predicting the most likely class, this method
    /// returns the probability of each class for each sample. This is useful when you need
    /// to know how confident the model is or want to set custom decision thresholds.
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
    /// <b>For Beginners:</b> This computes a probability distribution over all classes
    /// for one sample, using the similarity-weighted average of training samples' distributions.
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
    /// <b>For Beginners:</b> Class labels might be any values (1, 2, 3 or "cat", "dog", "bird").
    /// This method converts a label to its position in our internal list of classes, which
    /// is needed for the one-hot encoding used in label distributions.
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
    /// <b>For Beginners:</b> The RBF (Radial Basis Function) kernel is the most common choice
    /// for Label Propagation. It gives high similarity to points that are close together
    /// and low similarity to points that are far apart. The "gamma" parameter controls
    /// how quickly similarity drops off with distance.
    /// </para>
    /// </remarks>
    private IKernelFunction<T> CreateDefaultKernel()
    {
        // Use GaussianKernel with default sigma (which corresponds to gamma in RBF)
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
    /// <b>For Beginners:</b> Cloning creates an independent copy of the classifier.
    /// Changes to the clone won't affect the original, and vice versa. This is useful
    /// for ensemble methods or when you need to experiment without affecting your trained model.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new LabelPropagation<T>(
            _kernel,
            _maxIterations,
            _tolerance,
            _random.Next());

        // Copy state if trained
        if (_allFeatures is not null)
        {
            clone._allFeatures = new Matrix<T>(_allFeatures.Rows, _allFeatures.Columns);
            for (int i = 0; i < _allFeatures.Rows; i++)
            {
                for (int j = 0; j < _allFeatures.Columns; j++)
                {
                    clone._allFeatures[i, j] = _allFeatures[i, j];
                }
            }
        }

        if (_labelDistributions is not null)
        {
            clone._labelDistributions = new Matrix<T>(_labelDistributions.Rows, _labelDistributions.Columns);
            for (int i = 0; i < _labelDistributions.Rows; i++)
            {
                for (int j = 0; j < _labelDistributions.Columns; j++)
                {
                    clone._labelDistributions[i, j] = _labelDistributions[i, j];
                }
            }
        }

        clone._numLabeled = _numLabeled;

        return clone;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enum value for Label Propagation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This identifies what kind of model this is within the
    /// AiDotNet library's type system.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.LabelPropagation;
    }

    /// <summary>
    /// Gets all learnable parameters of the model as a single vector.
    /// </summary>
    /// <returns>An empty vector as Label Propagation is non-parametric.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike neural networks that have weights to learn, Label Propagation
    /// is a non-parametric method. It stores all training data and computes similarities at
    /// prediction time, so there are no "learned" parameters in the traditional sense.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Label Propagation is non-parametric - it stores training data, not learned weights
        return new Vector<T>(0);
    }

    /// <summary>
    /// Creates a new instance of the model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use (ignored for non-parametric models).</param>
    /// <returns>A new instance of the classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Propagation doesn't have learnable parameters,
    /// this just creates a new instance with the same configuration.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return new LabelPropagation<T>(_kernel, _maxIterations, _tolerance, _random.Next());
    }

    /// <summary>
    /// Sets the parameters of this model.
    /// </summary>
    /// <param name="parameters">The parameters to set (ignored for non-parametric models).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Propagation doesn't have learnable parameters,
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
    /// <returns>An empty gradient vector as Label Propagation doesn't use gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Label Propagation is not trained with gradient descent like neural
    /// networks. Instead, it uses graph-based iterative label spreading. Therefore, there are
    /// no gradients to compute.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Label Propagation is not gradient-based
        return new Vector<T>(0);
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">The gradients to apply (ignored).</param>
    /// <param name="learningRate">The learning rate (ignored).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since Label Propagation doesn't use gradient-based learning,
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
    /// <returns>A new LabelPropagation instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used internally for operations like cloning or serialization
    /// that need to create a fresh instance of the same type.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new LabelPropagation<T>(_kernel, _maxIterations, _tolerance, _random.Next());
    }

    #endregion
}
