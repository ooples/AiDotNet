using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Classification.SemiSupervised;

/// <summary>
/// Provides a base implementation for semi-supervised classification algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class extends ClassifierBase with semi-supervised learning capabilities,
/// allowing derived classes to leverage both labeled and unlabeled data during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides the foundation for building classifiers
/// that can learn from both labeled data (where you know the answers) and unlabeled data
/// (where you don't).
///
/// Think of it like a student learning with two types of study materials:
/// - A teacher's answer key (labeled data) - few examples but definitely correct
/// - Practice problems without answers (unlabeled data) - many examples to learn patterns from
///
/// By combining both, the student (classifier) can learn more effectively than using
/// just the answer key alone.
/// </para>
/// </remarks>
public abstract partial class SemiSupervisedClassifierBase<T> : ClassifierBase<T>, ISemiSupervisedClassifier<T>
{
    /// <summary>
    /// Gets or sets the number of labeled samples used in training.
    /// </summary>
    public int NumLabeledSamples { get; protected set; }

    /// <summary>
    /// Gets or sets the number of unlabeled samples used in training.
    /// </summary>
    public int NumUnlabeledSamples { get; protected set; }

    /// <summary>
    /// Stores the pseudo-labels assigned to unlabeled data during training.
    /// </summary>
    protected Vector<T>? PseudoLabels { get; set; }

    /// <summary>
    /// Stores the confidence scores for the pseudo-labels.
    /// </summary>
    protected Vector<T>? PseudoLabelConfidences { get; set; }

    /// <summary>
    /// The unlabeled feature matrix stored for prediction and analysis.
    /// </summary>
    protected Matrix<T>? UnlabeledData { get; set; }

    /// <summary>
    /// Initializes a new instance of the SemiSupervisedClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <param name="lossFunction">Loss function for gradient computation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the semi-supervised classifier with
    /// your specified settings. The options control how the algorithm behaves, regularization
    /// helps prevent the model from memorizing the training data (overfitting), and the loss
    /// function determines how prediction errors are measured.
    /// </para>
    /// </remarks>
    protected SemiSupervisedClassifierBase(
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        ILossFunction<T>? lossFunction = null)
        : base(options, regularization, lossFunction)
    {
    }

    /// <summary>
    /// Trains the classifier using both labeled and unlabeled data.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is where the semi-supervised magic happens.
    /// It first validates that your data is properly formatted, then calls the specific
    /// implementation in the derived class to actually perform the learning.
    ///
    /// The method ensures that:
    /// - Labeled features and labels have matching sample counts
    /// - All data has the same number of features
    /// - The data is stored for later analysis
    /// </para>
    /// </remarks>
    public virtual void TrainSemiSupervised(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        ValidateSemiSupervisedInput(labeledX, labeledY, unlabeledX);

        NumLabeledSamples = labeledX.Rows;
        NumUnlabeledSamples = unlabeledX.Rows;
        NumFeatures = labeledX.Columns;
        UnlabeledData = unlabeledX;

        // Extract class information from labeled data
        ClassLabels = ExtractClassLabels(labeledY);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(labeledY);

        // Call the derived class implementation
        TrainSemiSupervisedCore(labeledX, labeledY, unlabeledX);
    }

    /// <summary>
    /// Core implementation of semi-supervised training to be implemented by derived classes.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where each specific semi-supervised algorithm
    /// implements its unique learning strategy. For example:
    /// - Self-training iteratively labels high-confidence unlabeled samples
    /// - Label propagation spreads labels through a similarity graph
    /// - Co-training uses multiple views of the data
    /// </para>
    /// </remarks>
    protected abstract void TrainSemiSupervisedCore(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX);

    /// <summary>
    /// Validates the input data for semi-supervised training.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <exception cref="ArgumentException">Thrown when the input data is invalid.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks that your data is properly formatted
    /// before training begins. It catches common mistakes like:
    /// - Having different numbers of samples and labels
    /// - Having different numbers of features in labeled vs unlabeled data
    /// - Providing empty datasets
    /// </para>
    /// </remarks>
    protected virtual void ValidateSemiSupervisedInput(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        if (labeledX.Rows == 0)
        {
            throw new ArgumentException("Labeled feature matrix cannot be empty.", nameof(labeledX));
        }

        if (labeledX.Rows != labeledY.Length)
        {
            throw new ArgumentException(
                $"Number of labeled samples ({labeledX.Rows}) must match number of labels ({labeledY.Length}).",
                nameof(labeledY));
        }

        if (unlabeledX.Rows == 0)
        {
            throw new ArgumentException("Unlabeled feature matrix cannot be empty.", nameof(unlabeledX));
        }

        if (labeledX.Columns != unlabeledX.Columns)
        {
            throw new ArgumentException(
                $"Labeled data has {labeledX.Columns} features but unlabeled data has {unlabeledX.Columns} features. They must match.",
                nameof(unlabeledX));
        }
    }

    /// <summary>
    /// Gets the pseudo-labels assigned to the unlabeled data during training.
    /// </summary>
    /// <returns>A vector of predicted labels for the unlabeled samples, or null if not available.</returns>
    public virtual Vector<T>? GetPseudoLabels()
    {
        return PseudoLabels;
    }

    /// <summary>
    /// Gets the confidence scores for the pseudo-labels.
    /// </summary>
    /// <returns>A vector of confidence scores for each pseudo-label, or null if not available.</returns>
    public virtual Vector<T>? GetPseudoLabelConfidences()
    {
        return PseudoLabelConfidences;
    }

    /// <summary>
    /// Trains the classifier using only labeled data (standard supervised training).
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Semi-supervised classifiers can also be trained using only
    /// labeled data, just like regular classifiers. This is useful when you don't have
    /// any unlabeled data or want to compare performance with and without unlabeled data.
    ///
    /// When you call this method, the classifier treats all data as labeled and ignores
    /// the semi-supervised capabilities.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // For standard training, we can either:
        // 1. Train as a regular classifier (if the derived class supports it)
        // 2. Create an empty unlabeled set and use semi-supervised training
        NumLabeledSamples = x.Rows;
        NumUnlabeledSamples = 0;
        NumFeatures = x.Columns;

        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        TrainSupervisedCore(x, y);
    }

    /// <summary>
    /// Core implementation of standard supervised training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows the semi-supervised classifier to also
    /// work as a regular classifier when no unlabeled data is available.
    /// </para>
    /// </remarks>
    protected abstract void TrainSupervisedCore(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// The [a.Rows, b.Rows] similarity matrix S[i, j] = kernel(a_i, b_j).
    /// </summary>
    /// <param name="kernel">The kernel function.</param>
    /// <param name="a">The first set of samples (rows).</param>
    /// <param name="b">The second set of samples (rows).</param>
    /// <param name="graphWithoutSelfLoops">
    /// When true, a and b are the same samples: the diagonal is 0 and kernel(a_i, a_j) for j &gt; i
    /// is mirrored to [j, i].
    /// </param>
    /// <returns>The similarity matrix.</returns>
    private protected Tensor<T> KernelSimilarities(IKernelFunction<T> kernel, Matrix<T> a, Matrix<T> b, bool graphWithoutSelfLoops)
    {
        int n = a.Rows;
        int m = b.Rows;
        var similarities = new Tensor<T>(new[] { n, m });
        if (n == 0 || m == 0)
        {
            return similarities;
        }

        // IKernelFunction<T> only exposes a pairwise Calculate, so the Gram matrix is filled pair by pair.
        var left = new Vector<T>[n];
        var right = graphWithoutSelfLoops ? left : new Vector<T>[m];
        for (int i = 0; i < n; i++)
        {
            left[i] = a.GetRow(i);
        }

        if (!graphWithoutSelfLoops)
        {
            for (int j = 0; j < m; j++)
            {
                right[j] = b.GetRow(j);
            }
        }

        // One row of kernel values at a time; without self loops only the strict upper triangle is
        // evaluated (the rest is 0) and it is mirrored by adding its transpose
        T zero = NumOps.Zero;
        var rows = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            var row = left[i];
            int first = graphWithoutSelfLoops ? i + 1 : 0;
            rows[i] = new Vector<T>(Enumerable.Range(0, m).Select(j => j >= first ? kernel.Calculate(row, right[j]) : zero))
                .AsTensor().Reshape(1, m);
        }

        var engine = Engine;
        var upper = engine.TensorConcatenate(rows, 0);
        return graphWithoutSelfLoops ? engine.TensorAdd(upper, engine.TensorTranspose(upper)) : upper;
    }

    /// <summary>
    /// Divides every row with a positive sum by that sum. Other rows become uniform (1 / columns)
    /// when <paramref name="uniformWhenNotPositive"/> is true and are left unchanged otherwise.
    /// </summary>
    private protected Tensor<T> NormalizeRowSums(Tensor<T> weights, bool uniformWhenNotPositive)
    {
        var engine = Engine;
        int rows = weights.Shape[0];
        int columns = weights.Shape[1];
        if (rows == 0 || columns == 0)
        {
            return weights.Clone();
        }

        var sums = engine.ReduceSum(weights, new[] { 1 }, true);
        var positive = engine.TensorGreaterThan(sums, NumOps.Zero);
        var ones = new Tensor<T>(new[] { rows, 1 });
        engine.TensorFill(ones, NumOps.One);
        var normalized = engine.TensorDivide(weights, engine.TensorWhere(positive, sums, ones));
        Tensor<T> fallback = weights;
        if (uniformWhenNotPositive)
        {
            fallback = new Tensor<T>(new[] { rows, columns });
            engine.TensorFill(fallback, NumOps.Divide(NumOps.One, NumOps.FromDouble(columns)));
        }

        return engine.TensorWhere(engine.TensorTile(positive, new[] { 1, columns }), normalized, fallback);
    }

    /// <summary>
    /// One-hot rows for the labeled samples followed by uniform (1 / NumClasses) rows up to
    /// <paramref name="totalSamples"/>.
    /// </summary>
    private protected Matrix<T> InitialLabelDistributions(Vector<T> labeledY, int totalSamples)
    {
        var engine = Engine;
        var classIndices = GetClassIndexTensor(labeledY);
        int unknown = Array.IndexOf(classIndices.ToArray(), -1);
        if (unknown >= 0)
        {
            throw new ArgumentException($"Unknown class label: {labeledY[unknown]}");
        }

        int labeled = labeledY.Length;
        var parts = new List<Tensor<T>>();
        if (labeled > 0)
        {
            parts.Add(engine.TensorOneHot<T>(classIndices, NumClasses));
        }

        if (totalSamples > labeled)
        {
            var uniform = new Tensor<T>(new[] { totalSamples - labeled, NumClasses });
            engine.TensorFill(uniform, NumOps.Divide(NumOps.One, NumOps.FromDouble(NumClasses)));
            parts.Add(uniform);
        }

        if (parts.Count == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        return (parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 0)).ToMatrix();
    }

    /// <summary>
    /// The largest |a - b| over all entries (0 when empty).
    /// </summary>
    private protected T MaxAbsoluteChange(Tensor<T> a, Tensor<T> b)
    {
        return a.Length == 0 ? NumOps.Zero : Engine.TensorMaxValue(Engine.TensorAbs(Engine.TensorSubtract(a, b)));
    }

    /// <summary>
    /// Sets the pseudo-labels (argmax class, first wins ties) and their confidences (the max
    /// probability) for the rows after the first <paramref name="numLabeled"/>.
    /// </summary>
    private protected void SetPseudoLabelsFromDistributions(Matrix<T> distributions, int numLabeled, string missingLabelsMessage)
    {
        int numUnlabeled = distributions.Rows - numLabeled;
        if (numUnlabeled <= 0) return;

        if (ClassLabels is null)
        {
            throw new InvalidOperationException(missingLabelsMessage);
        }

        var unlabeled = Tensor<T>.FromMatrix(distributions).Slice(0, numLabeled, distributions.Rows);
        PseudoLabels = ClassIndicesToLabels(Engine.TensorArgMax(unlabeled, 1));
        PseudoLabelConfidences = Engine.ReduceMax(unlabeled, new[] { 1 }, false).ToVector();
    }

    /// <summary>
    /// Class distributions for new samples: kernel similarities to the training samples, row-normalized
    /// (uniform when a row has no positive similarity), times the training label distributions.
    /// </summary>
    private protected Matrix<T> PropagatedDistributions(IKernelFunction<T> kernel, Matrix<T> input, Matrix<T> trainFeatures, Matrix<T> labelDistributions)
    {
        if (input.Rows == 0)
        {
            return new Matrix<T>(0, labelDistributions.Columns);
        }

        var weights = NormalizeRowSums(KernelSimilarities(kernel, input, trainFeatures, false), true);
        return Engine.TensorMatMul(weights, Tensor<T>.FromMatrix(labelDistributions)).ToMatrix();
    }
}
