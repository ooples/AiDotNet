using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Implements the Label Powerset approach for multi-label classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Label Powerset transforms the multi-label problem into a single multi-class problem by treating
/// each unique combination of labels as a separate class.
/// </para>
/// <para>
/// <b>For Beginners:</b> Label Powerset takes a clever approach:
///
/// Instead of predicting labels independently, it treats each unique combination of labels as a
/// single class. For example, if you have 3 labels (action, comedy, romance):
///
/// - "none" becomes class 0
/// - "action only" becomes class 1
/// - "comedy only" becomes class 2
/// - "romance only" becomes class 3
/// - "action+comedy" becomes class 4
/// - "action+romance" becomes class 5
/// - "comedy+romance" becomes class 6
/// - "action+comedy+romance" becomes class 7
///
/// Now we train ONE multi-class classifier that directly predicts which combination applies.
///
/// Pros:
/// - Perfectly captures label correlations (impossible combinations never predicted)
/// - Only one classifier to train
/// - Naturally handles label interdependencies
///
/// Cons:
/// - Number of possible classes = 2^n (exponential in number of labels)
/// - Many classes may have very few examples (data sparsity)
/// - New label combinations unseen in training cannot be predicted
///
/// Works best when:
/// - Number of labels is small (≤10)
/// - Label combinations in test data were seen in training
/// - Label correlations are important
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create Label Powerset treating each label combination as a class
///
/// // Prepare features and multi-label targets
/// var features = new Matrix&lt;double&gt;(4, 2);
/// features[0, 0] = 1.0; features[0, 1] = 2.0;
/// features[1, 0] = 3.0; features[1, 1] = 4.0;
/// features[2, 0] = 5.0; features[2, 1] = 6.0;
/// features[3, 0] = 7.0; features[3, 1] = 8.0;
/// var labels = new Matrix&lt;double&gt;(4, 3);
/// labels[0, 0] = 1; labels[0, 1] = 0; labels[0, 2] = 1;
/// labels[1, 0] = 1; labels[1, 1] = 1; labels[1, 2] = 0;
/// labels[2, 0] = 0; labels[2, 1] = 1; labels[2, 2] = 1;
/// labels[3, 0] = 0; labels[3, 1] = 0; labels[3, 2] = 1;
///
/// // Train single multi-class classifier on label combination classes
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Matrix&lt;double&gt;&gt;()
///     .ConfigureModel(new LabelPowerset&lt;double&gt;())
///     .Build(features, labels);
///
/// // Predict complete label combination for new sample
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 2.0; newSample[0, 1] = 3.0;
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Learning multi-label scene classification", "https://doi.org/10.1016/j.patcog.2004.01.013", Year = 2004, Authors = "Matthew R. Boutell, Jiebo Luo, Xipeng Shen, Christopher M. Brown")]
public partial class LabelPowerset<T> : MultiLabelClassifierBase<T>
{

    /// <inheritdoc />
    /// <remarks>The single multi-class classifier this transformation trains over the powerset of label combinations. Registration remains null-tolerant before training creates the classifier.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_classifier);
    }
    #region Fields

    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier.
    /// </summary>
    public LabelPowerset()
        : this(() => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// Factory function to create the multi-class classifier.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Unlike Binary Relevance (which creates many binary classifiers),
    /// Label Powerset creates just ONE multi-class classifier that predicts among all possible
    /// label combinations.
    /// </para>
    /// </remarks>
    private readonly Func<IClassifier<T>> _classifierFactory;

    /// <summary>
    /// The trained multi-class classifier.
    /// </summary>
    private IClassifier<T>? _classifier;

    /// <summary>
    /// Maps class indices to label combinations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This dictionary remembers what each class number means.
    /// For example: class 4 → [1, 1, 0] meaning "action + comedy, but not romance".
    /// </para>
    /// </remarks>
    private Dictionary<int, bool[]>? _classToLabels;

    /// <summary>
    /// Maps label combinations to class indices.
    /// </summary>
    private Dictionary<string, int>? _labelsToClass;

    /// <summary>
    /// The number of unique label combinations (classes) found in training data.
    /// </summary>
    private int _numCombinations;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the LabelPowerset class with a classifier factory.
    /// </summary>
    /// <param name="classifierFactory">A function that creates a multi-class classifier.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> To create a Label Powerset classifier, provide a factory function
    /// that creates a multi-class classifier. The classifier should be able to handle many classes.
    ///
    /// Example usage:
    /// <code>
    /// // Using logistic regression (softmax for multi-class)
    /// var lp = new LabelPowerset&lt;double&gt;(() => new LogisticRegression&lt;double&gt;());
    ///
    /// // Using a neural network
    /// var lp2 = new LabelPowerset&lt;double&gt;(() => new NeuralNetworkClassifier&lt;double&gt;());
    /// </code>
    /// </para>
    /// </remarks>
    public LabelPowerset(
        Func<IClassifier<T>> classifierFactory,
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        Guard.NotNull(classifierFactory);
        _classifierFactory = classifierFactory;
    }

    #endregion

    #region Training

    /// <summary>
    /// Core implementation of multi-label training using Label Powerset.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The multi-label target matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method transforms multi-label data into multi-class:
    ///
    /// 1. Find all unique label combinations in the training data
    /// 2. Assign a class number to each unique combination
    /// 3. Convert the multi-label matrix y into a single-label vector
    /// 4. Train a multi-class classifier on this transformed data
    ///
    /// For example, if row 5 has labels [1, 0, 1] (action + romance), and this is assigned
    /// to class 5, then the transformed target for row 5 becomes just the number 5.
    /// </para>
    /// </remarks>
    protected override void TrainMultiLabelCore(Matrix<T> x, Matrix<T> y)
    {
        // Build mappings from label combinations to class indices
        BuildLabelCombinationMappings(y);

        // Transform multi-label y into single-label class indices
        var transformedY = TransformToClassLabels(y);

        // Train multi-class classifier
        _classifier = _classifierFactory();
        _classifier.Train(x, transformedY);
    }

    /// <summary>
    /// Builds mappings between label combinations and class indices.
    /// </summary>
    /// <param name="y">The multi-label target matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method scans through all training examples and creates
    /// a unique class number for each unique label combination it finds.
    ///
    /// Only combinations that appear in the training data get a class number.
    /// This means if certain combinations never appear in training, they can't be predicted.
    /// </para>
    /// </remarks>
    private void BuildLabelCombinationMappings(Matrix<T> y)
    {
        _labelsToClass = new Dictionary<string, int>();
        _classToLabels = new Dictionary<int, bool[]>();

        int nextClassIndex = 0;
        var bits = BinarizeLabels(y, requireExactOne: false);

        for (int i = 0; i < y.Rows; i++)
        {
            var labelKey = GetLabelKey(bits, i, y.Columns);

            if (!_labelsToClass.ContainsKey(labelKey))
            {
                _labelsToClass[labelKey] = nextClassIndex;
                _classToLabels[nextClassIndex] = GetLabelArray(bits, i, y.Columns);
                nextClassIndex++;
            }
        }

        _numCombinations = nextClassIndex;
    }

    /// <summary>
    /// Creates a string key representing a label combination.
    /// </summary>
    /// <param name="y">The multi-label target matrix.</param>
    /// <param name="row">The row index.</param>
    /// <returns>A string key like "1,0,1" representing the label combination.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> To quickly look up whether we've seen a label combination before,
    /// we convert it to a string like "1,0,1,0". This makes it easy to use as a dictionary key.
    /// </para>
    /// </remarks>
    private static string GetLabelKey(int[] bits, int row, int columns)
    {
        var parts = new string[columns];
        for (int j = 0; j < columns; j++)
        {
            parts[j] = bits[row * columns + j] == 1 ? "1" : "0";
        }
        return string.Join(",", parts);
    }

    /// <summary>
    /// Binarizes every label value at once (row-major 0/1 ints): value &gt;= 0.5, or value == 1 when
    /// <paramref name="requireExactOne"/> is set. NaN gives 0.
    /// </summary>
    private int[] BinarizeLabels(Matrix<T> y, bool requireExactOne)
    {
        if (y.Rows == 0 || y.Columns == 0)
        {
            return Array.Empty<int>();
        }

        var engine = Engine;
        var values = Tensor<T>.FromMatrix(y);
        Tensor<T> present;
        if (requireExactOne)
        {
            present = engine.TensorEquals(values, NumOps.One);
        }
        else
        {
            var half = new Tensor<T>(new[] { y.Rows, y.Columns });
            engine.TensorFill(half, NumOps.FromDouble(0.5));
            present = engine.TensorGreaterEqual(values, half);
        }

        // argmax over [1 - b, b] turns the exact 0/1 values into integer indicators
        var absent = engine.ScalarMinusTensor(NumOps.One, present);
        return engine.TensorArgMax(engine.TensorStack(new[] { absent, present }, 2), 2).ToArray();
    }

    /// <summary>
    /// Class index values 0..count-1 as T, gathered for each entry of <paramref name="classIndices"/>.
    /// </summary>
    private Vector<T> ClassIndexValues(int[] classIndices, int count)
    {
        if (classIndices.Length == 0)
        {
            return new Vector<T>(0);
        }

        var range = Engine.TensorArange<T>(0, Math.Max(1, count));
        return Engine.TensorGather(range, new Tensor<int>(classIndices, new[] { classIndices.Length }), 0).ToVector();
    }

    /// <summary>
    /// The [count + 1, NumLabels] 0/1 label matrix of every class (row <paramref name="count"/> and unknown
    /// classes are all zero).
    /// </summary>
    private Tensor<T> ClassLabelIndicators(int count)
    {
        var flags = new int[(count + 1) * NumLabels];
        if (_classToLabels is not null)
        {
            for (int c = 0; c < count; c++)
            {
                if (_classToLabels.TryGetValue(c, out var labels))
                {
                    for (int l = 0; l < NumLabels && l < labels.Length; l++)
                    {
                        flags[c * NumLabels + l] = labels[l] ? 1 : 0;
                    }
                }
            }
        }

        var zeroOne = Engine.TensorArange<T>(0, 2);
        return Engine.TensorGather(zeroOne, new Tensor<int>(flags, new[] { flags.Length }), 0)
            .Reshape(count + 1, NumLabels);
    }

    /// <summary>
    /// Extracts a boolean array representing the label combination for a row.
    /// </summary>
    /// <param name="y">The multi-label target matrix.</param>
    /// <param name="row">The row index.</param>
    /// <returns>A boolean array where true means the label is present.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts the numeric values in a row to a simple boolean array.
    /// [1.0, 0.0, 1.0] becomes [true, false, true].
    /// </para>
    /// </remarks>
    private static bool[] GetLabelArray(int[] bits, int row, int columns)
    {
        var labels = new bool[columns];
        for (int j = 0; j < columns; j++)
        {
            labels[j] = bits[row * columns + j] == 1;
        }
        return labels;
    }

    /// <summary>
    /// Transforms multi-label matrix into single-label class vector.
    /// </summary>
    /// <param name="y">The multi-label target matrix.</param>
    /// <returns>A vector of class indices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts the multi-label matrix into a simple vector where
    /// each value is the class number corresponding to that row's label combination.
    /// </para>
    /// </remarks>
    private Vector<T> TransformToClassLabels(Matrix<T> y)
    {
        var labelsToClass = _labelsToClass ?? throw new InvalidOperationException("_labelsToClass has not been initialized.");
        var bits = BinarizeLabels(y, requireExactOne: false);
        var classIndices = new int[y.Rows];

        for (int i = 0; i < y.Rows; i++)
        {
            classIndices[i] = labelsToClass[GetLabelKey(bits, i, y.Columns)];
        }

        return ClassIndexValues(classIndices, _numCombinations);
    }

    #endregion

    #region Prediction

    /// <summary>
    /// Predicts probabilities for each label for each sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A probability matrix where each row is a sample and each column is the probability of that label.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Prediction works by:
    ///
    /// 1. Get the classifier's prediction (which class/combination)
    /// 2. Convert that class back to label probabilities
    ///
    /// If the classifier supports probability predictions, we can combine probabilities from
    /// multiple classes. Otherwise, we just use the predicted class's labels.
    ///
    /// For example, if class 4 = [action, comedy] has 80% probability and class 1 = [action] has 20%,
    /// then P(action) = 80% + 20% = 100%, P(comedy) = 80%, P(romance) = 0%.
    /// </para>
    /// </remarks>
    public override Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_classifier is null || _classToLabels is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        if (input.Rows == 0 || NumLabels == 0)
        {
            return new Matrix<T>(input.Rows, NumLabels);
        }

        var engine = Engine;
        if (_classifier is IProbabilisticClassifier<T> probabilisticClassifier)
        {
            // Use class probabilities to compute label probabilities:
            // P(label) = sum of the probabilities of classes that include that label
            var classProbs = probabilisticClassifier.PredictProbabilities(input);
            int usable = Math.Min(_numCombinations, classProbs.Columns);
            if (usable == 0)
            {
                return new Matrix<T>(input.Rows, NumLabels);
            }

            var probs = Tensor<T>.FromMatrix(classProbs);
            if (usable != classProbs.Columns)
            {
                probs = probs.Slice(1, 0, usable);
            }

            var indicators = ClassLabelIndicators(usable).Slice(0, 0, usable);
            return engine.TensorMatMul(probs, indicators).ToMatrix();
        }

        // Fall back to hard predictions: the labels of the predicted class (unknown classes give no labels)
        var predictions = _classifier.Predict(input);
        var classValues = engine.TensorArange<T>(0, _numCombinations + 1);
        var rows = engine.TensorSubtractScalar(
            engine.TensorSearchSorted(classValues, predictions.AsTensor(), true), 1);
        var invalid = engine.TensorLessThan(rows, 0);
        var safeRows = engine.TensorWhere(invalid,
            engine.TensorAddScalar(engine.TensorMultiplyScalar(rows, 0), _numCombinations), rows);
        return engine.TensorGather(ClassLabelIndicators(_numCombinations), safeRows, 0).ToMatrix();
    }

    #endregion

    #region Serialization

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Creates a new instance of the model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use.</param>
    /// <returns>A new instance of the classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a new Label Powerset classifier and sets
    /// the underlying classifier's parameters.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Matrix<T>> WithParameters(Vector<T> parameters)
    {
        var clone = (LabelPowerset<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    /// <summary>
    /// Computes gradients for gradient-based optimization.
    /// </summary>
    /// <param name="input">The input features.</param>
    /// <param name="target">The target labels.</param>
    /// <param name="lossFunction">The loss function (optional).</param>
    /// <returns>The gradients from the underlying classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes gradients for the underlying multi-class classifier.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Matrix<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_classifier is null || _labelsToClass is null)
        {
            return new Vector<T>(0);
        }

        // Convert multi-label target to class index vector
        var bits = BinarizeLabels(target, requireExactOne: true);
        var classIndices = new int[target.Rows];
        for (int i = 0; i < target.Rows; i++)
        {
            var key = GetLabelKey(bits, i, target.Columns);
            if (!_labelsToClass.TryGetValue(key, out var classIdx))
            {
                throw new ArgumentException($"Label combination '{key}' was not seen during training.", nameof(target));
            }
            classIndices[i] = classIdx;
        }

        var classTargets = ClassIndexValues(classIndices, _numCombinations);

        return ((IGradientComputable<T, Matrix<T>, Vector<T>>)_classifier).ComputeGradients(input, classTargets, lossFunction);
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">The gradients to apply.</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This applies gradient updates to the underlying classifier.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        (_classifier as IGradientComputable<T, Matrix<T>, Vector<T>>)?.ApplyGradients(gradients, learningRate);
    }

    #endregion

    #region Properties

    /// <summary>
    /// Gets the number of unique label combinations found in training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many "classes" the multi-class classifier
    /// is actually predicting among. If you have 5 labels but only 12 unique combinations
    /// appeared in training, this would be 12.
    /// </para>
    /// </remarks>
    public int NumLabelCombinations => _numCombinations;

    /// <summary>
    /// Gets the label combination for a given class index.
    /// </summary>
    /// <param name="classIndex">The class index.</param>
    /// <returns>A boolean array representing which labels are present, or null if not found.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to understand what labels correspond to a predicted class.
    /// </para>
    /// </remarks>
    public bool[]? GetLabelsForClass(int classIndex)
    {
        return _classToLabels?.TryGetValue(classIndex, out var labels) == true ? labels.ToArray() : null;
    }

    #endregion
}
