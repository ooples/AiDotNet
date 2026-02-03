using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

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
public class LabelPowerset<T> : MultiLabelClassifierBase<T>
{
    #region Fields

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
    /// var lp = new LabelPowerset&lt;double&gt;(() => new NeuralNetworkClassifier&lt;double&gt;());
    /// </code>
    /// </para>
    /// </remarks>
    public LabelPowerset(
        Func<IClassifier<T>> classifierFactory,
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _classifierFactory = classifierFactory ?? throw new ArgumentNullException(nameof(classifierFactory));
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

        for (int i = 0; i < y.Rows; i++)
        {
            var labelKey = GetLabelKey(y, i);

            if (!_labelsToClass.ContainsKey(labelKey))
            {
                _labelsToClass[labelKey] = nextClassIndex;
                _classToLabels[nextClassIndex] = GetLabelArray(y, i);
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
    private string GetLabelKey(Matrix<T> y, int row)
    {
        var parts = new string[y.Columns];
        for (int j = 0; j < y.Columns; j++)
        {
            parts[j] = NumOps.Compare(y[row, j], NumOps.FromDouble(0.5)) >= 0 ? "1" : "0";
        }
        return string.Join(",", parts);
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
    private bool[] GetLabelArray(Matrix<T> y, int row)
    {
        var labels = new bool[y.Columns];
        for (int j = 0; j < y.Columns; j++)
        {
            labels[j] = NumOps.Compare(y[row, j], NumOps.FromDouble(0.5)) >= 0;
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
        var classLabels = new Vector<T>(y.Rows);

        for (int i = 0; i < y.Rows; i++)
        {
            var labelKey = GetLabelKey(y, i);
            int classIndex = _labelsToClass![labelKey];
            classLabels[i] = NumOps.FromDouble(classIndex);
        }

        return classLabels;
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

        var probabilities = new Matrix<T>(input.Rows, NumLabels);

        if (_classifier is IProbabilisticClassifier<T> probabilisticClassifier)
        {
            // Use class probabilities to compute label probabilities
            var classProbs = probabilisticClassifier.PredictProbabilities(input);

            for (int i = 0; i < input.Rows; i++)
            {
                // For each label, sum probabilities of classes that include that label
                for (int labelIdx = 0; labelIdx < NumLabels; labelIdx++)
                {
                    T labelProb = NumOps.Zero;

                    for (int classIdx = 0; classIdx < _numCombinations && classIdx < classProbs.Columns; classIdx++)
                    {
                        if (_classToLabels.TryGetValue(classIdx, out var labels) && labels[labelIdx])
                        {
                            labelProb = NumOps.Add(labelProb, classProbs[i, classIdx]);
                        }
                    }

                    probabilities[i, labelIdx] = labelProb;
                }
            }
        }
        else
        {
            // Fall back to hard predictions
            var predictions = _classifier.Predict(input);

            for (int i = 0; i < input.Rows; i++)
            {
                int classIdx = (int)NumOps.ToDouble(predictions[i]);

                if (_classToLabels.TryGetValue(classIdx, out var labels))
                {
                    for (int labelIdx = 0; labelIdx < NumLabels; labelIdx++)
                    {
                        probabilities[i, labelIdx] = labels[labelIdx] ? NumOps.One : NumOps.Zero;
                    }
                }
            }
        }

        return probabilities;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enum value for Label Powerset.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This identifies what kind of model this is within the
    /// AiDotNet library's type system.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.LabelPowersetClassifier;
    }

    /// <summary>
    /// Gets all learnable parameters of the model as a single vector.
    /// </summary>
    /// <returns>The parameters from the underlying multi-class classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Label Powerset has just one classifier, so this returns
    /// that classifier's parameters.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_classifier is null)
        {
            return new Vector<T>(0);
        }

        return _classifier.GetParameters();
    }

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
        var newClassifier = new LabelPowerset<T>(_classifierFactory, Options, Regularization);
        newClassifier.NumLabels = NumLabels;
        newClassifier.NumFeatures = NumFeatures;
        newClassifier._classToLabels = _classToLabels;
        newClassifier._labelsToClass = _labelsToClass;
        newClassifier._numCombinations = _numCombinations;
        newClassifier.SetParameters(parameters);
        return newClassifier;
    }

    /// <summary>
    /// Sets the parameters of this model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets the underlying classifier's parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        _classifier?.SetParameters(parameters);
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
        var classTargets = new Vector<T>(target.Rows);
        for (int i = 0; i < target.Rows; i++)
        {
            var labels = new bool[target.Columns];
            for (int j = 0; j < target.Columns; j++)
            {
                labels[j] = NumOps.ToDouble(target[i, j]) > 0.5;
            }

            var key = string.Join(",", labels.Select(l => l ? "1" : "0"));
            int classIdx = _labelsToClass.TryGetValue(key, out var idx) ? idx : 0;
            classTargets[i] = NumOps.FromDouble(classIdx);
        }

        return _classifier.ComputeGradients(input, classTargets, lossFunction);
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
        _classifier?.ApplyGradients(gradients, learningRate);
    }

    /// <summary>
    /// Creates a new instance of this classifier with default configuration.
    /// </summary>
    /// <returns>A new LabelPowerset instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used internally for operations like cloning or serialization.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Matrix<T>> CreateNewInstance()
    {
        return new LabelPowerset<T>(_classifierFactory, Options, Regularization);
    }

    /// <summary>
    /// Creates a deep copy of this classifier.
    /// </summary>
    /// <returns>A new instance with the same parameters and state.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cloning creates an independent copy of the classifier,
    /// including its label mappings and underlying classifier.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Matrix<T>> Clone()
    {
        var clone = new LabelPowerset<T>(_classifierFactory, Options, Regularization);
        clone.NumLabels = NumLabels;
        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone._numCombinations = _numCombinations;

        if (_classToLabels is not null)
        {
            clone._classToLabels = new Dictionary<int, bool[]>();
            foreach (var kvp in _classToLabels)
            {
                clone._classToLabels[kvp.Key] = kvp.Value.ToArray();
            }
        }

        if (_labelsToClass is not null)
        {
            clone._labelsToClass = new Dictionary<string, int>(_labelsToClass);
        }

        if (_classifier is not null)
        {
            clone._classifier = (IClassifier<T>)_classifier.Clone();
        }

        return clone;
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
