namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a node in a decision tree for machine learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// A decision tree is a flowchart-like structure where each internal node represents a decision based on a feature,
/// each branch represents an outcome of that decision, and each leaf node represents a prediction or classification.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a decision tree like a flowchart of questions. Starting at the top (root),
/// each question (node) splits the data based on a feature (like "Is temperature > 70°F?").
/// Following the answers (branches) leads you to more questions or eventually to a final answer (leaf node).
/// Decision trees are popular because they're easy to understand and visualize - they make decisions
/// similar to how humans think.
/// </para>
/// </remarks>
public class DecisionTreeNode<T>
{
    /// <summary>
    /// Gets or sets the index of the feature used for splitting at this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is like asking "Which question should I ask at this point?"
    /// For example, if your data has features like [temperature, humidity, wind speed],
    /// a FeatureIndex of 0 means this node is making a decision based on temperature.
    /// </remarks>
    public int FeatureIndex { get; set; }

    /// <summary>
    /// Gets or sets the threshold value used to determine the split direction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the specific value used in the question.
    /// For example, if FeatureIndex refers to temperature, Threshold might be 70°F,
    /// so the question becomes "Is temperature > 70°F?"
    /// </remarks>
    public T Threshold { get; set; }

    /// <summary>
    /// Gets or sets the value used to split the data at this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the actual value that was found to be the best point to split the data.
    /// While similar to Threshold, SplitValue is often the specific value from the dataset that was chosen
    /// as the optimal splitting point during tree construction.
    /// </remarks>
    public T SplitValue { get; set; }

    /// <summary>
    /// Gets or sets the prediction value for this node when it's a leaf node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If this node is a final answer (leaf node), this is the actual prediction.
    /// For example, in a tree predicting house prices, this might be "$250,000".
    /// </remarks>
    public T Prediction { get; set; }

    /// <summary>
    /// Gets or sets the left child node (typically represents the "less than" or "no" branch).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If the answer to the node's question is "no" or "less than" (e.g., "Is temperature > 70°F?" "No"),
    /// the decision tree follows this path to the next question or answer.
    /// </remarks>
    public DecisionTreeNode<T>? Left { get; set; }

    /// <summary>
    /// Gets or sets the right child node (typically represents the "greater than or equal to" or "yes" branch).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If the answer to the node's question is "yes" or "greater than or equal to" 
    /// (e.g., "Is temperature > 70°F?" "Yes"), the decision tree follows this path to the next question or answer.
    /// </remarks>
    public DecisionTreeNode<T>? Right { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether this node is a leaf node (has no children).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A leaf node is a final answer, not another question. If IsLeaf is true,
    /// this node doesn't lead to more questions - it provides a prediction directly.
    /// </remarks>
    public bool IsLeaf { get; set; }

    /// <summary>
    /// Gets or sets the list of data samples that reached this node during training.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the examples from your training data that ended up at this node
    /// after answering all the previous questions. The decision tree uses these samples to make decisions
    /// about how to structure itself or what prediction to make.
    /// </remarks>
    public List<Sample<T>> Samples { get; set; } = [];

    /// <summary>
    /// Gets or sets the number of samples that went to the left child after splitting.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This counts how many examples answered "no" or "less than" to this node's question
    /// and followed the left path.
    /// </remarks>
    public int LeftSampleCount { get; set; }

    /// <summary>
    /// Gets or sets the number of samples that went to the right child after splitting.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This counts how many examples answered "yes" or "greater than or equal to" to this node's question
    /// and followed the right path.
    /// </remarks>
    public int RightSampleCount { get; set; }

    /// <summary>
    /// Gets or sets the list of target values from the samples at this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the actual "answers" or "outcomes" from your training examples that reached this node.
    /// For instance, if predicting house prices, these would be the actual prices of houses in your training data that
    /// matched all the conditions to reach this node.
    /// </remarks>
    public List<T> SampleValues { get; set; } = [];

    /// <summary>
    /// Gets or sets the linear regression model for this node (used in some advanced tree variants).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Some advanced decision trees (like model trees) use a simple linear equation at the leaves
    /// instead of a single prediction value. This property holds that equation if used. Think of it as making a more
    /// nuanced prediction based on a formula rather than a single value.
    /// </remarks>
    public SimpleRegression<T>? LinearModel { get; set; }

    /// <summary>
    /// Gets or sets the vector of predictions for samples at this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a list of predictions made for each sample that reached this node.
    /// It's used during training to evaluate how well the node is performing.
    /// </remarks>
    public Vector<T>? Predictions { get; set; }

    /// <summary>
    /// Gets or sets the sum of squared errors for predictions at this node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This measures how accurate the node's predictions are. It calculates the difference
    /// between each prediction and the actual value, squares these differences (to make them positive), and adds them up.
    /// A smaller value means better predictions.
    /// </remarks>
    public T SumSquaredError { get; set; }

    /// <summary>
    /// Gets or sets the numeric operations helper for the generic type T.
    /// </summary>
    /// <remarks>
    /// This provides mathematical operations for the generic numeric type.
    /// </remarks>
    private INumericOperations<T> _numOps { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="DecisionTreeNode{T}"/> class as a leaf node.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new "final answer" node with default values.
    /// It's typically used when building a tree and creating new nodes that will be configured later.
    /// </remarks>
    public DecisionTreeNode()
    {
        Left = null;
        Right = null;
        IsLeaf = true;

        _numOps = MathHelper.GetNumericOperations<T>();
        SplitValue = _numOps.Zero;
        Prediction = _numOps.Zero;
        Threshold = _numOps.Zero;
        SumSquaredError = _numOps.Zero;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DecisionTreeNode{T}"/> class as an internal decision node.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to split on.</param>
    /// <param name="splitValue">The value to use for splitting.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new "question" node. You specify which feature to ask about
    /// (featureIndex) and what value to compare against (splitValue).
    /// </remarks>
    public DecisionTreeNode(int featureIndex, T splitValue)
    {
        FeatureIndex = featureIndex;
        SplitValue = splitValue;
        IsLeaf = false;

        _numOps = MathHelper.GetNumericOperations<T>();
        Prediction = _numOps.Zero;
        Threshold = _numOps.Zero;
        SumSquaredError = _numOps.Zero;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DecisionTreeNode{T}"/> class as a leaf node with a prediction.
    /// </summary>
    /// <param name="prediction">The prediction value for this leaf node.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new "final answer" node with a specific prediction value.
    /// For example, if your tree is predicting house prices, this might create a leaf node that predicts "$250,000".
    /// </remarks>
    public DecisionTreeNode(T prediction)
    {
        Prediction = prediction;
        IsLeaf = true;

        _numOps = MathHelper.GetNumericOperations<T>();
        SplitValue = _numOps.Zero;
        Threshold = _numOps.Zero;
        SumSquaredError = _numOps.Zero;
    }

    /// <summary>
    /// Updates statistical information for this node based on its samples.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method recalculates how well this node is performing by measuring
    /// the error between its predictions and the actual values. It's typically called after the node
    /// has been modified or when samples have been added or removed.
    /// </remarks>
    public void UpdateNodeStatistics()
    {
        SumSquaredError = CalculateSumSquaredError();
    }

    /// <summary>
    /// Calculates the sum of squared errors for the predictions at this node.
    /// </summary>
    /// <returns>The sum of squared errors.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method measures how accurate the node's predictions are by:
    /// 1. Taking each sample that reached this node
    /// 2. Finding the difference between the predicted value and the actual value
    /// 3. Squaring that difference (to make it positive)
    /// 4. Adding up all these squared differences
    /// 
    /// A smaller result means better predictions.
    /// </remarks>
    private T CalculateSumSquaredError()
    {
        if (Samples == null || Samples.Count == 0)
        {
            return _numOps.Zero;
        }

        return Samples.Aggregate(_numOps.Zero, (sum, sample) =>
        {
            var error = _numOps.Subtract(sample.Target, Prediction);
            return _numOps.Add(sum, _numOps.Multiply(error, error));
        });
    }
}
