namespace AiDotNet.Regression;

/// <summary>
/// Represents a decision tree regression model that predicts continuous values based on input features.
/// </summary>
/// <remarks>
/// <para>
/// Decision tree regression builds a model in the form of a tree structure where each internal node represents a 
/// decision based on a feature, each branch represents an outcome of that decision, and each leaf node 
/// represents a predicted value. The model is trained by recursively splitting the data based on the optimal 
/// feature and threshold that minimizes the prediction error.
/// </para>
/// <para><b>For Beginners:</b> A decision tree regression is like a flowchart that helps predict numerical values.
/// 
/// Think of it like answering a series of yes/no questions to reach a prediction:
/// - "Is the temperature above 75—F?"
/// - "Is the humidity below 50%?"
/// - "Is it a weekend?"
/// 
/// Each question splits the data into two groups, and the tree learns which questions to ask 
/// to make the most accurate predictions. For example, a decision tree might predict house prices 
/// based on features like square footage, number of bedrooms, and neighborhood.
/// 
/// The model is called a "tree" because it resembles an upside-down tree, with a single starting point (root) 
/// that branches out into multiple endpoints (leaves) where the final predictions are made.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DecisionTreeRegression<T> : DecisionTreeRegressionBase<T>
{
    /// <summary>
    /// The configuration options for the decision tree algorithm.
    /// </summary>
    private readonly DecisionTreeOptions _options;

    /// <summary>
    /// Random number generator used for feature selection and other randomized operations.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Vector storing the importance scores for each feature in the model.
    /// </summary>
    private Vector<T> _featureImportances;

    /// <summary>
    /// The regularization strategy applied to the model to prevent overfitting.
    /// </summary>
    private readonly IRegularization<T, Matrix<T>, Vector<T>> _regularization;

    /// <summary>
    /// Gets the number of trees in this model, which is always 1 for a single decision tree.
    /// </summary>
    /// <value>
    /// The number of trees in the model, which is 1 for this implementation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the number of decision trees used in the model. For the DecisionTreeRegression class, 
    /// this is always 1, as it implements a single decision tree. This property is provided for compatibility with 
    /// ensemble methods that may use multiple trees.
    /// </para>
    /// <para><b>For Beginners:</b> This property simply tells you how many trees are in the model.
    /// 
    /// A single decision tree model (like this one) always returns 1.
    /// 
    /// Other algorithms like Random Forests or Gradient Boosting use multiple trees 
    /// (sometimes hundreds or thousands) to make better predictions, but a basic 
    /// decision tree uses just one tree structure.
    /// </para>
    /// </remarks>
    public override int NumberOfTrees => 1;

    /// <summary>
    /// Initializes a new instance of the <see cref="DecisionTreeRegression{T}"/> class with optional configuration.
    /// </summary>
    /// <param name="options">Optional configuration options for the decision tree algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new decision tree regression model with the specified options and regularization strategy.
    /// If no options are provided, default values are used. If no regularization is specified, no regularization is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new decision tree prediction model.
    /// 
    /// When creating a decision tree, you can specify two main things:
    /// - Options: Controls how the tree grows (like its maximum depth or how many samples are needed to split)
    /// - Regularization: Helps prevent the model from becoming too complex and "memorizing" the training data
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a decision tree with default settings
    /// var tree = new DecisionTreeRegression&lt;double&gt;();
    /// 
    /// // Create a decision tree with custom options
    /// var options = new DecisionTreeOptions { MaxDepth = 5 };
    /// var customTree = new DecisionTreeRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public DecisionTreeRegression(DecisionTreeOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new DecisionTreeOptions();
        _regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        _featureImportances = Vector<T>.Empty();
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the decision tree model using the provided input features and target values.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This method builds the decision tree model by recursively splitting the data based on features and thresholds
    /// that best reduce the prediction error. Unlike traditional regression models, decision trees do not apply data
    /// regularization transformations. Instead, they control model complexity through structural parameters such as
    /// MaxDepth, MinSamplesSplit, and MinSamplesLeaf. After building the tree, feature importances are calculated.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the decision tree how to make predictions using your data.
    /// 
    /// You provide:
    /// - x: Your input data (features) - like house size, number of bedrooms, location, etc.
    /// - y: The values you want to predict - like house prices
    /// 
    /// The training process:
    /// 1. Looks at your data to find patterns
    /// 2. Decides which features are most useful for predictions
    /// 3. Creates a tree structure with decision rules
    /// 4. Figures out how important each feature is
    /// 
    /// After training, the model is ready to make predictions on new data.
    /// 
    /// Example:
    /// ```csharp
    /// // Create training data
    /// var features = new Matrix&lt;double&gt;(...); // Input features
    /// var targets = new Vector&lt;double&gt;(...);  // Target values
    /// 
    /// // Train the model
    /// decisionTree.Train(features, targets);
    /// ```
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Build the decision tree using the original data
        // Note: Decision tree regularization is handled through tree structure parameters
        // (MaxDepth, MinSamplesSplit, etc.), not through data transformation
        Root = BuildTree(x, y, 0);
        CalculateFeatureImportances(x);
    }

    /// <summary>
    /// Predicts target values for the provided input features using the trained decision tree model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method traverses the decision tree for each input sample to find the leaf node that corresponds to the sample's features.
    /// The prediction stored in the leaf node is then returned as the predicted value for that sample. Any specified regularization
    /// is applied to both the input data and the predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// How it works:
    /// 1. For each row of input data, the model starts at the top of the decision tree
    /// 2. At each decision point (node), it checks the value of a specific feature
    /// 3. Based on that value, it follows the appropriate branch
    /// 4. It continues until it reaches a leaf node (endpoint)
    /// 5. The value stored in that leaf node becomes the prediction
    /// 
    /// For example, if predicting house prices:
    /// - "Is square footage > 2000?" If yes, go left; if no, go right
    /// - "Is number of bedrooms > 3?" If yes, go left; if no, go right
    /// - Reach leaf node: Predict price = $350,000
    /// 
    /// Example:
    /// ```csharp
    /// // Create test data
    /// var newFeatures = new Matrix&lt;double&gt;(...);
    /// 
    /// // Make predictions
    /// var predictions = decisionTree.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = PredictSingle(input.GetRow(i), Root);
        }

        return new Vector<T>(predictions);
    }

    /// <summary>
    /// Gets metadata about the decision tree model and its configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type and configuration options. This information
    /// can be useful for model management, comparison, and documentation purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about your decision tree model.
    /// 
    /// The metadata includes:
    /// - The type of model (Decision Tree)
    /// - Maximum depth of the tree (how many questions it can ask)
    /// - Minimum samples required to split a node (how much data is needed to create a new decision point)
    /// - Maximum features considered at each split (how many features the model looks at when deciding how to split)
    /// 
    /// This information is helpful when:
    /// - Comparing different models
    /// - Documenting your model's configuration
    /// - Troubleshooting model performance
    /// 
    /// Example:
    /// ```csharp
    /// var metadata = decisionTree.GetModelMetadata();
    /// Console.WriteLine($"Model type: {metadata.ModelType}");
    /// Console.WriteLine($"Max depth: {metadata.AdditionalInfo["MaxDepth"]}");
    /// ```
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DecisionTree,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "MaxFeatures", _options.MaxFeatures }
            }
        };
    }

    /// <summary>
    /// Gets the importance score of a specific feature in the decision tree model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to get the importance score for.</param>
    /// <returns>The importance score of the specified feature.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the model hasn't been trained yet.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the feature index is invalid.</exception>
    /// <remarks>
    /// <para>
    /// This method returns the importance score of the specified feature in the trained decision tree model. Feature
    /// importance scores indicate how useful each feature was in building the tree, with higher values indicating 
    /// more important features. The scores are normalized to sum to 1.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you how important each feature is for making predictions.
    /// 
    /// Feature importance:
    /// - Measures how much each feature contributes to the model's predictions
    /// - Higher values mean the feature has more influence on the predictions
    /// - Values range from 0 to 1, and all feature importances sum to 1
    /// 
    /// For example, when predicting house prices:
    /// - Square footage might have importance 0.6 (very important)
    /// - Number of bedrooms might have importance 0.3 (somewhat important)
    /// - Year built might have importance 0.1 (less important)
    /// 
    /// This helps you understand which features matter most for your predictions.
    /// 
    /// Example:
    /// ```csharp
    /// // Get importance of the first feature (index 0)
    /// var importance = decisionTree.GetFeatureImportance(0);
    /// Console.WriteLine($"Feature 0 importance: {importance}");
    /// ```
    /// </para>
    /// </remarks>
    public T GetFeatureImportance(int featureIndex)
    {
        if (_featureImportances.Length == 0)
        {
            throw new InvalidOperationException("Feature importances are not available. Train the model first.");
        }

        if (featureIndex < 0 || featureIndex >= _featureImportances.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex), "Feature index is out of range.");
        }

        return _featureImportances[featureIndex];
    }

    /// <summary>
    /// Serializes the decision tree model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the decision tree model into a byte array that can be stored in a file, database,
    /// or transmitted over a network. The serialized data includes the model's configuration options and the
    /// complete tree structure.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves your trained model as a sequence of bytes.
    /// 
    /// Serialization allows you to:
    /// - Save your model to a file
    /// - Store your model in a database
    /// - Send your model over a network
    /// - Keep your model for later use without having to retrain it
    /// 
    /// The serialized data includes:
    /// - All the model's settings (like maximum depth)
    /// - The entire tree structure with all its decision rules
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = decisionTree.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("decisionTree.model", modelData);
    /// ```
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        // Serialize options
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(_options.MaxFeatures);
        writer.Write(_options.Seed ?? -1);

        // Serialize the tree structure
        SerializeNode(Root, writer);

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized decision tree model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a decision tree model from a byte array that was previously created using the
    /// Serialize method. It restores the model's configuration options and tree structure, allowing the model
    /// to be used for predictions without retraining.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize a model:
    /// - All settings are restored
    /// - The entire tree structure is reconstructed
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("decisionTree.model");
    /// 
    /// // Deserialize the model
    /// var decisionTree = new DecisionTreeRegression&lt;double&gt;();
    /// decisionTree.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = decisionTree.Predict(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        // Deserialize options
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        _options.MaxFeatures = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;

        // Deserialize the tree structure
        Root = DeserializeNode(reader);
    }

    /// <summary>
    /// Trains the decision tree model using the provided input features, target values, and sample weights.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <param name="sampleWeights">A vector of weights for each sample, indicating their importance during training.</param>
    /// <exception cref="ArgumentException">Thrown when input dimensions don't match.</exception>
    /// <remarks>
    /// <para>
    /// This method builds the decision tree model similar to the Train method, but allows specifying different weights
    /// for each training sample. Samples with higher weights have more influence on the training process, which can be 
    /// useful for handling imbalanced datasets or for boosting algorithms.
    /// </para>
    /// <para><b>For Beginners:</b> This method is similar to the regular Train method, but lets you specify how important each training example is.
    /// 
    /// Sample weights allow you to:
    /// - Give more importance to certain examples during training
    /// - Make the model pay more attention to rare cases
    /// - Balance uneven datasets (where some outcomes are much more common than others)
    /// 
    /// For example, when predicting house prices:
    /// - You might give higher weights to recent sales (more relevant)
    /// - You might give lower weights to unusual properties (potential outliers)
    /// - You might give higher weights to properties similar to the ones you'll make predictions for
    /// 
    /// Example:
    /// ```csharp
    /// // Create training data
    /// var features = new Matrix&lt;double&gt;(...);  // Input features
    /// var targets = new Vector&lt;double&gt;(...);   // Target values
    /// var weights = new Vector&lt;double&gt;(...);   // Sample weights
    /// 
    /// // Train the model with weights
    /// decisionTree.TrainWithWeights(features, targets, weights);
    /// ```
    /// </para>
    /// </remarks>
    public void TrainWithWeights(Matrix<T> x, Vector<T> y, Vector<T> sampleWeights)
    {
        // Validate inputs
        if (x.Rows != y.Length || x.Rows != sampleWeights.Length)
        {
            throw new ArgumentException("Input dimensions mismatch");
        }

        // Initialize the root node
        Root = new DecisionTreeNode<T>();

        // Build the tree recursively
        BuildTreeWithWeights(Root, x, y, sampleWeights, 0);

        // Calculate feature importances
        CalculateFeatureImportances(x);
    }

    /// <summary>
    /// Finds the best feature and threshold to split the data based on weighted samples.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <param name="weights">The sample weights vector.</param>
    /// <param name="featureIndices">The indices of features to consider for splitting.</param>
    /// <returns>A tuple containing the index of the best feature and the best threshold value.</returns>
    private (int featureIndex, T threshold) FindBestSplitWithWeights(Matrix<T> x, Vector<T> y, Vector<T> weights, IEnumerable<int> featureIndices)
    {
        int bestFeatureIndex = -1;
        T bestThreshold = NumOps.Zero;
        T bestScore = NumOps.MinValue;

        foreach (int featureIndex in featureIndices)
        {
            var featureValues = x.GetColumn(featureIndex);
            var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();

            for (int i = 0; i < uniqueValues.Count - 1; i++)
            {
                T threshold = NumOps.Divide(NumOps.Add(uniqueValues[i], uniqueValues[i + 1]), NumOps.FromDouble(2));
                T score = CalculateWeightedSplitScore(featureValues, y, weights, threshold);

                if (NumOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = threshold;
                }
            }
        }

        return (bestFeatureIndex, bestThreshold);
    }

    /// <summary>
    /// Calculates the score of a potential split based on weighted samples.
    /// </summary>
    /// <param name="featureValues">The values of the feature being considered for splitting.</param>
    /// <param name="y">The target vector.</param>
    /// <param name="weights">The sample weights vector.</param>
    /// <param name="threshold">The threshold value to evaluate for splitting.</param>
    /// <returns>The score of the split, with higher values indicating better splits.</returns>
    private T CalculateWeightedSplitScore(Vector<T> featureValues, Vector<T> y, Vector<T> weights, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < featureValues.Length; i++)
        {
            if (NumOps.LessThanOrEquals(featureValues[i], threshold))
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        if (leftIndices.Count == 0 || rightIndices.Count == 0)
        {
            return NumOps.MinValue;
        }

        T leftScore = CalculateWeightedVarianceReduction(y.GetElements(leftIndices), weights.GetElements(leftIndices));
        T rightScore = CalculateWeightedVarianceReduction(y.GetElements(rightIndices), weights.GetElements(rightIndices));

        T totalWeight = weights.Sum();
        T leftWeight = weights.GetElements(leftIndices).Sum();
        T rightWeight = weights.GetElements(rightIndices).Sum();

        return NumOps.Subtract(
            NumOps.Multiply(totalWeight, CalculateWeightedVarianceReduction(y, weights)),
            NumOps.Add(
                NumOps.Multiply(leftWeight, leftScore),
                NumOps.Multiply(rightWeight, rightScore)
            )
        );
    }

    /// <summary>
    /// Calculates the weighted variance reduction for a set of target values and weights.
    /// </summary>
    /// <param name="y">The target vector.</param>
    /// <param name="weights">The sample weights vector.</param>
    /// <returns>The weighted variance reduction value.</returns>
    private T CalculateWeightedVarianceReduction(Vector<T> y, Vector<T> weights)
    {
        T totalWeight = weights.Sum();
        // Protect against division by zero
        if (NumOps.Equals(totalWeight, NumOps.Zero))
        {
            return NumOps.Zero;
        }
        T weightedMean = NumOps.Divide(y.DotProduct(weights), totalWeight);
        T weightedVariance = NumOps.Zero;

        for (int i = 0; i < y.Length; i++)
        {
            T diff = NumOps.Subtract(y[i], weightedMean);
            weightedVariance = NumOps.Add(weightedVariance, NumOps.Multiply(weights[i], NumOps.Multiply(diff, diff)));
        }

        weightedVariance = NumOps.Divide(weightedVariance, totalWeight);
        return weightedVariance;
    }

    /// <summary>
    /// Builds a decision tree using weighted samples.
    /// </summary>
    /// <param name="node">The current node to build from.</param>
    /// <param name="x">The feature matrix for samples in this node.</param>
    /// <param name="y">The target vector for samples in this node.</param>
    /// <param name="weights">The sample weights vector for samples in this node.</param>
    /// <param name="depth">The current depth in the tree.</param>
    private void BuildTreeWithWeights(DecisionTreeNode<T> node, Matrix<T> x, Vector<T> y, Vector<T> weights, int depth)
    {
        if (depth >= Options.MaxDepth || x.Rows < Options.MinSamplesSplit)
        {
            // Create a leaf node
            node.IsLeaf = true;
            node.Prediction = CalculateWeightedLeafValue(y, weights);
            return;
        }

        int featuresToConsider = (int)Math.Min(x.Columns, Math.Max(1, Options.MaxFeatures * x.Columns));
        var featureIndices = Enumerable.Range(0, x.Columns).OrderBy(_ => _random.Next()).Take(featuresToConsider);

        // Find the best split
        var (featureIndex, threshold) = FindBestSplitWithWeights(x, y, weights, featureIndices);

        if (featureIndex == -1)
        {
            // No valid split found, create a leaf node
            node.IsLeaf = true;
            node.Prediction = CalculateWeightedLeafValue(y, weights);
            return;
        }

        // Split the data
        var (leftIndices, rightIndices) = SplitDataWithWeights(x, y, weights, featureIndex, threshold);

        if (leftIndices.Count == 0 || rightIndices.Count == 0)
        {
            // If split results in empty node, create a leaf
            node.IsLeaf = true;
            node.Prediction = CalculateWeightedLeafValue(y, weights);
            return;
        }

        // Create child nodes and continue building the tree
        node.IsLeaf = false;
        node.FeatureIndex = featureIndex;
        node.SplitValue = threshold;  // Use SplitValue since PredictSingle uses SplitValue
        node.Threshold = threshold;
        node.Left = new DecisionTreeNode<T>();
        node.Right = new DecisionTreeNode<T>();

        BuildTreeWithWeights(node.Left, x.GetRows(leftIndices), y.GetElements(leftIndices), weights.GetElements(leftIndices), depth + 1);
        BuildTreeWithWeights(node.Right, x.GetRows(rightIndices), y.GetElements(rightIndices), weights.GetElements(rightIndices), depth + 1);
    }

    /// <summary>
    /// Splits the data into left and right subsets based on a feature and threshold.
    /// </summary>
    /// <param name="x">The feature matrix.</param>
    /// <param name="y">The target vector.</param>
    /// <param name="weights">The sample weights vector.</param>
    /// <param name="featureIndex">The index of the feature to split on.</param>
    /// <param name="threshold">The threshold value to split on.</param>
    /// <returns>Two lists containing the indices of samples that go to the left and right child nodes.</returns>
    private (List<int> leftIndices, List<int> rightIndices) SplitDataWithWeights(Matrix<T> x, Vector<T> y, Vector<T> weights, int featureIndex, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.LessThanOrEquals(x[i, featureIndex], threshold))
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        return (leftIndices, rightIndices);
    }

    /// <summary>
    /// Calculates the weighted prediction value for a leaf node.
    /// </summary>
    /// <param name="y">The target vector for samples in this leaf.</param>
    /// <param name="weights">The sample weights vector for samples in this leaf.</param>
    /// <returns>The weighted average of target values, to be used as the leaf's prediction.</returns>
    private T CalculateWeightedLeafValue(Vector<T> y, Vector<T> weights)
    {
        T totalWeight = weights.Sum();
        // Protect against division by zero - if total weight is 0, return mean of y values
        if (NumOps.Equals(totalWeight, NumOps.Zero))
        {
            return StatisticsHelper<T>.CalculateMean(y);
        }
        return NumOps.Divide(y.DotProduct(weights), totalWeight);
    }

    /// <summary>
    /// Builds the decision tree recursively.
    /// </summary>
    /// <param name="x">The feature matrix for samples in this node.</param>
    /// <param name="y">The target vector for samples in this node.</param>
    /// <param name="depth">The current depth in the tree.</param>
    /// <returns>The root node of the built decision tree.</returns>
    private DecisionTreeNode<T>? BuildTree(Matrix<T> x, Vector<T> y, int depth)
    {
        if (depth >= _options.MaxDepth || x.Rows < _options.MinSamplesSplit)
        {
            return new DecisionTreeNode<T>
            {
                IsLeaf = true,
                Prediction = StatisticsHelper<T>.CalculateMean(y)
            };
        }

        int bestFeatureIndex = -1;
        T bestSplitValue = NumOps.Zero;
        T bestScore = NumOps.MinValue;

        int featuresToConsider = (int)Math.Min(x.Columns, Math.Max(1, _options.MaxFeatures * x.Columns));
        var featureIndices = Enumerable.Range(0, x.Columns).OrderBy(_ => _random.Next()).Take(featuresToConsider).ToList();

        foreach (int featureIndex in featureIndices)
        {
            var featureValues = x.GetColumn(featureIndex);
            var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();

            foreach (var splitValue in uniqueValues.Skip(1))
            {
                var leftIndices = new List<int>();
                var rightIndices = new List<int>();

                for (int i = 0; i < x.Rows; i++)
                {
                    if (NumOps.LessThan(x[i, featureIndex], splitValue))
                    {
                        leftIndices.Add(i);
                    }
                    else
                    {
                        rightIndices.Add(i);
                    }
                }

                if (leftIndices.Count == 0 || rightIndices.Count == 0) continue;

                T score = StatisticsHelper<T>.CalculateSplitScore(y, leftIndices, rightIndices, _options.SplitCriterion);

                if (NumOps.GreaterThan(score, bestScore))
                {
                    bestScore = score;
                    bestFeatureIndex = featureIndex;
                    bestSplitValue = splitValue;
                }
            }
        }

        if (bestFeatureIndex == -1)
        {
            return new DecisionTreeNode<T>
            {
                IsLeaf = true,
                Prediction = StatisticsHelper<T>.CalculateMean(y)
            };
        }

        var leftX = new List<Vector<T>>();
        var leftY = new List<T>();
        var rightX = new List<Vector<T>>();
        var rightY = new List<T>();

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.LessThan(x[i, bestFeatureIndex], bestSplitValue))
            {
                leftX.Add(x.GetRow(i));
                leftY.Add(y[i]);
            }
            else
            {
                rightX.Add(x.GetRow(i));
                rightY.Add(y[i]);
            }
        }

        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = bestFeatureIndex,
            SplitValue = bestSplitValue,
            IsLeaf = false, // Mark as internal node - required since default is true
            Left = BuildTree(new Matrix<T>(leftX), new Vector<T>(leftY), depth + 1),
            Right = BuildTree(new Matrix<T>(rightX), new Vector<T>(rightY), depth + 1),
            LeftSampleCount = leftX.Count,
            RightSampleCount = rightX.Count,
            Samples = [.. x.GetRows().Select((_, i) => new Sample<T>(x.GetRow(i), y[i]))]
        };

        return node;
    }

    /// <summary>
    /// Predicts the target value for a single sample by traversing the decision tree.
    /// </summary>
    /// <param name="input">The feature vector of the sample to predict.</param>
    /// <param name="node">The current node in the traversal.</param>
    /// <returns>The predicted value for the input sample.</returns>
    private T PredictSingle(Vector<T> input, DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            return NumOps.Zero;
        }

        if (node.IsLeaf)
        {
            return node.Prediction;
        }

        if (NumOps.LessThan(input[node.FeatureIndex], node.SplitValue))
        {
            return PredictSingle(input, node.Left);
        }
        else
        {
            return PredictSingle(input, node.Right);
        }
    }

    /// <summary>
    /// Calculates the importance scores for all features based on their contribution to the tree.
    /// </summary>
    /// <param name="x">The feature matrix used for training.</param>
    private void CalculateFeatureImportances(Matrix<T> x)
    {
        _featureImportances = new Vector<T>([.. Enumerable.Repeat(NumOps.Zero, x.Columns)]);
        CalculateFeatureImportancesRecursive(Root, x.Columns);
        NormalizeFeatureImportances();
        // Copy to the public property from base class so ensemble methods can access it
        FeatureImportances = _featureImportances;
    }

    /// <summary>
    /// Recursively calculates feature importances by traversing the tree.
    /// </summary>
    /// <param name="node">The current node in the traversal.</param>
    /// <param name="numFeatures">The total number of features in the model.</param>
    private void CalculateFeatureImportancesRecursive(DecisionTreeNode<T>? node, int numFeatures)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        T nodeImportance = CalculateNodeImportance(node);
        _featureImportances[node.FeatureIndex] = NumOps.Add(_featureImportances[node.FeatureIndex], nodeImportance);

        CalculateFeatureImportancesRecursive(node.Left, numFeatures);
        CalculateFeatureImportancesRecursive(node.Right, numFeatures);
    }

    /// <summary>
    /// Calculates the importance of a single node based on the variance reduction it achieves.
    /// </summary>
    /// <param name="node">The node to calculate importance for.</param>
    /// <returns>The importance score of the node.</returns>
    private T CalculateNodeImportance(DecisionTreeNode<T> node)
    {
        if (node.IsLeaf || node.Left == null || node.Right == null)
        {
            return NumOps.Zero;
        }

        T parentVariance = StatisticsHelper<T>.CalculateVariance(node.Samples.Select(s => s.Target));
        T leftVariance = StatisticsHelper<T>.CalculateVariance(node.Left.Samples.Select(s => s.Target));
        T rightVariance = StatisticsHelper<T>.CalculateVariance(node.Right.Samples.Select(s => s.Target));

        T leftWeight = NumOps.Divide(NumOps.FromDouble(node.Left.Samples.Count), NumOps.FromDouble(node.Samples.Count));
        T rightWeight = NumOps.Divide(NumOps.FromDouble(node.Right.Samples.Count), NumOps.FromDouble(node.Samples.Count));

        T varianceReduction = NumOps.Subtract(parentVariance, NumOps.Add(NumOps.Multiply(leftWeight, leftVariance), NumOps.Multiply(rightWeight, rightVariance)));

        // Normalize by the number of samples to give less weight to deeper nodes
        return NumOps.Multiply(varianceReduction, NumOps.FromDouble(node.Samples.Count));
    }

    /// <summary>
    /// Normalizes feature importance scores to sum to 1.
    /// </summary>
    private void NormalizeFeatureImportances()
    {
        T sum = _featureImportances.Aggregate(NumOps.Zero, (acc, x) => NumOps.Add(acc, x));

        if (NumOps.Equals(sum, NumOps.Zero))
        {
            return;
        }

        for (int i = 0; i < _featureImportances.Length; i++)
        {
            _featureImportances[i] = NumOps.Divide(_featureImportances[i], sum);
        }
    }

    /// <summary>
    /// Serializes a tree node to a binary writer.
    /// </summary>
    /// <param name="node">The node to serialize.</param>
    /// <param name="writer">The binary writer to write to.</param>
    private void SerializeNode(DecisionTreeNode<T>? node, BinaryWriter writer)
    {
        if (node == null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);
        writer.Write(node.FeatureIndex);
        writer.Write(Convert.ToDouble(node.SplitValue));
        writer.Write(Convert.ToDouble(node.Prediction));
        writer.Write(node.IsLeaf);

        SerializeNode(node.Left, writer);
        SerializeNode(node.Right, writer);
    }

    /// <summary>
    /// Deserializes a tree node from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>The deserialized node.</returns>
    private DecisionTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        bool nodeExists = reader.ReadBoolean();
        if (!nodeExists)
        {
            return null;
        }

        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = reader.ReadInt32(),
            SplitValue = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T)),
            Prediction = (T)Convert.ChangeType(reader.ReadDouble(), typeof(T)),
            IsLeaf = reader.ReadBoolean(),
            Left = DeserializeNode(reader),
            Right = DeserializeNode(reader)
        };

        return node;
    }

    /// <summary>
    /// Calculates feature importances based on the number of features.
    /// </summary>
    /// <param name="featureCount">The total number of features.</param>
    protected override void CalculateFeatureImportances(int featureCount)
    {
        _featureImportances = new Vector<T>(featureCount);
        T totalImportance = NumOps.Zero;

        // Traverse the tree and calculate feature importances
        CalculateNodeImportance(Root, NumOps.One);

        // Normalize feature importances
        if (!NumOps.Equals(totalImportance, NumOps.Zero))
        {
            for (int i = 0; i < _featureImportances.Length; i++)
            {
                _featureImportances[i] = NumOps.Divide(_featureImportances[i], totalImportance);
            }
        }

        void CalculateNodeImportance(DecisionTreeNode<T>? node, T nodeWeight)
        {
            if (node == null || node.IsLeaf)
            {
                return;
            }

            T improvement = CalculateImpurityImprovement(node);
            T weightedImprovement = NumOps.Multiply(improvement, nodeWeight);

            _featureImportances[node.FeatureIndex] = NumOps.Add(_featureImportances[node.FeatureIndex], weightedImprovement);
            totalImportance = NumOps.Add(totalImportance, weightedImprovement);

            T leftWeight = NumOps.Multiply(nodeWeight, NumOps.FromDouble(node.LeftSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount)));
            T rightWeight = NumOps.Multiply(nodeWeight, NumOps.FromDouble(node.RightSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount)));

            CalculateNodeImportance(node.Left, leftWeight);
            CalculateNodeImportance(node.Right, rightWeight);
        }

        T CalculateImpurityImprovement(DecisionTreeNode<T> node)
        {
            T parentImpurity = CalculateNodeImpurity(node);
            T leftImpurity = CalculateNodeImpurity(node.Left);
            T rightImpurity = CalculateNodeImpurity(node.Right);

            T leftWeight = NumOps.FromDouble(node.LeftSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount));
            T rightWeight = NumOps.FromDouble(node.RightSampleCount / (double)(node.LeftSampleCount + node.RightSampleCount));

            T weightedChildImpurity = NumOps.Add(
                NumOps.Multiply(leftWeight, leftImpurity),
                NumOps.Multiply(rightWeight, rightImpurity)
            );

            return NumOps.Subtract(parentImpurity, weightedChildImpurity);
        }

        T CalculateNodeImpurity(DecisionTreeNode<T>? node)
        {
            if (node == null || node.IsLeaf)
            {
                return NumOps.Zero;
            }

            // For regression trees, we use variance as the impurity measure
            T variance = NumOps.Zero;
            T mean = node.Prediction;
            int sampleCount = node.Samples.Count;

            foreach (var sample in node.Samples)
            {
                T diff = NumOps.Subtract(sample.Target, mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }

            return NumOps.Divide(variance, NumOps.FromDouble(sampleCount));
        }

        // Copy to the public property from base class so ensemble methods can access it
        FeatureImportances = _featureImportances;
    }

    /// <summary>
    /// Creates a new instance of the decision tree regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the decision tree regression model with the same configuration
    /// options and regularization method as the current instance, but without copying the trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters.
    /// 
    /// Think of it like getting a blank notepad with the same paper quality and size, 
    /// but without any writing on it yet. The new model has the same:
    /// - Maximum depth setting
    /// - Minimum samples split setting
    /// - Split criterion (how nodes decide which feature to split on)
    /// - Random seed (if specified)
    /// - Regularization method
    /// 
    /// But it doesn't have any of the actual tree structure that was learned from data.
    /// 
    /// This is mainly used internally when doing things like cross-validation or 
    /// creating ensembles of similar models with different training data.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        return new DecisionTreeRegression<T>(_options, _regularization);
    }
}
