namespace AiDotNet.Regression;

/// <summary>
/// Represents a conditional inference tree regression model that builds decision trees based on statistical tests.
/// </summary>
/// <remarks>
/// <para>
/// A conditional inference tree is a type of decision tree that uses statistical tests to determine optimal
/// splits in the data. Unlike traditional decision trees that use measures like Gini impurity or information gain,
/// conditional inference trees use statistical significance testing to create unbiased trees that don't favor
/// features with many possible split points.
/// </para>
/// <para><b>For Beginners:</b> This class creates a special type of decision tree for predicting numerical values.
/// 
/// Think of a decision tree like a flowchart of yes/no questions that helps you make predictions:
/// - The tree starts with a question (like "Is temperature > 70°F?")
/// - Based on the answer, it follows different branches
/// - It continues asking questions until it reaches a final prediction
/// 
/// What makes this tree special is how it chooses the questions:
/// - It uses statistical tests to find the most meaningful questions to ask
/// - It avoids favoring certain types of data unfairly
/// - It provides a measurement of confidence (p-value) for each split
/// 
/// This approach tends to create more reliable and fair prediction models.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ConditionalInferenceTreeRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// The options that control the behavior of the conditional inference tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the configuration settings for the conditional inference tree, including maximum tree depth,
    /// minimum samples required for splitting, statistical significance threshold, and parallelism settings.
    /// These options determine how the tree is constructed during training.
    /// </para>
    /// <para><b>For Beginners:</b> This stores the settings that control how the decision tree works:
    /// 
    /// - How deep (complex) the tree can grow
    /// - How many data points are needed before asking a question
    /// - How confident the model must be that a pattern is real before creating a split
    /// - How many calculations can run at the same time for better performance
    /// 
    /// These settings shape how the model learns from your data and makes predictions.
    /// </para>
    /// </remarks>
    private readonly ConditionalInferenceTreeOptions _options;

    /// <summary>
    /// The root node of the decision tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the root node of the conditional inference tree, which is the starting point for both
    /// traversing the tree during prediction and for accessing the entire tree structure. The tree is built
    /// during training and consists of decision nodes (with feature index and threshold) and leaf nodes (with predictions).
    /// A null root indicates that the model has not been trained yet.
    /// </para>
    /// <para><b>For Beginners:</b> This is the starting point of the decision tree.
    /// 
    /// Think of a decision tree like a flowchart that starts with a single question:
    /// - The root node is the first question at the top of the flowchart
    /// - From there, you follow branches based on your answers
    /// - Eventually you reach a final prediction at the bottom
    /// 
    /// All predictions begin by evaluating this root node, then following the appropriate path
    /// through the tree based on the feature values of the data point being predicted.
    /// </para>
    /// </remarks>
    private ConditionalInferenceTreeNode<T>? _root;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalInferenceTreeRegression{T}"/> class.
    /// </summary>
    /// <param name="options">The options that control the tree building process.</param>
    /// <param name="regularization">Optional regularization to apply to input data to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes a new conditional inference tree regression model with the specified options
    /// and optional regularization. The options control various aspects of tree building, such as the maximum
    /// depth, minimum number of samples required to split a node, and the significance level for statistical tests.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new prediction model with your chosen settings.
    /// 
    /// The options parameter controls things like:
    /// - How deep (complex) the tree can grow
    /// - How many data points are needed before making a split
    /// - How confident the model must be before creating a split
    /// 
    /// The regularization parameter is optional and helps prevent "overfitting" - a problem where
    /// the model learns the training data too perfectly and performs poorly on new data.
    /// Think of it like teaching a student the principles rather than just memorizing specific examples.
    /// </para>
    /// </remarks>
    public ConditionalInferenceTreeRegression(ConditionalInferenceTreeOptions options, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
    }

    /// <summary>
    /// Asynchronously trains the regression model on the provided training data.
    /// </summary>
    /// <param name="x">The input features matrix where rows represent samples and columns represent features.</param>
    /// <param name="y">The target values vector corresponding to each sample in the input matrix.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the conditional inference tree regression model using the provided input features and
    /// target values. It first applies regularization to the data if specified, then builds the tree recursively
    /// starting from the root node. After building the tree, it calculates feature importances to identify which
    /// features have the most impact on predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions based on your data.
    /// 
    /// During training:
    /// - The data is first prepared by applying any regularization
    /// - A decision tree is built by finding the best questions to ask at each step
    /// - The model looks for patterns that connect your input features to the values you want to predict
    /// - The importance of each feature is calculated, showing which inputs matter most for predictions
    /// 
    /// The "Async" in the method name means it can run in the background while your program does other things,
    /// which is helpful for large datasets.
    /// </para>
    /// </remarks>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        // Note: Tree-based methods handle regularization through tree structure parameters
        // (MaxDepth, MinSamplesSplit, etc.), not through data transformation
        _root = await BuildTreeAsync(x, y, 0);
        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Recursively builds the decision tree by finding optimal splits.
    /// </summary>
    /// <param name="x">The input features matrix for the current node.</param>
    /// <param name="y">The target values vector for the current node.</param>
    /// <param name="depth">The current depth of the tree.</param>
    /// <returns>A task representing the asynchronous tree building operation, returning the node or null.</returns>
    /// <remarks>
    /// <para>
    /// This private method recursively builds the decision tree by finding the best split at each node. If the node
    /// meets stopping criteria (minimum samples, maximum depth), a leaf node is created with a prediction value.
    /// Otherwise, a split is found, the data is divided, and the method is called recursively for the left and right
    /// child nodes. The method leverages parallel processing to build subtrees simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the decision tree piece by piece.
    /// 
    /// At each step:
    /// - It checks if we should stop building (reached maximum depth or have too few samples)
    /// - If stopping, it creates a leaf node that gives a final prediction (average of target values)
    /// - Otherwise, it finds the best feature and value to split the data on
    /// - It divides the data into two groups based on this split
    /// - It repeats this process for each group, building left and right branches
    /// 
    /// The process creates a tree structure where each internal node asks a question about the data,
    /// and each leaf node provides a prediction.
    /// </para>
    /// </remarks>
    private async Task<ConditionalInferenceTreeNode<T>?> BuildTreeAsync(Matrix<T> x, Vector<T> y, int depth)
    {
        if (x.Rows < _options.MinSamplesSplit || depth >= _options.MaxDepth)
        {
            return new ConditionalInferenceTreeNode<T> { IsLeaf = true, Prediction = y.Mean() };
        }

        var splitResult = await FindBestSplitAsync(x, y);
        if (splitResult == null)
        {
            return new ConditionalInferenceTreeNode<T> { IsLeaf = true, Prediction = y.Mean() };
        }

        var (leftX, leftY, rightX, rightY) = SplitData(x, y, splitResult.Value.Feature, splitResult.Value.Threshold);

        var node = new ConditionalInferenceTreeNode<T>
        {
            FeatureIndex = splitResult.Value.Feature,
            Threshold = splitResult.Value.Threshold,
            PValue = splitResult.Value.PValue,
            IsLeaf = false // Mark as internal node
        };

        var buildTasks = new[]
        {
            new Func<Task<ConditionalInferenceTreeNode<T>?>>(() => BuildTreeAsync(leftX, leftY, depth + 1)),
            new Func<Task<ConditionalInferenceTreeNode<T>?>>(() => BuildTreeAsync(rightX, rightY, depth + 1))
        };

        var results = await ParallelProcessingHelper.ProcessTasksInParallel(buildTasks, _options.MaxDegreeOfParallelism);

        node.Left = await results[0];
        node.Right = await results[1];

        return node;
    }

    /// <summary>
    /// Splits the data into left and right subsets based on the feature and threshold.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <param name="feature">The index of the feature to split on.</param>
    /// <param name="threshold">The threshold value for the split.</param>
    /// <returns>A tuple containing the left and right data subsets for both features and targets.</returns>
    /// <remarks>
    /// <para>
    /// This private method splits the input data into two subsets based on the specified feature and threshold.
    /// Samples where the feature value is less than or equal to the threshold go to the left subset, while
    /// samples where the feature value is greater than the threshold go to the right subset.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides data into two groups based on a question.
    /// 
    /// For example, if the feature is "Temperature" and the threshold is 70:
    /// - The question is: "Is Temperature = 70?"
    /// - The left group contains all data points where the answer is "Yes"
    /// - The right group contains all data points where the answer is "No"
    /// 
    /// This division creates the branching structure of the decision tree,
    /// where each branch represents a different path based on the answer to a question.
    /// </para>
    /// </remarks>
    private (Matrix<T> LeftX, Vector<T> LeftY, Matrix<T> RightX, Vector<T> RightY) SplitData(Matrix<T> x, Vector<T> y, int feature, T threshold)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        for (int i = 0; i < x.Rows; i++)
        {
            if (NumOps.LessThanOrEquals(x[i, feature], threshold))
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        return (
            x.GetRows(leftIndices),
            y.GetElements(leftIndices),
            x.GetRows(rightIndices),
            y.GetElements(rightIndices)
        );
    }

    /// <summary>
    /// Asynchronously finds the best feature and threshold to split the data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A task representing the asynchronous operation, returning the best split parameters or null if no significant split is found.</returns>
    /// <remarks>
    /// <para>
    /// This private method asynchronously evaluates all features to find the best split for the current node.
    /// It uses parallel processing to evaluate multiple features simultaneously, then selects the split with
    /// the lowest p-value (highest statistical significance).
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the most meaningful question to ask about the data.
    /// 
    /// To find the best question:
    /// - It examines each feature (like temperature, humidity, etc.)
    /// - For each feature, it finds the best threshold value to split the data
    /// - It calculates a "p-value" for each potential split (lower is better)
    /// - It picks the feature and threshold with the lowest p-value
    /// 
    /// The p-value measures statistical significance - a lower value means we're more confident
    /// that the split reveals a real pattern rather than random chance.
    /// 
    /// This method can examine multiple features at once to save time on large datasets.
    /// </para>
    /// </remarks>
    private async Task<(int Feature, T Threshold, T PValue)?> FindBestSplitAsync(Matrix<T> x, Vector<T> y)
    {
        var tasks = Enumerable.Range(0, x.Columns)
            .Select(feature => new Func<(int Feature, T Threshold, T PValue)?>(() => FindBestSplitForFeature(x, y, feature)))
            .ToArray();

        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks, _options.MaxDegreeOfParallelism);
        return results.Where(r => r.HasValue)
                      .OrderBy(r => r!.Value.PValue)
                      .FirstOrDefault();
    }

    /// <summary>
    /// Finds the best threshold to split data for a specific feature.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <param name="feature">The index of the feature to evaluate.</param>
    /// <returns>The best split parameters for the feature, or null if no significant split is found.</returns>
    /// <remarks>
    /// <para>
    /// This private method finds the best threshold to split the data for a specific feature. It evaluates
    /// potential thresholds by taking the midpoint between adjacent unique feature values, calculating the p-value
    /// of each potential split, and selecting the one with the lowest p-value. If no split has a p-value below
    /// the significance level, null is returned.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best value to use when asking a question about one feature.
    /// 
    /// For example, if our feature is "Temperature":
    /// - We gather all unique temperature values in the data
    /// - We sort them from smallest to largest
    /// - We try splitting between each pair of values (e.g., "Is Temperature = 68?", "Is Temperature = 72?")
    /// - For each potential split, we calculate how well it separates the data
    /// - We pick the split that creates the most statistically significant separation
    /// - If no split is significant enough, we decide this feature isn't useful
    /// 
    /// The p-value measures how likely it is that any pattern we see could have happened by chance.
    /// Lower p-values mean we're more confident the pattern is real.
    /// </para>
    /// </remarks>
    private (int Feature, T Threshold, T PValue)? FindBestSplitForFeature(Matrix<T> x, Vector<T> y, int feature)
    {
        var featureValues = x.GetColumn(feature);
        var uniqueValues = featureValues.Distinct().OrderBy(v => v).ToList();

        if (uniqueValues.Count < 2)
        {
            return null;
        }

        var bestSplit = (Threshold: default(T), PValue: NumOps.MaxValue);

        for (int i = 0; i < uniqueValues.Count - 1; i++)
        {
            var threshold = NumOps.Divide(NumOps.Add(uniqueValues[i], uniqueValues[i + 1]), NumOps.FromDouble(2));
            var leftIndices = featureValues.Select((v, idx) => (Value: v, Index: idx))
                .Where(pair => NumOps.LessThanOrEquals(pair.Value, threshold))
                .Select(pair => pair.Index)
                .ToList();
            var rightIndices = Enumerable.Range(0, y.Length).Except(leftIndices).ToList();

            if (leftIndices.Count == 0 || rightIndices.Count == 0)
            {
                continue;
            }

            var leftY = y.GetElements(leftIndices);
            var rightY = y.GetElements(rightIndices);

            var pValue = StatisticsHelper<T>.CalculatePValue(leftY, rightY, _options.StatisticalTest);

            if (NumOps.LessThan(pValue, bestSplit.PValue))
            {
                bestSplit = (threshold, pValue);
            }
        }

        return NumOps.LessThan(bestSplit.PValue, NumOps.FromDouble(_options.SignificanceLevel))
        ? (Feature: feature, Threshold: bestSplit.Threshold ?? NumOps.Zero, PValue: bestSplit.PValue)
        : null;
    }

    /// <summary>
    /// Asynchronously predicts target values for new input data.
    /// </summary>
    /// <param name="input">The input features matrix for prediction.</param>
    /// <returns>A task representing the asynchronous operation, returning a vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method asynchronously predicts target values for new input data by traversing the decision tree
    /// for each input sample. It first applies regularization to the input data if specified, then uses
    /// parallel processing to make predictions for multiple samples simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the trained model to make predictions on new data.
    /// 
    /// When making predictions:
    /// - The new data is first prepared using the same process as during training
    /// - For each data point, the model follows the decision tree from top to bottom
    /// - At each node, it answers a yes/no question and follows the appropriate branch
    /// - When it reaches a leaf node, it returns the prediction stored there
    /// 
    /// The method can process multiple data points at the same time to make predictions faster.
    /// The result is a set of predicted values, one for each input row.
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // Note: Tree-based methods handle regularization through tree structure parameters
        // (MaxDepth, MinSamplesSplit, etc.), not through data transformation
        var tasks = Enumerable.Range(0, input.Rows)
            .Select(i => new Func<T>(() => PredictSingle(input.GetRow(i))));

        return new Vector<T>(await ParallelProcessingHelper.ProcessTasksInParallel(tasks, _options.MaxDegreeOfParallelism));
    }

    /// <summary>
    /// Predicts a target value for a single input sample.
    /// </summary>
    /// <param name="input">The input feature vector.</param>
    /// <returns>The predicted target value.</returns>
    /// <remarks>
    /// <para>
    /// This private method predicts a target value for a single input sample by traversing the decision tree
    /// from the root to a leaf node. At each internal node, it compares the value of the specified feature
    /// to the node's threshold and follows the appropriate branch until it reaches a leaf node with a prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a prediction for a single data point.
    /// 
    /// To make a prediction:
    /// - It starts at the top of the decision tree (the root)
    /// - At each node, it checks a feature value against a threshold (e.g., "Is Temperature = 70?")
    /// - Based on the answer, it follows either the left branch (Yes) or right branch (No)
    /// - It continues until it reaches a leaf node, which contains the final prediction
    /// - It returns this prediction as the answer
    /// 
    /// Think of it like following a flowchart until you reach a final result.
    /// </para>
    /// </remarks>
    private T PredictSingle(Vector<T> input)
    {
        var node = _root;
        while (node != null && !node.IsLeaf)
        {
            if (NumOps.LessThanOrEquals(input[node.FeatureIndex], node.Threshold))
            {
                node = (ConditionalInferenceTreeNode<T>?)node.Left;
            }
            else
            {
                node = (ConditionalInferenceTreeNode<T>?)node.Right;
            }
        }

        return node != null ? node.Prediction : NumOps.Zero;
    }

    /// <summary>
    /// Asynchronously calculates the importance of each feature in the model.
    /// </summary>
    /// <param name="featureCount">The total number of features.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para>
    /// This protected method asynchronously calculates the importance of each feature in the model by traversing
    /// the decision tree. Feature importance is based on the statistical significance of splits using that feature,
    /// where features used in splits with lower p-values (higher significance) receive higher importance scores.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out which features are most important for predictions.
    /// 
    /// Feature importance:
    /// - Tells you which inputs have the biggest impact on the predictions
    /// - Is calculated by looking at all the splits in the tree that use each feature
    /// - Gives higher scores to features used in more statistically significant splits
    /// - Helps you understand what factors most influence the outcome you're predicting
    /// 
    /// For example, if "temperature" has a high importance score, it means that knowing the
    /// temperature gives you a lot of information about what the prediction will be.
    /// </para>
    /// </remarks>
    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[featureCount];
        await CalculateFeatureImportancesRecursiveAsync(_root, importances);
        FeatureImportances = new Vector<T>(importances);
    }

    /// <summary>
    /// Recursively calculates feature importances by traversing the tree.
    /// </summary>
    /// <param name="node">The current tree node.</param>
    /// <param name="importances">The array to store feature importance values.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para>
    /// This private method recursively traverses the decision tree to calculate feature importances.
    /// For each internal node, it adds the complement of the p-value (1 - p-value) to the importance
    /// score of the feature used at that node. Lower p-values (higher statistical significance) result
    /// in higher importance scores.
    /// </para>
    /// <para><b>For Beginners:</b> This method walks through the entire tree to calculate importance scores.
    /// 
    /// For each decision node in the tree:
    /// - It identifies which feature is being used for the split
    /// - It calculates a score based on how statistically significant the split is
    /// - It adds this score to the running total for that feature
    /// - It continues through all branches of the tree
    /// 
    /// Features that appear in more nodes and have more significant splits will receive
    /// higher importance scores.
    /// </para>
    /// </remarks>
    private async Task CalculateFeatureImportancesRecursiveAsync(ConditionalInferenceTreeNode<T>? node, T[] importances)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        importances[node.FeatureIndex] = NumOps.Add(importances[node.FeatureIndex], NumOps.Subtract(NumOps.One, node.PValue));

        var tasks = new[]
        {
            new Func<Task>(() => CalculateFeatureImportancesRecursiveAsync((ConditionalInferenceTreeNode<T>?)node.Left, importances)),
            new Func<Task>(() => CalculateFeatureImportancesRecursiveAsync((ConditionalInferenceTreeNode<T>?)node.Right, importances))
        };

        await ParallelProcessingHelper.ProcessTasksInParallel(tasks, _options.MaxDegreeOfParallelism);
    }

    /// <summary>
    /// Gets metadata about the regression model.
    /// </summary>
    /// <returns>A <see cref="ModelMetaData{T}"/> object containing model information.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the regression model, including its type, hyperparameters,
    /// and feature importances. This information can be useful for model comparison, logging, and
    /// generating reports about model performance.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides a summary of the model's settings and characteristics.
    /// 
    /// The metadata includes:
    /// - The type of model (Conditional Inference Tree)
    /// - The maximum depth of the tree
    /// - The minimum number of samples required to create a split
    /// - The significance level used for statistical tests
    /// - The importance scores for each feature
    /// 
    /// This information is useful for:
    /// - Comparing different models
    /// - Documenting what settings were used
    /// - Understanding the model's behavior
    /// - Generating reports about the model's performance
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.ConditionalInferenceTree,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "SignificanceLevel", _options.SignificanceLevel }
            }
        };

        // Add feature importances to AdditionalInfo if available
        if (FeatureImportances != null && FeatureImportances.Any())
        {
            metadata.AdditionalInfo["FeatureImportances"] = FeatureImportances.ToList();
        }

        return metadata;
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model to a byte array for storage or transmission.
    /// It includes all necessary information to reconstruct the model, including options,
    /// the tree structure, and feature importances.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the model to binary data that can be stored or shared.
    /// 
    /// Serialization converts the model into a compact format that:
    /// - Can be saved to a file
    /// - Can be sent over a network
    /// - Can be stored in a database
    /// - Can be loaded later to make predictions without retraining
    /// 
    /// The saved data includes:
    /// - All the model's settings (like max depth)
    /// - The entire structure of the decision tree
    /// - The feature importance scores
    /// 
    /// This is like taking a snapshot of the model for future use.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(_options.SignificanceLevel);
        writer.Write(_options.Seed ?? -1);

        // Serialize the tree structure
        SerializeNode(writer, _root);

        // Serialize feature importances
        writer.Write(FeatureImportances.Length);
        foreach (var importance in FeatureImportances)
        {
            writer.Write(Convert.ToDouble(importance));
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model from a serialized byte array. It reads the options,
    /// tree structure, and feature importances from the byte array and rebuilds the model.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from binary data.
    /// 
    /// Deserialization converts the binary data back into a working model:
    /// - It loads all the model's settings
    /// - It reconstructs the entire decision tree
    /// - It restores the feature importance scores
    /// 
    /// This allows you to:
    /// - Use a model that was trained earlier
    /// - Share models between different applications
    /// - Deploy models to production environments
    /// 
    /// It's like restoring the model from a snapshot so you can use it again without retraining.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        _options.SignificanceLevel = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;

        // Deserialize the tree structure
        _root = DeserializeNode(reader);

        // Deserialize feature importances
        int importanceCount = reader.ReadInt32();
        var importances = new T[importanceCount];
        for (int i = 0; i < importanceCount; i++)
        {
            importances[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        FeatureImportances = new Vector<T>(importances);
    }

    /// <summary>
    /// Serializes a tree node to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <param name="node">The node to serialize.</param>
    /// <remarks>
    /// <para>
    /// This private method serializes a single tree node to a binary writer. It writes a flag indicating
    /// whether the node exists, then if it does, it writes whether it's a leaf node, its prediction value,
    /// and for internal nodes, the feature index, threshold, p-value, and recursively serializes child nodes.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves a single node of the decision tree to binary data.
    /// 
    /// For each node in the tree:
    /// - It first writes whether the node exists
    /// - If it's a leaf node, it saves the prediction value
    /// - If it's a decision node, it saves the feature index, threshold value, and confidence level (p-value)
    /// - It then continues to save any child nodes
    /// 
    /// This process maps the tree structure into a linear sequence of data
    /// that can be stored and later reconstructed.
    /// </para>
    /// </remarks>
    private void SerializeNode(BinaryWriter writer, ConditionalInferenceTreeNode<T>? node)
    {
        if (node == null)
        {
            writer.Write(false);
            return;
        }

        writer.Write(true);
        writer.Write(node.IsLeaf);
        writer.Write(Convert.ToDouble(node.Prediction));

        if (!node.IsLeaf)
        {
            writer.Write(node.FeatureIndex);
            writer.Write(Convert.ToDouble(node.Threshold));
            writer.Write(Convert.ToDouble(node.PValue));
            SerializeNode(writer, (ConditionalInferenceTreeNode<T>?)node.Left);
            SerializeNode(writer, (ConditionalInferenceTreeNode<T>?)node.Right);
        }
    }

    /// <summary>
    /// Deserializes a tree node from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>The deserialized node, or null if no node exists at the current position.</returns>
    /// <remarks>
    /// <para>
    /// This private method deserializes a single tree node from a binary reader. It reads a flag indicating
    /// whether a node exists, then if it does, it reads whether it's a leaf node, its prediction value,
    /// and for internal nodes, the feature index, threshold, p-value, and recursively deserializes child nodes.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs a single node of the decision tree from binary data.
    /// 
    /// When reading each node:
    /// - It first checks if a node exists at this position
    /// - If not, it returns null
    /// - If a node exists, it reads whether it's a leaf node or a decision node
    /// - For leaf nodes, it reads the prediction value
    /// - For decision nodes, it reads the feature index, threshold value, and confidence level
    /// - It then recursively reconstructs any child nodes
    /// 
    /// This process transforms the linear sequence of data back into the
    /// original tree structure with all its branches and leaves.
    /// </para>
    /// </remarks>
    private ConditionalInferenceTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        if (!reader.ReadBoolean())
        {
            return null;
        }

        var node = new ConditionalInferenceTreeNode<T>
        {
            IsLeaf = reader.ReadBoolean(),
            Prediction = NumOps.FromDouble(reader.ReadDouble())
        };

        if (!node.IsLeaf)
        {
            node.FeatureIndex = reader.ReadInt32();
            node.Threshold = NumOps.FromDouble(reader.ReadDouble());
            node.PValue = NumOps.FromDouble(reader.ReadDouble());
            node.Left = DeserializeNode(reader);
            node.Right = DeserializeNode(reader);
        }

        return node;
    }

    /// <summary>
    /// Creates a new instance of the conditional inference tree regression model with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="ConditionalInferenceTreeRegression{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new conditional inference tree regression model that has the same configuration 
    /// as the current instance. It's used for model persistence, cloning, and transferring the model's 
    /// configuration to new instances.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your model that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar models
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its settings
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create and return a new instance with the same configuration
        return new ConditionalInferenceTreeRegression<T>(_options, Regularization);
    }
}
