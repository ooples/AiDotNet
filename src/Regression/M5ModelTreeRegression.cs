namespace AiDotNet.Regression;

/// <summary>
/// Represents an M5 model tree for regression problems, combining decision tree structure with linear models at the leaves.
/// </summary>
/// <remarks>
/// <para>
/// The M5 model tree is an advanced regression technique that combines the benefits of decision trees and linear regression.
/// Instead of using a single value at each leaf node (as in standard regression trees), M5 model trees fit linear regression
/// models at each leaf. This allows the tree to capture both global patterns through its structure and local patterns through
/// the linear models, often resulting in more accurate predictions compared to standard regression trees.
/// </para>
/// <para><b>For Beginners:</b> An M5 model tree is like a smart decision-making system for predicting numbers.
/// 
/// Think of it like a flowchart for home price prediction:
/// - The tree asks questions about the home (Is it bigger than 2000 sq ft? Is it in neighborhood A?)
/// - Based on the answers, you follow different paths down the tree
/// - When you reach the end (a leaf), instead of getting a single price value, you get a mini-calculator (linear model)
/// - This mini-calculator uses the home's features to make a more precise prediction for that specific group of homes
/// 
/// For example, for small homes in urban areas, the price might depend more on location,
/// while for large homes in suburbs, the number of bathrooms might be more important.
/// The M5 model tree captures these different patterns for different groups of data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class M5ModelTree<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// The configuration options for the M5 model tree algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options control the behavior of the M5 model tree during training and prediction, including parameters
    /// such as the maximum tree depth, minimum number of instances per leaf, pruning settings, and whether to use
    /// linear regression models at the leaves.
    /// </para>
    /// <para><b>For Beginners:</b> These are the settings that control how the tree grows and makes predictions.
    /// 
    /// Key settings include:
    /// - How deep the tree can grow (MaxDepth)
    /// - How many data points must be in each leaf (MinInstancesPerLeaf)
    /// - Whether to simplify the tree after building it (UsePruning)
    /// - Whether to use mini-calculators (linear models) at the leaves
    /// 
    /// These settings help balance between a model that fits your training data perfectly
    /// and one that will work well on new, unseen data.
    /// </para>
    /// </remarks>
    private readonly M5ModelTreeOptions _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="M5ModelTree{T}"/> class with optional custom options and regularization.
    /// </summary>
    /// <param name="options">Custom options for the M5 model tree algorithm. If null, default options are used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization is applied.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new M5 model tree with the specified options and regularization. If no options are provided,
    /// default values are used. Regularization helps prevent overfitting by penalizing complex models.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new M5 model tree with your chosen settings.
    /// 
    /// When creating an M5 model tree:
    /// - You can provide custom settings (options) or use the defaults
    /// - You can add regularization, which helps prevent the model from memorizing the training data too closely
    /// 
    /// Regularization is like adding guardrails that prevent the model from becoming too complex
    /// or fitting too closely to the training data, which helps it perform better on new data.
    /// </para>
    /// </remarks>
    public M5ModelTree(M5ModelTreeOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new M5ModelTreeOptions();
    }

    /// <summary>
    /// Asynchronously trains the M5 model tree using the provided features and target values.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target vector containing the continuous values to predict.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds the M5 model tree structure recursively, finding the best feature splits at each node to minimize
    /// prediction error. If enabled, it applies pruning to reduce tree complexity and prevent overfitting. Finally, it
    /// calculates feature importances to provide insights into which features are most influential in making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model learns from your data.
    /// 
    /// During training:
    /// 1. The tree is built from top to bottom by finding the best questions to ask about your data
    /// 2. If pruning is enabled, the tree is simplified by removing unnecessary branches
    /// 3. The model calculates which features are most important for predictions
    /// 
    /// The "Async" in the name means this method can run efficiently without blocking other operations,
    /// which is especially helpful when training with large datasets.
    /// </para>
    /// </remarks>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        Root = await BuildTreeAsync(x, y, 0);
        if (_options.UsePruning)
        {
            await PruneTreeAsync(Root);
        }
        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Asynchronously generates predictions for new data points using the trained M5 model tree.
    /// </summary>
    /// <param name="input">The feature matrix where each row is a sample to predict.</param>
    /// <returns>A task that represents the asynchronous operation, containing a vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method traverses the tree for each input sample, following the decision path until reaching a leaf node.
    /// At the leaf, it either uses the stored constant value or applies the linear regression model to generate the prediction.
    /// The predictions for multiple samples are processed in parallel for improved performance.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the model makes predictions on new data.
    /// 
    /// For each data point:
    /// 1. The model follows the decision tree path based on the feature values
    /// 2. When it reaches a leaf node, it either:
    ///    - Returns the average value for that leaf (simple approach)
    ///    - Uses a mini-calculator (linear model) for a more precise prediction
    /// 3. All data points are processed at the same time (in parallel) for speed
    /// 
    /// For example, when predicting house prices, each house's features guide it through different
    /// paths in the tree until reaching the appropriate pricing model for that type of house.
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);
        var tasks = Enumerable.Range(0, input.Rows).Select(i => Task.Run(() => PredictSingle(input.GetRow(i))));
        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);
        for (int i = 0; i < results.Count; i++)
        {
            predictions[i] = results[i];
        }

        return predictions;
    }

    /// <summary>
    /// Asynchronously builds the decision tree structure recursively.
    /// </summary>
    /// <param name="x">The feature matrix for the current node.</param>
    /// <param name="y">The target vector for the current node.</param>
    /// <param name="depth">The current depth in the tree.</param>
    /// <returns>A task that represents the asynchronous operation, containing the constructed tree node.</returns>
    /// <remarks>
    /// <para>
    /// This method constructs the tree recursively by finding the best feature and threshold to split the data at each node.
    /// It stops splitting when the stopping criteria are met (minimum instances per leaf, maximum depth) or when no good split
    /// can be found. At each leaf node, depending on the options, it either stores the average target value or fits a linear
    /// regression model.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the decision tree structure one node at a time.
    /// 
    /// When building each part of the tree:
    /// - It looks for the best question to ask about the data (which feature and what threshold)
    /// - It splits the data into two groups based on the answer
    /// - It continues this process for each group, creating branches
    /// - It stops when groups are too small or the tree gets too deep
    /// - At the end points (leaves), it creates either a simple average or a mini-calculator
    /// 
    /// The "Async" part means multiple branches can be built at the same time, making it faster.
    /// </para>
    /// </remarks>
    private async Task<DecisionTreeNode<T>> BuildTreeAsync(Matrix<T> x, Vector<T> y, int depth)
    {
        if (x.Rows <= _options.MinInstancesPerLeaf || depth >= _options.MaxDepth)
        {
            return CreateLeafNode(x, y);
        }

        var bestSplit = await FindBestSplitAsync(x, y);
        if (bestSplit == null)
        {
            return CreateLeafNode(x, y);
        }

        var (leftX, leftY, rightX, rightY) = SplitData(x, y, bestSplit.Value.Feature, bestSplit.Value.Threshold);

        var leftChildTask = BuildTreeAsync(leftX, leftY, depth + 1);
        var rightChildTask = BuildTreeAsync(rightX, rightY, depth + 1);

        await Task.WhenAll(leftChildTask, rightChildTask);

        return new DecisionTreeNode<T>(bestSplit.Value.Feature, bestSplit.Value.Threshold)
        {
            Left = await leftChildTask,
            Right = await rightChildTask
        };
    }

    /// <summary>
    /// Asynchronously finds the best feature and threshold to split the data.
    /// </summary>
    /// <param name="x">The feature matrix for the current node.</param>
    /// <param name="y">The target vector for the current node.</param>
    /// <returns>A task that represents the asynchronous operation, containing the best split information or null if no good split is found.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates all features in parallel to find the one that provides the best split according to the standard
    /// deviation reduction (SDR) criterion. For each feature, it considers various thresholds and selects the one that
    /// maximizes the reduction in variance after splitting the data.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best question to ask about your data.
    /// 
    /// For each possible feature (like house size, number of bedrooms, etc.):
    /// - It tries different threshold values to split the data
    /// - It measures how much each split reduces the variability in predictions
    /// - It picks the feature and threshold that gives the biggest improvement
    /// 
    /// The "Async" means it checks multiple features at the same time for speed.
    /// The goal is to find the question that best separates the data into two groups
    /// with more similar values within each group.
    /// </para>
    /// </remarks>
    private async Task<(int Feature, T Threshold)?> FindBestSplitAsync(Matrix<T> x, Vector<T> y)
    {
        var tasks = Enumerable.Range(0, x.Columns).Select(feature =>
            Task.Run(() => FindBestSplitForFeature(x, y, feature)));

        var results = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

        var bestSplit = results.OrderByDescending(r => r.SDR).FirstOrDefault();
        return bestSplit.Feature == -1 ? null : (bestSplit.Feature, bestSplit.Threshold);
    }

    /// <summary>
    /// Finds the best threshold for a specific feature to split the data.
    /// </summary>
    /// <param name="x">The feature matrix for the current node.</param>
    /// <param name="y">The target vector for the current node.</param>
    /// <param name="feature">The feature index to evaluate.</param>
    /// <returns>Information about the best split for this feature, including the feature index, threshold, and standard deviation reduction.</returns>
    /// <remarks>
    /// <para>
    /// This method evaluates different thresholds for the specified feature to find the one that maximizes the standard
    /// deviation reduction (SDR). It sorts the data by the feature value and considers the midpoint between adjacent
    /// values as potential thresholds.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best threshold value for a specific feature.
    /// 
    /// For a single feature (like house size):
    /// - It sorts all the data points by this feature's value
    /// - It tries splitting the data between each pair of adjacent values
    /// - For each split, it calculates how much the split reduces the variability
    /// - It returns the threshold that gives the biggest improvement
    /// 
    /// For example, when looking at house size, it might find that splitting at 2000 sq ft
    /// creates the most uniform groups in terms of price prediction.
    /// </para>
    /// </remarks>
    private (int Feature, T Threshold, T SDR) FindBestSplitForFeature(Matrix<T> x, Vector<T> y, int feature)
    {
        var featureValues = x.GetColumn(feature);
        var sortedIndices = featureValues.Select((value, index) => (value, index))
                                            .OrderBy(pair => pair.value)
                                            .Select(pair => pair.index)
                                            .ToArray();
        var bestSplit = (Threshold: NumOps.Zero, SDR: NumOps.Zero);

        for (int i = 1; i < sortedIndices.Length; i++)
        {
            var threshold = NumOps.Divide(NumOps.Add(featureValues[sortedIndices[i - 1]], featureValues[sortedIndices[i]]), NumOps.FromDouble(2));
            var sdr = CalculateSDR(x, y, feature, threshold);

            if (NumOps.GreaterThan(sdr, bestSplit.SDR))
            {
                bestSplit = (threshold, sdr);
            }
        }

        return (feature, bestSplit.Threshold, bestSplit.SDR);
    }

    /// <summary>
    /// Calculates the standard deviation reduction (SDR) for a potential split.
    /// </summary>
    /// <param name="x">The feature matrix for the current node.</param>
    /// <param name="y">The target vector for the current node.</param>
    /// <param name="feature">The feature index for the split.</param>
    /// <param name="threshold">The threshold value for the split.</param>
    /// <returns>The standard deviation reduction value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates how much a particular split reduces the variability in the target values. It computes
    /// the total variance of the target values before splitting and the weighted average of the variances after splitting,
    /// and returns the difference as the standard deviation reduction (SDR).
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how good a potential split is.
    /// 
    /// It works by:
    /// - Calculating how spread out (variable) the target values are before splitting
    /// - Splitting the data into two groups based on the feature and threshold
    /// - Calculating how spread out the values are within each group
    /// - Comparing the before and after spread to see how much improvement the split provides
    /// 
    /// A higher SDR value means the split does a better job of creating groups with similar values,
    /// which leads to more accurate predictions.
    /// </para>
    /// </remarks>
    private T CalculateSDR(Matrix<T> x, Vector<T> y, int feature, T threshold)
    {
        var (leftX, leftY, rightX, rightY) = SplitData(x, y, feature, threshold);
        var totalVariance = StatisticsHelper<T>.CalculateVariance(y);
        var leftVariance = StatisticsHelper<T>.CalculateVariance(leftY);
        var rightVariance = StatisticsHelper<T>.CalculateVariance(rightY);

        var leftWeight = NumOps.Divide(NumOps.FromDouble(leftY.Length), NumOps.FromDouble(y.Length));
        var rightWeight = NumOps.Divide(NumOps.FromDouble(rightY.Length), NumOps.FromDouble(y.Length));

        var weightedVariance = NumOps.Add(
            NumOps.Multiply(leftWeight, leftVariance),
            NumOps.Multiply(rightWeight, rightVariance)
        );

        return NumOps.Subtract(totalVariance, weightedVariance);
    }

    /// <summary>
    /// Creates a leaf node for the decision tree.
    /// </summary>
    /// <param name="x">The feature matrix for the leaf node.</param>
    /// <param name="y">The target vector for the leaf node.</param>
    /// <returns>A new leaf node with either a constant prediction value or a linear regression model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a leaf node for the decision tree, which can either store a constant prediction value (the mean
    /// of the target values) or a linear regression model depending on the configuration. Using linear models at the leaves
    /// is the distinctive feature of M5 model trees that allows them to capture more complex patterns than standard
    /// regression trees.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an endpoint (leaf) for the decision tree.
    /// 
    /// When creating a leaf node:
    /// - If linear models are enabled, it creates a mini-calculator (linear model) that uses the features
    ///   to make precise predictions for this specific group of data
    /// - If linear models are disabled, it simply uses the average value of all training samples in this group
    /// 
    /// The linear model approach is like having specialized experts for each subgroup of your data,
    /// while the simple average is like having a general rule for each subgroup.
    /// </para>
    /// </remarks>
    private DecisionTreeNode<T> CreateLeafNode(Matrix<T> x, Vector<T> y)
    {
        DecisionTreeNode<T> node;

        // Linear regression requires at least 2 samples to avoid singular matrix
        // (with 1 sample and intercept, xTx has determinant 0)
        if (_options.UseLinearRegressionAtLeaves && y.Length >= 2)
        {
            var model = FitLinearModel(x, y);
            // Use mean as Prediction fallback (used after deserialization when LinearModel is null)
            var mean = StatisticsHelper<T>.CalculateMean(y);
            node = new DecisionTreeNode<T>(mean) { LinearModel = model };
        }
        else
        {
            var mean = StatisticsHelper<T>.CalculateMean(y);
            node = new DecisionTreeNode<T>(mean);
        }

        // Populate Samples for use in pruning calculations
        for (int i = 0; i < y.Length; i++)
        {
            node.Samples.Add(new Sample<T>(x.GetRow(i), y[i]));
        }

        return node;
    }

    /// <summary>
    /// Fits a linear regression model to the data for a leaf node.
    /// </summary>
    /// <param name="x">The feature matrix for the leaf node.</param>
    /// <param name="y">The target vector for the leaf node.</param>
    /// <returns>A trained simple linear regression model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates and trains a simple linear regression model for a leaf node using the provided features and
    /// target values. The regularization setting from the M5 model tree is passed to the linear regression model to
    /// prevent overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a mini-calculator for a leaf node.
    /// 
    /// The linear model:
    /// - Finds the relationship between features and the target value for this specific group
    /// - Uses the equation: target = (weight1 × feature1) + (weight2 × feature2) + ... + constant
    /// - Calculates the optimal weights to make accurate predictions
    /// - Inherits the regularization settings to prevent overcomplicating the equation
    /// 
    /// This approach allows the model to capture more precise relationships within each leaf,
    /// rather than just using a single average value.
    /// </para>
    /// </remarks>
    private SimpleRegression<T> FitLinearModel(Matrix<T> x, Vector<T> y)
    {
        var regression = new SimpleRegression<T>(regularization: Regularization);
        regression.Train(x, y);

        return regression;
    }

    /// <summary>
    /// Asynchronously prunes the tree to reduce complexity and prevent overfitting.
    /// </summary>
    /// <param name="node">The current node to evaluate for pruning.</param>
    /// <returns>A task representing the asynchronous pruning operation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements post-pruning on the trained tree. It works bottom-up, first pruning the subtrees recursively
    /// and then deciding whether to convert the current node into a leaf based on the error comparison. A node is converted
    /// to a leaf if the error of the pruned subtree (with a penalty factor) is less than or equal to the error of the
    /// unpruned subtree.
    /// </para>
    /// <para><b>For Beginners:</b> This method simplifies the tree to prevent it from becoming too complex.
    /// 
    /// During pruning:
    /// - The method starts at the bottom of the tree and works upward
    /// - For each branch, it compares the error if the branch is kept versus if it's removed
    /// - If removing the branch (and replacing it with a leaf) doesn't increase the error too much,
    ///   the branch is pruned
    /// - The pruning factor controls how aggressive the pruning is
    /// 
    /// This process is like editing a long document - removing unnecessary details
    /// while keeping the important parts that affect the overall message.
    /// </para>
    /// </remarks>
    private async Task PruneTreeAsync(DecisionTreeNode<T>? node)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        await Task.WhenAll(
            PruneTreeAsync(node.Left),
            PruneTreeAsync(node.Right)
        );

        // Collect samples from subtree leaves before calculating errors
        // This is needed because internal nodes don't have Samples populated
        if (node.Samples.Count == 0)
        {
            CollectSamplesFromSubtree(node, node.Samples);
        }

        var subtreeError = CalculateSubtreeError(node);
        var leafError = CalculateLeafError(node);

        // Apply pruning factor
        var adjustedLeafError = NumOps.Multiply(leafError, NumOps.FromDouble(1 + _options.PruningFactor));

        if (NumOps.LessThanOrEquals(adjustedLeafError, subtreeError))
        {
            // Convert to leaf node
            node.Left = null;
            node.Right = null;
            node.IsLeaf = true;
            node.Prediction = CalculateAveragePrediction(node);
            node.Predictions = Vector<T>.CreateDefault(node.Samples.Count, node.Prediction);
            node.SumSquaredError = leafError;
        }
    }

    /// <summary>
    /// Collects all samples from leaf nodes in a subtree.
    /// </summary>
    /// <param name="node">The root of the subtree to collect samples from.</param>
    /// <param name="samples">The list to add samples to.</param>
    private void CollectSamplesFromSubtree(DecisionTreeNode<T>? node, List<Sample<T>> samples)
    {
        if (node == null) return;

        if (node.IsLeaf)
        {
            samples.AddRange(node.Samples);
        }
        else
        {
            CollectSamplesFromSubtree(node.Left, samples);
            CollectSamplesFromSubtree(node.Right, samples);
        }
    }

    /// <summary>
    /// Predicts a value for a single input vector by traversing the tree.
    /// </summary>
    /// <param name="input">The feature vector to predict.</param>
    /// <returns>The predicted value.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the tree structure is invalid.</exception>
    /// <remarks>
    /// <para>
    /// This method traverses the decision tree for a single input sample until reaching a leaf node. At the leaf,
    /// it either returns the stored constant prediction or applies the linear regression model to generate a prediction,
    /// depending on the tree configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method predicts a value for a single data point.
    /// 
    /// The prediction process:
    /// - Starts at the top of the tree (the root)
    /// - At each branch, it asks a question about a feature (Is it greater than the threshold?)
    /// - Based on the answer, it follows either the left or right path
    /// - When it reaches a leaf node, it produces a prediction using either:
    ///   - The simple average value stored in the leaf, or
    ///   - The mini-calculator (linear model) that calculates a custom value
    /// 
    /// It's like following a choose-your-own-adventure book where the features of your
    /// data point determine which page you turn to next.
    /// </para>
    /// </remarks>
    private T PredictSingle(Vector<T> input)
    {
        var node = Root;
        while (node != null && !node.IsLeaf)
        {
            // Use SplitValue which is set by the two-parameter constructor
            if (NumOps.LessThanOrEquals(input[node.FeatureIndex], node.SplitValue))
            {
                node = node?.Left;
            }
            else
            {
                node = node?.Right;
            }
        }

        if (node != null)
        {
            if (_options.UseLinearRegressionAtLeaves && node.LinearModel != null)
            {
                // Convert the input vector to a single-column matrix using the new method
                var inputMatrix = Matrix<T>.FromVector(input);
                return node.LinearModel.Predict(inputMatrix)[0];
            }
            else
            {
                return node.Prediction;
            }
        }

        throw new InvalidOperationException("Invalid tree structure");
    }

    /// <summary>
    /// Asynchronously calculates the importance of each feature in the model.
    /// </summary>
    /// <param name="featureCount">The total number of features.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the importance of each feature in the decision tree by assigning weights to nodes based on
    /// their position in the tree. Features used closer to the root receive higher importance scores. The resulting
    /// feature importance values are normalized to sum to 1.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out which features are most important for predictions.
    /// 
    /// The process:
    /// - It examines the whole tree structure
    /// - Features used near the top of the tree (root) are considered more important
    /// - Features used multiple times throughout the tree gain importance
    /// - The values are adjusted so they add up to 1 (or 100%)
    /// 
    /// This helps you understand which factors have the biggest impact on predictions.
    /// For example, you might learn that for house prices, location affects the prediction
    /// more than the number of bedrooms.
    /// </para>
    /// </remarks>
    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        FeatureImportances = new Vector<T>(featureCount);
        await CalculateFeatureImportancesRecursiveAsync(Root, NumOps.One);
        FeatureImportances = FeatureImportances.Divide(FeatureImportances.Sum());
    }

    /// <summary>
    /// Recursively calculates feature importances throughout the tree.
    /// </summary>
    /// <param name="node">The current node being evaluated.</param>
    /// <param name="weight">The current weight for this level in the tree.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para>
    /// This method recursively traverses the tree to calculate feature importances. It assigns a weight to the feature
    /// used at the current node and then recursively processes child nodes with a reduced weight. Features used at higher
    /// levels of the tree (closer to the root) receive greater weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method walks through the tree to calculate feature importance scores.
    /// 
    /// For each node in the tree:
    /// - If it's a decision node, the feature used gets some importance points
    /// - Features used near the top get more points (they're more important)
    /// - The method then continues to the child nodes with half the weight
    /// - The process repeats until reaching the leaves of the tree
    /// 
    /// This creates a ranking of features based on how frequently and how early they appear in the tree.
    /// </para>
    /// </remarks>
    private async Task CalculateFeatureImportancesRecursiveAsync(DecisionTreeNode<T>? node, T weight)
    {
        if (node == null || node.IsLeaf)
        {
            return;
        }

        FeatureImportances[node.FeatureIndex] = NumOps.Add(FeatureImportances[node.FeatureIndex], weight);

        var leftWeight = NumOps.Multiply(weight, NumOps.FromDouble(0.5));
        var rightWeight = NumOps.Multiply(weight, NumOps.FromDouble(0.5));

        await Task.WhenAll(
            CalculateFeatureImportancesRecursiveAsync(node.Left, leftWeight),
            CalculateFeatureImportancesRecursiveAsync(node?.Right, rightWeight)
        );
    }

    /// <summary>
    /// Gets metadata about the trained model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a metadata object containing information about the trained model, including the model type,
    /// hyperparameters, and feature importances. This metadata can be useful for model inspection, comparison, and
    /// serialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides a summary of the model and its settings.
    /// 
    /// The metadata includes:
    /// - The type of model (M5ModelTree)
    /// - All the settings used to create the model
    /// - Information about the importance of each feature
    /// - The type of regularization used (if any)
    /// 
    /// This is useful for:
    /// - Documenting how the model was built
    /// - Comparing different models
    /// - Sharing model information with others
    /// - Saving important details along with the model
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.M5ModelTree,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MaxDepth", _options.MaxDepth },
                { "MinInstancesPerLeaf", _options.MinInstancesPerLeaf },
                { "PruningFactor", _options.PruningFactor },
                { "UseLinearRegressionAtLeaves", _options.UseLinearRegressionAtLeaves },
                { "UsePruning", _options.UsePruning },
                { "SmoothingConstant", _options.SmoothingConstant },
                { "FeatureImportances", FeatureImportances },
                { "Regularization", Regularization.GetType().Name }
            }
        };
    }

    /// <summary>
    /// Calculates the prediction error for a subtree.
    /// </summary>
    /// <param name="node">The root node of the subtree.</param>
    /// <returns>The sum of squared errors for the subtree.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the total prediction error for a subtree by recursively summing the errors from all leaf nodes.
    /// It is used during the pruning process to determine whether pruning a node would increase or decrease the overall error.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how accurate a branch of the tree is.
    /// 
    /// Error calculation:
    /// - If the node is a leaf, it returns the error stored in that leaf
    /// - If the node has branches, it adds up the errors from all branches below it
    /// - The error represents how far off the predictions are from the actual values
    /// 
    /// This is used during pruning to decide whether removing branches would make the model better or worse.
    /// </para>
    /// </remarks>
    private T CalculateSubtreeError(DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            return NumOps.Zero;
        }

        if (node.IsLeaf)
        {
            return node.SumSquaredError;
        }

        return NumOps.Add(
            CalculateSubtreeError(node.Left),
            CalculateSubtreeError(node.Right)
        );
    }

    /// <summary>
    /// Calculates the error if a node were converted to a leaf.
    /// </summary>
    /// <param name="node">The node to evaluate.</param>
    /// <returns>The sum of squared errors if the node were a leaf.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates what the prediction error would be if the current node were converted to a leaf node.
    /// It computes the mean of the target values for all samples at this node and calculates the sum of squared differences
    /// between this mean and the actual target values. If regularization is used and the node has a linear model,
    /// a regularization term is added to the error.
    /// </para>
    /// <para><b>For Beginners:</b> This method predicts how accurate the model would be if a branch were simplified to a leaf.
    /// 
    /// It works by:
    /// - Calculating the average target value for all data points in this part of the tree
    /// - Measuring how far each actual value is from this average
    /// - Squaring these differences and adding them up to get the total error
    /// - If the node uses a linear model, it adds a penalty for complex models (regularization)
    /// 
    /// This helps decide whether keeping branches is worth the added complexity,
    /// or if a simple average would work almost as well.
    /// </para>
    /// </remarks>
    private T CalculateLeafError(DecisionTreeNode<T>? node)
    {
        if (node == null || node.Samples == null)
        {
            return NumOps.Zero;
        }

        var meanPrediction = CalculateAveragePrediction(node);
        var error = node.Samples.Select(sample =>
            NumOps.Multiply(
                NumOps.Subtract(sample.Target, meanPrediction),
                NumOps.Subtract(sample.Target, meanPrediction)
            )
        ).Aggregate(NumOps.Zero, NumOps.Add);

        // Apply regularization if a linear model exists
        if (node.LinearModel != null && node.LinearModel.Coefficients != null)
        {
            var regularizedCoefficients = Regularization.Regularize(node.LinearModel.Coefficients);
            var regularizationTerm = regularizedCoefficients.Subtract(node.LinearModel.Coefficients).L2Norm();
            error = NumOps.Add(error, regularizationTerm);
        }

        return error;
    }

    /// <summary>
    /// Calculates the average prediction value for a node.
    /// </summary>
    /// <param name="node">The node to calculate the average for.</param>
    /// <returns>The average of all target values in the node.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the average of all target values for the samples at the current node. This average is used
    /// as the prediction value when converting a node to a leaf during pruning or when using simple constant predictions
    /// instead of linear models at the leaves.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the average target value for all data points in a node.
    /// 
    /// It simply:
    /// - Adds up all the target values from the training samples that reached this node
    /// - Divides by the number of samples to get the average
    /// 
    /// This average becomes the prediction value when:
    /// - The model doesn't use linear models at leaves
    /// - A node is being converted to a leaf during pruning
    /// 
    /// For example, if all houses that reached this node had prices of $250K, $300K, and $280K,
    /// the average prediction would be $276,667.
    /// </para>
    /// </remarks>
    private T CalculateAveragePrediction(DecisionTreeNode<T> node)
    {
        if (node.Samples == null || node.Samples.Count == 0)
        {
            return NumOps.Zero;
        }

        var sum = node.Samples.Aggregate(NumOps.Zero, (acc, sample) => NumOps.Add(acc, sample.Target));
        return NumOps.Divide(sum, NumOps.FromDouble(node.Samples.Count));
    }

    /// <summary>
    /// Calculates the maximum depth of the tree.
    /// </summary>
    /// <param name="node">The root node of the tree or subtree.</param>
    /// <returns>The maximum depth of the tree.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the maximum depth of the tree by recursively finding the longest path from the root to any leaf.
    /// The depth of a tree provides insight into its complexity and is useful for model analysis and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how many levels deep the tree goes.
    /// 
    /// Tree depth:
    /// - Starts at 0 for leaf nodes
    /// - For branch nodes, it's 1 plus the maximum depth of its deepest child
    /// - A deeper tree can make more fine-grained distinctions but risks overfitting
    /// - A shallower tree is simpler but might miss important patterns
    /// 
    /// This is like measuring how many questions you have to ask at most before getting an answer.
    /// A depth of 5 means some data points need to answer 5 questions before getting a prediction.
    /// </para>
    /// </remarks>
    private int CalculateTreeDepth(DecisionTreeNode<T>? node)
    {
        if (node == null || node.IsLeaf)
        {
            return 0;
        }
        return 1 + Math.Max(CalculateTreeDepth(node.Left), CalculateTreeDepth(node?.Right));
    }

    /// <summary>
    /// Counts the total number of nodes in the tree.
    /// </summary>
    /// <param name="node">The root node of the tree or subtree.</param>
    /// <returns>The total number of nodes in the tree.</returns>
    /// <remarks>
    /// <para>
    /// This method counts the total number of nodes in the tree by recursively traversing all branches and leaves.
    /// The node count provides a measure of the tree's size and complexity, which is useful for model analysis
    /// and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This method counts how many decision points and endpoints the tree has.
    /// 
    /// Node counting:
    /// - Counts each decision point (branch) and endpoint (leaf)
    /// - A larger number indicates a more complex tree
    /// - Used to understand the size and complexity of the model
    /// - Can help diagnose if a tree is too large or too simple
    /// 
    /// This is like counting all the boxes in a flowchart to see how complex it is.
    /// More nodes generally means more detailed decision rules.
    /// </para>
    /// </remarks>
    private int CountNodes(DecisionTreeNode<T>? node)
    {
        if (node == null)
        {
            return 0;
        }

        return 1 + CountNodes(node?.Left) + CountNodes(node?.Right);
    }

    /// <summary>
    /// Splits the data into left and right subsets based on a feature and threshold.
    /// </summary>
    /// <param name="x">The feature matrix to split.</param>
    /// <param name="y">The target vector to split.</param>
    /// <param name="feature">The feature index to split on.</param>
    /// <param name="threshold">The threshold value for the split.</param>
    /// <returns>Tuples containing the split feature matrices and target vectors.</returns>
    /// <remarks>
    /// <para>
    /// This method divides the data into two subsets based on the values of a specific feature compared to a threshold.
    /// Samples with feature values less than or equal to the threshold go to the left subset, while samples with feature
    /// values greater than the threshold go to the right subset. This is a fundamental operation in building a decision tree.
    /// </para>
    /// <para><b>For Beginners:</b> This method divides data into two groups based on a feature value.
    /// 
    /// The splitting process:
    /// - Looks at each data point's value for the selected feature
    /// - If the value is less than or equal to the threshold, it goes to the left group
    /// - If the value is greater than the threshold, it goes to the right group
    /// - Both features (X) and target values (Y) are kept together in their respective groups
    /// 
    /// For example, if splitting on house size with a threshold of 2000 sq ft:
    /// - Houses smaller than or equal to 2000 sq ft go to the left group
    /// - Houses larger than 2000 sq ft go to the right group
    /// 
    /// This is the fundamental operation that creates the branches in the decision tree.
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
    /// Creates a new instance of the M5ModelTree with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new M5ModelTree instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the abstract method from the base class, allowing the creation of a new model
    /// with the same configuration options and regularization settings. This is useful for model cloning,
    /// ensemble methods, or cross-validation scenarios where multiple instances of the same model type
    /// with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, if you're testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new M5ModelTree<T>(_options, Regularization);
    }

    /// <summary>
    /// Serializes the M5 model tree to a byte array, including linear models at leaf nodes.
    /// </summary>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(double.IsNaN(_options.MaxFeatures) ? -1 : (int)_options.MaxFeatures);
        writer.Write(_options.Seed ?? -1);
        writer.Write((int)_options.SplitCriterion);
        writer.Write(_options.MinInstancesPerLeaf);
        writer.Write(_options.UsePruning);
        writer.Write(_options.PruningFactor);
        writer.Write(_options.UseLinearRegressionAtLeaves);
        writer.Write(_options.SmoothingConstant);

        // Serialize feature importances
        writer.Write(FeatureImportances.Length);
        for (int i = 0; i < FeatureImportances.Length; i++)
        {
            writer.Write(Convert.ToDouble(FeatureImportances[i]));
        }

        // Serialize tree structure including linear models
        SerializeM5Node(writer, Root);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the M5 model tree from a byte array, including linear models at leaf nodes.
    /// </summary>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        int maxFeatures = reader.ReadInt32();
        _options.MaxFeatures = maxFeatures == -1 ? double.NaN : maxFeatures;
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;
        _options.SplitCriterion = (SplitCriterion)reader.ReadInt32();
        _options.MinInstancesPerLeaf = reader.ReadInt32();
        _options.UsePruning = reader.ReadBoolean();
        _options.PruningFactor = reader.ReadDouble();
        _options.UseLinearRegressionAtLeaves = reader.ReadBoolean();
        _options.SmoothingConstant = reader.ReadDouble();

        // Deserialize feature importances
        int featureCount = reader.ReadInt32();
        var importances = new T[featureCount];
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        FeatureImportances = new Vector<T>(importances);

        // Deserialize tree structure including linear models
        Root = DeserializeM5Node(reader);
    }

    /// <summary>
    /// Serializes an M5 tree node including its linear model if present.
    /// </summary>
    private void SerializeM5Node(BinaryWriter writer, DecisionTreeNode<T>? node)
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

        // Serialize linear model if present
        bool hasLinearModel = node.LinearModel != null;
        writer.Write(hasLinearModel);
        if (hasLinearModel)
        {
            var coefficients = node.LinearModel!.Coefficients;
            writer.Write(coefficients.Length);
            for (int i = 0; i < coefficients.Length; i++)
            {
                writer.Write(Convert.ToDouble(coefficients[i]));
            }
            writer.Write(Convert.ToDouble(node.LinearModel.Intercept));
        }

        SerializeM5Node(writer, node.Left);
        SerializeM5Node(writer, node.Right);
    }

    /// <summary>
    /// Deserializes an M5 tree node including its linear model if present.
    /// </summary>
    private DecisionTreeNode<T>? DeserializeM5Node(BinaryReader reader)
    {
        bool hasNode = reader.ReadBoolean();
        if (!hasNode) return null;

        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = reader.ReadInt32(),
            SplitValue = NumOps.FromDouble(reader.ReadDouble()),
            Prediction = NumOps.FromDouble(reader.ReadDouble()),
            IsLeaf = reader.ReadBoolean()
        };

        // Deserialize linear model if present
        bool hasLinearModel = reader.ReadBoolean();
        if (hasLinearModel)
        {
            int coeffCount = reader.ReadInt32();
            var coefficients = new T[coeffCount];
            for (int i = 0; i < coeffCount; i++)
            {
                coefficients[i] = NumOps.FromDouble(reader.ReadDouble());
            }
            var intercept = NumOps.FromDouble(reader.ReadDouble());

            // Create a SimpleRegression and set its coefficients
            var regression = new SimpleRegression<T>(regularization: Regularization);
            regression.SetCoefficientsAndIntercept(new Vector<T>(coefficients), intercept);
            node.LinearModel = regression;
        }

        node.Left = DeserializeM5Node(reader);
        node.Right = DeserializeM5Node(reader);

        return node;
    }
}
