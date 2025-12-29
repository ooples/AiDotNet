using Newtonsoft.Json;

namespace AiDotNet.Regression;

/// <summary>
/// Implements the AdaBoost.R2 algorithm for regression problems, an ensemble learning method that combines
/// multiple decision tree regressors to improve prediction accuracy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// AdaBoost.R2 (Adaptive Boosting for Regression) is an extension of the AdaBoost algorithm for regression tasks.
/// It works by training a sequence of weak regressors (decision trees) on repeatedly modified versions of the data.
/// The predictions from all regressors are then combined through a weighted majority vote to produce the final prediction.
/// </para>
/// <para>
/// In AdaBoost.R2, each training sample is assigned a weight that determines its importance during training.
/// Initially, all weights are equal. For each iteration, the weights of incorrectly predicted samples are increased
/// so that subsequent weak regressors focus more on difficult cases. The algorithm stops when the specified number
/// of estimators is reached or when the error rate exceeds 0.5.
/// </para>
/// <para><b>For Beginners:</b> AdaBoost.R2 is a powerful machine learning technique for predicting numeric values
/// (like prices, temperatures, or ages) rather than categories.
/// 
/// Think of AdaBoost.R2 as a team of experts (decision trees) working together to make predictions:
/// 1. The first "expert" makes predictions on all the training data
/// 2. The algorithm identifies which samples were predicted poorly
/// 3. The next expert pays special attention to those difficult samples
/// 4. This process repeats, creating a team of experts that each specialize in different aspects of the problem
/// 5. When making predictions, all experts "vote" on the final answer, but experts who performed better get more voting power
/// 
/// This approach is particularly effective because:
/// - It can turn a collection of "weak" learners (simple decision trees) into a "strong" learner
/// - It automatically focuses on the hardest parts of the problem
/// - It's less prone to overfitting than a single, complex model
/// 
/// AdaBoost.R2 is ideal for problems where you need high prediction accuracy and have enough training data
/// to build multiple models.
/// </para>
/// </remarks>
public class AdaBoostR2Regression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Options for configuring the AdaBoost.R2 regression algorithm.
    /// </summary>
    private AdaBoostR2RegressionOptions _options;

    /// <summary>
    /// The ensemble of decision trees and their corresponding weights.
    /// </summary>
    private List<(DecisionTreeRegression<T> Tree, T Weight)> _ensemble;

    /// <summary>
    /// Random number generator for creating diverse decision trees.
    /// </summary>
    private Random _random;

    /// <summary>
    /// Gets the number of decision trees in the ensemble.
    /// </summary>
    public override int NumberOfTrees => _options.NumberOfEstimators;

    /// <summary>
    /// Gets the maximum depth of each decision tree in the ensemble.
    /// </summary>
    public override int MaxDepth => _options.MaxDepth;

    /// <summary>
    /// Initializes a new instance of the <see cref="AdaBoostR2Regression{T}"/> class with specified options and regularization.
    /// </summary>
    /// <param name="options">The options for configuring the AdaBoost.R2 algorithm.</param>
    /// <param name="regularization">Optional regularization to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the AdaBoost.R2 regression model with the specified configuration options
    /// and regularization. The options control parameters such as the number of estimators (trees) to use,
    /// the maximum depth of each tree, and the minimum number of samples required to split a node.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new AdaBoost.R2 regression model with specific settings.
    /// 
    /// The options parameter controls important settings like:
    /// - How many decision trees to create (NumberOfEstimators)
    /// - How complex each tree can be (MaxDepth)
    /// - How much data is needed to make decisions in the trees (MinSamplesSplit)
    /// 
    /// The regularization parameter helps prevent "overfitting" - a situation where the model works well
    /// on training data but poorly on new data because it's too closely tailored to the specific examples
    /// it was trained on.
    /// 
    /// If you're not sure what values to use, the default options typically provide a good starting point
    /// for many regression problems.
    /// </para>
    /// </remarks>
    public AdaBoostR2Regression(AdaBoostR2RegressionOptions options, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _ensemble = new List<(DecisionTreeRegression<T> Tree, T Weight)>();
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the AdaBoost.R2 regression model on the provided input data and target values asynchronously.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to the input samples.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the AdaBoost.R2 algorithm for regression. It trains multiple decision trees
    /// sequentially, where each tree focuses more on samples that previous trees predicted poorly.
    /// The training process consists of the following steps:
    /// 1. Initialize sample weights equally for all training samples.
    /// 2. For the specified number of estimators:
    ///    a. Train a decision tree on the weighted data.
    ///    b. Calculate prediction errors for each sample.
    ///    c. Compute the weighted average error.
    ///    d. If the average error is â‰¥ 0.5, stop the training (the learner is too weak).
    ///    e. Calculate the weight for the current tree based on its error.
    ///    f. Update sample weights to focus more on poorly predicted samples.
    /// 3. Calculate feature importances across all trees in the ensemble.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model to make predictions based on your training data.
    /// 
    /// Here's what happens during training:
    /// 1. The method starts by giving equal importance to all training examples
    /// 2. For each new tree to be added to the ensemble:
    ///    - It trains a decision tree that pays attention to the importance weights
    ///    - It checks how well the tree performed on each example
    ///    - It calculates an overall error rate for the tree
    ///    - If the tree is too inaccurate (error â‰¥ 0.5), it stops adding more trees
    ///    - Otherwise, it calculates how much voting power this tree should get
    ///    - It updates the importance weights to focus more on examples that were predicted poorly
    /// 3. Finally, it calculates how important each feature (input variable) is for making predictions
    /// 
    /// This iterative process creates a diverse ensemble of trees that work together to make accurate predictions,
    /// with each tree specializing in different aspects of the problem.
    /// </para>
    /// </remarks>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _ensemble.Clear();
        var sampleWeights = Vector<T>.CreateDefault(y.Length, NumOps.One);
        var numSamples = x.Rows;

        for (int i = 0; i < _options.NumberOfEstimators; i++)
        {
            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                Seed = _random.Next(),
                SplitCriterion = _options.SplitCriterion
            };

            var tree = new DecisionTreeRegression<T>(treeOptions, Regularization);
            tree.TrainWithWeights(x, y, sampleWeights);

            var predictions = tree.Predict(x);
            var errors = CalculateErrors(y, predictions);
            var averageError = CalculateAverageError(errors, sampleWeights);

            if (NumOps.GreaterThanOrEquals(averageError, NumOps.FromDouble(0.5)))
            {
                break; // Stop if the error is too high
            }

            // Handle edge case: if averageError is 0, set a small value to avoid division by 0
            T effectiveError = NumOps.LessThan(averageError, NumOps.FromDouble(1e-10))
                ? NumOps.FromDouble(1e-10)
                : averageError;

            var beta = NumOps.Divide(effectiveError, NumOps.Subtract(NumOps.One, effectiveError));
            // Clamp beta to avoid Log(0) or Log(infinity)
            beta = MathHelper.Max(beta, NumOps.FromDouble(1e-10));
            var weight = NumOps.Log(NumOps.Divide(NumOps.One, beta));

            _ensemble.Add((tree, weight));

            // Update sample weights
            sampleWeights = UpdateSampleWeights(sampleWeights, errors, beta);
        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Makes predictions on new data using the trained ensemble of decision trees asynchronously.
    /// </summary>
    /// <param name="input">The input features matrix where each row is a sample to predict.</param>
    /// <returns>A task representing the asynchronous operation, containing the predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method makes predictions for new data points using the trained AdaBoost.R2 ensemble.
    /// The prediction process consists of the following steps:
    /// 1. Regularize the input data (if regularization is enabled).
    /// 2. For each decision tree in the ensemble:
    ///    a. Generate predictions for all input samples.
    ///    b. Multiply the predictions by the tree's weight.
    /// 3. Compute the weighted average of all tree predictions for each sample.
    /// 4. Apply regularization to the final predictions (if regularization is enabled).
    /// 
    /// The predictions are processed in parallel to improve performance on multi-core systems.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the trained model to make predictions on new data.
    /// 
    /// Here's how the prediction works:
    /// 1. Each decision tree in the ensemble makes its own prediction for each input sample
    /// 2. These predictions are weighted by how well each tree performed during training
    ///    (better trees have more influence on the final result)
    /// 3. The weighted predictions are averaged to produce the final prediction for each sample
    /// 
    /// The method uses parallel processing to make predictions faster on computers with multiple
    /// processing cores. This means that multiple trees can make their predictions simultaneously,
    /// speeding up the overall prediction process.
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // Handle empty ensemble case
        if (_ensemble.Count == 0)
        {
            return new Vector<T>(new T[input.Rows]); // Return zeros
        }

        // Note: Tree-based methods handle regularization through tree structure parameters
        // (MaxDepth, MinSamplesSplit, etc.), not through data transformation
        var sumWeights = _ensemble.Aggregate(NumOps.Zero, (acc, e) => NumOps.Add(acc, e.Weight));

        // Handle case where all weights sum to 0
        if (NumOps.Equals(sumWeights, NumOps.Zero))
        {
            // Use unweighted average instead
            sumWeights = NumOps.FromDouble(_ensemble.Count);
            var uniformWeight = NumOps.One;

            var result0 = new T[input.Rows];
            foreach (var (tree, _) in _ensemble)
            {
                var prediction = tree.Predict(input);
                for (int i = 0; i < input.Rows; i++)
                    result0[i] = NumOps.Add(result0[i], prediction[i]);
            }
            for (int i = 0; i < input.Rows; i++)
                result0[i] = NumOps.Divide(result0[i], sumWeights);
            return new Vector<T>(result0);
        }

        var result = new T[input.Rows];

        var tasks = _ensemble.Select(treeWeight => Task.Run(() =>
        {
            var (tree, weight) = treeWeight;
            var prediction = tree.Predict(input);
            return (prediction, weight);
        }));

        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = predictions.Aggregate(NumOps.Zero, (acc, p) =>
                NumOps.Add(acc, NumOps.Multiply(p.prediction[i], p.weight)));
            result[i] = NumOps.Divide(result[i], sumWeights);
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Calculates the absolute error between the target values and predictions.
    /// </summary>
    /// <param name="y">The true target values.</param>
    /// <param name="predictions">The predicted values.</param>
    /// <returns>A vector of absolute errors for each sample.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the absolute error between the true target values and the predicted values
    /// for each sample. The absolute error is used in AdaBoost.R2 to evaluate the performance of
    /// each weak learner and to update the sample weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how far off each prediction is from the true value.
    /// 
    /// The absolute error is simply the absolute value of the difference between the actual value
    /// and the predicted value. For example:
    /// - If the true value is 10 and the prediction is 12, the error is |10-12| = 2
    /// - If the true value is 10 and the prediction is 7, the error is |10-7| = 3
    /// 
    /// This error measure treats over-predictions and under-predictions equally, focusing only on
    /// the magnitude of the error. These error values are used to determine which samples need more
    /// attention in the next round of training.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateErrors(Vector<T> y, Vector<T> predictions)
    {
        return new Vector<T>(y.Select((yi, i) => NumOps.Abs(NumOps.Subtract(yi, predictions[i]))));
    }

    /// <summary>
    /// Calculates the weighted average error used in the AdaBoost.R2 algorithm.
    /// </summary>
    /// <param name="errors">The absolute errors for each sample.</param>
    /// <param name="sampleWeights">The weights assigned to each sample.</param>
    /// <returns>The weighted average error.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the weighted average error as defined in the AdaBoost.R2 algorithm.
    /// It first normalizes the errors by dividing by the maximum error, then computes a weighted
    /// average using the sample weights. This average error is used to determine the weight of
    /// the current weak learner in the ensemble and to decide whether to continue training.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how well (or poorly) a decision tree performed overall.
    /// 
    /// The process involves:
    /// 1. Finding the largest error in the predictions
    /// 2. Dividing all errors by this maximum error to normalize them to a 0-1 scale
    /// 3. Weighting these normalized errors by the importance of each sample
    /// 4. Calculating the weighted average to get a single error value for the tree
    /// 
    /// This average error value has an important role:
    /// - It determines how much influence the tree will have in the final ensemble
    /// - If it's too high (â‰¥ 0.5), the tree is considered too weak and training stops
    /// </para>
    /// </remarks>
    private T CalculateAverageError(Vector<T> errors, Vector<T> sampleWeights)
    {
        var maxError = errors.Max();

        // If all predictions are perfect, return 0 (no error)
        if (NumOps.Equals(maxError, NumOps.Zero))
        {
            return NumOps.Zero;
        }

        var weightedErrors = errors.Select((e, i) =>
            NumOps.Multiply(NumOps.Divide(e, maxError), sampleWeights[i]));

        return NumOps.Divide(weightedErrors.Aggregate(NumOps.Zero, NumOps.Add), sampleWeights.Sum());
    }

    /// <summary>
    /// Updates the sample weights for the next iteration of AdaBoost.R2.
    /// </summary>
    /// <param name="sampleWeights">The current sample weights.</param>
    /// <param name="errors">The absolute errors for each sample.</param>
    /// <param name="beta">The beta value calculated from the weighted average error.</param>
    /// <returns>The updated sample weights.</returns>
    /// <remarks>
    /// <para>
    /// This method updates the sample weights for the next iteration of the AdaBoost.R2 algorithm.
    /// The weights are updated according to the formula: w_i = w_i * beta^(1 - e_i/max(e)),
    /// where w_i is the weight of sample i, e_i is the error of sample i, and max(e) is the maximum error.
    /// The weights are then normalized to sum to 1. This update increases the weights of samples
    /// with high relative errors, causing the next weak learner to focus more on those samples.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts how important each training example will be 
    /// for the next decision tree.
    /// 
    /// The weight update process works like this:
    /// 1. Examples that were predicted accurately (small errors) get their weights reduced
    /// 2. Examples that were predicted poorly (large errors) get their weights increased
    /// 3. The beta parameter (which is based on the overall error) controls how aggressive this adjustment is
    /// 4. All weights are normalized to sum to 1, maintaining their relative importance
    /// 
    /// This weighting scheme is the key to AdaBoost's power:
    /// - It forces each new tree to focus on the examples that previous trees struggled with
    /// - It creates diversity in the ensemble, with each tree specializing in different aspects of the problem
    /// - It helps the overall model learn from its mistakes and continuously improve
    /// </para>
    /// </remarks>
    private Vector<T> UpdateSampleWeights(Vector<T> sampleWeights, Vector<T> errors, T beta)
    {
        var maxError = errors.Max();

        // If all errors are 0, keep weights unchanged
        if (NumOps.Equals(maxError, NumOps.Zero))
        {
            return sampleWeights;
        }

        var updatedWeights = sampleWeights.Select((w, i) =>
            NumOps.Multiply(w, NumOps.Power(beta, NumOps.Subtract(NumOps.One, NumOps.Divide(errors[i], maxError)))));
        var sumWeights = updatedWeights.Aggregate(NumOps.Zero, NumOps.Add);

        // Avoid division by 0 when normalizing
        if (NumOps.Equals(sumWeights, NumOps.Zero))
        {
            return sampleWeights;
        }

        return new Vector<T>(updatedWeights.Select(w => NumOps.Divide(w, sumWeights)));
    }

    /// <summary>
    /// Calculates the feature importances across all trees in the ensemble asynchronously.
    /// </summary>
    /// <param name="numFeatures">The number of features in the input data.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the importance of each feature in making predictions, based on
    /// the trained ensemble of decision trees. The feature importance for each tree is weighted
    /// by the tree's weight in the ensemble, and then the weighted importances are summed across
    /// all trees. This provides insight into which features are most influential in the model's
    /// predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how important each input feature is
    /// for making accurate predictions.
    /// 
    /// Feature importance tells you which input variables have the most influence on the model's predictions.
    /// For example, if you're predicting house prices:
    /// - High feature importance for "square footage" would indicate that size strongly affects price
    /// - Low feature importance for "house color" would suggest color doesn't matter much for price
    /// 
    /// In AdaBoost.R2, the feature importance calculation:
    /// 1. Gets the importance of each feature from each decision tree
    /// 2. Weights these importances by how much influence each tree has in the ensemble
    /// 3. Combines them to get an overall importance score for each feature
    /// 
    /// This information is valuable for:
    /// - Understanding which factors drive your predictions
    /// - Simplifying your model by potentially removing unimportant features
    /// - Gaining insights into the underlying patterns in your data
    /// </para>
    /// </remarks>
    protected override async Task CalculateFeatureImportancesAsync(int numFeatures)
    {
        var importances = new T[numFeatures];
        var totalWeight = _ensemble.Aggregate(NumOps.Zero, (acc, e) => NumOps.Add(acc, e.Weight));

        var tasks = _ensemble.Select(treeWeight => Task.Run(() =>
        {
            var (tree, weight) = treeWeight;
            var treeImportances = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                treeImportances[i] = NumOps.Multiply(tree.GetFeatureImportance(i), NumOps.Divide(weight, totalWeight));
            }
            return treeImportances;
        }));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(tasks);

        for (int i = 0; i < numFeatures; i++)
        {
            importances[i] = allImportances.Aggregate(NumOps.Zero, (acc, imp) => NumOps.Add(acc, imp[i]));
        }

        FeatureImportances = new Vector<T>(importances);
    }

    /// <summary>
    /// Gets metadata about the trained model.
    /// </summary>
    /// <returns>A <see cref="ModelMetaData{T}"/> object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the trained AdaBoost.R2 regression model, including
    /// the model type, configuration options, feature importances, and regularization type.
    /// This information can be useful for model management, comparison, and documentation.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about the trained model,
    /// which can be useful for documentation or comparison with other models.
    /// 
    /// The metadata includes:
    /// - The type of model (AdaBoost.R2)
    /// - Configuration settings like the number of trees and their maximum depth
    /// - Feature importance scores
    /// - The type of regularization used (if any)
    /// 
    /// This information helps you keep track of different models you've trained and understand
    /// their characteristics without having to retrain or examine the internal structure.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.AdaBoostR2,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfEstimators", _options.NumberOfEstimators },
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "MaxFeatures", _options.MaxFeatures },
                { "FeatureImportances", FeatureImportances },
                { "RegularizationType", Regularization.GetType().Name }
            }
        };
    }

    /// <summary>
    /// Serializes the model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the AdaBoost.R2 regression model to a byte array, including the
    /// configuration options, the ensemble of trees with their weights, and the regularization type.
    /// The serialization is performed using JSON, with the decision trees serialized to Base64 strings.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the trained model into a format that can be
    /// saved to a file or database.
    /// 
    /// Serializing a model allows you to:
    /// - Save it for later use without having to retrain
    /// - Share it with others
    /// - Deploy it to production environments
    /// 
    /// The serialized data includes everything needed to recreate the model:
    /// - All configuration settings
    /// - The entire ensemble of decision trees and their weights
    /// - Information about the regularization used
    /// 
    /// After serializing, you can store the resulting byte array in a file or database,
    /// and later restore the model using the Deserialize method.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        var serializableModel = new
        {
            Options = _options,
            Ensemble = _ensemble.Select(e => new
            {
                Tree = Convert.ToBase64String(e.Tree.Serialize()),
                Weight = e.Weight
            }).ToList(),
            Regularization = Regularization.GetType().Name
        };

        var json = JsonConvert.SerializeObject(serializableModel, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">A byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes an AdaBoost.R2 regression model from a byte array, restoring
    /// the configuration options, the ensemble of trees with their weights, and initializing
    /// the random number generator. The deserialization is performed using JSON, with the
    /// decision trees deserialized from Base64 strings.
    /// </para>
    /// <para><b>For Beginners:</b> This method restores a previously saved model from its
    /// serialized format.
    /// 
    /// Deserializing allows you to:
    /// - Load a previously trained model without having to retrain it
    /// - Use models trained by others
    /// - Deploy pre-trained models to new environments
    /// 
    /// The process reconstructs:
    /// - All configuration settings
    /// - The entire ensemble of decision trees and their weights
    /// - The appropriate random number generator state
    /// 
    /// After deserialization, the model is ready to use for making predictions,
    /// just as if you had just finished training it.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        var json = Encoding.UTF8.GetString(data);
        var deserializedModel = JsonConvert.DeserializeAnonymousType(json, new
        {
            Options = new AdaBoostR2RegressionOptions(),
            Ensemble = new List<dynamic>(),
            Regularization = ""
        });

        if (deserializedModel == null)
        {
            throw new InvalidOperationException("Failed to deserialize the model");
        }

        _options = deserializedModel.Options;

        _ensemble = [.. deserializedModel.Ensemble.Select(e =>
        {
            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                Seed = _options.Seed,
                SplitCriterion = _options.SplitCriterion
            };
            var tree = new DecisionTreeRegression<T>(treeOptions, Regularization);
            tree.Deserialize(Convert.FromBase64String((string)e.Tree));
            return (Tree: tree, Weight: (T)e.Weight);
        })];

        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Creates a new instance of the AdaBoostR2Regression with the same configuration as the current instance.
    /// </summary>
    /// <returns>A new AdaBoostR2Regression instance with the same options and regularization as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the AdaBoostR2Regression model with the same configuration options
    /// and regularization settings as the current instance. This is useful for model cloning, ensemble methods, or
    /// cross-validation scenarios where multiple instances of the same model with identical configurations are needed.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model's blueprint.
    /// 
    /// When you need multiple versions of the same type of model with identical settings:
    /// - This method creates a new, empty model with the same configuration
    /// - It's like making a copy of a recipe before you start cooking
    /// - The new model has the same settings but no trained data
    /// - This is useful for techniques that need multiple models, like cross-validation
    /// 
    /// For example, when testing your model on different subsets of data,
    /// you'd want each test to use a model with identical settings.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new AdaBoostR2Regression<T>(_options, Regularization);
    }

    #region IJitCompilable Implementation Override

    /// <summary>
    /// Gets whether this AdaBoost.R2 model supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when soft tree mode is enabled and the ensemble has been trained;
    /// <c>false</c> otherwise.
    /// </value>
    /// <remarks>
    /// <para>
    /// AdaBoost.R2 supports JIT compilation when soft tree mode is enabled. In soft mode,
    /// each tree in the ensemble uses sigmoid-based soft gating instead of hard if-then splits,
    /// making the weighted ensemble differentiable.
    /// </para>
    /// <para>
    /// The computation graph follows the weighted averaging formula:
    /// <code>prediction = Σ(weight_i × tree_i(input)) / Σ(weight_i)</code>
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation is available when soft tree mode is enabled.
    ///
    /// In soft tree mode:
    /// - Each tree in the AdaBoost ensemble uses smooth transitions
    /// - Tree weights (based on training error) are embedded in the computation graph
    /// - The weighted average is computed just like regular AdaBoost
    ///
    /// This gives you adaptive boosting benefits with JIT-compiled speed.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        UseSoftTree && _ensemble.Count > 0;

    /// <summary>
    /// Exports the AdaBoost.R2 model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The root node of the exported computation graph.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when soft tree mode is not enabled.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the ensemble has not been trained.
    /// </exception>
    /// <remarks>
    /// <para>
    /// When soft tree mode is enabled, this exports the entire AdaBoost.R2 ensemble as a
    /// differentiable computation graph. The graph implements weighted averaging:
    /// <code>output = Σ(weight_i × tree_i(input)) / Σ(weight_i)</code>
    /// where each tree uses soft split operations.
    /// </para>
    /// <para><b>For Beginners:</b> This exports the AdaBoost ensemble as a computation graph.
    ///
    /// AdaBoost uses weighted trees where:
    /// - Each tree has a weight based on how well it performed during training
    /// - Better-performing trees get higher weights
    /// - The final prediction is a weighted average of all tree predictions
    ///
    /// The exported graph includes these weights for optimized inference.
    /// </para>
    /// </remarks>
    public override AiDotNet.Autodiff.ComputationNode<T> ExportComputationGraph(
        List<AiDotNet.Autodiff.ComputationNode<T>> inputNodes)
    {
        if (!UseSoftTree)
        {
            throw new NotSupportedException(
                "AdaBoost.R2 does not support JIT compilation in hard tree mode because " +
                "decision trees use discrete branching logic.\n\n" +
                "To enable JIT compilation, set UseSoftTree = true to use soft (differentiable) " +
                "decision trees with sigmoid-based gating.");
        }

        if (_ensemble.Count == 0)
        {
            throw new InvalidOperationException(
                "Cannot export computation graph: the AdaBoost.R2 model has not been trained. " +
                "Call Train() or TrainAsync() first to build the ensemble.");
        }

        // Ensure all trees have soft mode enabled
        foreach (var (tree, _) in _ensemble)
        {
            tree.UseSoftTree = true;
            tree.SoftTreeTemperature = SoftTreeTemperature;
        }

        // Compute total weight for normalization
        T totalWeight = NumOps.Zero;
        foreach (var (_, weight) in _ensemble)
        {
            totalWeight = NumOps.Add(totalWeight, weight);
        }

        // Export first tree to get input node
        var tempInputNodes = new List<AiDotNet.Autodiff.ComputationNode<T>>();
        var (firstTree, firstWeight) = _ensemble[0];
        var firstTreeGraph = firstTree.ExportComputationGraph(tempInputNodes);

        if (tempInputNodes.Count > 0)
        {
            inputNodes.Add(tempInputNodes[0]);
        }

        // Create weighted first tree contribution
        var firstWeightTensor = new Tensor<T>(new[] { 1 });
        firstWeightTensor[0] = firstWeight;
        var firstWeightNode = TensorOperations<T>.Constant(firstWeightTensor, "weight_0");
        var weightedSum = TensorOperations<T>.ElementwiseMultiply(firstWeightNode, firstTreeGraph);

        // Add weighted contributions from remaining trees
        for (int i = 1; i < _ensemble.Count; i++)
        {
            var (tree, weight) = _ensemble[i];
            var treeInputNodes = new List<AiDotNet.Autodiff.ComputationNode<T>>();
            var treeGraph = tree.ExportComputationGraph(treeInputNodes);

            // Create weight constant
            var weightTensor = new Tensor<T>(new[] { 1 });
            weightTensor[0] = weight;
            var weightNode = TensorOperations<T>.Constant(weightTensor, $"weight_{i}");

            // weighted contribution: weight * tree_output
            var weightedTree = TensorOperations<T>.ElementwiseMultiply(weightNode, treeGraph);

            // Accumulate
            weightedSum = TensorOperations<T>.Add(weightedSum, weightedTree);
        }

        // Normalize by total weight: weighted_sum / total_weight
        var totalWeightTensor = new Tensor<T>(new[] { 1 });
        totalWeightTensor[0] = totalWeight;
        var totalWeightNode = TensorOperations<T>.Constant(totalWeightTensor, "total_weight");

        return TensorOperations<T>.Divide(weightedSum, totalWeightNode);
    }

    #endregion
}
