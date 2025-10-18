namespace AiDotNet.Regression;

/// <summary>
/// Implements Random Forest Regression, an ensemble learning method that operates by constructing multiple
/// decision trees during training and outputting the average prediction of the individual trees.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Random Forest Regression combines multiple decision trees to improve prediction accuracy and control overfitting.
/// Each tree is trained on a bootstrap sample of the training data, and at each node, only a random subset of
/// features is considered for splitting. The final prediction is the average of predictions from all trees.
/// </para>
/// <para>
/// The algorithm's key strengths include robustness to outliers, good performance on high-dimensional data,
/// and the ability to capture non-linear relationships without requiring extensive hyperparameter tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of Random Forest as a committee of decision trees, where each tree votes on the prediction.
/// By combining many trees, each trained slightly differently, the model becomes more robust and accurate
/// than any single tree. It's like asking multiple experts for their opinion and taking the average.
/// </para>
/// </remarks>
public class RandomForestRegression<T> : AsyncDecisionTreeRegressionModelBase<T>
{
    /// <summary>
    /// Configuration options for the Random Forest regression model.
    /// </summary>
    /// <value>
    /// Contains settings like number of trees, maximum depth, minimum samples to split, and maximum features.
    /// </value>
    private RandomForestRegressionOptions _options = default!;

    /// <summary>
    /// The collection of decision trees that make up the forest.
    /// </summary>
    /// <value>
    /// A list of decision tree regression models.
    /// </value>
    private List<DecisionTreeRegression<T>> _trees = default!;

    /// <summary>
    /// Random number generator used for bootstrap sampling and feature selection.
    /// </summary>
    /// <value>
    /// An instance of the Random class.
    /// </value>
    private Random _random = default!;

    /// <summary>
    /// Gets the number of trees in the forest.
    /// </summary>
    /// <value>
    /// The number of trees specified in the options.
    /// </value>
    public override int NumberOfTrees => _options.NumberOfTrees;

    /// <summary>
    /// Gets the maximum depth of the trees in the forest.
    /// </summary>
    /// <value>
    /// The maximum depth specified in the options.
    /// </value>
    public override int MaxDepth => _options.MaxDepth;

    /// <summary>
    /// Initializes a new instance of the RandomForestRegression class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the Random Forest regression model.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with the provided options and sets up the random number generator.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the Random Forest model with your specified settings. The options control things
    /// like how many trees to build, how deep each tree can be, and how many features to consider at each split.
    /// Regularization is an optional technique to prevent the model from becoming too complex and overfitting
    /// to the training data.
    /// </para>
    /// </remarks>
    public RandomForestRegression(RandomForestRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new(), regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>())
    {
        _options = options ?? new();
        _trees = [];
        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
    }

    /// <summary>
    /// Asynchronously trains the Random Forest regression model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <returns>A task that represents the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds multiple decision trees in parallel, each trained on a bootstrap sample of the training data
    /// and considering a random subset of features at each split. The steps are:
    /// 1. Clear any existing trees
    /// 2. Calculate the number of features to consider at each split
    /// 3. For each tree:
    ///    a. Generate a bootstrap sample of the training data
    ///    b. Create a new decision tree with the specified options
    ///    c. Train the tree on the bootstrap sample
    /// 4. Calculate feature importances by averaging across all trees
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Training is the process where the model learns from your data. The algorithm builds multiple decision trees,
    /// each on a slightly different version of your data (created by random sampling with replacement). Each tree
    /// also considers only a random subset of features at each split, which helps to make the trees more diverse.
    /// By building many diverse trees and combining their predictions, the model can capture complex relationships
    /// and provide more robust predictions than a single tree.
    /// </para>
    /// </remarks>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _trees.Clear();
        var numFeatures = x.Columns;
        var numSamples = x.Rows;
        var featuresToConsider = (int)Math.Max(1, Math.Round(_options.MaxFeatures * numFeatures));

        var treeTasks = Enumerable.Range(0, _options.NumberOfTrees).Select(_ => Task.Run(() =>
        {
            var bootstrapIndices = GetBootstrapSampleIndices(numSamples);
            var bootstrapX = x.GetRows(bootstrapIndices);
            var bootstrapY = y.GetElements(bootstrapIndices);

            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = featuresToConsider / (double)numFeatures,
                Seed = _random.Next(),
                SplitCriterion = _options.SplitCriterion
            };
            var tree = new DecisionTreeRegression<T>(treeOptions, Regularization);
            tree.Train(bootstrapX, bootstrapY);
            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks).ConfigureAwait(false);

        await CalculateFeatureImportancesAsync(x.Columns).ConfigureAwait(false);
    }

    /// <summary>
    /// Asynchronously makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A task that represents the asynchronous prediction operation, containing a vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method makes predictions by averaging the predictions from all trees in the forest.
    /// The steps are:
    /// 1. Apply regularization to the input matrix
    /// 2. Get predictions from all trees in parallel
    /// 3. Average the predictions for each input example
    /// 4. Apply regularization to the averaged predictions
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// After training, this method is used to make predictions on new data. It gets a prediction from each tree
    /// in the forest and then averages these predictions to produce the final result. This averaging helps to
    /// reduce the variance (randomness) in the predictions, making the model more stable and accurate than
    /// any single decision tree.
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var regularizedInput = Regularization.Regularize(input);
        var predictionTasks = _trees.Select(tree => Task.Run(() => tree.Predict(regularizedInput)));
        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(predictionTasks).ConfigureAwait(false);

        var result = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = NumOps.Divide(
                predictions.Aggregate(NumOps.Zero, (acc, p) => NumOps.Add(acc, p[i])),
                NumOps.FromDouble(_trees.Count)
            );
        }

        var regularizedPredictions = new Vector<T>(result);
        return Regularization.Regularize(regularizedPredictions);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, number of trees, maximum depth,
    /// minimum samples to split, maximum features, feature importances, and regularization type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Model metadata provides information about the model itself, rather than the predictions it makes.
    /// This includes details about how the model is configured (like how many trees it uses and how deep they are)
    /// and information about the importance of different features. This can help you understand which input
    /// variables are most influential in making predictions.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.RandomForest,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _options.NumberOfTrees },
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "MaxFeatures", _options.MaxFeatures },
                { "FeatureImportances", FeatureImportances },
                { "RegularizationType", Regularization.GetType().Name }
            }
        };
    }

    /// <summary>
    /// Generates indices for a bootstrap sample of the training data.
    /// </summary>
    /// <param name="numSamples">The number of samples in the original dataset.</param>
    /// <returns>An array of indices representing the bootstrap sample.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a bootstrap sample by randomly selecting indices with replacement,
    /// meaning the same index can be selected multiple times.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Bootstrap sampling is a technique where we create a new dataset by randomly selecting examples from the
    /// original dataset, with the possibility of selecting the same example multiple times. This creates slightly
    /// different versions of the data for each tree, which helps the forest capture different aspects of the
    /// relationships in the data and reduces overfitting.
    /// </para>
    /// </remarks>
    private int[] GetBootstrapSampleIndices(int numSamples)
    {
        var indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            indices[i] = _random.Next(numSamples);
        }

        return indices;
    }

    /// <summary>
    /// Asynchronously calculates the importance of each feature in the model.
    /// </summary>
    /// <param name="numFeatures">The number of features in the input data.</param>
    /// <returns>A task that represents the asynchronous calculation operation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates feature importances by averaging the importances across all trees in the forest
    /// and then normalizing them so they sum to 1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Feature importance tells you which input variables have the most influence on the predictions.
    /// In Random Forests, this is calculated by measuring how much each feature reduces the prediction error
    /// when used in the trees. Higher values indicate more important features. The importances are normalized
    /// to sum to 1, so you can interpret them as percentages of total importance.
    /// </para>
    /// </remarks>
    protected override async Task CalculateFeatureImportancesAsync(int numFeatures)
    {
        var importances = new T[numFeatures];

        // Calculate importances in parallel for each tree
        var importanceTasks = _trees.Select(tree => Task.Run(() =>
        {
            var treeImportances = new T[numFeatures];
            for (int i = 0; i < numFeatures; i++)
            {
                treeImportances[i] = tree.GetFeatureImportance(i);
            }
            return treeImportances;
        }));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks).ConfigureAwait(false);

        // Aggregate importances
        for (int i = 0; i < numFeatures; i++)
        {
            importances[i] = allImportances.Aggregate(NumOps.Zero, (acc, treeImportance) => NumOps.Add(acc, treeImportance[i]));
        }

        // Normalize importances
        T sum = importances.Aggregate(NumOps.Zero, NumOps.Add);
        for (int i = 0; i < numFeatures; i++)
        {
            importances[i] = NumOps.Divide(importances[i], sum);
        }

        FeatureImportances = new Vector<T>(importances);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including options, trees, and regularization type,
    /// to a JSON format and then converts it to a byte array.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        var serializableModel = new
        {
            Options = _options,
            Trees = _trees.Select(tree => Convert.ToBase64String(tree.Serialize())).ToList(),
            Regularization = Regularization.GetType().Name
        };

        var json = JsonConvert.SerializeObject(serializableModel, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model data.</param>
    /// <exception cref="InvalidOperationException">Thrown when deserialization fails.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including options,
    /// trees, and regularization type.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] data)
    {
        var json = Encoding.UTF8.GetString(data);
        var deserializedModel = JsonConvert.DeserializeAnonymousType(json, new
        {
            Options = new RandomForestRegressionOptions(),
            Trees = new List<string>(),
            Regularization = ""
        });

        if (deserializedModel == null)
        {
            throw new InvalidOperationException("Failed to deserialize the model");
        }

        _options = deserializedModel.Options;

        _trees = [.. deserializedModel.Trees.Select(treeData =>
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
            tree.Deserialize(Convert.FromBase64String(treeData));
            return tree;
        })];

        // Reinitialize other fields
        _random = _options.Seed.HasValue ? new Random(_options.Seed.Value) : new Random();
    }

    /// <summary>
    /// Creates a new instance of the Random Forest regression model with the same options.
    /// </summary>
    /// <returns>A new instance of the model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the Random Forest regression model with the same configuration
    /// options and regularization method as the current instance, but without copying the trained trees
    /// or other learned parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters.
    /// 
    /// Think of it like getting a blank forest template with the same settings, 
    /// but without any of the trained trees. The new model has the same:
    /// - Number of trees setting
    /// - Maximum depth setting
    /// - Minimum samples split setting
    /// - Maximum features ratio
    /// - Split criterion (how nodes decide which feature to split on)
    /// - Regularization method
    /// 
    /// But it doesn't have any of the actual trained trees that were learned from data.
    /// 
    /// This is mainly used internally when doing things like cross-validation or 
    /// creating ensembles of similar models with different training data.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        // Create a new instance with the same options and regularization
        return new RandomForestRegression<T>(_options, Regularization);
    }
}