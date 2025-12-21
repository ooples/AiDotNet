namespace AiDotNet.Regression;

/// <summary>
/// Implements Quantile Regression Forests, an extension of Random Forests that can predict conditional quantiles
/// of the target variable, not just the conditional mean.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Quantile Regression Forests extend the Random Forests algorithm to estimate the full conditional distribution
/// of the response variable, not just its mean. This allows for prediction of any quantile of the response variable,
/// providing a more complete picture of the relationship between predictors and the response.
/// </para>
/// <para>
/// The algorithm works by building multiple decision trees on bootstrap samples of the training data, similar to
/// Random Forests. However, instead of averaging the predictions, it uses the empirical distribution of the predictions
/// from all trees to estimate quantiles.
/// </para>
/// <para>
/// <b>For Beginners:</b> While standard Random Forests tell you the average prediction, Quantile Regression Forests can tell you about
/// the entire range of possible outcomes. For example, they can predict not just the expected value, but also the
/// 10th percentile (a pessimistic scenario) or the 90th percentile (an optimistic scenario). This is particularly
/// useful when you need to understand the uncertainty in your predictions or when the relationship between variables
/// varies across different parts of the distribution.
/// </para>
/// </remarks>
public class QuantileRegressionForests<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Configuration options for the Quantile Regression Forests model.
    /// </summary>
    /// <value>
    /// Contains settings like number of trees, maximum depth, minimum samples to split, and maximum features.
    /// </value>
    private readonly QuantileRegressionForestsOptions _options;

    /// <summary>
    /// The collection of decision trees that make up the forest.
    /// </summary>
    /// <value>
    /// A list of decision tree regression models.
    /// </value>
    private List<DecisionTreeRegression<T>> _trees;

    /// <summary>
    /// Random number generator used for bootstrap sampling.
    /// </summary>
    /// <value>
    /// An instance of the Random class.
    /// </value>
    private Random _random;

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
    /// Initializes a new instance of the QuantileRegressionForests class with the specified options and regularization.
    /// </summary>
    /// <param name="options">Configuration options for the Quantile Regression Forests model.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <remarks>
    /// <para>
    /// The constructor initializes the model with the provided options and sets up the random number generator.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Quantile Regression Forests model with your specified settings.
    /// The options control things like how many trees to build, how deep each tree can be, and how many
    /// features to consider at each split. Regularization is an optional technique to prevent the model
    /// from becoming too complex and overfitting to the training data.
    /// </para>
    /// </remarks>
    public QuantileRegressionForests(QuantileRegressionForestsOptions options, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _trees = new List<DecisionTreeRegression<T>>();
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Asynchronously trains the Quantile Regression Forests model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix where each row is a training example and each column is a feature.</param>
    /// <param name="y">The target values vector corresponding to each training example.</param>
    /// <returns>A task that represents the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds multiple decision trees in parallel, each trained on a bootstrap sample of the training data.
    /// The steps are:
    /// 1. Clear any existing trees
    /// 2. For each tree:
    ///    a. Create a new decision tree with the specified options
    ///    b. Generate a bootstrap sample of the training data
    ///    c. Train the tree on the bootstrap sample
    /// 3. Calculate feature importances by averaging across all trees
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Training is the process where the model learns from your data. The algorithm builds multiple decision trees,
    /// each on a slightly different version of your data (created by random sampling with replacement). Each tree
    /// learns to predict the target variable based on the features. By building many trees and combining their
    /// predictions, the model can capture complex relationships and provide estimates of different quantiles
    /// (percentiles) of the target variable.
    /// </para>
    /// </remarks>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _trees.Clear();
        var treeTasks = Enumerable.Range(0, _options.NumberOfTrees).Select(_ => new Func<DecisionTreeRegression<T>>(() =>
        {
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                Seed = _random.Next()
            }, Regularization);

            var (sampledX, sampledY) = SampleWithReplacement(x, y);
            tree.Train(sampledX, sampledY);
            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks, _options.MaxDegreeOfParallelism);
        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Asynchronously predicts a specific quantile of the target variable for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <param name="quantile">The quantile to predict, a value between 0 and 1.</param>
    /// <returns>A task that represents the asynchronous prediction operation, containing a vector of predicted quantile values.</returns>
    /// <exception cref="ArgumentException">Thrown when the quantile is not between 0 and 1.</exception>
    /// <remarks>
    /// <para>
    /// This method predicts the specified quantile of the conditional distribution for each input example.
    /// The steps are:
    /// 1. Validate that the quantile is between 0 and 1
    /// 2. Apply regularization to the input matrix
    /// 3. Get predictions from all trees in parallel
    /// 4. For each input example:
    ///    a. Sort the predictions from all trees
    ///    b. Select the value at the position corresponding to the specified quantile
    /// 5. Apply regularization to the quantile predictions
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method predicts a specific percentile of the possible outcomes for each example in your input data.
    /// For instance, if you specify quantile=0.5, it predicts the median (middle value); if you specify quantile=0.9,
    /// it predicts the value below which 90% of the outcomes would fall. This is useful for understanding the range
    /// of possible outcomes and the uncertainty in your predictions.
    /// </para>
    /// </remarks>
    public async Task<Vector<T>> PredictQuantileAsync(Matrix<T> input, double quantile)
    {
        if (quantile < 0 || quantile > 1)
        {
            throw new ArgumentException("Quantile must be between 0 and 1", nameof(quantile));
        }

        var regularizedInput = Regularization.Regularize(input);
        var predictionTasks = _trees.Select(tree => new Func<Vector<T>>(() => tree.Predict(regularizedInput)));
        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(predictionTasks, _options.MaxDegreeOfParallelism);

        var result = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            var samplePredictions = predictions.Select(p => p[i]).OrderBy(v => v).ToList();
            int index = (int)Math.Floor(quantile * (samplePredictions.Count - 1));
            result[i] = samplePredictions[index];
        }

        var quantilePredictions = new Vector<T>(result);
        return Regularization.Regularize(quantilePredictions);
    }

    /// <summary>
    /// Asynchronously makes predictions for the given input data.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A task that represents the asynchronous prediction operation, containing a vector of predicted values.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts the median (0.5 quantile) of the conditional distribution for each input example.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> After training, this method is used to make predictions on new data. By default, it predicts the median
    /// value (the middle of the distribution), which is often a good central estimate. If you need a different
    /// percentile, you can use the PredictQuantileAsync method instead.
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // For standard prediction, we'll use the median (0.5 quantile)
        return await PredictQuantileAsync(input, 0.5);
    }

    /// <summary>
    /// Creates a bootstrap sample of the training data by sampling with replacement.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A tuple containing the sampled features matrix and target values vector.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a bootstrap sample by randomly selecting examples from the original data with replacement,
    /// meaning the same example can be selected multiple times.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Bootstrap sampling is a technique where we create a new dataset by randomly selecting examples from the
    /// original dataset, with the possibility of selecting the same example multiple times. This creates slightly
    /// different versions of the data for each tree, which helps the forest capture different aspects of the
    /// relationships in the data and reduces overfitting.
    /// </para>
    /// </remarks>
    private (Matrix<T> sampledX, Vector<T> sampledY) SampleWithReplacement(Matrix<T> x, Vector<T> y)
    {
        var sampledIndices = new List<int>();
        for (int i = 0; i < x.Rows; i++)
        {
            sampledIndices.Add(_random.Next(0, x.Rows));
        }

        var sampledX = new Matrix<T>(sampledIndices.Select(i => x.GetRow(i)).ToList());
        var sampledY = new Vector<T>([.. sampledIndices.Select(i => y[i])]);

        return (sampledX, sampledY);
    }

    /// <summary>
    /// Asynchronously calculates the importance of each feature in the model.
    /// </summary>
    /// <param name="numFeatures">The number of features in the input data.</param>
    /// <returns>A task that represents the asynchronous calculation operation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates feature importances by averaging the importances across all trees in the forest.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Feature importance tells you which input variables have the most influence on the predictions.
    /// In Quantile Regression Forests, this is calculated by averaging the feature importances from all
    /// the individual trees. Higher values indicate more important features.
    /// </para>
    /// </remarks>
    protected override async Task CalculateFeatureImportancesAsync(int numFeatures)
    {
        FeatureImportances = new Vector<T>(new T[numFeatures]);
        var importanceTasks = _trees.Select(tree => new Func<Vector<T>>(() => tree.FeatureImportances));
        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks, _options.MaxDegreeOfParallelism);

        for (int i = 0; i < numFeatures; i++)
        {
            FeatureImportances[i] = NumOps.Divide(
                allImportances.Aggregate(NumOps.Zero, (acc, imp) => NumOps.Add(acc, imp[i])),
                NumOps.FromDouble(_trees.Count)
            );
        }
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, number of trees, maximum depth,
    /// and feature importances.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Model metadata provides information about the model itself, rather than the predictions it makes.
    /// This includes details about how the model is configured (like how many trees it uses and how deep they are)
    /// and information about the importance of different features. This can help you understand which input
    /// variables are most influential in making predictions.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.QuantileRegressionForests,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _trees.Count },
                { "MaxDepth", _options.MaxDepth }
            }
        };

        // Add feature importances to AdditionalInfo if available
        if (FeatureImportances != null && FeatureImportances.Length > 0)
        {
            metadata.AdditionalInfo["FeatureImportances"] = FeatureImportances.ToList();
        }

        return metadata;
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model data.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model's parameters, including options, feature importances, and all trees in the forest.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Serialization converts the model's internal state into a format that can be saved to disk or
    /// transmitted over a network. This allows you to save a trained model and load it later without
    /// having to retrain it. Think of it like saving your progress in a video game.
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        writer.Write(_options.NumberOfTrees);
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(_options.MaxFeatures);
        writer.Write(_options.Seed ?? -1);
        writer.Write((int)_options.SplitCriterion);
        writer.Write(_options.MaxDegreeOfParallelism);

        // Serialize feature importances
        writer.Write(FeatureImportances.Length);
        foreach (var importance in FeatureImportances)
        {
            writer.Write(Convert.ToDouble(importance));
        }

        // Serialize trees
        writer.Write(_trees.Count);
        foreach (var tree in _trees)
        {
            var treeData = tree.Serialize();
            writer.Write(treeData.Length);
            writer.Write(treeData);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the model's parameters from a serialized byte array, including options,
    /// feature importances, and all trees in the forest.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Deserialization is the opposite of serialization - it takes the saved model data and reconstructs
    /// the model's internal state. This allows you to load a previously trained model and use it to make
    /// predictions without having to retrain it. It's like loading a saved game to continue where you left off.
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        _options.NumberOfTrees = reader.ReadInt32();
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        _options.MaxFeatures = reader.ReadDouble();
        int seed = reader.ReadInt32();
        _options.Seed = seed == -1 ? null : seed;
        _options.SplitCriterion = (SplitCriterion)reader.ReadInt32();
        _options.MaxDegreeOfParallelism = reader.ReadInt32();

        // Deserialize feature importances
        int featureCount = reader.ReadInt32();
        var importances = new T[featureCount];
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        FeatureImportances = new Vector<T>(importances);

        // Deserialize trees
        int treeCount = reader.ReadInt32();
        _trees = new List<DecisionTreeRegression<T>>(treeCount);
        for (int i = 0; i < treeCount; i++)
        {
            int treeDataLength = reader.ReadInt32();
            byte[] treeData = reader.ReadBytes(treeDataLength);
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions(), Regularization);
            tree.Deserialize(treeData);
            _trees.Add(tree);
        }

        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Creates a new instance of the Quantile Regression Forests model with the same configuration.
    /// </summary>
    /// <returns>A new instance of the Quantile Regression Forests model.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the creation fails or required components are null.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the current model, including its configuration options, 
    /// trained trees, feature importances, and regularization settings. The new instance is completely 
    /// independent of the original, allowing modifications without affecting the original model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method creates an exact copy of your trained model.
    /// 
    /// Think of it like making a perfect clone of your forest model:
    /// - It copies all the configuration settings (number of trees, max depth, etc.)
    /// - It duplicates all the individual decision trees that make up the forest
    /// - It preserves the feature importance values that show which inputs matter most
    /// - It maintains all regularization settings that help prevent overfitting
    /// 
    /// Creating a copy is useful when you want to:
    /// - Create a backup before further modifying the model
    /// - Create variations of the same model for different purposes
    /// - Share the model with others while keeping your original intact
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        var newModel = new QuantileRegressionForests<T>(_options, Regularization);

        // Copy feature importances if they exist
        if (FeatureImportances != null)
        {
            newModel.FeatureImportances = new Vector<T>([.. FeatureImportances]);
        }

        // Deep copy all the trees
        newModel._trees = new List<DecisionTreeRegression<T>>(_trees.Count);
        foreach (var tree in _trees)
        {
            // Create a deep copy of each tree by serializing and deserializing
            var treeData = tree.Serialize();
            var treeCopy = new DecisionTreeRegression<T>(new DecisionTreeOptions(), Regularization);
            treeCopy.Deserialize(treeData);
            newModel._trees.Add(treeCopy);
        }

        // Initialize the random number generator with the same seed if available
        if (_options.Seed.HasValue)
        {
            newModel._random = RandomHelper.CreateSeededRandom(_options.Seed.Value);
        }

        return newModel;
    }
}
