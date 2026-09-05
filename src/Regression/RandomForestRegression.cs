using AiDotNet.Attributes;
using AiDotNet.Enums;
using Newtonsoft.Json;

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
/// <example>
/// <code>
/// // Create a random forest regression with ensemble of decision trees
/// var options = new RandomForestRegressionOptions();
/// var model = new RandomForestRegression&lt;double&gt;(options);
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train the ensemble model
/// model.Train(features, targets);
///
/// // Predict for a new sample (averages predictions from all trees)
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = model.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Random Forests", "https://doi.org/10.1023/A:1010933404324", Year = 2001, Authors = "Leo Breiman")]
public partial class RandomForestRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public RandomForestRegression()
        : this(new RandomForestRegressionOptions())
    {
    }

    /// <summary>
    /// Configuration options for the Random Forest regression model.
    /// </summary>
    /// <value>
    /// Contains settings like number of trees, maximum depth, minimum samples to split, and maximum features.
    /// </value>
    private RandomForestRegressionOptions _options;

    /// <summary>
    /// The collection of decision trees that make up the forest.
    /// </summary>
    /// <value>
    /// A list of decision tree regression models.
    /// </value>
    private List<DecisionTreeRegression<T>> _trees;

    /// <summary>
    /// Random number generator used for bootstrap sampling and feature selection.
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
    public RandomForestRegression(RandomForestRegressionOptions options, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _trees = new List<DecisionTreeRegression<T>>();
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
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

        // Pre-generate seeds for reproducibility (calling _random.Next() in parallel is non-deterministic)
        var seeds = Enumerable.Range(0, _options.NumberOfTrees).Select(_ => _random.Next()).ToArray();
        var bootstrapSamples = Enumerable.Range(0, _options.NumberOfTrees)
            .Select(_ => GetBootstrapSampleIndices(numSamples)).ToArray();

        var treeTasks = Enumerable.Range(0, _options.NumberOfTrees).Select(i => Task.Run(() =>
        {
            var bootstrapX = x.GetRows(bootstrapSamples[i]);
            var bootstrapY = y.GetElements(bootstrapSamples[i]);

            var treeOptions = new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = featuresToConsider / (double)numFeatures,
                Seed = seeds[i],
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
        // Note: Tree-based methods handle regularization through tree structure parameters
        // (MaxDepth, MinSamplesSplit, etc.), not through data transformation
        var predictionTasks = _trees.Select(tree => Task.Run(() => tree.Predict(input)));
        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(predictionTasks).ConfigureAwait(false);

        var result = new T[input.Rows];
        for (int i = 0; i < input.Rows; i++)
        {
            result[i] = NumOps.Divide(
                predictions.Aggregate(NumOps.Zero, (acc, p) => NumOps.Add(acc, p[i])),
                NumOps.FromDouble(_trees.Count)
            );
        }

        return new Vector<T>(result);
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

}
