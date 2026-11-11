using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Models.Parameters;

namespace AiDotNet.Regression;

/// <summary>
/// Histogram-based Gradient Boosting Regression for fast training on large datasets.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Histogram-based Gradient Boosting discretizes continuous features into a fixed number of bins,
/// then builds histograms of gradients and hessians for each bin. This approach dramatically
/// reduces the time complexity of finding the best split from O(n*features) to O(bins*features),
/// making it suitable for large datasets with millions of samples.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional gradient boosting looks at every possible split point
/// for every feature, which is slow for large datasets. Histogram-based methods group similar
/// values into "bins" first, then only consider splits between bins.
///
/// Think of it like sorting students by height:
/// - Traditional method: Consider every student's exact height as a potential grouping point
/// - Histogram method: First group students into height ranges (5'0"-5'2", 5'2"-5'4", etc.),
///   then only consider splitting between groups
///
/// This is much faster because there are far fewer groups than individual heights.
///
/// Key advantages:
/// - 10-100x faster than traditional gradient boosting on large datasets
/// - Memory efficient (stores bin indices, not raw values)
/// - Handles missing values naturally
/// - Similar accuracy to traditional methods
///
/// This is the same approach used by LightGBM, XGBoost (hist mode), and scikit-learn's
/// HistGradientBoostingRegressor.
///
/// Usage:
/// <code>
/// var options = new HistGradientBoostingOptions { NumberOfIterations = 100, LearningRate = 0.1 };
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new HistGradientBoostingRegression&lt;double&gt;(options))
///     .Build(features, targets);
/// var predictions = result.Predict(X_test);
/// </code>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a histogram-based gradient boosting regression for fast large-scale training
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train through the facade: the builder runs validation, preprocessing and the training pipeline,
/// // then hands back a result you predict on.
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new HistGradientBoostingRegression&lt;double&gt;(new HistGradientBoostingOptions()))
///     .Build(features, targets);
///
/// // Predict for a new sample
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree", Year = 2017, Authors = "Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu")]
public partial class HistGradientBoostingRegression<T> : ModelBase<T, Matrix<T>, Vector<T>>, IConfigurableModel<T>
{
    #region Fields

    /// <summary>
    /// Configuration options for the histogram gradient boosting algorithm.
    /// </summary>
    private readonly HistGradientBoostingOptions _options;

    /// <inheritdoc/>
    public ModelOptions GetOptions() => _options;

    /// <summary>
    /// Bin thresholds for each feature (jagged array).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the "boundaries" between bins for each feature.
    /// For example, if a feature is temperature with thresholds [30, 50, 70, 90],
    /// then values 0-30 go in bin 0, 30-50 in bin 1, etc.
    /// </para>
    /// </remarks>
    private Vector<T>[]? _binThresholds;

    /// <summary>
    /// Binned feature values for training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of storing raw feature values, we store which
    /// bin each value falls into. This is more memory efficient and faster to process.
    /// </para>
    /// </remarks>
    private byte[,]? _binnedData;

    /// <summary>
    /// The collection of histogram-based trees.
    /// </summary>
    private List<HistTreeNode>? _trees;

    /// <summary>
    /// The initial prediction (mean of target values).
    /// </summary>
    private T _initialPrediction;

    /// <summary>
    /// Feature importance scores accumulated during training.
    /// </summary>
    private Vector<T>? _featureImportances;

    /// <summary>
    /// Random number generator for subsampling.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Number of features in the training data.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Active feature indices that are actually used by the model.
    /// </summary>
    private HashSet<int>? _activeFeatureIndices;

    /// <summary>
    /// The default loss function for gradient computation.
    /// </summary>
    private readonly ILossFunction<T> _defaultLossFunction;

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of the HistGradientBoostingRegression class.
    /// </summary>
    /// <param name="options">Configuration options for the algorithm.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a new histogram-based gradient boosting model.
    /// You can customize the behavior by providing options, or use defaults.
    ///
    /// Example with defaults:
    /// <code>
    /// var model = new HistGradientBoostingRegression&lt;double&gt;();
    /// </code>
    ///
    /// Example with custom options:
    /// <code>
    /// var options = new HistGradientBoostingOptions
    /// {
    ///     NumberOfIterations = 200,
    ///     LearningRate = 0.05,
    ///     MaxDepth = 4
    /// };
    /// var model = new HistGradientBoostingRegression&lt;double&gt;(options);
    /// </code>
    /// </para>
    /// </remarks>
    public HistGradientBoostingRegression(HistGradientBoostingOptions? options = null)
    {
        _options = options ?? new HistGradientBoostingOptions();
        _initialPrediction = NumOps.Zero;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _defaultLossFunction = new MeanSquaredErrorLoss<T>();
    }

    // ParameterCount override is below (returns 0 to prevent optimizer parameter injection)

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Trains the model on the provided data.
    /// </summary>
    /// <param name="x">Feature matrix where each row is a sample.</param>
    /// <param name="y">Target values.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the model learns from your data.
    /// The algorithm:
    /// 1. Bins the feature values into discrete groups
    /// 2. Computes the initial prediction (mean of targets)
    /// 3. For each iteration:
    ///    a. Compute residuals (how wrong current predictions are)
    ///    b. Build a tree to predict the residuals
    ///    c. Add the tree's predictions to the ensemble
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;

        // Step 1: Bin the features
        BinFeatures(x);

        var engine = AiDotNetEngine.Current;

        // Step 2: Compute initial prediction (mean of y)
        _initialPrediction = NumOps.Divide(engine.Sum(y), NumOps.FromDouble(y.Length));

        // Step 3: Initialize predictions and residuals
        var predictions = engine.Fill(y.Length, _initialPrediction);
        var residuals = engine.Subtract(y, predictions);

        // Step 4: Initialize trees and feature importances
        _trees = new List<HistTreeNode>(_options.NumberOfIterations);
        _featureImportances = engine.FillZero(_numFeatures);

        // Step 5: Build trees iteratively
        for (int iteration = 0; iteration < _options.NumberOfIterations; iteration++)
        {
            // Subsample indices
            int[] sampleIndices = GetSubsampleIndices(y.Length);

            // Build tree on residuals
            var tree = BuildTree(residuals, sampleIndices);
            _trees.Add(tree);

            // Update predictions
            T lr = NumOps.FromDouble(_options.LearningRate);
            var binnedData = _binnedData ?? throw new InvalidOperationException("Binned data not computed.");
            var treePredictions = PredictTree(tree, binnedData, y.Length);
            predictions = engine.Add(predictions, engine.Multiply(treePredictions, lr));
            residuals = engine.Subtract(y, predictions);
        }

        // Normalize feature importances
        NormalizeFeatureImportances();
    }

    /// <summary>
    /// Makes predictions for new data.
    /// </summary>
    /// <param name="input">Feature matrix for prediction.</param>
    /// <returns>Predicted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After training, use this to make predictions on new data.
    /// The prediction is: initial_prediction + learning_rate * sum(tree_predictions)
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_trees is null || _binThresholds is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var engine = AiDotNetEngine.Current;
        T lr = NumOps.FromDouble(_options.LearningRate);

        // Bin the input features
        var binnedInput = BinMatrix(input);

        // Add the contribution of each tree, in tree order
        var predictions = engine.Fill(input.Rows, _initialPrediction);
        foreach (var tree in _trees)
        {
            predictions = engine.Add(predictions, engine.Multiply(PredictTree(tree, binnedInput, input.Rows), lr));
        }

        return predictions;
    }

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _trees?.Count ?? 0 },
                { "NumberOfIterations", _options.NumberOfIterations },
                { "LearningRate", _options.LearningRate },
                { "MaxBins", _options.MaxBins },
                { "MaxDepth", _options.MaxDepth },
                { "MaxLeafNodes", _options.MaxLeafNodes ?? -1 },
                { "MinSamplesLeaf", _options.MinSamplesLeaf },
                { "L2Regularization", _options.L2Regularization }
            }
        };
    }

    /// <summary>
    /// Gets the feature importance scores.
    /// </summary>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        if (_featureImportances is null)
        {
            return result;
        }

        int i = 0;
        foreach (var importance in _featureImportances)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = importance;
            i++;
        }

        return result;
    }

    /// <summary>
    /// The model's one continuous parameter: the initial prediction the boosted trees correct.
    /// </summary>
    /// <remarks>
    /// The surface this replaces returned TWO values -- the initial prediction and the tree COUNT.
    /// A count is structure, not a parameter: SetParameters read only the first slot and ignored
    /// the second, so the vector advertised a value it could never restore, and any caller pairing
    /// the two by length was silently working with one dead slot. The trees themselves are not a
    /// flat vector at all; they round-trip through serialization.
    /// </remarks>
    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(new ScalarParameterSource<T>(
            () => _initialPrediction,
            value => _initialPrediction = value));
    }

    // Replaced by the declared parameter source below. Removed under AIDN082.

    // Replaced by the declared parameter source below. Removed under AIDN082.

    /// <summary>
    /// Creates a new instance with the given parameters.
    /// </summary>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new HistGradientBoostingRegression<T>(_options);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Gets the default loss function used for gradient computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Histogram Gradient Boosting uses Mean Squared Error (MSE)
    /// as its default loss function for regression tasks. MSE measures the average
    /// squared difference between predictions and actual values.
    /// </para>
    /// </remarks>
    public override ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <summary>
    /// Gets the number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For histogram-based gradient boosting, the "parameters"
    /// include the initial prediction and all leaf values across all trees.
    /// This is a simplification since the actual model is tree-structured.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Returns 0 to prevent optimizer random parameter injection.
    /// Histogram gradient boosting builds trees internally.
    /// </summary>
        /// <remarks>
    /// Expressed as a capability, not as a count. A zero ParameterCount also suppresses
    /// injection -- that is why this was written that way -- but it overloads a COUNT to carry
    /// a CAPABILITY: the model does have parameters (the base getter returns its coefficients
    /// and intercept), so the count contradicted the vector and anything pairing the two by
    /// length saw parameters the model claimed not to have.
    /// </remarks>
    public override bool SupportsParameterInitialization => false;

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This saves your trained model to a file so you can
    /// load it later without retraining.
    /// </para>
    /// </remarks>
    public override void SaveModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            byte[] data = Serialize();
            File.WriteAllBytes(filePath, data);
        }
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    /// <param name="filePath">The path to the saved model file.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This loads a previously saved model so you can use
    /// it for predictions without retraining.
    /// </para>
    /// </remarks>
    public override void LoadModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            byte[] data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
    }

    /// <summary>
    /// Saves the model state to a stream.
    /// </summary>
    /// <param name="stream">The stream to write to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is useful for checkpointing during training
    /// or for storing models in databases/memory.
    /// </para>
    /// </remarks>
    public override void SaveState(Stream stream)
    {
        byte[] data = Serialize();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(data.Length);
        writer.Write(data);
        writer.Flush();
    }

    /// <summary>
    /// Loads the model state from a stream.
    /// </summary>
    /// <param name="stream">The stream to read from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This restores a model from a checkpoint or database.
    /// </para>
    /// </remarks>
    public override void LoadState(Stream stream)
    {
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        int length = reader.ReadInt32();
        byte[] data = reader.ReadBytes(length);
        Deserialize(data);
    }

    /// <summary>
    /// Gets the indices of features that are actively used by the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns which features are actually used in the trees.
    /// Features not used by any split may be irrelevant for predictions.
    /// </para>
    /// </remarks>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        if (_activeFeatureIndices is null)
        {
            _activeFeatureIndices = new HashSet<int>();
            if (_trees is not null)
            {
                foreach (var tree in _trees)
                {
                    CollectActiveFeatures(tree, _activeFeatureIndices);
                }
            }
        }
        return _activeFeatureIndices;
    }

    /// <summary>
    /// Sets the active feature indices for the model.
    /// </summary>
    /// <param name="featureIndices">The feature indices to mark as active.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This allows you to manually specify which features
    /// the model should consider. Usually computed automatically during training.
    /// </para>
    /// </remarks>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _activeFeatureIndices = new HashSet<int>(featureIndices);
    }

    /// <summary>
    /// Checks if a specific feature is used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used in any tree split.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you if a specific feature contributes to
    /// predictions. Unused features can be removed from future training data.
    /// </para>
    /// </remarks>
    public override bool IsFeatureUsed(int featureIndex)
    {
        var activeFeatures = GetActiveFeatureIndices();
        return activeFeatures.Contains(featureIndex);
    }

    /// <summary>
    /// Computes gradients without updating parameters.
    /// </summary>
    /// <param name="input">The input features.</param>
    /// <param name="target">The target values.</param>
    /// <param name="lossFunction">Optional loss function (uses default if null).</param>
    /// <returns>Gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient boosting computes gradients (residuals) as
    /// the direction to improve predictions. For MSE loss, the gradient is simply
    /// the difference between predictions and targets.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? _defaultLossFunction;
        var predictions = Predict(input);

        // Compute gradients (negative of loss derivative with respect to predictions)
        // For MSE: derivative = 2 * (prediction - target)
        // We return the negative gradient (direction of improvement)
        var engine = AiDotNetEngine.Current;
        return engine.Multiply(engine.Subtract(target, predictions), NumOps.FromDouble(2.0));
    }

    /// <summary>
    /// Applies gradients to update the model.
    /// </summary>
    /// <param name="gradients">The gradient vector to apply.</param>
    /// <param name="learningRate">The learning rate for the update.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For tree-based models, applying gradients means
    /// adjusting the initial prediction and leaf values. This is a simplified
    /// update that shifts predictions in the direction of the gradients.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // For gradient boosting, we adjust the initial prediction based on average gradient
        if (gradients.Length > 0)
        {
            T sum = AiDotNetEngine.Current.Sum(gradients);
            T avgGradient = NumOps.Divide(sum, NumOps.FromDouble(gradients.Length));
            T update = NumOps.Multiply(learningRate, avgGradient);
            _initialPrediction = NumOps.Add(_initialPrediction, update);
        }
    }

    /// <summary>
    /// Counts the number of leaves in a tree.
    /// </summary>
    private int CountLeaves(HistTreeNode node)
    {
        if (node.IsLeaf)
        {
            return 1;
        }
        int count = 0;
        if (node.Left is not null)
        {
            count += CountLeaves(node.Left);
        }
        if (node.Right is not null)
        {
            count += CountLeaves(node.Right);
        }
        return count;
    }

    /// <summary>
    /// Recursively collects active feature indices from a tree.
    /// </summary>
    private void CollectActiveFeatures(HistTreeNode node, HashSet<int> features)
    {
        if (node.IsLeaf)
        {
            return;
        }
        features.Add(node.FeatureIndex);
        if (node.Left is not null)
        {
            CollectActiveFeatures(node.Left, features);
        }
        if (node.Right is not null)
        {
            CollectActiveFeatures(node.Right, features);
        }
    }

    #endregion

    #region Binning

    /// <summary>
    /// Bins all features in the training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This converts continuous feature values into discrete bins.
    /// For each feature:
    /// 1. Find the unique values
    /// 2. Determine bin boundaries (quantiles or uniform)
    /// 3. Assign each value to a bin (0 to MaxBins-1)
    ///
    /// This is a key optimization that makes the algorithm fast.
    /// </para>
    /// </remarks>
    private void BinFeatures(Matrix<T> x)
    {
        int numSamples = x.Rows;
        int numFeatures = x.Columns;

        var engine = AiDotNetEngine.Current;
        _binThresholds = new Vector<T>[numFeatures];

        for (int f = 0; f < numFeatures; f++)
        {
            // Extract and sort feature values
            var column = engine.GetColumn(x, f);
            var (sorted, _) = engine.TensorSort(Tensor<T>.FromVector(column));

            // Compute bin thresholds using quantiles
            _binThresholds[f] = ComputeQuantileThresholds(sorted.ToVector());
        }

        // Bin each value
        _binnedData = BinMatrix(x);
    }

    /// <summary>
    /// Computes quantile-based bin thresholds.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of uniform bins, we use quantiles to ensure
    /// each bin has roughly the same number of samples. This is more effective
    /// when feature values are not uniformly distributed.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeQuantileThresholds(Vector<T> sortedValues)
    {
        int n = sortedValues.Length;
        int maxBins = Math.Min(_options.MaxBins, n);

        if (maxBins <= 1)
        {
            return new Vector<T>(0);
        }

        var engine = AiDotNetEngine.Current;

        // Get unique values (the input is sorted, so this keeps ascending order)
        var uniqueValues = new Vector<T>(sortedValues.Distinct());

        if (uniqueValues.Length <= maxBins)
        {
            // Use midpoints between unique values as thresholds
            int count = uniqueValues.Length - 1;
            if (count <= 0)
            {
                return new Vector<T>(0);
            }

            return engine.Divide(
                engine.Add(uniqueValues.GetSubVector(0, count), uniqueValues.GetSubVector(1, count)),
                NumOps.FromDouble(2.0));
        }

        // Use quantiles: the values at positions floor(i / maxBins * (n - 1)) for i = 1..maxBins-1
        var quantileIndices = new int[maxBins - 1];
        for (int i = 1; i < maxBins; i++)
        {
            double quantile = (double)i / maxBins;
            quantileIndices[i - 1] = (int)(quantile * (n - 1));
        }

        // The picked values are non-decreasing, so dropping repeats keeps exactly the strictly
        // increasing thresholds (avoids duplicate thresholds).
        var picked = RegressionEngineOps.Gather(sortedValues, quantileIndices);
        return new Vector<T>(picked.Distinct());
    }

    /// <summary>
    /// Bins every value of <paramref name="x"/> against the learned per-feature thresholds.
    /// </summary>
    /// <remarks>
    /// A value's bin is the first threshold index it does not exceed (thresholds.Length when it exceeds
    /// all of them or is NaN), which is a left-sided sorted search.
    /// </remarks>
    private byte[,] BinMatrix(Matrix<T> x)
    {
        var binThresholds = _binThresholds ?? throw new InvalidOperationException("Bin thresholds not computed.");
        var engine = AiDotNetEngine.Current;
        var binned = new byte[x.Rows, _numFeatures];

        for (int f = 0; f < _numFeatures; f++)
        {
            var thresholds = binThresholds[f];
            if (thresholds.Length == 0 || x.Rows == 0)
            {
                continue;
            }

            int[] bins = engine.TensorSearchSorted(
                Tensor<T>.FromVector(thresholds),
                Tensor<T>.FromVector(engine.GetColumn(x, f)),
                right: false).ToArray();
            for (int i = 0; i < bins.Length; i++)
            {
                binned[i, f] = (byte)bins[i];
            }
        }

        return binned;
    }



    #endregion

    #region Tree Building

    /// <summary>
    /// Gets subsample indices for stochastic gradient boosting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If SubsampleRatio is less than 1.0, we randomly
    /// select a subset of samples to train each tree. This adds randomness
    /// that can improve generalization.
    /// </para>
    /// </remarks>
    private int[] GetSubsampleIndices(int totalSamples)
    {
        if (_options.SubsampleRatio >= 1.0)
        {
            return Enumerable.Range(0, totalSamples).ToArray();
        }

        int subsampleSize = (int)(totalSamples * _options.SubsampleRatio);
        var indices = new HashSet<int>();

        while (indices.Count < subsampleSize)
        {
            indices.Add(_random.Next(totalSamples));
        }

        return indices.ToArray();
    }

    /// <summary>
    /// Builds a single histogram-based tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This builds a decision tree using histograms for
    /// efficient split finding. The algorithm:
    /// 1. Start with all samples at the root
    /// 2. Find the best split using histograms
    /// 3. Split into left and right children
    /// 4. Recursively build children until stopping criteria
    /// </para>
    /// </remarks>
    private HistTreeNode BuildTree(Vector<T> residuals, int[] sampleIndices)
    {
        var root = new HistTreeNode(NumOps.Zero)
        {
            SampleIndices = sampleIndices.ToList(),
            Depth = 0
        };

        // Use a priority queue for best-first growth
        var queue = new List<HistTreeNode> { root };
        int leafCount = 1;

        var importanceFeatures = new List<int>();
        var importanceGains = new List<T>();

        while (queue.Count > 0)
        {
            // Find the node with the best potential gain
            int bestIdx = -1;
            SplitInfo? bestSplit = null;

            var candidateNodes = new List<int>();
            var candidateSplits = new List<SplitInfo>();
            for (int i = 0; i < queue.Count; i++)
            {
                var node = queue[i];

                // Check stopping criteria
                if (node.Depth >= _options.MaxDepth) continue;
                if (node.SampleIndices.Count < 2 * _options.MinSamplesLeaf) continue;

                // Find best split for this node
                var split = FindBestSplit(node, residuals);
                if (split is not null)
                {
                    candidateNodes.Add(i);
                    candidateSplits.Add(split);
                }
            }

            // The first node with the largest gain wins
            if (candidateSplits.Count > 0)
            {
                int best = RegressionEngineOps.ArgMaxFirst(candidateSplits.Select(s => s.Gain));
                bestSplit = candidateSplits[best];
                bestIdx = candidateNodes[best];
            }

            // If no beneficial split found or max leaves reached, stop
            if (bestIdx < 0 || bestSplit is null ||
                (_options.MaxLeafNodes.HasValue && leafCount >= _options.MaxLeafNodes.Value))
            {
                break;
            }

            // Apply the best split
            var nodeToSplit = queue[bestIdx];
            queue.RemoveAt(bestIdx);

            ApplySplit(nodeToSplit, bestSplit, residuals);

            // Add children to queue
            if (nodeToSplit.Left is not null)
            {
                queue.Add(nodeToSplit.Left);
            }
            if (nodeToSplit.Right is not null)
            {
                queue.Add(nodeToSplit.Right);
            }

            // Record the gain for the feature importance of the split feature
            importanceFeatures.Add(bestSplit.FeatureIndex);
            importanceGains.Add(bestSplit.Gain);

            leafCount++;
        }

        // Update feature importance
        if (_featureImportances is not null && importanceFeatures.Count > 0)
        {
            var engine = AiDotNetEngine.Current;
            var treeImportance = engine.TensorScatterAdd(
                Tensor<T>.FromVector(engine.FillZero(_featureImportances.Length)),
                new Tensor<int>(importanceFeatures.ToArray(), new[] { importanceFeatures.Count }),
                Tensor<T>.FromVector(new Vector<T>(importanceGains)),
                0).ToVector();
            _featureImportances = engine.Add(_featureImportances, treeImportance);
        }

        // Set leaf values for remaining nodes
        foreach (var node in queue)
        {
            SetLeafValue(node, residuals);
        }

        // Also ensure root has a leaf value if it was never split
        if (root.Left is null && root.Right is null)
        {
            SetLeafValue(root, residuals);
        }

        return root;
    }

    /// <summary>
    /// Finds the best split for a node using histograms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the core of histogram-based gradient boosting.
    /// Instead of checking every possible split point:
    /// 1. Build a histogram counting gradient sums for each bin
    /// 2. Only check splits between bins
    ///
    /// This reduces complexity from O(n) to O(bins).
    /// </para>
    /// </remarks>
    private SplitInfo? FindBestSplit(HistTreeNode node, Vector<T> residuals)
    {
        // Get features to consider (column subsampling)
        var featuresToConsider = GetFeaturesToConsider();

        // Every returned split already beats MinGainToSplit; the first feature with the largest gain wins.
        var splits = new List<SplitInfo>();
        foreach (int featureIdx in featuresToConsider)
        {
            // Build histogram for this feature
            var histogram = BuildHistogram(node.SampleIndices, featureIdx, residuals);

            // Find best split point in histogram
            var split = FindBestSplitInHistogram(histogram, featureIdx, node.SampleIndices.Count);
            if (split is not null)
            {
                splits.Add(split);
            }
        }

        return splits.Count == 0
            ? null
            : splits[RegressionEngineOps.ArgMaxFirst(splits.Select(s => s.Gain))];
    }

    /// <summary>
    /// Gets feature indices to consider for splitting (column subsampling).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If ColsampleByTree is less than 1.0, we randomly
    /// select a subset of features to consider for each split. This adds
    /// diversity to the trees.
    /// </para>
    /// </remarks>
    private List<int> GetFeaturesToConsider()
    {
        if (_options.ColsampleByTree >= 1.0)
        {
            return Enumerable.Range(0, _numFeatures).ToList();
        }

        int numFeaturesToUse = Math.Max(1, (int)(_numFeatures * _options.ColsampleByTree));
        var allFeatures = Enumerable.Range(0, _numFeatures).ToList();

        // Shuffle and take first numFeaturesToUse
        for (int i = allFeatures.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (allFeatures[i], allFeatures[j]) = (allFeatures[j], allFeatures[i]);
        }

        return allFeatures.Take(numFeaturesToUse).ToList();
    }

    /// <summary>
    /// Builds a gradient histogram for a feature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A histogram accumulates the sum of residuals (gradients)
    /// for each bin. This allows us to quickly compute the gain for any split point.
    /// </para>
    /// </remarks>
    private Histogram BuildHistogram(List<int> sampleIndices, int featureIdx, Vector<T> residuals)
    {
        int numBins = (_binThresholds ?? throw new InvalidOperationException("Bin thresholds not computed."))[featureIdx].Length + 1;
        var binnedData = _binnedData ?? throw new InvalidOperationException("Binned data not computed.");
        var engine = AiDotNetEngine.Current;

        var bins = new int[sampleIndices.Count];
        var counts = new int[numBins];
        for (int k = 0; k < bins.Length; k++)
        {
            int bin = binnedData[sampleIndices[k], featureIdx];
            bins[k] = bin;
            counts[bin]++;
        }

        var gradientSum = engine.FillZero(numBins);
        var hessianSum = engine.FillZero(numBins);
        if (bins.Length > 0)
        {
            var binIndex = new Tensor<int>(bins, new[] { bins.Length });
            gradientSum = engine.TensorScatterAdd(
                Tensor<T>.FromVector(gradientSum),
                binIndex,
                Tensor<T>.FromVector(RegressionEngineOps.Gather(residuals, sampleIndices.ToArray())),
                0).ToVector();

            // Squared loss has hessian = 1
            hessianSum = engine.TensorScatterAdd(
                Tensor<T>.FromVector(hessianSum),
                binIndex,
                Tensor<T>.FromVector(engine.Fill(bins.Length, NumOps.One)),
                0).ToVector();
        }

        return new Histogram(gradientSum, hessianSum, counts);
    }

    /// <summary>
    /// Finds the best split point within a histogram.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given the histogram, we scan through all possible
    /// split points (between bins) and compute the gain for each. The gain
    /// measures how much the split reduces prediction error.
    ///
    /// Gain = (left_gradient²/left_hessian + right_gradient²/right_hessian - total_gradient²/total_hessian) / 2
    ///        - regularization_penalty
    /// </para>
    /// </remarks>
    private SplitInfo? FindBestSplitInHistogram(Histogram histogram, int featureIdx, int totalCount)
    {
        // Split after bin b sends bins 0..b left; b ranges over 0..numBins-2
        int candidates = histogram.Count.Length - 1;
        if (candidates <= 0)
        {
            return null;
        }

        var engine = AiDotNetEngine.Current;

        // Compute totals
        T totalGrad = engine.Sum(histogram.GradientSum);
        T totalHess = engine.Sum(histogram.HessianSum);

        // Running (left) sums for every split point at once
        var leftGrad = engine.TensorCumSum(Tensor<T>.FromVector(histogram.GradientSum), 0).ToVector().GetSubVector(0, candidates);
        var leftHess = engine.TensorCumSum(Tensor<T>.FromVector(histogram.HessianSum), 0).ToVector().GetSubVector(0, candidates);
        var leftCounts = new int[candidates];
        var allowed = new bool[candidates];
        int leftCount = 0;
        for (int bin = 0; bin < candidates; bin++)
        {
            leftCount += histogram.Count[bin];
            leftCounts[bin] = leftCount;

            // Check minimum samples constraint
            int rightCount = totalCount - leftCount;
            allowed[bin] = leftCount >= _options.MinSamplesLeaf && rightCount >= _options.MinSamplesLeaf;
        }

        var rightGrad = engine.Subtract(engine.Fill(candidates, totalGrad), leftGrad);
        var rightHess = engine.Subtract(engine.Fill(candidates, totalHess), leftHess);

        // Compute gain with L2 regularization
        // gain = 0.5 * (leftGrad^2/(leftHess+lambda) + rightGrad^2/(rightHess+lambda) - totalGrad^2/(totalHess+lambda))
        T lambda = NumOps.FromDouble(_options.L2Regularization);
        var lambdas = engine.Fill(candidates, lambda);
        var leftTerm = engine.Divide(engine.Multiply(leftGrad, leftGrad), engine.Add(leftHess, lambdas));
        var rightTerm = engine.Divide(engine.Multiply(rightGrad, rightGrad), engine.Add(rightHess, lambdas));
        T totalTerm = NumOps.Divide(NumOps.Multiply(totalGrad, totalGrad), NumOps.Add(totalHess, lambda));
        var gains = engine.Multiply(
            engine.Subtract(engine.Add(leftTerm, rightTerm), engine.Fill(candidates, totalTerm)),
            NumOps.FromDouble(0.5));

        // The first split point with the largest gain wins; disallowed or NaN gains never win.
        var (bestBin, bestGain) = RegressionEngineOps.MaskedArgMaxFirst(gains, allowed);
        if (bestBin < 0 || !NumOps.GreaterThan(bestGain, NumOps.FromDouble(_options.MinGainToSplit)))
        {
            return null;
        }

        // Left and right counts for the best split
        int bestLeftCount = leftCounts[bestBin];

        return new SplitInfo(NumOps.Zero)
        {
            FeatureIndex = featureIdx,
            BinThreshold = bestBin,
            Gain = bestGain,
            LeftCount = bestLeftCount,
            RightCount = totalCount - bestLeftCount
        };
    }

    /// <summary>
    /// Applies a split to a node, creating left and right children.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After finding the best split, this method:
    /// 1. Creates left and right child nodes
    /// 2. Assigns samples to each child based on their bin values
    /// 3. Sets the split threshold on the parent node
    /// </para>
    /// </remarks>
    private void ApplySplit(HistTreeNode node, SplitInfo split, Vector<T> residuals)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        foreach (int idx in node.SampleIndices)
        {
            int bin = (_binnedData ?? throw new InvalidOperationException("Binned data not computed."))[idx, split.FeatureIndex];
            if (bin <= split.BinThreshold)
            {
                leftIndices.Add(idx);
            }
            else
            {
                rightIndices.Add(idx);
            }
        }

        node.IsLeaf = false;
        node.FeatureIndex = split.FeatureIndex;
        node.BinThreshold = split.BinThreshold;

        // Convert bin threshold to actual value for prediction
        if (split.BinThreshold < (_binThresholds ?? throw new InvalidOperationException("Bin thresholds not computed."))[split.FeatureIndex].Length)
        {
            node.Threshold = _binThresholds[split.FeatureIndex][split.BinThreshold];
        }
        else
        {
            node.Threshold = NumOps.MaxValue;
        }

        node.Left = new HistTreeNode(NumOps.Zero)
        {
            SampleIndices = leftIndices,
            Depth = node.Depth + 1
        };

        node.Right = new HistTreeNode(NumOps.Zero)
        {
            SampleIndices = rightIndices,
            Depth = node.Depth + 1
        };

        // Set leaf values (may be overwritten if children are split further)
        SetLeafValue(node.Left, residuals);
        SetLeafValue(node.Right, residuals);
    }

    /// <summary>
    /// Sets the prediction value for a leaf node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For squared error loss, the optimal leaf value is
    /// the mean of the residuals at that leaf. With L2 regularization, this
    /// becomes: sum(residuals) / (count + regularization)
    /// </para>
    /// </remarks>
    private void SetLeafValue(HistTreeNode node, Vector<T> residuals)
    {
        node.IsLeaf = true;

        if (node.SampleIndices.Count == 0)
        {
            node.LeafValue = NumOps.Zero;
            return;
        }

        T sum = AiDotNetEngine.Current.Sum(RegressionEngineOps.Gather(residuals, node.SampleIndices.ToArray()));

        // Optimal leaf value with L2 regularization
        T lambda = NumOps.FromDouble(_options.L2Regularization);
        node.LeafValue = NumOps.Divide(sum, NumOps.Add(NumOps.FromDouble(node.SampleIndices.Count), lambda));
    }

    #endregion

    #region Prediction

    /// <summary>
    /// Predicts using a single tree on binned training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, we use the already-binned data
    /// for faster prediction. This avoids re-binning at each iteration.
    /// </para>
    /// </remarks>
    private T PredictSingleTree(HistTreeNode tree, byte[,] binnedData, int sampleIndex)
    {
        var node = tree;

        while (!node.IsLeaf)
        {
            int bin = binnedData[sampleIndex, node.FeatureIndex];
            if (node.Left is null || node.Right is null)
            {
                throw new InvalidOperationException("Internal tree node has null children.");
            }
            node = bin <= node.BinThreshold ? node.Left : node.Right;
        }

        return node.LeafValue;
    }

    /// <summary>
    /// Predicts using a single tree from binned features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For new data that has been binned, traverse the
    /// tree using bin indices to find the leaf prediction.
    /// </para>
    /// </remarks>
    private Vector<T> PredictTree(HistTreeNode tree, byte[,] binnedData, int rows)
    {
        // Each row walks the tree over its integer bin codes; the leaf values form the prediction vector.
        return new Vector<T>(Enumerable.Range(0, rows).Select(i => PredictSingleTree(tree, binnedData, i)));
    }

    #endregion

    #region Feature Importance

    /// <summary>
    /// Normalizes feature importances to sum to 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feature importance scores are accumulated during
    /// training (total gain from splits using each feature). Normalizing makes
    /// them easier to interpret as relative importance percentages.
    /// </para>
    /// </remarks>
    private void NormalizeFeatureImportances()
    {
        if (_featureImportances is null) return;

        var engine = AiDotNetEngine.Current;
        T sum = engine.Sum(_featureImportances);

        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            _featureImportances = engine.Divide(_featureImportances, sum);
        }
    }

    #endregion

    #region Serialization Helpers

    /// <summary>
    /// Serializes a tree node recursively.
    /// </summary>
    private void SerializeTree(BinaryWriter writer, HistTreeNode node)
    {
        writer.Write(node.IsLeaf);
        writer.Write(NumOps.ToDouble(node.LeafValue));
        writer.Write(node.FeatureIndex);
        writer.Write(node.BinThreshold);
        writer.Write(NumOps.ToDouble(node.Threshold));
        writer.Write(node.Depth);

        if (!node.IsLeaf)
        {
            if (node.Left is null || node.Right is null)
            {
                throw new InvalidOperationException("Internal tree node has null children during serialization.");
            }
            SerializeTree(writer, node.Left);
            SerializeTree(writer, node.Right);
        }
    }

    /// <summary>
    /// Deserializes a tree node recursively.
    /// </summary>
    private HistTreeNode DeserializeTree(BinaryReader reader)
    {
        var node = new HistTreeNode(NumOps.Zero)
        {
            IsLeaf = reader.ReadBoolean(),
            LeafValue = NumOps.FromDouble(reader.ReadDouble()),
            FeatureIndex = reader.ReadInt32(),
            BinThreshold = reader.ReadInt32(),
            Threshold = NumOps.FromDouble(reader.ReadDouble()),
            Depth = reader.ReadInt32()
        };

        if (!node.IsLeaf)
        {
            node.Left = DeserializeTree(reader);
            node.Right = DeserializeTree(reader);
        }

        return node;
    }

    #endregion

    #region JIT Compilation Support

    /// <summary>
    /// Exports a soft decision tree as a computation graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hard decision trees use if-then-else logic which
    /// isn't differentiable. Soft trees use sigmoid functions to smoothly
    /// blend between branches, making them differentiable and suitable for
    /// JIT compilation and hardware acceleration.
    /// </para>
    /// </remarks>
    private ComputationNode<T> ExportSoftTree(ComputationNode<T> inputNode, HistTreeNode tree, double temperature)
    {
        return ExportSoftTreeNode(inputNode, tree, temperature);
    }

    /// <summary>
    /// Recursively exports a tree node as a soft computation graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For each internal node:
    /// output = sigmoid(temp * (threshold - x[feature])) * left_output +
    ///          (1 - sigmoid(...)) * right_output
    ///
    /// As temperature → ∞, this approaches a hard split.
    /// </para>
    /// </remarks>
    private ComputationNode<T> ExportSoftTreeNode(ComputationNode<T> inputNode, HistTreeNode node, double temperature)
    {
        if (node.IsLeaf)
        {
            // Create constant for leaf value
            var leafTensor = new Tensor<T>(new int[] { 1, 1 });
            leafTensor[0, 0] = node.LeafValue;
            return TensorOperations<T>.Constant(leafTensor, $"leaf_{node.GetHashCode()}");
        }

        // Get the feature value: x[:, featureIndex]
        var featureSliceNode = TensorOperations<T>.Slice(inputNode, 0, node.FeatureIndex, node.FeatureIndex + 1);

        // Create threshold constant
        var thresholdTensor = new Tensor<T>(new int[] { 1, 1 });
        thresholdTensor[0, 0] = node.Threshold;
        var thresholdNode = TensorOperations<T>.Constant(thresholdTensor, $"threshold_{node.GetHashCode()}");

        // Create temperature constant
        var tempTensor = new Tensor<T>(new int[] { 1, 1 });
        tempTensor[0, 0] = NumOps.FromDouble(temperature);
        var tempNode = TensorOperations<T>.Constant(tempTensor, $"temp_{node.GetHashCode()}");

        // Compute sigmoid(temperature * (threshold - x))
        var diffNode = TensorOperations<T>.Subtract(thresholdNode, featureSliceNode);
        var scaledDiffNode = TensorOperations<T>.ElementwiseMultiply(tempNode, diffNode);
        var sigmoidNode = TensorOperations<T>.Sigmoid(scaledDiffNode);

        // Get left and right subtree outputs
        if (node.Left is null || node.Right is null)
        {
            throw new InvalidOperationException("Internal tree node has null children during graph export.");
        }
        var leftOutput = ExportSoftTreeNode(inputNode, node.Left, temperature);
        var rightOutput = ExportSoftTreeNode(inputNode, node.Right, temperature);

        // Compute: sigmoid * left + (1 - sigmoid) * right
        var leftWeighted = TensorOperations<T>.ElementwiseMultiply(sigmoidNode, leftOutput);

        var onesTensor = new Tensor<T>(new int[] { 1, 1 });
        onesTensor[0, 0] = NumOps.One;
        var onesNode = TensorOperations<T>.Constant(onesTensor, "ones");
        var oneMinusSigmoid = TensorOperations<T>.Subtract(onesNode, sigmoidNode);

        var rightWeighted = TensorOperations<T>.ElementwiseMultiply(oneMinusSigmoid, rightOutput);

        return TensorOperations<T>.Add(leftWeighted, rightWeighted);
    }

    #endregion

    #region Helper Classes

    /// <summary>
    /// Represents a node in the histogram-based decision tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each node is either:
    /// - A leaf node: makes a prediction (LeafValue)
    /// - An internal node: splits based on a feature and threshold
    /// </para>
    /// </remarks>
    private class HistTreeNode
    {
        public bool IsLeaf { get; set; } = true;
        public T LeafValue { get; set; }
        public int FeatureIndex { get; set; }
        public int BinThreshold { get; set; }
        public T Threshold { get; set; }
        public HistTreeNode? Left { get; set; }
        public HistTreeNode? Right { get; set; }
        public List<int> SampleIndices { get; set; } = [];
        public int Depth { get; set; }

        public HistTreeNode(T zero)
        {
            LeafValue = zero;
            Threshold = zero;
        }

        public HistTreeNode() : this(MathHelper.GetNumericOperations<T>().Zero) { }
    }

    /// <summary>
    /// Represents a single bin in a gradient histogram.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each bin accumulates statistics about the samples
    /// that fall into that bin:
    /// - GradientSum: sum of residuals (for finding optimal leaf values)
    /// - HessianSum: sum of hessians (for regularization)
    /// - Count: number of samples
    /// </para>
    /// </remarks>
    private sealed class Histogram
    {
        public Vector<T> GradientSum { get; }
        public Vector<T> HessianSum { get; }
        public int[] Count { get; }

        public Histogram(Vector<T> gradientSum, Vector<T> hessianSum, int[] count)
        {
            GradientSum = gradientSum;
            HessianSum = hessianSum;
            Count = count;
        }
    }

    /// <summary>
    /// Contains information about a potential split.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When evaluating splits, we track:
    /// - Which feature to split on
    /// - Which bin threshold to use
    /// - How much gain (error reduction) the split provides
    /// - How many samples go to each child
    /// </para>
    /// </remarks>
    private class SplitInfo
    {
        public int FeatureIndex { get; set; }
        public int BinThreshold { get; set; }
        public T Gain { get; set; }
        public int LeftCount { get; set; }
        public int RightCount { get; set; }

        public SplitInfo(T zero)
        {
            Gain = zero;
        }

        public SplitInfo() : this(MathHelper.GetNumericOperations<T>().Zero) { }
    }

    #endregion
}
