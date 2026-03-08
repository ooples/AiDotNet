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
/// var model = new HistGradientBoostingRegression&lt;double&gt;(options);
/// model.Train(X, y);
/// var predictions = model.Predict(X_test);
/// </code>
/// </para>
/// </remarks>
public class HistGradientBoostingRegression<T> : IFullModel<T, Matrix<T>, Vector<T>>, IConfigurableModel<T>
{
    #region Fields

    /// <summary>
    /// Provides numeric operations for generic type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This allows the algorithm to perform math operations
    /// (add, subtract, multiply, etc.) on the generic type T, which could be
    /// float, double, or other numeric types.
    /// </para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

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
    private T[][]? _binThresholds;

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
    private T[]? _featureImportances;

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
        NumOps = MathHelper.GetNumericOperations<T>();
        _initialPrediction = NumOps.Zero;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _defaultLossFunction = new MeanSquaredErrorLoss<T>();
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    public ModelType ModelType => ModelType.HistGradientBoosting;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets whether JIT compilation is supported.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Histogram-based gradient boosting supports JIT compilation
    /// when we represent the tree traversal using soft (differentiable) splits.
    /// This allows the entire ensemble to be exported as a computation graph.
    /// </para>
    /// </remarks>
    public bool SupportsJitCompilation => true;

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
    public void Train(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;

        // Step 1: Bin the features
        BinFeatures(x);

        // Step 2: Compute initial prediction (mean of y)
        T sum = NumOps.Zero;
        for (int i = 0; i < y.Length; i++)
        {
            sum = NumOps.Add(sum, y[i]);
        }
        _initialPrediction = NumOps.Divide(sum, NumOps.FromDouble(y.Length));

        // Step 3: Initialize predictions and residuals
        var predictions = new double[y.Length];
        var residuals = new double[y.Length];
        double initialPred = NumOps.ToDouble(_initialPrediction);

        for (int i = 0; i < y.Length; i++)
        {
            predictions[i] = initialPred;
            residuals[i] = NumOps.ToDouble(y[i]) - predictions[i];
        }

        // Step 4: Initialize trees and feature importances
        _trees = new List<HistTreeNode>(_options.NumberOfIterations);
        _featureImportances = new T[_numFeatures];
        for (int i = 0; i < _numFeatures; i++)
        {
            _featureImportances[i] = NumOps.Zero;
        }

        // Step 5: Build trees iteratively
        for (int iteration = 0; iteration < _options.NumberOfIterations; iteration++)
        {
            // Subsample indices
            int[] sampleIndices = GetSubsampleIndices(y.Length);

            // Build tree on residuals
            var tree = BuildTree(residuals, sampleIndices);
            _trees.Add(tree);

            // Update predictions
            for (int i = 0; i < y.Length; i++)
            {
                double treePred = PredictSingleTree(tree, i);
                predictions[i] += _options.LearningRate * treePred;
                residuals[i] = NumOps.ToDouble(y[i]) - predictions[i];
            }
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
    public Vector<T> Predict(Matrix<T> input)
    {
        if (_trees is null || _binThresholds is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var predictions = new Vector<T>(input.Rows);
        double initialPred = NumOps.ToDouble(_initialPrediction);

        for (int i = 0; i < input.Rows; i++)
        {
            double pred = initialPred;

            // Bin the input features
            var binnedRow = BinRow(input, i);

            // Add contribution from each tree
            foreach (var tree in _trees)
            {
                pred += _options.LearningRate * PredictSingleTreeFromBins(tree, binnedRow);
            }

            predictions[i] = NumOps.FromDouble(pred);
        }

        return predictions;
    }

    /// <summary>
    /// Gets model metadata.
    /// </summary>
    public ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.HistGradientBoosting,
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
    public Dictionary<string, T> GetFeatureImportance()
    {
        var result = new Dictionary<string, T>();

        if (_featureImportances is null)
        {
            return result;
        }

        for (int i = 0; i < _featureImportances.Length; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = _featureImportances[i];
        }

        return result;
    }

    /// <summary>
    /// Gets model parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        // Return initial prediction and tree count as parameters
        var parameters = new Vector<T>(2);
        parameters[0] = _initialPrediction;
        parameters[1] = NumOps.FromDouble(_trees?.Count ?? 0);
        return parameters;
    }

    /// <summary>
    /// Sets model parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length >= 1)
        {
            _initialPrediction = parameters[0];
        }
    }

    /// <summary>
    /// Creates a new instance with the given parameters.
    /// </summary>
    public IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = new HistGradientBoostingRegression<T>(_options);
        newModel.SetParameters(parameters);
        return newModel;
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write options
        writer.Write(_options.NumberOfIterations);
        writer.Write(_options.LearningRate);
        writer.Write(_options.MaxBins);
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MaxLeafNodes ?? -1);
        writer.Write(_options.MinSamplesLeaf);
        writer.Write(_options.L2Regularization);
        writer.Write(_options.SubsampleRatio);

        // Write model state
        writer.Write(NumOps.ToDouble(_initialPrediction));
        writer.Write(_numFeatures);

        // Write bin thresholds
        if (_binThresholds is not null)
        {
            writer.Write(_binThresholds.Length);
            foreach (var featureThresholds in _binThresholds)
            {
                writer.Write(featureThresholds.Length);
                foreach (var threshold in featureThresholds)
                {
                    writer.Write(NumOps.ToDouble(threshold));
                }
            }
        }
        else
        {
            writer.Write(0);
        }

        // Write trees
        if (_trees is not null)
        {
            writer.Write(_trees.Count);
            foreach (var tree in _trees)
            {
                SerializeTree(writer, tree);
            }
        }
        else
        {
            writer.Write(0);
        }

        // Write feature importances
        if (_featureImportances is not null)
        {
            writer.Write(_featureImportances.Length);
            foreach (var importance in _featureImportances)
            {
                writer.Write(NumOps.ToDouble(importance));
            }
        }
        else
        {
            writer.Write(0);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    public void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Read options
        _options.NumberOfIterations = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.MaxBins = reader.ReadInt32();
        _options.MaxDepth = reader.ReadInt32();
        int maxLeaf = reader.ReadInt32();
        _options.MaxLeafNodes = maxLeaf >= 0 ? maxLeaf : null;
        _options.MinSamplesLeaf = reader.ReadInt32();
        _options.L2Regularization = reader.ReadDouble();
        _options.SubsampleRatio = reader.ReadDouble();

        // Read model state
        _initialPrediction = NumOps.FromDouble(reader.ReadDouble());
        _numFeatures = reader.ReadInt32();

        // Read bin thresholds
        int numFeatureThresholds = reader.ReadInt32();
        if (numFeatureThresholds > 0)
        {
            _binThresholds = new T[numFeatureThresholds][];
            for (int i = 0; i < numFeatureThresholds; i++)
            {
                int numThresholds = reader.ReadInt32();
                _binThresholds[i] = new T[numThresholds];
                for (int j = 0; j < numThresholds; j++)
                {
                    _binThresholds[i][j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
        }

        // Read trees
        int numTrees = reader.ReadInt32();
        _trees = new List<HistTreeNode>(numTrees);
        for (int i = 0; i < numTrees; i++)
        {
            _trees.Add(DeserializeTree(reader));
        }

        // Read feature importances
        int numImportances = reader.ReadInt32();
        if (numImportances > 0)
        {
            _featureImportances = new T[numImportances];
            for (int i = 0; i < numImportances; i++)
            {
                _featureImportances[i] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This exports the entire gradient boosting ensemble as a
    /// differentiable computation graph. Each tree's decision path is approximated
    /// using soft (sigmoid) splits to make it differentiable.
    ///
    /// The computation graph represents:
    /// output = initial_prediction + learning_rate * sum(soft_tree_predictions)
    ///
    /// The soft tree uses sigmoid functions to approximate hard splits:
    /// - Hard split: if (x &lt; threshold) then left else right
    /// - Soft split: sigmoid(temperature * (threshold - x)) * left + (1 - sigmoid(...)) * right
    ///
    /// This allows gradient-based optimization and hardware acceleration.
    /// </para>
    /// </remarks>
    public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (_trees is null || _trees.Count == 0 || _binThresholds is null)
        {
            throw new InvalidOperationException(
                "Model must be trained before exporting computation graph.");
        }

        // Create input placeholder for features: [batchSize, numFeatures]
        var inputTensor = new Tensor<T>(new int[] { 1, _numFeatures });
        var inputNode = TensorOperations<T>.Variable(inputTensor, "features");
        inputNodes.Add(inputNode);

        // Create initial prediction constant
        var initialTensor = new Tensor<T>(new int[] { 1, 1 });
        initialTensor[0, 0] = _initialPrediction;
        var initialNode = TensorOperations<T>.Constant(initialTensor, "initial_prediction");

        // Create learning rate constant
        var lrTensor = new Tensor<T>(new int[] { 1, 1 });
        lrTensor[0, 0] = NumOps.FromDouble(_options.LearningRate);
        var lrNode = TensorOperations<T>.Constant(lrTensor, "learning_rate");

        // Export each tree as a soft decision tree
        ComputationNode<T>? treeSumNode = null;
        double temperature = 10.0; // Steepness of soft sigmoid splits

        foreach (var tree in _trees)
        {
            var treeNode = ExportSoftTree(inputNode, tree, temperature);

            if (treeSumNode is null)
            {
                treeSumNode = treeNode;
            }
            else
            {
                treeSumNode = TensorOperations<T>.Add(treeSumNode, treeNode);
            }
        }

        // Multiply by learning rate
        var scaledTreesNode = TensorOperations<T>.ElementwiseMultiply(lrNode, treeSumNode!);

        // Add initial prediction
        var outputNode = TensorOperations<T>.Add(initialNode, scaledTreesNode);
        outputNode.Name = "prediction";

        return outputNode;
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
    public ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

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
    public int ParameterCount
    {
        get
        {
            int count = 1; // Initial prediction
            if (_trees is not null)
            {
                foreach (var tree in _trees)
                {
                    count += CountLeaves(tree);
                }
            }
            return count;
        }
    }

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
    public void SaveModel(string filePath)
    {
        byte[] data = Serialize();
        File.WriteAllBytes(filePath, data);
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
    public void LoadModel(string filePath)
    {
        byte[] data = File.ReadAllBytes(filePath);
        Deserialize(data);
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
    public void SaveState(Stream stream)
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
    public void LoadState(Stream stream)
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
    public IEnumerable<int> GetActiveFeatureIndices()
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
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
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
    public bool IsFeatureUsed(int featureIndex)
    {
        var activeFeatures = GetActiveFeatureIndices();
        return activeFeatures.Contains(featureIndex);
    }

    /// <summary>
    /// Creates a deep copy of the model.
    /// </summary>
    /// <returns>A new instance with all data copied.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a complete independent copy of the model.
    /// Changes to the copy won't affect the original.
    /// </para>
    /// </remarks>
    public IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        var copy = new HistGradientBoostingRegression<T>(_options);
        copy.Deserialize(Serialize());
        return copy;
    }

    /// <summary>
    /// Creates a shallow copy of the model.
    /// </summary>
    /// <returns>A new instance sharing data with the original.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy that shares some internal data
    /// with the original. For this model, we use deep copy for safety.
    /// </para>
    /// </remarks>
    public IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        return DeepCopy();
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
    public Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? _defaultLossFunction;
        var predictions = Predict(input);

        // Compute gradients (negative of loss derivative with respect to predictions)
        var gradients = new Vector<T>(target.Length);
        for (int i = 0; i < target.Length; i++)
        {
            // For MSE: derivative = 2 * (prediction - target)
            // We return the negative gradient (direction of improvement)
            var diff = NumOps.Subtract(target[i], predictions[i]);
            gradients[i] = NumOps.Multiply(NumOps.FromDouble(2.0), diff);
        }

        return gradients;
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
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // For gradient boosting, we adjust the initial prediction based on average gradient
        if (gradients.Length > 0)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < gradients.Length; i++)
            {
                sum = NumOps.Add(sum, gradients[i]);
            }
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

        _binThresholds = new T[numFeatures][];
        _binnedData = new byte[numSamples, numFeatures];

        for (int f = 0; f < numFeatures; f++)
        {
            // Extract and sort feature values
            var values = new List<double>(numSamples);
            for (int i = 0; i < numSamples; i++)
            {
                values.Add(NumOps.ToDouble(x[i, f]));
            }
            values.Sort();

            // Compute bin thresholds using quantiles
            var thresholds = ComputeQuantileThresholds(values);
            _binThresholds[f] = thresholds.Select(t => NumOps.FromDouble(t)).ToArray();

            // Bin each value
            for (int i = 0; i < numSamples; i++)
            {
                double val = NumOps.ToDouble(x[i, f]);
                _binnedData[i, f] = (byte)FindBin(val, thresholds);
            }
        }
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
    private List<double> ComputeQuantileThresholds(List<double> sortedValues)
    {
        var thresholds = new List<double>();
        int n = sortedValues.Count;
        int maxBins = Math.Min(_options.MaxBins, n);

        if (maxBins <= 1)
        {
            return thresholds;
        }

        // Get unique values
        var uniqueValues = sortedValues.Distinct().ToList();

        if (uniqueValues.Count <= maxBins)
        {
            // Use midpoints between unique values as thresholds
            for (int i = 0; i < uniqueValues.Count - 1; i++)
            {
                thresholds.Add((uniqueValues[i] + uniqueValues[i + 1]) / 2.0);
            }
        }
        else
        {
            // Use quantiles
            for (int i = 1; i < maxBins; i++)
            {
                double quantile = (double)i / maxBins;
                int index = (int)(quantile * (n - 1));
                double threshold = sortedValues[index];

                // Avoid duplicate thresholds
                if (thresholds.Count == 0 || threshold > thresholds[^1])
                {
                    thresholds.Add(threshold);
                }
            }
        }

        return thresholds;
    }

    /// <summary>
    /// Finds the bin index for a given value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given a value and the bin thresholds, this finds
    /// which bin the value belongs to using binary search for efficiency.
    /// </para>
    /// </remarks>
    private int FindBin(double value, List<double> thresholds)
    {
        // Binary search for the correct bin
        int left = 0;
        int right = thresholds.Count;

        while (left < right)
        {
            int mid = (left + right) / 2;
            if (value <= thresholds[mid])
            {
                right = mid;
            }
            else
            {
                left = mid + 1;
            }
        }

        return left;
    }

    /// <summary>
    /// Bins a single row of new data for prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When making predictions on new data, we need to
    /// bin the features using the same thresholds learned during training.
    /// </para>
    /// </remarks>
    private byte[] BinRow(Matrix<T> x, int row)
    {
        var binned = new byte[_numFeatures];

        for (int f = 0; f < _numFeatures; f++)
        {
            double val = NumOps.ToDouble(x[row, f]);
            var thresholds = _binThresholds![f].Select(t => NumOps.ToDouble(t)).ToList();
            binned[f] = (byte)FindBin(val, thresholds);
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
    private HistTreeNode BuildTree(double[] residuals, int[] sampleIndices)
    {
        var root = new HistTreeNode
        {
            SampleIndices = sampleIndices.ToList(),
            Depth = 0
        };

        // Use a priority queue for best-first growth
        var queue = new List<HistTreeNode> { root };
        int leafCount = 1;

        while (queue.Count > 0)
        {
            // Find the node with the best potential gain
            int bestIdx = -1;
            double bestGain = double.MinValue;
            SplitInfo? bestSplit = null;

            for (int i = 0; i < queue.Count; i++)
            {
                var node = queue[i];

                // Check stopping criteria
                if (node.Depth >= _options.MaxDepth) continue;
                if (node.SampleIndices.Count < 2 * _options.MinSamplesLeaf) continue;

                // Find best split for this node
                var split = FindBestSplit(node, residuals);

                if (split is not null && split.Gain > bestGain)
                {
                    bestGain = split.Gain;
                    bestSplit = split;
                    bestIdx = i;
                }
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

            // Update feature importance
            if (_featureImportances is not null)
            {
                _featureImportances[bestSplit.FeatureIndex] = NumOps.Add(
                    _featureImportances[bestSplit.FeatureIndex],
                    NumOps.FromDouble(bestSplit.Gain));
            }

            leafCount++;
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
    private SplitInfo? FindBestSplit(HistTreeNode node, double[] residuals)
    {
        SplitInfo? bestSplit = null;
        double bestGain = _options.MinGainToSplit;

        // Get features to consider (column subsampling)
        var featuresToConsider = GetFeaturesToConsider();

        foreach (int featureIdx in featuresToConsider)
        {
            // Build histogram for this feature
            var histogram = BuildHistogram(node.SampleIndices, featureIdx, residuals);

            // Find best split point in histogram
            var split = FindBestSplitInHistogram(histogram, featureIdx, node.SampleIndices.Count);

            if (split is not null && split.Gain > bestGain)
            {
                bestGain = split.Gain;
                bestSplit = split;
            }
        }

        return bestSplit;
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
    private HistogramBin[] BuildHistogram(List<int> sampleIndices, int featureIdx, double[] residuals)
    {
        int numBins = _binThresholds![featureIdx].Length + 1;
        var histogram = new HistogramBin[numBins];

        for (int i = 0; i < numBins; i++)
        {
            histogram[i] = new HistogramBin();
        }

        foreach (int idx in sampleIndices)
        {
            int bin = _binnedData![idx, featureIdx];
            histogram[bin].GradientSum += residuals[idx];
            histogram[bin].HessianSum += 1.0; // Squared loss has hessian = 1
            histogram[bin].Count++;
        }

        return histogram;
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
    private SplitInfo? FindBestSplitInHistogram(HistogramBin[] histogram, int featureIdx, int totalCount)
    {
        // Compute totals
        double totalGrad = 0, totalHess = 0;
        foreach (var bin in histogram)
        {
            totalGrad += bin.GradientSum;
            totalHess += bin.HessianSum;
        }

        double bestGain = _options.MinGainToSplit;
        int bestBin = -1;

        double leftGrad = 0, leftHess = 0;
        int leftCount = 0;

        // Try each split point
        for (int bin = 0; bin < histogram.Length - 1; bin++)
        {
            leftGrad += histogram[bin].GradientSum;
            leftHess += histogram[bin].HessianSum;
            leftCount += histogram[bin].Count;

            // Check minimum samples constraint
            int rightCount = totalCount - leftCount;
            if (leftCount < _options.MinSamplesLeaf || rightCount < _options.MinSamplesLeaf)
            {
                continue;
            }

            double rightGrad = totalGrad - leftGrad;
            double rightHess = totalHess - leftHess;

            // Compute gain with L2 regularization
            double lambda = _options.L2Regularization;
            double gain = 0.5 * (
                (leftGrad * leftGrad) / (leftHess + lambda) +
                (rightGrad * rightGrad) / (rightHess + lambda) -
                (totalGrad * totalGrad) / (totalHess + lambda));

            if (gain > bestGain)
            {
                bestGain = gain;
                bestBin = bin;
            }
        }

        if (bestBin < 0)
        {
            return null;
        }

        // Compute left and right counts for the best split
        int bestLeftCount = 0;
        for (int bin = 0; bin <= bestBin; bin++)
        {
            bestLeftCount += histogram[bin].Count;
        }

        return new SplitInfo
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
    private void ApplySplit(HistTreeNode node, SplitInfo split, double[] residuals)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        foreach (int idx in node.SampleIndices)
        {
            int bin = _binnedData![idx, split.FeatureIndex];
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
        if (split.BinThreshold < _binThresholds![split.FeatureIndex].Length)
        {
            node.Threshold = NumOps.ToDouble(_binThresholds[split.FeatureIndex][split.BinThreshold]);
        }
        else
        {
            node.Threshold = double.MaxValue;
        }

        node.Left = new HistTreeNode
        {
            SampleIndices = leftIndices,
            Depth = node.Depth + 1
        };

        node.Right = new HistTreeNode
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
    private void SetLeafValue(HistTreeNode node, double[] residuals)
    {
        node.IsLeaf = true;

        if (node.SampleIndices.Count == 0)
        {
            node.LeafValue = 0;
            return;
        }

        double sum = 0;
        foreach (int idx in node.SampleIndices)
        {
            sum += residuals[idx];
        }

        // Optimal leaf value with L2 regularization
        double lambda = _options.L2Regularization;
        node.LeafValue = sum / (node.SampleIndices.Count + lambda);
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
    private double PredictSingleTree(HistTreeNode tree, int sampleIndex)
    {
        var node = tree;

        while (!node.IsLeaf)
        {
            int bin = _binnedData![sampleIndex, node.FeatureIndex];
            node = bin <= node.BinThreshold ? node.Left! : node.Right!;
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
    private double PredictSingleTreeFromBins(HistTreeNode tree, byte[] binnedRow)
    {
        var node = tree;

        while (!node.IsLeaf)
        {
            int bin = binnedRow[node.FeatureIndex];
            node = bin <= node.BinThreshold ? node.Left! : node.Right!;
        }

        return node.LeafValue;
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

        T sum = NumOps.Zero;
        foreach (var importance in _featureImportances)
        {
            sum = NumOps.Add(sum, importance);
        }

        if (NumOps.ToDouble(sum) > 0)
        {
            for (int i = 0; i < _featureImportances.Length; i++)
            {
                _featureImportances[i] = NumOps.Divide(_featureImportances[i], sum);
            }
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
        writer.Write(node.LeafValue);
        writer.Write(node.FeatureIndex);
        writer.Write(node.BinThreshold);
        writer.Write(node.Threshold);
        writer.Write(node.Depth);

        if (!node.IsLeaf)
        {
            SerializeTree(writer, node.Left!);
            SerializeTree(writer, node.Right!);
        }
    }

    /// <summary>
    /// Deserializes a tree node recursively.
    /// </summary>
    private HistTreeNode DeserializeTree(BinaryReader reader)
    {
        var node = new HistTreeNode
        {
            IsLeaf = reader.ReadBoolean(),
            LeafValue = reader.ReadDouble(),
            FeatureIndex = reader.ReadInt32(),
            BinThreshold = reader.ReadInt32(),
            Threshold = reader.ReadDouble(),
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
            leafTensor[0, 0] = NumOps.FromDouble(node.LeafValue);
            return TensorOperations<T>.Constant(leafTensor, $"leaf_{node.GetHashCode()}");
        }

        // Get the feature value: x[:, featureIndex]
        var featureSliceTensor = new Tensor<T>(new int[] { 1, 1 });
        var featureSliceNode = TensorOperations<T>.Slice(inputNode, 0, node.FeatureIndex, node.FeatureIndex + 1);

        // Create threshold constant
        var thresholdTensor = new Tensor<T>(new int[] { 1, 1 });
        thresholdTensor[0, 0] = NumOps.FromDouble(node.Threshold);
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
        var leftOutput = ExportSoftTreeNode(inputNode, node.Left!, temperature);
        var rightOutput = ExportSoftTreeNode(inputNode, node.Right!, temperature);

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
        public double LeafValue { get; set; }
        public int FeatureIndex { get; set; }
        public int BinThreshold { get; set; }
        public double Threshold { get; set; }
        public HistTreeNode? Left { get; set; }
        public HistTreeNode? Right { get; set; }
        public List<int> SampleIndices { get; set; } = [];
        public int Depth { get; set; }
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
    private class HistogramBin
    {
        public double GradientSum { get; set; }
        public double HessianSum { get; set; }
        public int Count { get; set; }
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
        public double Gain { get; set; }
        public int LeftCount { get; set; }
        public int RightCount { get; set; }
    }

    #endregion
}
