namespace AiDotNet.Regression;

/// <summary>
/// Implements an Extremely Randomized Trees regression model, which is an ensemble method that uses multiple decision trees
/// with additional randomization for improved prediction accuracy and reduced overfitting.
/// </summary>
/// <remarks>
/// <para>
/// Extremely Randomized Trees (also known as Extra Trees) is an ensemble method that builds multiple decision trees
/// and averages their predictions to improve accuracy and reduce overfitting. Unlike Random Forests, which use the best
/// split for each feature, Extra Trees selects random thresholds for each feature and chooses the best among these
/// random thresholds, adding an additional layer of randomization that can further reduce variance.
/// </para>
/// <para><b>For Beginners:</b> This model works like a committee of decision trees that vote on predictions.
/// 
/// While a single decision tree might make mistakes due to its specific structure, 
/// a group of different trees can work together to make more reliable predictions:
/// 
/// - Each tree sees a random subset of the training data
/// - Each tree uses random thresholds for making decisions
/// - The final prediction is the average of all individual tree predictions
/// 
/// The key advantage is that by adding extra randomness in how the trees are built,
/// the model avoids "memorizing" the training data and becomes better at generalizing
/// to new data. This is similar to how asking many different people for their opinion
/// often leads to better decisions than relying on just one person.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ExtremelyRandomizedTreesRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// The configuration options for the Extremely Randomized Trees algorithm.
    /// </summary>
    private readonly ExtremelyRandomizedTreesRegressionOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Collection of individual decision trees that make up the ensemble.
    /// </summary>
    private List<DecisionTreeRegression<T>> _trees;

    /// <summary>
    /// Random number generator used for bootstrapping and feature selection.
    /// </summary>
    private Random _random;

    /// <summary>
    /// Gets the number of trees in the ensemble model.
    /// </summary>
    /// <value>
    /// The number of decision trees used in the ensemble.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the number of individual decision trees that make up the Extremely Randomized Trees
    /// ensemble. A larger number of trees typically improves prediction accuracy but increases training and
    /// prediction time.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many individual decision trees work together in this model.
    /// 
    /// Think of it as the size of your "committee of experts":
    /// - A small number (10-50): Faster to train but might be less accurate
    /// - A medium number (50-200): Good balance of accuracy and speed
    /// - A large number (200+): More accurate but slower to train and use
    /// 
    /// Unlike a single decision tree model, which has just one tree, ensemble methods like
    /// Extremely Randomized Trees use multiple trees working together to make better predictions.
    /// The final prediction is the average of what all these trees predict.
    /// </para>
    /// </remarks>
    public override int NumberOfTrees => _options.NumberOfTrees;

    /// <summary>
    /// Gets the maximum depth of the decision trees in the ensemble.
    /// </summary>
    /// <value>
    /// The maximum number of levels in each tree, from the root to the deepest leaf.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the maximum depth of the individual decision trees in the ensemble. This is one of the
    /// most important parameters for controlling the complexity of the model. Deeper trees can capture more complex
    /// patterns but are more prone to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you how many levels of questions each tree in the ensemble can ask.
    /// 
    /// Just like with a single decision tree:
    /// - A smaller MaxDepth (e.g., 3-5): Creates simpler trees that might miss some patterns but are less likely to memorize the training data
    /// - A larger MaxDepth (e.g., 10-20): Creates more complex trees that can capture detailed patterns but might learn noise in the training data
    /// 
    /// One advantage of ensemble methods is that you can often use slightly deeper trees than you would with a single decision tree,
    /// because the averaging of multiple trees helps prevent overfitting.
    /// </para>
    /// </remarks>
    public override int MaxDepth => _options.MaxDepth;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExtremelyRandomizedTreesRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the Extremely Randomized Trees algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Extremely Randomized Trees regression model with the specified options and
    /// regularization strategy. The options control parameters such as the number of trees, maximum tree depth,
    /// and feature selection. If no regularization is specified, no regularization is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new Extremely Randomized Trees model.
    /// 
    /// You need to provide:
    /// - options: Controls how the ensemble works (like how many trees to use, how deep each tree can be)
    /// - regularization: Optional setting to help prevent the model from "memorizing" the training data
    /// 
    /// Example:
    /// ```csharp
    /// // Create options for an Extra Trees model with 100 trees
    /// var options = new ExtremelyRandomizedTreesRegressionOptions {
    ///     NumberOfTrees = 100,
    ///     MaxDepth = 10
    /// };
    /// 
    /// // Create the model
    /// var extraTrees = new ExtremelyRandomizedTreesRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public ExtremelyRandomizedTreesRegression(ExtremelyRandomizedTreesRegressionOptions options, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options;
        _trees = new List<DecisionTreeRegression<T>>();
        _random = _options.Seed.HasValue ? RandomHelper.CreateSeededRandom(_options.Seed.Value) : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Asynchronously trains the Extremely Randomized Trees model using the provided input features and target values.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds the Extremely Randomized Trees ensemble by training multiple decision trees in parallel.
    /// Each tree is trained on a randomly sampled subset of the training data (bootstrap sampling). The trees
    /// are built with additional randomization in feature selection and threshold determination. The level of
    /// parallelism can be controlled through the options.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model how to make predictions using your data.
    /// 
    /// During training:
    /// 1. The model creates multiple decision trees (as specified in NumberOfTrees)
    /// 2. Each tree is given a random sample of your training data (some examples may be repeated, others left out)
    /// 3. Each tree learns independently but with extra randomness in how it makes decisions
    /// 4. The trees are trained in parallel to save time (using multiple CPU cores)
    /// 
    /// The "Async" in the name means this method can run without blocking other operations in your program,
    /// which is especially helpful when training large models that take significant time.
    /// 
    /// Example:
    /// ```csharp
    /// // Train the model
    /// await extraTrees.TrainAsync(features, targets);
    /// ```
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
    }

    /// <summary>
    /// Asynchronously predicts target values for the provided input features using the trained ensemble model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A task that returns a vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data by averaging the predictions from all decision trees
    /// in the ensemble. Each tree's prediction is computed in parallel, and the results are then averaged to form
    /// the final prediction. Any specified regularization is applied to both the input data and the predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// The prediction process:
    /// 1. Each individual tree in the ensemble makes its own prediction
    /// 2. These predictions happen in parallel to save time
    /// 3. The final prediction is the average of all the individual tree predictions
    /// 
    /// For example, if you're predicting house prices and have 100 trees:
    /// - Tree 1 predicts: $250,000
    /// - Tree 2 predicts: $275,000
    /// - ...
    /// - Tree 100 predicts: $260,000
    /// - Final prediction: Average of all 100 predictions
    /// 
    /// The "Async" in the name means this method returns a Task, allowing your program to do other things
    /// while waiting for predictions to complete.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = await extraTrees.PredictAsync(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // Note: Tree-based methods handle regularization through tree structure parameters
        // (MaxDepth, MinSamplesSplit, etc.), not through data transformation
        var predictionTasks = _trees.Select(tree => new Func<Vector<T>>(() => tree.Predict(input)));
        var predictions = await ParallelProcessingHelper.ProcessTasksInParallel(predictionTasks, _options.MaxDegreeOfParallelism);

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
    /// Creates a random sample of the training data with replacement (bootstrap sampling).
    /// </summary>
    /// <param name="x">The feature matrix to sample from.</param>
    /// <param name="y">The target vector to sample from.</param>
    /// <returns>A tuple containing the sampled feature matrix and target vector.</returns>
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
    /// Calculates the average feature importances across all trees in the ensemble.
    /// </summary>
    /// <param name="numFeatures">The number of features in the model.</param>
    /// <returns>A task representing the asynchronous calculation operation.</returns>
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
    /// Gets metadata about the Extremely Randomized Trees model and its configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, number of trees, maximum tree depth,
    /// and feature importances if available. This information can be useful for model management, comparison,
    /// and documentation purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about your Extremely Randomized Trees model.
    /// 
    /// The metadata includes:
    /// - The type of model (Extremely Randomized Trees)
    /// - How many trees are in the ensemble
    /// - Maximum depth of each tree
    /// - How important each feature is for making predictions (if available)
    /// 
    /// This information is helpful when:
    /// - Comparing different models
    /// - Documenting your model's configuration
    /// - Troubleshooting model performance
    /// - Understanding which features have the biggest impact on predictions
    /// 
    /// Example:
    /// ```csharp
    /// var metadata = extraTrees.GetModelMetadata();
    /// Console.WriteLine($"Model type: {metadata.ModelType}");
    /// Console.WriteLine($"Number of trees: {metadata.AdditionalInfo["NumberOfTrees"]}");
    /// ```
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.ExtremelyRandomizedTrees,
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
    /// Serializes the Extremely Randomized Trees model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the Extremely Randomized Trees model into a byte array that can be stored in a file,
    /// database, or transmitted over a network. The serialized data includes the model's configuration options,
    /// feature importances, and all individual decision trees in the ensemble.
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
    /// - All the model's settings (like number of trees and maximum depth)
    /// - The importance of each feature
    /// - Every individual decision tree in the ensemble
    /// 
    /// Because Extremely Randomized Trees models contain multiple trees, the serialized data
    /// can be quite large compared to a single decision tree model.
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = extraTrees.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("extraTrees.model", modelData);
    /// ```
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
    /// Loads a previously serialized Extremely Randomized Trees model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs an Extremely Randomized Trees model from a byte array that was previously created
    /// using the Serialize method. It restores the model's configuration options, feature importances, and all
    /// individual decision trees in the ensemble, allowing the model to be used for predictions without retraining.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved model from a sequence of bytes.
    /// 
    /// Deserialization allows you to:
    /// - Load a model that was saved earlier
    /// - Use a model without having to retrain it
    /// - Share models between different applications
    /// 
    /// When you deserialize an Extremely Randomized Trees model:
    /// - All settings are restored
    /// - Feature importances are recovered
    /// - All individual trees in the ensemble are reconstructed
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("extraTrees.model");
    /// 
    /// // Deserialize the model
    /// var extraTrees = new ExtremelyRandomizedTreesRegression&lt;double&gt;(options);
    /// extraTrees.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = await extraTrees.PredictAsync(newFeatures);
    /// ```
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
    /// Creates a new instance of the extremely randomized trees regression model with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="ExtremelyRandomizedTreesRegression{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new extremely randomized trees regression model that has the same configuration 
    /// as the current instance. It's used for model persistence, cloning, and transferring the model's 
    /// configuration to new instances.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like making a blueprint copy of your model that can be used to:
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
        return new ExtremelyRandomizedTreesRegression<T>(_options, Regularization);
    }

    #region IJitCompilable Implementation Override

    /// <summary>
    /// Gets whether this Extremely Randomized Trees model supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when soft tree mode is enabled and trees have been trained;
    /// <c>false</c> otherwise.
    /// </value>
    /// <remarks>
    /// <para>
    /// Extremely Randomized Trees supports JIT compilation when soft tree mode is enabled.
    /// In soft mode, each tree in the ensemble uses sigmoid-based soft gating instead of
    /// hard if-then splits, making the entire ensemble differentiable.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation is available when soft tree mode is enabled.
    ///
    /// In soft tree mode:
    /// - Each tree in the Extra Trees ensemble uses smooth transitions
    /// - All trees can be exported as a single computation graph
    /// - The final prediction averages all tree outputs
    ///
    /// This gives you the benefits of extra randomization with JIT-compiled speed.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        UseSoftTree && _trees.Count > 0;

    /// <summary>
    /// Exports the Extremely Randomized Trees model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The root node of the exported computation graph.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when soft tree mode is not enabled.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the forest has not been trained (no trees).
    /// </exception>
    /// <remarks>
    /// <para>
    /// When soft tree mode is enabled, this exports the entire Extra Trees ensemble as a
    /// differentiable computation graph. Each tree is exported individually, and their
    /// outputs are averaged to produce the final prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This exports the Extra Trees ensemble as a computation graph.
    ///
    /// Extra Trees, like Random Forest, averages predictions from all trees.
    /// The main difference is how trees are built (random thresholds instead of optimal),
    /// but for JIT compilation the averaging formula is the same.
    /// </para>
    /// </remarks>
    public override AiDotNet.Autodiff.ComputationNode<T> ExportComputationGraph(
        List<AiDotNet.Autodiff.ComputationNode<T>> inputNodes)
    {
        if (!UseSoftTree)
        {
            throw new NotSupportedException(
                "Extremely Randomized Trees does not support JIT compilation in hard tree mode because " +
                "decision trees use discrete branching logic.\n\n" +
                "To enable JIT compilation, set UseSoftTree = true to use soft (differentiable) " +
                "decision trees with sigmoid-based gating.");
        }

        if (_trees.Count == 0)
        {
            throw new InvalidOperationException(
                "Cannot export computation graph: the Extra Trees model has not been trained. " +
                "Call Train() or TrainAsync() first to build the trees.");
        }

        // Ensure all trees have soft mode enabled
        foreach (var tree in _trees)
        {
            tree.UseSoftTree = true;
            tree.SoftTreeTemperature = SoftTreeTemperature;
        }

        // Export first tree to get input node
        var tempInputNodes = new List<AiDotNet.Autodiff.ComputationNode<T>>();
        var firstTreeGraph = _trees[0].ExportComputationGraph(tempInputNodes);

        if (tempInputNodes.Count > 0)
        {
            inputNodes.Add(tempInputNodes[0]);
        }

        // If there's only one tree, return its graph directly
        if (_trees.Count == 1)
        {
            return firstTreeGraph;
        }

        // Sum all tree outputs
        var sumNode = firstTreeGraph;
        for (int i = 1; i < _trees.Count; i++)
        {
            var treeInputNodes = new List<AiDotNet.Autodiff.ComputationNode<T>>();
            var treeGraph = _trees[i].ExportComputationGraph(treeInputNodes);
            sumNode = TensorOperations<T>.Add(sumNode, treeGraph);
        }

        // Divide by number of trees to get average
        var numTreesTensor = new Tensor<T>(new[] { 1 });
        numTreesTensor[0] = NumOps.FromDouble(_trees.Count);
        var numTreesNode = TensorOperations<T>.Constant(numTreesTensor, "num_trees");

        return TensorOperations<T>.Divide(sumNode, numTreesNode);
    }

    #endregion
}
