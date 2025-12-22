namespace AiDotNet.Regression;

/// <summary>
/// Implements a Gradient Boosting Regression model, which combines multiple decision trees
/// sequentially to create a powerful ensemble that learns from the errors of previous trees.
/// </summary>
/// <remarks>
/// <para>
/// Gradient Boosting is an ensemble technique that builds decision trees sequentially, with each tree
/// correcting the errors made by the previous trees. The model starts with a simple prediction (typically
/// the mean of the target values) and iteratively adds trees that predict the residuals (errors) of the
/// current ensemble. These predictions are added to the ensemble with a learning rate that controls the
/// contribution of each tree, helping to prevent overfitting.
/// </para>
/// <para><b>For Beginners:</b> Gradient Boosting is like having a team of experts who learn from each other's mistakes.
/// 
/// Imagine you're trying to predict house prices:
/// - You start with a simple guess (the average price of all houses)
/// - You build a decision tree to predict where your guess was wrong
/// - You adjust your prediction a little bit based on this tree
/// - You build another tree to predict where you're still making mistakes
/// - You keep adding trees, each one focusing on fixing the remaining errors
/// 
/// The "gradient" part refers to how it identifies mistakes, and "boosting" means it builds trees 
/// sequentially, with each tree boosting the performance of the ensemble.
/// 
/// This approach is very powerful because:
/// - It learns complex patterns gradually
/// - It focuses its effort on the hard-to-predict cases
/// - It combines many simple models (trees) into a strong predictive model
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GradientBoostingRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Collection of decision trees that make up the ensemble.
    /// </summary>
    private List<DecisionTreeRegression<T>> _trees;

    /// <summary>
    /// The initial prediction value, typically the mean of the target values.
    /// </summary>
    private T _initialPrediction;

    /// <summary>
    /// Configuration options for the Gradient Boosting algorithm.
    /// </summary>
    private readonly GradientBoostingRegressionOptions _options;

    /// <summary>
    /// Gets the number of trees in the ensemble model.
    /// </summary>
    /// <value>
    /// The number of trees in the ensemble.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the number of decision trees in the Gradient Boosting ensemble. This is an important
    /// characteristic of the model as it represents the number of boosting stages that have been performed.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many individual decision trees are in your model.
    /// 
    /// In Gradient Boosting:
    /// - Each tree corrects errors made by all previous trees
    /// - More trees generally means better predictions (up to a point)
    /// - However, too many trees can lead to overfitting
    /// 
    /// Typical gradient boosting models might use anywhere from 50 to 1000 trees, 
    /// depending on the complexity of the problem and the depth of each tree.
    /// </para>
    /// </remarks>
    public override int NumberOfTrees => _trees.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="GradientBoostingRegression{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the Gradient Boosting algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Gradient Boosting Regression model with the specified options and regularization
    /// strategy. If no options are provided, default values are used. If no regularization is specified, no regularization
    /// is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This is how you create a new Gradient Boosting model.
    /// 
    /// When creating a model, you can specify:
    /// - Options: Controls how many trees to build, how complex each tree can be, and how quickly the model learns
    /// - Regularization: Helps prevent the model from becoming too specialized to the training data
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// Example:
    /// ```csharp
    /// // Create a Gradient Boosting model with default settings
    /// var gbr = new GradientBoostingRegression&lt;double&gt;();
    /// 
    /// // Create a model with custom options
    /// var options = new GradientBoostingRegressionOptions { 
    ///     NumberOfTrees = 100,
    ///     LearningRate = 0.1,
    ///     MaxDepth = 3
    /// };
    /// var customGbr = new GradientBoostingRegression&lt;double&gt;(options);
    /// ```
    /// </para>
    /// </remarks>
    public GradientBoostingRegression(GradientBoostingRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new();
        _trees = new List<DecisionTreeRegression<T>>();
        _initialPrediction = NumOps.Zero;
    }

    /// <summary>
    /// Asynchronously trains the Gradient Boosting Regression model using the provided input features and target values.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the Gradient Boosting Regression model by first calculating an initial prediction (typically
    /// the mean of the target values), and then sequentially building decision trees that predict the residuals
    /// (errors) of the current ensemble. The trees are built in parallel to improve training efficiency, but the
    /// sequential nature of the algorithm is maintained by updating the residuals after each tree is built.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the model how to make predictions using your data.
    /// 
    /// The training process works like this:
    /// 1. Start with a simple prediction (the average of all target values)
    /// 2. Calculate how wrong this prediction is for each training example (the "residuals")
    /// 3. Build a decision tree that tries to predict these residuals
    /// 4. Add this tree's predictions (scaled by the learning rate) to the current model
    /// 5. Update the residuals based on the new predictions
    /// 6. Repeat steps 3-5 until you've built the desired number of trees
    /// 
    /// The "Async" in the name means this method can run without blocking other operations in your program,
    /// and it uses parallel processing to build trees more quickly when possible.
    /// 
    /// Example:
    /// ```csharp
    /// // Train the model
    /// await gbr.TrainAsync(features, targets);
    /// ```
    /// </para>
    /// </remarks>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        // Apply regularization to the feature matrix
        x = Regularization.Regularize(x);

        _initialPrediction = NumOps.Divide(y.Sum(), NumOps.FromDouble(y.Length)); // Mean of y
        var residuals = y.Subtract(Vector<T>.CreateDefault(y.Length, _initialPrediction));

        FeatureImportances = new Vector<T>(x.Columns);

        var treeTasks = Enumerable.Range(0, _options.NumberOfTrees).Select(_ => Task.Run(() =>
        {
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions
            {
                MaxDepth = _options.MaxDepth,
                MinSamplesSplit = _options.MinSamplesSplit,
                MaxFeatures = _options.MaxFeatures,
                SplitCriterion = _options.SplitCriterion,
                Seed = Random.Next()
            });

            // Subsample the data if SubsampleRatio < 1
            Matrix<T> xSubsample = x;
            Vector<T> ySubsample = residuals;

            if (_options.SubsampleRatio < 1)
            {
                int subsampleSize = (int)(x.Rows * _options.SubsampleRatio);
                int[] sampleIndices = SamplingHelper.SampleWithoutReplacement(x.Rows, subsampleSize);
                xSubsample = x.GetRows(sampleIndices);
                ySubsample = residuals.GetElements(sampleIndices);
            }

            tree.Train(xSubsample, ySubsample);

            // Update residuals
            var predictions = tree.Predict(x);
            for (int i = 0; i < residuals.Length; i++)
            {
                residuals[i] = NumOps.Subtract(residuals[i], NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), predictions[i]));
            }

            return tree;
        }));

        _trees = await ParallelProcessingHelper.ProcessTasksInParallel(treeTasks);

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Asynchronously predicts target values for the provided input features using the trained Gradient Boosting model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A task that returns a vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This method predicts target values for new input data by combining the initial prediction with the
    /// weighted contributions of all trees in the ensemble. The predictions from each tree are scaled by the
    /// learning rate before being added to the ensemble prediction. The method uses parallel processing to
    /// generate tree predictions efficiently.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// The prediction process works like this:
    /// 1. Start with the initial prediction (the average of all target values in the training data)
    /// 2. For each tree in the model:
    ///    - Get the tree's prediction
    ///    - Scale it by the learning rate (to control how much influence each tree has)
    ///    - Add it to the running total
    /// 3. The final prediction is the sum of the initial prediction plus all the scaled tree predictions
    /// 
    /// The "Async" in the name means this method returns a Task, allowing your program to do other things
    /// while waiting for predictions to complete. It also uses parallel processing to get predictions from 
    /// multiple trees simultaneously, making it faster.
    /// 
    /// Example:
    /// ```csharp
    /// // Make predictions
    /// var predictions = await gbr.PredictAsync(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        // Apply regularization to the input matrix
        input = Regularization.Regularize(input);

        var predictions = Vector<T>.CreateDefault(input.Rows, _initialPrediction);

        var treePredictions = await ParallelProcessingHelper.ProcessTasksInParallel(
            _trees.Select(tree => Task.Run(() => tree.Predict(input))));

        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < _trees.Count; j++)
            {
                predictions[i] = NumOps.Add(predictions[i], NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), treePredictions[j][i]));
            }
        }

        // Apply regularization to the final predictions
        predictions = Regularization.Regularize(predictions);

        return predictions;
    }

    /// <summary>
    /// Calculates the importance scores for all features used in the model.
    /// </summary>
    /// <param name="featureCount">The number of features in the model.</param>
    /// <returns>A task representing the asynchronous calculation operation.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the importance of each feature in the Gradient Boosting model by aggregating
    /// the importance scores across all trees in the ensemble. The importance scores are normalized to sum to 1,
    /// making it easier to compare the relative importance of different features.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out which input features matter most for predictions.
    /// 
    /// Feature importance helps you understand:
    /// - Which variables have the biggest impact on your predictions
    /// - Which features might be redundant or irrelevant
    /// - What the model is focusing on when making decisions
    /// 
    /// The calculation works by:
    /// 1. Getting the importance scores from each individual tree
    /// 2. Adding up these scores across all trees for each feature
    /// 3. Normalizing the scores so they sum to 1 (making them easier to compare)
    /// 
    /// This information can help you interpret the model and potentially simplify future models
    /// by focusing on the most important features.
    /// </para>
    /// </remarks>
    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[featureCount];

        // Calculate importances in parallel for each tree
        var importanceTasks = _trees.Select(tree => Task.Run(() =>
        {
            var treeImportances = new T[featureCount];
            for (int i = 0; i < featureCount; i++)
            {
                treeImportances[i] = tree.FeatureImportances[i];
            }
            return treeImportances;
        }));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks);

        // Aggregate importances
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = allImportances.Aggregate(NumOps.Zero, (acc, treeImportance) => NumOps.Add(acc, treeImportance[i]));
        }

        // Normalize feature importances
        T sum = importances.Aggregate(NumOps.Zero, NumOps.Add);
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = NumOps.Divide(importances[i], sum);
        }

        FeatureImportances = new Vector<T>(importances);
    }

    /// <summary>
    /// Gets metadata about the Gradient Boosting Regression model and its configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type and configuration options such as
    /// the number of trees, maximum tree depth, learning rate, and subsampling ratio. This information can be
    /// useful for model management, comparison, and documentation purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about your Gradient Boosting model.
    /// 
    /// The metadata includes:
    /// - The type of model (Gradient Boosting)
    /// - How many trees are in the ensemble
    /// - How deep each tree is allowed to grow
    /// - The learning rate (how quickly the model incorporates new trees)
    /// - Subsampling ratio (what fraction of the data is used for each tree)
    /// - Other configuration settings
    /// 
    /// This information is helpful when:
    /// - Comparing different models
    /// - Documenting your model's configuration
    /// - Troubleshooting model performance
    /// - Replicating your results
    /// 
    /// Example:
    /// ```csharp
    /// var metadata = gbr.GetModelMetadata();
    /// Console.WriteLine($"Model type: {metadata.ModelType}");
    /// Console.WriteLine($"Number of trees: {metadata.AdditionalInfo["NumberOfTrees"]}");
    /// Console.WriteLine($"Learning rate: {metadata.AdditionalInfo["LearningRate"]}");
    /// ```
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.GradientBoosting,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfTrees", _options.NumberOfTrees },
                { "MaxDepth", _options.MaxDepth },
                { "MinSamplesSplit", _options.MinSamplesSplit },
                { "LearningRate", _options.LearningRate },
                { "SubsampleRatio", _options.SubsampleRatio },
                { "MaxFeatures", _options.MaxFeatures }
            }
        };
    }

    /// <summary>
    /// Serializes the Gradient Boosting Regression model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the Gradient Boosting Regression model into a byte array that can be stored in a file,
    /// database, or transmitted over a network. The serialized data includes the base class data, model-specific
    /// options, the initial prediction, and all the trees in the ensemble.
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
    /// - All the model's settings (like number of trees and learning rate)
    /// - The initial prediction (the starting point for all predictions)
    /// - Every individual decision tree in the ensemble
    /// 
    /// Because Gradient Boosting models contain multiple trees, the serialized data
    /// can be quite large for complex models.
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = gbr.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("gradientBoosting.model", modelData);
    /// ```
    /// </para>
    /// </remarks>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        // Serialize base class data
        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Serialize GradientBoostingRegression specific data
        writer.Write(_options.NumberOfTrees);
        writer.Write(_options.LearningRate);
        writer.Write(_options.SubsampleRatio);
        writer.Write(Convert.ToDouble(_initialPrediction));

        // Serialize trees
        writer.Write(_trees.Count);
        foreach (var tree in _trees)
        {
            byte[] treeData = tree.Serialize();
            writer.Write(treeData.Length);
            writer.Write(treeData);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Loads a previously serialized Gradient Boosting Regression model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a Gradient Boosting Regression model from a byte array that was previously created
    /// using the Serialize method. It restores the base class data, model-specific options, the initial prediction,
    /// and all the trees in the ensemble, allowing the model to be used for predictions without retraining.
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
    /// - The initial prediction is recovered
    /// - All the individual trees are reconstructed
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("gradientBoosting.model");
    /// 
    /// // Deserialize the model
    /// var gbr = new GradientBoostingRegression&lt;double&gt;();
    /// gbr.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = await gbr.PredictAsync(newFeatures);
    /// ```
    /// </para>
    /// </remarks>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);
        // Deserialize base class data
        int baseDataLength = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseDataLength);
        base.Deserialize(baseData);

        // Deserialize GradientBoostingRegression specific data
        _options.NumberOfTrees = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.SubsampleRatio = reader.ReadDouble();
        _initialPrediction = NumOps.FromDouble(reader.ReadDouble());

        // Deserialize trees
        int treeCount = reader.ReadInt32();
        _trees = new List<DecisionTreeRegression<T>>(treeCount);
        for (int i = 0; i < treeCount; i++)
        {
            int treeDataLength = reader.ReadInt32();
            byte[] treeData = reader.ReadBytes(treeDataLength);
            var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions());
            tree.Deserialize(treeData);
            _trees.Add(tree);
        }
    }

    /// <summary>
    /// Creates a new instance of the gradient boosting regression model with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="GradientBoostingRegression{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new gradient boosting regression model that has the same configuration 
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
        return new GradientBoostingRegression<T>(_options, Regularization);
    }

    #region IJitCompilable Implementation Override

    /// <summary>
    /// Gets whether this Gradient Boosting model supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when soft tree mode is enabled and trees have been trained;
    /// <c>false</c> otherwise.
    /// </value>
    /// <remarks>
    /// <para>
    /// Gradient Boosting supports JIT compilation when soft tree mode is enabled. In soft mode,
    /// each tree in the ensemble uses sigmoid-based soft gating instead of hard if-then splits,
    /// making the entire sequential ensemble differentiable.
    /// </para>
    /// <para>
    /// The computation graph follows the gradient boosting formula:
    /// <code>prediction = initial_prediction + learning_rate × Σ tree_i(input)</code>
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation is available when soft tree mode is enabled.
    ///
    /// In soft tree mode:
    /// - Each tree in the boosted ensemble uses smooth transitions
    /// - The sequential ensemble can be exported as a single computation graph
    /// - The learning rate and initial prediction are embedded in the graph
    ///
    /// This gives you the benefits of gradient boosting with JIT-compiled speed.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        UseSoftTree && _trees.Count > 0;

    /// <summary>
    /// Exports the Gradient Boosting model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The root node of the exported computation graph.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when soft tree mode is not enabled.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model has not been trained (no trees).
    /// </exception>
    /// <remarks>
    /// <para>
    /// When soft tree mode is enabled, this exports the entire Gradient Boosting ensemble as a
    /// differentiable computation graph. The graph follows the formula:
    /// <code>output = initial_prediction + learning_rate × (tree1 + tree2 + ... + treeN)</code>
    /// where each tree uses soft split operations.
    /// </para>
    /// <para><b>For Beginners:</b> This exports the gradient boosted ensemble as a computation graph.
    ///
    /// Unlike Random Forest (which averages tree outputs), Gradient Boosting:
    /// - Starts with an initial prediction (mean of training targets)
    /// - Adds contributions from each tree scaled by the learning rate
    /// - Each tree predicts "residuals" (errors from previous trees)
    ///
    /// The exported graph combines all these elements into optimized code.
    /// </para>
    /// </remarks>
    public override AiDotNet.Autodiff.ComputationNode<T> ExportComputationGraph(
        List<AiDotNet.Autodiff.ComputationNode<T>> inputNodes)
    {
        if (!UseSoftTree)
        {
            throw new NotSupportedException(
                "Gradient Boosting does not support JIT compilation in hard tree mode because " +
                "decision trees use discrete branching logic.\n\n" +
                "To enable JIT compilation, set UseSoftTree = true to use soft (differentiable) " +
                "decision trees with sigmoid-based gating.");
        }

        if (_trees.Count == 0)
        {
            throw new InvalidOperationException(
                "Cannot export computation graph: the Gradient Boosting model has not been trained. " +
                "Call Train() or TrainAsync() first to build the trees.");
        }

        // Ensure all trees have soft mode enabled
        foreach (var tree in _trees)
        {
            tree.UseSoftTree = true;
            tree.SoftTreeTemperature = SoftTreeTemperature;
        }

        // Create initial prediction constant
        var initialTensor = new Tensor<T>(new[] { 1 });
        initialTensor[0] = _initialPrediction;
        var initialNode = TensorOperations<T>.Constant(initialTensor, "initial_prediction");

        // Create learning rate constant
        var lrTensor = new Tensor<T>(new[] { 1 });
        lrTensor[0] = NumOps.FromDouble(_options.LearningRate);
        var learningRateNode = TensorOperations<T>.Constant(lrTensor, "learning_rate");

        // Export first tree to get input node
        var tempInputNodes = new List<AiDotNet.Autodiff.ComputationNode<T>>();
        var firstTreeGraph = _trees[0].ExportComputationGraph(tempInputNodes);

        if (tempInputNodes.Count > 0)
        {
            inputNodes.Add(tempInputNodes[0]);
        }

        // Sum all tree outputs
        var treeSumNode = firstTreeGraph;
        for (int i = 1; i < _trees.Count; i++)
        {
            var treeInputNodes = new List<AiDotNet.Autodiff.ComputationNode<T>>();
            var treeGraph = _trees[i].ExportComputationGraph(treeInputNodes);
            treeSumNode = TensorOperations<T>.Add(treeSumNode, treeGraph);
        }

        // Scale by learning rate: learning_rate * sum_of_trees
        var scaledTreesNode = TensorOperations<T>.ElementwiseMultiply(learningRateNode, treeSumNode);

        // Final prediction: initial_prediction + learning_rate * sum_of_trees
        return TensorOperations<T>.Add(initialNode, scaledTreesNode);
    }

    #endregion
}
