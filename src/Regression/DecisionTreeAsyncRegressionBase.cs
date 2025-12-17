using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Regression;

/// <summary>
/// Represents an abstract base class for asynchronous decision tree regression models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This class provides a foundation for implementing decision tree regression models that can be trained
/// and used for predictions asynchronously. It includes methods for training, prediction, serialization,
/// and deserialization of the model.
/// </para>
/// <para><b>For Beginners:</b> A decision tree is a type of machine learning model that makes predictions
/// by following a series of yes/no questions about the input data. It's like a flowchart that helps the
/// computer decide what prediction to make.
/// 
/// For example, if you're trying to predict if it will rain:
/// - Is the humidity high? If yes, go to next question. If no, predict no rain.
/// - Are there clouds? If yes, predict rain. If no, predict no rain.
/// 
/// This class provides the basic structure for building these types of models, but with more complex
/// questions and answers based on numerical data.
/// </para>
/// </remarks>
public abstract class AsyncDecisionTreeRegressionBase<T> : IAsyncTreeBasedModel<T>
{
    /// <summary>
    /// Gets the numeric operations for the type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets or sets the root node of the decision tree.
    /// </summary>
    protected DecisionTreeNode<T>? Root;

    /// <summary>
    /// Gets the options used to configure the decision tree.
    /// </summary>
    protected DecisionTreeOptions Options { get; private set; }

    /// <summary>
    /// Gets the regularization method used to prevent overfitting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regularization is a technique used to prevent the model from becoming
    /// too complex and fitting the training data too closely. This helps the model generalize better
    /// to new, unseen data.
    ///
    /// Think of it like learning to ride a bike:
    /// - Without regularization, you might only learn to ride on one specific path.
    /// - With regularization, you learn general bike-riding skills that work on many different paths.
    /// </para>
    /// </remarks>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

    /// <summary>
    /// Gets the default loss function for this async tree-based regression model.
    /// </summary>
    /// <value>
    /// The loss function used for gradient computation.
    /// </value>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets the maximum depth of the decision tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The maximum depth is the longest path from the root of the tree to a leaf.
    /// A deeper tree can capture more complex patterns but may also overfit to the training data.
    /// 
    /// Imagine a game of "20 Questions":
    /// - A shallow tree is like having only 5 questions (less detailed, but quicker).
    /// - A deep tree is like having all 20 questions (more detailed, but might be too specific).
    /// </para>
    /// </remarks>
    public virtual int MaxDepth => Options.MaxDepth;

    /// <summary>
    /// Gets or sets the importance of each feature in making predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feature importance tells you how much each input variable (feature)
    /// contributes to the model's predictions. Higher values mean that feature is more important.
    /// 
    /// For example, if predicting house prices:
    /// - Location might have high importance (big impact on price)
    /// - Wall color might have low importance (small impact on price)
    /// </para>
    /// </remarks>
    public Vector<T> FeatureImportances { get; protected set; }

    /// <summary>
    /// Gets the number of trees in the model. For a single decision tree, this is always 1.
    /// </summary>
    public virtual int NumberOfTrees => 1;

    /// <summary>
    /// Random number generator used for tree building and sampling.
    /// </summary>
    protected Random Random => new(Options.Seed ?? Environment.TickCount);

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    /// <value>
    /// An array of feature names. If not set, feature indices will be used as names.
    /// </value>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Initializes a new instance of the AsyncDecisionTreeRegressionBase class.
    /// </summary>
    /// <param name="options">The options for configuring the decision tree.</param>
    /// <param name="regularization">The regularization method to use.</param>
    /// <param name="lossFunction">Loss function for gradient computation. If null, defaults to Mean Squared Error.</param>
    protected AsyncDecisionTreeRegressionBase(DecisionTreeOptions? options, IRegularization<T, Matrix<T>, Vector<T>>? regularization, ILossFunction<T>? lossFunction = null)
    {
        Options = options ?? new();
        NumOps = MathHelper.GetNumericOperations<T>();
        FeatureImportances = new Vector<T>(0);
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Asynchronously trains the decision tree model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training is the process where the model learns from the data you provide.
    /// It's like teaching the model to recognize patterns in your data.
    /// 
    /// - x: This is your input data, like the features of houses (size, location, etc.)
    /// - y: This is what you're trying to predict, like house prices
    /// 
    /// After training, the model will have learned how to use the features to predict the target values.
    /// </para>
    /// </remarks>
    public abstract Task TrainAsync(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Asynchronously makes predictions using the trained model.
    /// </summary>
    /// <param name="input">The input features matrix to make predictions for.</param>
    /// <returns>A task representing the asynchronous operation, which resolves to a vector of predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prediction is using the trained model to estimate outcomes for new data.
    /// It's like using what the model learned to make educated guesses about new situations.
    /// 
    /// For example, if you trained a model on house features and prices:
    /// - Input: Features of a new house (size, location, etc.)
    /// - Output: The model's estimate of what that house might cost
    /// </para>
    /// </remarks>
    public abstract Task<Vector<T>> PredictAsync(Matrix<T> input);

    /// <summary>
    /// Gets metadata about the trained model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Model metadata is information about the trained model itself,
    /// not about the predictions it makes. This can include things like:
    /// 
    /// - How well the model performs
    /// - How complex the model is
    /// - What settings were used to train it
    /// 
    /// It's like getting a report card for your model, showing how well it learned and what it learned.
    /// </para>
    /// </remarks>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <summary>
    /// Asynchronously calculates the importance of each feature in the model.
    /// </summary>
    /// <param name="featureCount">The number of features in the input data.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method figures out how much each input feature contributes
    /// to the model's predictions. It helps you understand which pieces of information are most
    /// useful for making accurate predictions.
    /// 
    /// For example, in predicting house prices:
    /// - It might find that location is very important
    /// - While the house's paint color is less important
    /// 
    /// This can help you focus on collecting the most relevant data for future predictions.
    /// </para>
    /// </remarks>
    protected abstract Task CalculateFeatureImportancesAsync(int featureCount);

    /// <summary>
    /// Trains the decision tree model on the provided data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target values vector.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is a synchronous wrapper around the asynchronous TrainAsync method.
    /// It does the same thing as TrainAsync, but it waits for the training to complete before moving on.
    ///
    /// Use this method when you want to train the model and wait for it to finish before doing anything else.
    /// It's like waiting for a cake to finish baking before you start decorating it.
    /// </para>
    /// </remarks>
    public void Train(Matrix<T> x, Vector<T> y)
    {
        Task.Run(() => TrainAsync(x, y)).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Makes predictions using the trained model.
    /// </summary>
    /// <param name="input">The input features matrix to make predictions for.</param>
    /// <returns>A vector of predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is a synchronous wrapper around the asynchronous PredictAsync method.
    /// It does the same thing as PredictAsync, but it waits for the predictions to be made before moving on.
    ///
    /// Use this method when you want to get predictions immediately and wait for them to be ready.
    /// It's like asking a question and waiting for the answer before you do anything else.
    /// </para>
    /// </remarks>
    public Vector<T> Predict(Matrix<T> input)
    {
        return Task.Run(() => PredictAsync(input)).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized model.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Serialization is the process of converting the model into a format
    /// that can be easily stored or transmitted. It's like packing up the model into a suitcase so you
    /// can take it with you or save it for later.
    /// 
    /// This method saves:
    /// - The model's settings (like max depth)
    /// - The importance of each feature
    /// - The structure of the decision tree
    /// 
    /// You can use this to save your trained model and load it later without having to retrain.
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize options
        writer.Write(Options.MaxDepth);
        writer.Write(Options.MinSamplesSplit);
        writer.Write(double.IsNaN(Options.MaxFeatures) ? -1 : Options.MaxFeatures);
        writer.Write(Options.Seed ?? -1);
        writer.Write((int)Options.SplitCriterion);

        // Serialize feature importances
        writer.Write(FeatureImportances.Length);
        foreach (var importance in FeatureImportances)
        {
            writer.Write(Convert.ToDouble(importance));
        }

        // Serialize tree structure
        SerializeNode(writer, Root);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model data.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deserialization is the process of reconstructing the model from
    /// saved data. It's like unpacking the model from the suitcase you packed it in earlier.
    /// 
    /// This method:
    /// - Reads the saved settings and restores them
    /// - Rebuilds the decision tree structure
    /// - Sets up the feature importances
    /// 
    /// After calling this method, your model will be ready to use for making predictions,
    /// just like it was before you serialized it.
    /// </para>
    /// </remarks>
    public virtual void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        // Deserialize options
        Options.MaxDepth = reader.ReadInt32();
        Options.MinSamplesSplit = reader.ReadInt32();
        int maxFeatures = reader.ReadInt32();
        Options.MaxFeatures = maxFeatures == -1 ? double.NaN : maxFeatures;
        int seed = reader.ReadInt32();
        Options.Seed = seed == -1 ? null : seed;
        Options.SplitCriterion = (SplitCriterion)reader.ReadInt32();

        // Deserialize feature importances
        int featureCount = reader.ReadInt32();
        var importances = new T[featureCount];
        for (int i = 0; i < featureCount; i++)
        {
            importances[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        FeatureImportances = new Vector<T>(importances);

        // Deserialize tree structure
        Root = DeserializeNode(reader);
    }

    /// <summary>
    /// Serializes a single node of the decision tree.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the serialized data to.</param>
    /// <param name="node">The node to serialize.</param>
    private void SerializeNode(BinaryWriter writer, DecisionTreeNode<T>? node)
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

        SerializeNode(writer, node.Left);
        SerializeNode(writer, node.Right);
    }

    /// <summary>
    /// Deserializes a single node of the decision tree.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the serialized data from.</param>
    /// <returns>The deserialized DecisionTreeNode, or null if the node was not present.</returns>
    private DecisionTreeNode<T>? DeserializeNode(BinaryReader reader)
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

        node.Left = DeserializeNode(reader);
        node.Right = DeserializeNode(reader);

        return node;
    }

    /// <summary>
    /// Gets the model parameters as a vector representation.
    /// </summary>
    /// <returns>A vector containing a serialized representation of the decision tree structure.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a vector representation of the asynchronous decision tree model. Decision trees
    /// have a hierarchical structure that doesn't naturally fit into a flat vector format,
    /// so this representation is a simplified encoding of the tree structure suitable for
    /// certain optimization algorithms or model comparison techniques.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the tree structure into a flat list of numbers.
    /// 
    /// Decision trees are complex structures with branches and nodes, which don't naturally fit
    /// into a simple list of parameters like linear models do. This method creates a specialized
    /// representation of the tree that can be used by certain algorithms or for model comparison.
    /// 
    /// The exact format of this representation depends on the specific implementation, but
    /// generally includes information about:
    /// - Each node's feature index (which feature it splits on)
    /// - Each node's split value (the threshold for the decision)
    /// - Each node's prediction value (for leaf nodes)
    /// - The tree structure (how nodes connect to each other)
    /// 
    /// This is primarily used by advanced algorithms and not typically needed for regular use.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        // Get the total number of nodes in the tree
        int nodeCount = CountNodes(Root);

        // For each node, we store:
        // 1. Feature index (as converted double)
        // 2. Split value 
        // 3. Prediction value
        // 4. IsLeaf flag (as converted double: 1.0 for leaf, 0.0 for non-leaf)
        // Plus we need one additional parameter for the node count
        Vector<T> parameters = new(nodeCount * 4 + 1);

        // Store the node count as the first parameter
        parameters[0] = NumOps.FromDouble(nodeCount);

        // If the tree is empty, return just the node count
        if (Root == null)
        {
            return parameters;
        }

        // Traverse the tree and store each node's parameters
        int currentIndex = 1;
        SerializeNodeToVector(Root, parameters, ref currentIndex);

        return parameters;
    }

    /// <summary>
    /// Creates a new instance of the model with the specified parameters.
    /// </summary>
    /// <param name="parameters">A vector containing a serialized representation of the decision tree structure.</param>
    /// <returns>A new model instance with the reconstructed tree structure.</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs a decision tree model from a parameter vector that was previously
    /// created using the GetParameters method. Due to the complex nature of tree structures,
    /// this reconstruction is approximate and is primarily intended for use with optimization
    /// algorithms or model comparison techniques.
    /// </para>
    /// <para><b>For Beginners:</b> This method rebuilds a decision tree from a flat list of numbers.
    /// 
    /// It takes the specialized vector representation created by GetParameters() and attempts
    /// to reconstruct a decision tree from it. This is challenging because decision trees
    /// are complex structures that don't easily convert to and from simple lists of numbers.
    /// 
    /// This method is primarily used by advanced algorithms and not typically needed for regular use.
    /// For most purposes, the Serialize and Deserialize methods provide a more reliable way to
    /// save and load tree models.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        // Create a new instance with the same options
        var newModel = CreateNewInstance();

        // If the parameter vector is empty or invalid, return the empty model
        if (parameters.Length < 1)
        {
            return newModel;
        }

        // Get the node count from the first parameter
        int nodeCount = NumOps.ToInt32(parameters[0]);

        // If there are no nodes, return the empty model
        if (nodeCount == 0)
        {
            return newModel;
        }

        // Check if the parameter vector has the expected length
        if (parameters.Length != nodeCount * 4 + 1)
        {
            throw new ArgumentException("Invalid parameter vector length");
        }

        // Reconstruct the tree from the parameter vector
        int currentIndex = 1;
        ((AsyncDecisionTreeRegressionBase<T>)newModel).Root = DeserializeNodeFromVector(parameters, ref currentIndex);

        // Assume the feature importances are already calculated and stored in the parameters
        // or recalculate them based on the reconstructed tree
        if (FeatureImportances.Length > 0)
        {
            ((AsyncDecisionTreeRegressionBase<T>)newModel).FeatureImportances = new Vector<T>(FeatureImportances);
        }

        return newModel;
    }

    /// <summary>
    /// Gets the indices of all features that are used in the decision tree.
    /// </summary>
    /// <returns>An enumerable collection of indices for features used in the tree.</returns>
    /// <remarks>
    /// <para>
    /// This method identifies all features that are used as split criteria in the decision tree.
    /// Features that don't appear in any decision node are not considered active.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you which input features are actually used in the tree.
    /// 
    /// Decision trees often don't use all available features - they select the most informative ones
    /// during training. This method returns the positions (indices) of features that are actually
    /// used in decision nodes throughout the tree.
    /// 
    /// For example, if your dataset has 10 features but the tree only uses features at positions
    /// 2, 5, and 7, this method would return [2, 5, 7].
    /// 
    /// This is useful for:
    /// - Feature selection (identifying which features matter)
    /// - Model simplification (removing unused features)
    /// - Understanding which inputs actually affect the prediction
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        var activeFeatures = new HashSet<int>();
        CollectActiveFeatures(Root, activeFeatures);
        return activeFeatures;
    }

    /// <summary>
    /// Determines whether a specific feature is used in the decision tree.
    /// </summary>
    /// <param name="featureIndex">The zero-based index of the feature to check.</param>
    /// <returns>True if the feature is used in at least one decision node; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks whether a specific feature is used as a split criterion in any node of the decision tree.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a specific input feature is used in the tree.
    /// 
    /// You provide the position (index) of a feature, and the method tells you whether that feature
    /// is used in any decision node throughout the tree.
    /// 
    /// For example, if feature #3 is never used to make a decision in the tree, this method would
    /// return false because that feature doesn't affect the model's predictions.
    /// 
    /// This is useful when you want to check a specific feature's importance rather than
    /// getting all important features at once.
    /// </para>
    /// </remarks>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        return IsFeatureUsedInSubtree(Root, featureIndex);
    }

    /// <summary>
    /// Sets the parameters for this model.
    /// </summary>
    /// <param name="parameters">A vector containing the model parameters.</param>
    public virtual void SetParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Decision trees do not support direct parameter setting. Use WithParameters to create a new model with different parameters.");
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to activate.</param>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        throw new NotSupportedException("Decision trees do not support setting active features after training. Features are selected during tree construction.");
    }

    /// <summary>
    /// Gets the feature importance scores as a dictionary.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        if (Root == null)
        {
            return new Dictionary<string, T>();
        }

        var importanceScores = new Dictionary<int, T>();
        CalculateFeatureImportanceRecursive(Root, importanceScores);

        var result = new Dictionary<string, T>();
        foreach (var kvp in importanceScores)
        {
            string featureName = FeatureNames != null && kvp.Key < FeatureNames.Length
                ? FeatureNames[kvp.Key]
                : $"Feature_{kvp.Key}";
            result[featureName] = kvp.Value;
        }

        return result;
    }

    private void CalculateFeatureImportanceRecursive(DecisionTreeNode<T>? node, Dictionary<int, T> importanceScores)
    {
        if (node == null || node.IsLeaf)
            return;

        if (!importanceScores.ContainsKey(node.FeatureIndex))
        {
            importanceScores[node.FeatureIndex] = NumOps.Zero;
        }

        // NOTE: This is a simple count-based approach to feature importance.
        // It increments the score for each time a feature is used to split a node,
        // but does NOT account for the quality of the split (e.g., reduction in impurity or error).
        // This limitation means the importance scores may not reflect the true predictive power of each feature.
        importanceScores[node.FeatureIndex] = NumOps.Add(importanceScores[node.FeatureIndex], NumOps.One);

        CalculateFeatureImportanceRecursive(node.Left, importanceScores);
        CalculateFeatureImportanceRecursive(node.Right, importanceScores);
    }

    /// <summary>
    /// Creates a deep copy of the decision tree model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and tree structure.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the asynchronous decision tree model, including all nodes, connections,
    /// and learned parameters. The copy is independent of the original model, so modifications to one
    /// don't affect the other.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact independent copy of your model.
    /// 
    /// The copy has the same:
    /// - Tree structure (all nodes and their connections)
    /// - Decision rules (which features to split on and at what values)
    /// - Prediction values at leaf nodes
    /// - Feature importance scores
    /// 
    /// But it's completely separate from the original model - changes to one won't affect the other.
    /// 
    /// This is useful when you want to:
    /// - Experiment with modifying a model without affecting the original
    /// - Create multiple similar models to use in different contexts
    /// - Save a "checkpoint" of your model before making changes
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> DeepCopy()
    {
        return Clone();
    }

    /// <summary>
    /// Creates a clone of the decision tree model.
    /// </summary>
    /// <returns>A new instance of the model with the same parameters and tree structure.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete copy of the asynchronous decision tree model, including the entire tree structure
    /// and all learned parameters. Specific implementations should override this method to ensure all
    /// implementation-specific properties are properly copied.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact independent copy of your model.
    /// 
    /// Cloning a model means creating a new model that's exactly the same as the original,
    /// including all its learned parameters and settings. However, the clone is independent -
    /// changes to one model won't affect the other.
    /// 
    /// Think of it like photocopying a document - the copy has all the same information,
    /// but you can mark up the copy without changing the original.
    /// 
    /// Note: Specific decision tree algorithms will customize this method to ensure all their
    /// unique properties are properly copied.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        // Create a new instance with the same options
        var clone = CreateNewInstance();

        // Deep copy the tree structure
        if (Root != null)
        {
            ((AsyncDecisionTreeRegressionBase<T>)clone).Root = DeepCloneNode(Root);
        }

        // Copy feature importances
        if (FeatureImportances.Length > 0)
        {
            ((AsyncDecisionTreeRegressionBase<T>)clone).FeatureImportances = new Vector<T>(FeatureImportances);
        }

        return clone;
    }

    /// <summary>
    /// Creates a new instance of the async decision tree model with the same options.
    /// </summary>
    /// <returns>A new instance of the model with the same configuration but no trained parameters.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to create a new instance of the specific
    /// decision tree model type with the same configuration options but without copying the trained parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a fresh copy of the model configuration without 
    /// any learned parameters.
    /// 
    /// Think of it like getting a blank notepad with the same paper quality and size, 
    /// but without any writing on it yet. The new model has the same:
    /// - Maximum depth setting
    /// - Minimum samples split setting
    /// - Split criterion (how nodes decide which feature to split on)
    /// - Other configuration options
    /// 
    /// But it doesn't have any of the actual tree structure that was learned from data.
    /// 
    /// This is mainly used internally when doing things like cross-validation or 
    /// creating ensembles of similar models with different training data.
    /// </para>
    /// </remarks>
    protected abstract IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance();

    /// <summary>
    /// Counts the total number of nodes in the tree.
    /// </summary>
    /// <param name="node">The current node being counted.</param>
    /// <returns>The total number of nodes in the subtree rooted at the given node.</returns>
    private int CountNodes(DecisionTreeNode<T>? node)
    {
        if (node == null)
            return 0;

        return 1 + CountNodes(node.Left) + CountNodes(node.Right);
    }

    /// <summary>
    /// Serializes a node and its children to a parameter vector.
    /// </summary>
    /// <param name="node">The node to serialize.</param>
    /// <param name="parameters">The parameter vector to write to.</param>
    /// <param name="currentIndex">The current index in the parameter vector.</param>
    private void SerializeNodeToVector(DecisionTreeNode<T> node, Vector<T> parameters, ref int currentIndex)
    {
        // Store the node's parameters
        parameters[currentIndex++] = NumOps.FromDouble(node.FeatureIndex);
        parameters[currentIndex++] = node.SplitValue;
        parameters[currentIndex++] = node.Prediction;
        parameters[currentIndex++] = NumOps.FromDouble(node.IsLeaf ? 1.0 : 0.0);

        // Recursively serialize child nodes
        if (node.Left != null)
            SerializeNodeToVector(node.Left, parameters, ref currentIndex);

        if (node.Right != null)
            SerializeNodeToVector(node.Right, parameters, ref currentIndex);
    }

    /// <summary>
    /// Deserializes a node and its children from a parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector to read from.</param>
    /// <param name="currentIndex">The current index in the parameter vector.</param>
    /// <returns>The reconstructed node.</returns>
    private DecisionTreeNode<T>? DeserializeNodeFromVector(Vector<T> parameters, ref int currentIndex)
    {
        // Read the node's parameters
        int featureIndex = NumOps.ToInt32(parameters[currentIndex++]);
        T splitValue = parameters[currentIndex++];
        T prediction = parameters[currentIndex++];
        bool isLeaf = NumOps.ToInt32(parameters[currentIndex++]) == 1;

        // Create the node
        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = featureIndex,
            SplitValue = splitValue,
            Prediction = prediction,
            IsLeaf = isLeaf
        };

        // If it's not a leaf node, recursively deserialize child nodes
        if (!isLeaf)
        {
            node.Left = DeserializeNodeFromVector(parameters, ref currentIndex);
            node.Right = DeserializeNodeFromVector(parameters, ref currentIndex);
        }

        return node;
    }

    /// <summary>
    /// Collects all feature indices used in the tree.
    /// </summary>
    /// <param name="node">The current node being examined.</param>
    /// <param name="activeFeatures">The set of active feature indices.</param>
    private void CollectActiveFeatures(DecisionTreeNode<T>? node, HashSet<int> activeFeatures)
    {
        if (node == null)
            return;

        // If it's not a leaf node, add its feature index to the set
        if (!node.IsLeaf)
        {
            activeFeatures.Add(node.FeatureIndex);
        }

        // Recursively collect features from child nodes
        CollectActiveFeatures(node.Left, activeFeatures);
        CollectActiveFeatures(node.Right, activeFeatures);
    }

    /// <summary>
    /// Checks if a specific feature is used in a subtree.
    /// </summary>
    /// <param name="node">The current node being examined.</param>
    /// <param name="featureIndex">The index of the feature to check for.</param>
    /// <returns>True if the feature is used in the subtree; otherwise, false.</returns>
    private bool IsFeatureUsedInSubtree(DecisionTreeNode<T>? node, int featureIndex)
    {
        if (node == null)
            return false;

        // Check if this node uses the feature
        if (!node.IsLeaf && node.FeatureIndex == featureIndex)
            return true;

        // Recursively check child nodes
        return IsFeatureUsedInSubtree(node.Left, featureIndex) ||
               IsFeatureUsedInSubtree(node.Right, featureIndex);
    }

    /// <summary>
    /// Creates a deep clone of a node and its children.
    /// </summary>
    /// <param name="node">The node to clone.</param>
    /// <returns>A new node that is an exact copy of the original node and its subtree.</returns>
    private DecisionTreeNode<T> DeepCloneNode(DecisionTreeNode<T> node)
    {
        var clone = new DecisionTreeNode<T>
        {
            FeatureIndex = node.FeatureIndex,
            SplitValue = node.SplitValue,
            Prediction = node.Prediction,
            IsLeaf = node.IsLeaf
        };

        // Recursively clone child nodes
        if (node.Left != null)
            clone.Left = DeepCloneNode(node.Left);

        if (node.Right != null)
            clone.Right = DeepCloneNode(node.Right);

        return clone;
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));
        }
        var data = Serialize();
        // Ensure directory exists and handle IO exceptions with clearer context
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        try
        {
            File.WriteAllBytes(filePath, data);
        }
        catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException || ex is System.Security.SecurityException)
        {
            throw new InvalidOperationException($"Failed to save model to '{filePath}': {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Loads the model from a file.
    /// </summary>
    /// <param name="filePath">The path from which to load the model.</param>
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));
        }
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"The specified model file does not exist: {filePath}", filePath);
        }
        try
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load or deserialize model from file '{filePath}'.", ex);
        }
    }

    public virtual int ParameterCount
    {
        get { return CountNodes(Root) * 4 + 1; }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For async tree-based regression models, the default loss function is Mean Squared Error (MSE).
    /// This can be customized by passing a different loss function to the constructor.
    /// </para>
    /// </remarks>
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>IMPORTANT NOTE:</b> Decision trees (including async variants) are not continuously differentiable.
    /// This gradient computation provides a numerical approximation for compatibility with
    /// gradient-based distributed training, but it is NOT the gradient in the traditional sense.
    ///
    /// For proper tree-based distributed training with async models, consider:
    /// - Gradient Boosting (which uses gradients of the loss, not the tree)
    /// - Model averaging approaches (Random Forests, Extremely Randomized Trees)
    /// - Ensemble-based distributed training
    /// </para>
    /// <para><b>For Beginners:</b> Async tree models (like Random Forests) use multiple trees together.
    ///
    /// Each tree makes decisions using "if-then" rules, not smooth math functions.
    /// This method provides an approximation for compatibility, but true tree training
    /// happens through splitting algorithms, not gradient descent.
    ///
    /// For gradient-based training, use Gradient Boosting or other ensemble methods
    /// that explicitly use gradients.
    /// </para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        var loss = lossFunction ?? DefaultLossFunction;
        var predictions = Predict(input);

        // For gradient boosting: compute per-sample loss derivatives (pseudo-residuals)
        // These are ∂Loss/∂predictions, NOT ∂Loss/∂parameters
        // In gradient boosting, subsequent trees are fit to these negative gradients
        var sampleGradients = loss.CalculateDerivative(predictions, target);

        // Map per-sample gradients to per-parameter gradients
        // For decision trees, parameters typically represent leaf values or split thresholds
        // We aggregate sample gradients into ParameterCount buckets
        var gradients = new Vector<T>(ParameterCount);

        if (sampleGradients.Length == 0 || ParameterCount == 0)
        {
            return gradients; // Return zeros
        }

        // Distribute samples across parameters
        int samplesPerParam = Math.Max(1, (sampleGradients.Length + ParameterCount - 1) / ParameterCount);

        for (int paramIdx = 0; paramIdx < ParameterCount; paramIdx++)
        {
            T sum = NumOps.Zero;
            int count = 0;

            // Aggregate gradients for samples mapped to this parameter
            int startIdx = paramIdx * samplesPerParam;
            int endIdx = Math.Min((paramIdx + 1) * samplesPerParam, sampleGradients.Length);

            for (int sampleIdx = startIdx; sampleIdx < endIdx; sampleIdx++)
            {
                sum = NumOps.Add(sum, sampleGradients[sampleIdx]);
                count++;
            }

            // Average the gradients for this parameter bucket
            gradients[paramIdx] = count > 0
                ? NumOps.Divide(sum, NumOps.FromDouble(count))
                : NumOps.Zero;
        }

        return gradients;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>IMPORTANT NOTE:</b> Async tree models are not trained via gradient descent.
    /// This method is provided for interface compatibility.
    ///
    /// Async tree models are typically trained using:
    /// - Parallel tree construction algorithms
    /// - Bootstrap aggregating (Bagging) for Random Forests
    /// - Gradient-based ensemble methods for Gradient Boosting
    /// </para>
    /// <para><b>For Beginners:</b> This is a no-op for tree-based models.
    ///
    /// Trees are built differently than neural networks. They don't learn by
    /// adjusting weights with gradients. Instead, they:
    /// 1. Find the best feature to split on at each node
    /// 2. Build the tree structure recursively
    /// 3. Combine multiple trees in ensembles (Random Forests, etc.)
    ///
    /// This method exists for interface compatibility but doesn't perform gradient updates.
    /// </para>
    /// </remarks>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} gradients, but got {gradients.Length}", nameof(gradients));
        }

        // No-op for async tree models - trees are trained via splitting algorithms
        // Derived classes like GradientBoostingRegression can override with proper gradient-based updates
    }

    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    public virtual void SaveState(Stream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream must be writable.", nameof(stream));
        var data = Serialize();
        stream.Write(data, 0, data.Length);
        stream.Flush();
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    public virtual void LoadState(Stream stream)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead) throw new ArgumentException("Stream must be readable.", nameof(stream));
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        var data = ms.ToArray();
        if (data.Length == 0) throw new InvalidOperationException("Stream contains no data.");
        Deserialize(data);
    }

    #region Soft Tree Mode for JIT Compilation

    /// <summary>
    /// Gets or sets whether to use soft (differentiable) tree mode for JIT compilation support.
    /// </summary>
    /// <value><c>true</c> to enable soft tree mode; <c>false</c> (default) for traditional hard decision trees.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the decision tree uses sigmoid-based soft gating instead of hard if-then splits.
    /// This makes the tree differentiable and enables JIT compilation support.
    /// </para>
    /// <para>
    /// Formula at each split: output = σ((threshold - x[feature]) / temperature) * left + (1 - σ) * right
    /// where σ is the sigmoid function.
    /// </para>
    /// <para><b>For Beginners:</b> Soft tree mode allows the decision tree to be JIT compiled for faster inference.
    ///
    /// Traditional decision trees make hard yes/no decisions:
    /// - "If feature &gt; 5, go LEFT, otherwise go RIGHT"
    ///
    /// Soft trees use smooth transitions instead:
    /// - Near the boundary, the output blends both left and right paths
    /// - This creates a smooth, differentiable function that can be JIT compiled
    /// </para>
    /// </remarks>
    public bool UseSoftTree
    {
        get => Options.UseSoftTree;
        set => Options.UseSoftTree = value;
    }

    /// <summary>
    /// Gets or sets the temperature parameter for soft decision tree mode.
    /// </summary>
    /// <value>
    /// The temperature for sigmoid gating. Lower values produce sharper decisions.
    /// Default is 1.0.
    /// </value>
    /// <remarks>
    /// <para>
    /// Only used when <see cref="UseSoftTree"/> is enabled. Controls the smoothness of
    /// the soft split operations:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Lower temperature (e.g., 0.1) = sharper, more discrete decisions</description></item>
    /// <item><description>Higher temperature (e.g., 10.0) = softer, more blended decisions</description></item>
    /// </list>
    /// </remarks>
    public T SoftTreeTemperature
    {
        get => NumOps.FromDouble(Options.SoftTreeTemperature);
        set => Options.SoftTreeTemperature = Convert.ToDouble(value);
    }

    #endregion

    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether this model currently supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> when <see cref="UseSoftTree"/> is enabled and the tree has been trained;
    /// <c>false</c> otherwise.
    /// </value>
    /// <remarks>
    /// <para>
    /// When <see cref="UseSoftTree"/> is enabled, the decision tree can be exported as a
    /// differentiable computation graph using soft (sigmoid-based) gating. This enables
    /// JIT compilation for optimized inference.
    /// </para>
    /// <para>
    /// When <see cref="UseSoftTree"/> is disabled, JIT compilation is not supported because
    /// traditional hard decision trees use branching logic that cannot be represented as
    /// a static computation graph.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation is available when soft tree mode is enabled.
    ///
    /// In soft tree mode, the discrete if-then decisions are replaced with smooth sigmoid
    /// functions that can be compiled into an optimized computation graph. This gives you
    /// the interpretability of decision trees with the speed of JIT-compiled models.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation => UseSoftTree && Root != null;

    /// <summary>
    /// Exports the model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The root node of the exported computation graph.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when <see cref="UseSoftTree"/> is false.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the tree has not been trained (Root is null).
    /// </exception>
    /// <remarks>
    /// <para>
    /// When soft tree mode is enabled, this exports the tree as a differentiable computation
    /// graph using <see cref="Autodiff.TensorOperations{T}.SoftSplit"/> operations. Each internal
    /// node becomes a soft split operation that computes sigmoid-weighted combinations of
    /// left and right subtree outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the decision tree into a computation graph.
    ///
    /// In soft tree mode, each decision node becomes a smooth blend:
    /// - Instead of "go left OR right", it computes "X% left + Y% right"
    /// - The percentages are determined by the sigmoid function
    /// - This creates a smooth, differentiable function that can be JIT compiled
    /// </para>
    /// </remarks>
    public virtual AiDotNet.Autodiff.ComputationNode<T> ExportComputationGraph(List<AiDotNet.Autodiff.ComputationNode<T>> inputNodes)
    {
        if (!UseSoftTree)
        {
            throw new NotSupportedException(
                "Async decision trees do not support JIT compilation in hard tree mode because they use " +
                "discrete branching logic (if-then-else rules).\n\n" +
                "To enable JIT compilation, set UseSoftTree = true to use soft (differentiable) decision trees " +
                "with sigmoid-based gating.");
        }

        if (Root == null)
        {
            throw new InvalidOperationException(
                "Cannot export computation graph: the decision tree has not been trained. " +
                "Call Train() or TrainAsync() first to build the tree structure.");
        }

        // Get the number of features from the tree structure
        int numFeatures = GetMaxFeatureIndexFromTree(Root) + 1;

        // Create input variable node
        var inputTensor = new Tensor<T>(new[] { numFeatures });
        var input = Autodiff.TensorOperations<T>.Variable(inputTensor, "input");
        inputNodes.Add(input);

        // Recursively export the tree as soft split operations
        return ExportNodeAsComputationGraph(Root, input);
    }

    /// <summary>
    /// Gets the maximum feature index used in the tree.
    /// </summary>
    /// <param name="node">The root node of the tree to scan.</param>
    /// <returns>The maximum feature index found.</returns>
    private int GetMaxFeatureIndexFromTree(DecisionTreeNode<T>? node)
    {
        if (node == null || node.IsLeaf)
            return -1;

        int maxIndex = node.FeatureIndex;
        int leftMax = GetMaxFeatureIndexFromTree(node.Left);
        int rightMax = GetMaxFeatureIndexFromTree(node.Right);

        return Math.Max(maxIndex, Math.Max(leftMax, rightMax));
    }

    /// <summary>
    /// Recursively exports a tree node as a computation graph.
    /// </summary>
    /// <param name="node">The node to export.</param>
    /// <param name="input">The input computation node.</param>
    /// <returns>A computation node representing this subtree.</returns>
    private Autodiff.ComputationNode<T> ExportNodeAsComputationGraph(
        DecisionTreeNode<T> node,
        Autodiff.ComputationNode<T> input)
    {
        if (node.IsLeaf)
        {
            // Leaf node: return constant prediction value
            var leafTensor = new Tensor<T>(new[] { 1 });
            leafTensor[0] = node.Prediction;
            return Autodiff.TensorOperations<T>.Constant(leafTensor, $"leaf_{node.GetHashCode()}");
        }

        // Internal node: export as SoftSplit operation
        // Recursively export left and right subtrees
        var leftOutput = ExportNodeAsComputationGraph(node.Left!, input);
        var rightOutput = ExportNodeAsComputationGraph(node.Right!, input);

        // Use SoftSplit operation: output = sigmoid((threshold - x[feature]) / temp) * left + (1 - sigmoid) * right
        return Autodiff.TensorOperations<T>.SoftSplit(
            input,
            leftOutput,
            rightOutput,
            node.FeatureIndex,
            node.SplitValue,
            SoftTreeTemperature);
    }

    #endregion
}
