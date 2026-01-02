using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Regression;

/// <summary>
/// Provides a base implementation for decision tree regression models that predict continuous values.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for decision tree regression models, providing a framework
/// for building predictive models based on decision trees. It manages the tree structure, handles serialization
/// and deserialization, and defines the interface that concrete implementations must support.
/// </para>
/// <para><b>For Beginners:</b> This is a template for creating decision tree models that predict numerical values.
/// 
/// A decision tree works like a flowchart of yes/no questions to make predictions:
/// - Start at the top (root) of the tree
/// - At each step, answer a question about your data
/// - Follow the appropriate path based on your answer
/// - Continue until you reach an endpoint that provides a prediction
/// 
/// This base class provides the common structure and behaviors that all decision tree models share,
/// while allowing specific implementations to customize how the tree is built and used.
/// 
/// Think of it like a blueprint for building different types of decision trees, where specific 
/// implementations can fill in the details according to their requirements.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public abstract class DecisionTreeRegressionBase<T> : ITreeBasedRegression<T>
{
    /// <summary>
    /// Provides operations for performing numeric calculations appropriate for the type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field contains implementations of basic mathematical operations (addition, subtraction, etc.)
    /// that work with the specific numeric type T. It allows the decision tree algorithm to perform calculations
    /// independently of the specific numeric type being used.
    /// </para>
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The root node of the decision tree.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the root node of the decision tree structure. All predictions start from this node
    /// and follow a path through the tree based on the feature values of the input sample.
    /// </para>
    /// </remarks>
    protected DecisionTreeNode<T>? Root;

    /// <summary>
    /// Gets the configuration options used by the decision tree algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the configuration options that control how the decision tree is built,
    /// such as maximum depth, minimum samples required for splitting, and split criteria.
    /// </para>
    /// </remarks>
    protected DecisionTreeOptions Options { get; private set; }

    /// <summary>
    /// Gets the regularization strategy applied to the model to prevent overfitting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property provides access to the regularization strategy used to prevent the model from overfitting
    /// to the training data. Regularization helps improve the model's ability to generalize to new, unseen data.
    /// </para>
    /// </remarks>
    protected IRegularization<T, Matrix<T>, Vector<T>> Regularization { get; private set; }

    /// <summary>
    /// Gets the default loss function for this tree-based regression model.
    /// </summary>
    /// <value>
    /// The loss function used for gradient computation.
    /// </value>
    private readonly ILossFunction<T> _defaultLossFunction;

    /// <summary>
    /// Gets the maximum depth of the decision tree.
    /// </summary>
    /// <value>
    /// The maximum number of levels in the tree, from the root to the deepest leaf.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the maximum depth of the decision tree, which is one of the most important parameters
    /// for controlling the complexity of the model. Deeper trees can capture more complex patterns but are more
    /// prone to overfitting.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you how many levels of questions the tree can ask.
    /// 
    /// Think of MaxDepth as the maximum number of questions that can be asked before making a prediction:
    /// - A tree with MaxDepth = 1 can only ask one question (very simple model)
    /// - A tree with MaxDepth = 10 can ask up to 10 nested questions (more complex model)
    /// 
    /// Setting an appropriate maximum depth helps prevent the model from becoming too complex:
    /// - Too shallow (small MaxDepth): The model might be too simple to capture important patterns
    /// - Too deep (large MaxDepth): The model might "memorize" the training data instead of learning
    ///   general patterns, making it perform poorly on new data
    /// </para>
    /// </remarks>
    public int MaxDepth => Options.MaxDepth;

    /// <summary>
    /// Gets or sets the feature names.
    /// </summary>
    /// <value>
    /// An array of feature names. If not set, feature indices will be used as names.
    /// </value>
    public string[]? FeatureNames { get; set; }

    /// <summary>
    /// Gets the importance scores for each feature used in the model.
    /// </summary>
    /// <value>
    /// A vector of values representing the relative importance of each feature, normalized to sum to 1.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property provides access to the calculated importance of each feature in the trained model.
    /// Feature importance scores indicate how useful or valuable each feature was in building the decision tree.
    /// Higher values indicate features that were more important for making predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you which input features have the biggest impact on predictions.
    /// 
    /// Feature importance helps you understand:
    /// - Which factors matter most for your predictions
    /// - Which features might be redundant or irrelevant
    /// - How the model is making its decisions
    /// 
    /// For example, when predicting house prices:
    /// - Location might have importance 0.7 (very important)
    /// - Square footage might have importance 0.2 (somewhat important)
    /// - Year built might have importance 0.1 (less important)
    /// 
    /// These values always add up to 1, making it easy to compare the relative importance
    /// of different features.
    /// </para>
    /// </remarks>
    public Vector<T> FeatureImportances { get; protected set; }

    /// <summary>
    /// Gets the number of trees in this model, which is always 1 for a single decision tree.
    /// </summary>
    /// <value>
    /// The number of trees in the model, which is 1 for a standard decision tree implementation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the number of decision trees used in the model. For standard decision tree
    /// implementations, this is always 1. This property exists primarily for compatibility with ensemble methods
    /// that may use multiple trees.
    /// </para>
    /// <para><b>For Beginners:</b> This property indicates how many trees make up this model.
    /// 
    /// A single decision tree model always returns 1 here.
    /// 
    /// Some more advanced models (like Random Forests or Gradient Boosting) use multiple trees 
    /// working together to make better predictions. In those cases, this property would return
    /// the number of trees in the ensemble.
    /// </para>
    /// </remarks>
    public virtual int NumberOfTrees => 1;

    /// <summary>
    /// Initializes a new instance of the <see cref="DecisionTreeRegressionBase{T}"/> class.
    /// </summary>
    /// <param name="options">Optional configuration options for the decision tree algorithm.</param>
    /// <param name="regularization">Optional regularization strategy to prevent overfitting.</param>
    /// <param name="lossFunction">Loss function for gradient computation. If null, defaults to Mean Squared Error.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new base class for decision tree regression with the specified options
    /// and regularization strategy. If no options are provided, default values are used. If no regularization
    /// is specified, no regularization is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the foundation for a decision tree model.
    ///
    /// When creating a decision tree, you can specify three main things:
    /// - Options: Controls how the tree grows (like its maximum depth or minimum samples needed to split)
    /// - Regularization: Helps prevent the model from becoming too complex and "memorizing" the training data
    /// - Loss Function: Determines how prediction errors are measured (defaults to Mean Squared Error)
    ///
    /// If you don't specify these parameters, the model will use reasonable default settings.
    ///
    /// This constructor is typically not called directly but is used by specific implementations
    /// of decision tree models.
    /// </para>
    /// </remarks>
    protected DecisionTreeRegressionBase(DecisionTreeOptions? options, IRegularization<T, Matrix<T>, Vector<T>>? regularization, ILossFunction<T>? lossFunction = null)
    {
        Options = options ?? new();
        NumOps = MathHelper.GetNumericOperations<T>();
        FeatureImportances = new Vector<T>(0);
        Regularization = regularization ?? new NoRegularization<T, Matrix<T>, Vector<T>>();
        _defaultLossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        SoftTreeTemperature = NumOps.One; // Default temperature = 1.0
    }

    /// <summary>
    /// Trains the decision tree model using the provided input features and target values.
    /// </summary>
    /// <param name="x">A matrix where each row represents a sample and each column represents a feature.</param>
    /// <param name="y">A vector of target values corresponding to each sample in x.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to build a decision tree model using
    /// the provided training data. The specific algorithm for building the tree is defined by the implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the decision tree how to make predictions.
    /// 
    /// You provide:
    /// - x: Your input data (features) - like house size, number of bedrooms, location, etc.
    /// - y: The values you want to predict - like house prices
    /// 
    /// Each specific implementation of a decision tree will provide its own version of this method,
    /// which defines exactly how the tree learns from your data.
    /// 
    /// After training, the model will be ready to make predictions on new data.
    /// </para>
    /// </remarks>
    public abstract void Train(Matrix<T> x, Vector<T> y);

    /// <summary>
    /// Predicts target values for the provided input features using the trained decision tree model.
    /// </summary>
    /// <param name="input">A matrix where each row represents a sample to predict and each column represents a feature.</param>
    /// <returns>A vector of predicted values corresponding to each input sample.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to predict target values for new input data
    /// using the trained decision tree model. The specific algorithm for making predictions is defined by the implementation.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses your trained model to make predictions on new data.
    /// 
    /// You provide:
    /// - input: New data points for which you want predictions
    /// 
    /// The model will use the decision tree it learned during training to predict values for each
    /// row of input data. The way it navigates the tree to make predictions will depend on the
    /// specific implementation of the decision tree model.
    /// 
    /// For example, if you trained the model to predict house prices, you could use this method
    /// to predict prices for a new set of houses based on their features.
    /// </para>
    /// </remarks>
    public abstract Vector<T> Predict(Matrix<T> input);

    /// <summary>
    /// Gets metadata about the decision tree model and its configuration.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to provide metadata about the model,
    /// including its type and configuration options. This information can be useful for model management,
    /// comparison, and documentation purposes.
    /// </para>
    /// <para><b>For Beginners:</b> This method provides information about your decision tree model.
    /// 
    /// The metadata typically includes:
    /// - The type of model (e.g., Decision Tree, Random Forest)
    /// - Configuration settings (like maximum depth)
    /// - Other relevant information about the model
    /// 
    /// This information is helpful when:
    /// - Comparing different models
    /// - Documenting your model's configuration
    /// - Troubleshooting model performance
    /// 
    /// Each implementation of a decision tree will provide its own version of this method,
    /// returning the specific metadata relevant to that implementation.
    /// </para>
    /// </remarks>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <summary>
    /// Calculates the importance scores for all features used in the model.
    /// </summary>
    /// <param name="featureCount">The number of features in the model.</param>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to calculate the importance of each feature
    /// in the trained model. Feature importance indicates how valuable each feature was in building the decision tree.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out which input features matter most for predictions.
    /// 
    /// Different decision tree implementations might calculate feature importance in different ways,
    /// but the general idea is to measure how much each feature contributes to improving predictions
    /// when it's used in decision nodes throughout the tree.
    /// 
    /// After this method runs, the FeatureImportances property will contain a score for each feature,
    /// allowing you to see which features have the biggest impact on your model's predictions.
    /// </para>
    /// </remarks>
    protected abstract void CalculateFeatureImportances(int featureCount);

    /// <summary>
    /// Serializes the decision tree model to a byte array for storage or transmission.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method converts the decision tree model into a byte array that can be stored in a file, database,
    /// or transmitted over a network. The serialized data includes the model's configuration options, feature
    /// importances, and the complete tree structure.
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
    /// - All the model's settings (like maximum depth)
    /// - The importance of each feature
    /// - The entire tree structure with all its decision rules
    /// 
    /// Example:
    /// ```csharp
    /// // Serialize the model
    /// byte[] modelData = decisionTree.Serialize();
    /// 
    /// // Save to a file
    /// File.WriteAllBytes("decisionTree.model", modelData);
    /// ```
    /// </para>
    /// </remarks>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        // Serialize options
        writer.Write(Options.MaxDepth);
        writer.Write(Options.MinSamplesSplit);
        // Write MaxFeatures as int (-1 for NaN, cast to int otherwise)
        writer.Write(double.IsNaN(Options.MaxFeatures) ? -1 : (int)Options.MaxFeatures);
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
    /// Loads a previously serialized decision tree model from a byte array.
    /// </summary>
    /// <param name="modelData">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs a decision tree model from a byte array that was previously created using the
    /// Serialize method. It restores the model's configuration options, feature importances, and tree structure,
    /// allowing the model to be used for predictions without retraining.
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
    /// - Feature importances are recovered
    /// - The entire tree structure is reconstructed
    /// - The model is ready to make predictions immediately
    /// 
    /// Example:
    /// ```csharp
    /// // Load from a file
    /// byte[] modelData = File.ReadAllBytes("decisionTree.model");
    /// 
    /// // Deserialize the model
    /// decisionTree.Deserialize(modelData);
    /// 
    /// // Now you can use the model for predictions
    /// var predictions = decisionTree.Predict(newFeatures);
    /// ```
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
    /// Serializes a tree node to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
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
    /// Deserializes a tree node from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <returns>The deserialized node.</returns>
    private DecisionTreeNode<T>? DeserializeNode(BinaryReader reader)
    {
        bool hasNode = reader.ReadBoolean();
        if (!hasNode) return null;
        var node = new DecisionTreeNode<T>
        {
            FeatureIndex = reader.ReadInt32(),
            SplitValue = NumOps.FromDouble(reader.ReadDouble()),
            Prediction = NumOps.FromDouble(reader.ReadDouble()),
            IsLeaf = reader.ReadBoolean(),
            Left = DeserializeNode(reader),
            Right = DeserializeNode(reader)
        };

        return node;
    }

    /// <summary>
    /// Gets the model parameters as a vector representation.
    /// </summary>
    /// <returns>A vector containing a serialized representation of the decision tree structure.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a vector representation of the decision tree model. Decision trees
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
        ((DecisionTreeRegressionBase<T>)newModel).Root = DeserializeNodeFromVector(parameters, ref currentIndex);

        // Assume the feature importances are already calculated and stored in the parameters
        // or recalculate them based on the reconstructed tree
        if (FeatureImportances.Length > 0)
        {
            ((DecisionTreeRegressionBase<T>)newModel).FeatureImportances = new Vector<T>(FeatureImportances);
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
        // Decision trees don't have traditional parameters like linear models
        // This is a stub implementation to satisfy the interface
        // Actual parameter setting would require reconstructing the tree
        throw new NotSupportedException("Decision trees do not support direct parameter setting. Use WithParameters to create a new model with different parameters.");
    }

    /// <summary>
    /// Sets the active feature indices for this model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to activate.</param>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        // Decision trees select features during training
        // This is a stub implementation to satisfy the interface
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

        // Calculate weighted importance based on the quality of the split.
        // For regression trees, we use variance reduction as the weight:
        // importance = (n_node / n_total) * (variance_parent - (n_left * variance_left + n_right * variance_right) / n_node)
        //
        // Since we don't have access to the sample counts and variances at each node in this base class,
        // we use a simplified approach: count each split where the feature is used.
        // Derived classes that track sample counts and variances can override GetFeatureImportance()
        // to provide variance-weighted importance scores.
        //
        // This count-based approach provides a reasonable approximation:
        // - Features used more frequently in splits tend to be more important
        // - Features used higher in the tree (closer to root) are counted more times
        // - This correlates well with true variance reduction in practice
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
    /// This method creates a complete copy of the decision tree model, including all nodes, connections,
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
    /// This method creates a complete copy of the decision tree model, including the entire tree structure
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
            ((DecisionTreeRegressionBase<T>)clone).Root = DeepCloneNode(Root);
        }

        // Copy feature importances
        if (FeatureImportances.Length > 0)
        {
            ((DecisionTreeRegressionBase<T>)clone).FeatureImportances = new Vector<T>(FeatureImportances);
        }

        return clone;
    }

    /// <summary>
    /// Creates a new instance of the decision tree model with the same options.
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

    public virtual int ParameterCount
    {
        get { return CountNodes(Root) * 4 + 1; }
    }

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

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For tree-based regression models, the default loss function is Mean Squared Error (MSE).
    /// This can be customized by passing a different loss function to the constructor.
    /// </para>
    /// </remarks>
    public virtual ILossFunction<T> DefaultLossFunction => _defaultLossFunction;

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>IMPORTANT NOTE:</b> Decision trees are not continuously differentiable models.
    /// This gradient computation provides a numerical approximation for compatibility with
    /// gradient-based distributed training, but it is NOT the gradient in the traditional sense.
    ///
    /// For proper tree-based distributed training, consider using:
    /// - Gradient Boosting (which uses gradients of the loss, not the tree)
    /// - Model averaging approaches (like Random Forests in distributed mode)
    /// - Ensemble-based distributed training
    /// </para>
    /// <para><b>For Beginners:</b> Decision trees make decisions using "if-then" rules, not smooth math functions.
    ///
    /// Because of this, they don't have traditional gradients (rates of change). This method
    /// provides an approximation using numerical differentiation, which means:
    /// - It estimates gradients by making small changes to predictions
    /// - It's computationally expensive (requires many predictions)
    /// - It's not as accurate as true gradients
    ///
    /// If you need gradient-based training for trees, consider Gradient Boosting, which
    /// uses gradients of the loss function rather than gradients of the tree structure.
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
    /// <para><b>IMPORTANT NOTE:</b> Decision trees are not trained via gradient descent.
    /// This method is provided for interface compatibility but has limited effectiveness.
    ///
    /// Trees are typically trained using:
    /// - Greedy splitting algorithms (CART, ID3, C4.5)
    /// - Ensemble methods (Random Forests, Gradient Boosting)
    /// - Specialized tree modification algorithms
    /// </para>
    /// <para><b>For Beginners:</b> Applying gradients to a decision tree is unusual.
    ///
    /// Normal trees are built by:
    /// 1. Finding the best feature to split on
    /// 2. Recursively building child nodes
    /// 3. Stopping when nodes are pure enough or deep enough
    ///
    /// This method is provided for compatibility with distributed training interfaces,
    /// but it doesn't perform traditional gradient descent on the tree structure.
    /// Instead, it's primarily useful for ensemble methods like Gradient Boosting.
    /// </para>
    /// </remarks>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} gradients, but got {gradients.Length}", nameof(gradients));
        }

        // For tree models, applying gradients directly is not standard practice
        // This is a no-op implementation for interface compatibility
        // Derived classes (like GradientBoostingRegression) can override with proper gradient-based updates

        // Note: Actual tree parameter updates would require modifying node split points and leaf values
        // which is algorithm-specific and typically handled during the Train() method instead
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

    // ===== Soft Decision Tree Support for JIT Compilation =====

    /// <summary>
    /// Gets or sets whether to use soft (differentiable) tree mode for JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable soft tree mode with sigmoid gating for JIT support;
    /// <c>false</c> (default) for traditional hard decision tree.
    /// </value>
    /// <remarks>
    /// <para><b>Soft Decision Trees:</b> Instead of hard branching (if-then-else), soft trees use
    /// sigmoid gating to compute a smooth probability of going left or right at each node.
    /// This makes the tree differentiable and JIT-compilable.</para>
    /// <para><b>Formula:</b> p_left = σ((threshold - x[feature]) / temperature)</para>
    /// <para><b>Output:</b> weighted_output = p_left * left_value + (1 - p_left) * right_value</para>
    /// <para><b>Trade-offs:</b></para>
    /// <list type="bullet">
    /// <item><description>Soft trees are differentiable and JIT-compilable</description></item>
    /// <item><description>Results are smooth approximations of hard decisions</description></item>
    /// <item><description>Lower temperature = sharper (closer to hard) decisions</description></item>
    /// <item><description>Higher temperature = softer (more averaged) decisions</description></item>
    /// </list>
    /// </remarks>
    public bool UseSoftTree { get; set; } = false;

    /// <summary>
    /// Gets or sets the temperature parameter for soft decision tree mode.
    /// </summary>
    /// <value>
    /// The temperature for sigmoid gating. Lower values produce sharper decisions.
    /// Default is 1.0.
    /// </value>
    public T SoftTreeTemperature { get; set; }

    /// <summary>
    /// Gets a value indicating whether this model supports JIT (Just-In-Time) compilation.
    /// </summary>
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
    /// </remarks>
    public virtual bool SupportsJitCompilation => UseSoftTree && Root != null;

    /// <summary>
    /// Exports the model's computation as a graph of operations.
    /// </summary>
    /// <param name="inputNodes">The input nodes for the computation graph.</param>
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
    /// </remarks>
    public virtual AiDotNet.Autodiff.ComputationNode<T> ExportComputationGraph(List<AiDotNet.Autodiff.ComputationNode<T>> inputNodes)
    {
        if (!UseSoftTree)
        {
            throw new NotSupportedException(
                "Decision tree regression models do not support JIT compilation in hard tree mode because they use:\n" +
                "- Tree-based branching logic with dynamic conditions\n" +
                "- Recursive tree traversal that depends on input values\n" +
                "- Conditional splits that cannot be represented as static tensor operations\n\n" +
                "To enable JIT compilation, set UseSoftTree = true to use soft (differentiable) decision trees " +
                "with sigmoid-based gating.");
        }

        if (Root == null)
        {
            throw new InvalidOperationException(
                "Cannot export computation graph: the decision tree has not been trained. " +
                "Call Train() first to build the tree structure.");
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
    /// Recursively exports a tree node as a computation graph.
    /// </summary>
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

    /// <summary>
    /// Gets the maximum feature index used in the tree.
    /// </summary>
    private int GetMaxFeatureIndexFromTree(DecisionTreeNode<T>? node)
    {
        if (node == null || node.IsLeaf)
            return -1;

        int maxIndex = node.FeatureIndex;
        int leftMax = GetMaxFeatureIndexFromTree(node.Left);
        int rightMax = GetMaxFeatureIndexFromTree(node.Right);

        return Math.Max(maxIndex, Math.Max(leftMax, rightMax));
    }
}
