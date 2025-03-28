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
public abstract class DecisionTreeRegressionBase<T> : ITreeBasedModel<T>
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
    protected IRegularization<T> Regularization { get; private set; }
    
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
    /// <remarks>
    /// <para>
    /// This constructor initializes a new base class for decision tree regression with the specified options
    /// and regularization strategy. If no options are provided, default values are used. If no regularization
    /// is specified, no regularization is applied.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the foundation for a decision tree model.
    /// 
    /// When creating a decision tree, you can specify two main things:
    /// - Options: Controls how the tree grows (like its maximum depth or minimum samples needed to split)
    /// - Regularization: Helps prevent the model from becoming too complex and "memorizing" the training data
    /// 
    /// If you don't specify these parameters, the model will use reasonable default settings.
    /// 
    /// This constructor is typically not called directly but is used by specific implementations
    /// of decision tree models.
    /// </para>
    /// </remarks>
    protected DecisionTreeRegressionBase(DecisionTreeOptions? options, IRegularization<T>? regularization)
    {
        Options = options ?? new();
        NumOps = MathHelper.GetNumericOperations<T>();
        FeatureImportances = new Vector<T>(0);
        Regularization = regularization ?? new NoRegularization<T>();
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
}