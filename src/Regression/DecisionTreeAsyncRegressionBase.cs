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
    protected IRegularization<T> Regularization { get; private set; }

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
    /// Initializes a new instance of the AsyncDecisionTreeRegressionBase class.
    /// </summary>
    /// <param name="options">The options for configuring the decision tree.</param>
    /// <param name="regularization">The regularization method to use.</param>
    protected AsyncDecisionTreeRegressionBase(DecisionTreeOptions? options, IRegularization<T>? regularization)
    {
        Options = options ?? new();
        NumOps = MathHelper.GetNumericOperations<T>();
        FeatureImportances = new Vector<T>(0);
        Regularization = regularization ?? new NoRegularization<T>();
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
        TrainAsync(x, y).GetAwaiter().GetResult();
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
        return PredictAsync(input).GetAwaiter().GetResult();
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
}