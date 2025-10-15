using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using AiDotNet.Interpretability;

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
public abstract class AsyncDecisionTreeRegressionModelBase<T> : IAsyncTreeBasedModel<T>
{
    /// <summary>
    /// Gets the numeric operations for the type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This field stores which input features have been manually
    /// selected as important for the decision tree model, overriding the automatic detection
    /// based on the trained tree structure.
    /// 
    /// When this is set, it takes precedence over the features actually used in the tree.
    /// This can be useful for feature selection experiments or when you want to force the
    /// model to consider specific features as important.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

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
    /// Initializes a new instance of the AsyncDecisionTreeRegressionModelBase class.
    /// </summary>
    /// <param name="options">The options for configuring the decision tree.</param>
    /// <param name="regularization">The regularization method to use.</param>
    protected AsyncDecisionTreeRegressionModelBase(DecisionTreeOptions options, IRegularization<T, Matrix<T>, Vector<T>> regularization)
    {
        Options = options;
        NumOps = MathHelper.GetNumericOperations<T>();
        FeatureImportances = new Vector<T>(0);
        Regularization = regularization;
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
        ((AsyncDecisionTreeRegressionModelBase<T>)newModel).Root = DeserializeNodeFromVector(parameters, ref currentIndex);
    
        // Assume the feature importances are already calculated and stored in the parameters
        // or recalculate them based on the reconstructed tree
        if (FeatureImportances.Length > 0)
        {
            ((AsyncDecisionTreeRegressionModelBase<T>)newModel).FeatureImportances = new Vector<T>(FeatureImportances);
        }
    
        return newModel;
    }

    /// <summary>
    /// Sets the parameters of the model from a vector representation.
    /// </summary>
    /// <param name="parameters">A vector containing a serialized representation of the decision tree structure.</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the parameter vector has an invalid length.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the decision tree model from a parameter vector that was previously
    /// created using the GetParameters method. The current tree structure is replaced with the
    /// structure defined in the parameter vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method rebuilds the decision tree from a flat list of numbers.
    /// 
    /// It takes the specialized vector representation created by GetParameters() and reconstructs
    /// the decision tree from it, replacing the current tree structure. This is challenging because
    /// decision trees are complex structures that don't easily convert to and from simple lists of numbers.
    /// 
    /// This method is primarily used when:
    /// - Loading a saved model
    /// - Applying parameter updates from optimization algorithms
    /// - Transferring parameters between models
    /// 
    /// For most purposes, the Serialize and Deserialize methods provide a more reliable way to
    /// save and load tree models.
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        // If the parameter vector is empty or invalid, clear the tree
        if (parameters.Length < 1)
        {
            Root = null;
            return;
        }

        // Get the node count from the first parameter
        int nodeCount = NumOps.ToInt32(parameters[0]);

        // If there are no nodes, clear the tree
        if (nodeCount == 0)
        {
            Root = null;
            return;
        }

        // Check if the parameter vector has the expected length
        if (parameters.Length != nodeCount * 4 + 1)
        {
            throw new ArgumentException($"Invalid parameter vector length. Expected {nodeCount * 4 + 1} but got {parameters.Length}.", nameof(parameters));
        }

        // Reconstruct the tree from the parameter vector
        int currentIndex = 1;
        Root = DeserializeNodeFromVector(parameters, ref currentIndex);
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
        // If we have explicitly set active features, return those
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0)
        {
            return _explicitlySetActiveFeatures.OrderBy(i => i);
        }

        // Otherwise, continue with the existing implementation
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
        // If feature index is explicitly set as active, return true immediately
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Contains(featureIndex))
        {
            return true;
        }

        // If explicitly set active features exist but don't include this index, it's not used
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0)
        {
            return false;
        }

        return IsFeatureUsedInSubtree(Root, featureIndex);
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
            ((AsyncDecisionTreeRegressionModelBase<T>)clone).Root = DeepCloneNode(Root);
        }
    
        // Copy feature importances
        if (FeatureImportances.Length > 0)
        {
            ((AsyncDecisionTreeRegressionModelBase<T>)clone).FeatureImportances = new Vector<T>(FeatureImportances);
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
    /// Sets which features should be considered active in the model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is negative.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which features should be considered active in the
    /// decision tree model, overriding the automatic determination based on the tree structure.
    /// Any features not included in the provided collection will be considered inactive,
    /// regardless of whether they are used in decision nodes.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you manually tell the model which input features
    /// are important, regardless of what the decision tree actually learned during training.
    /// 
    /// For example, if you have features like age, income, and education level, you can use
    /// this method to specify that only age and education level should be considered active,
    /// even if the trained tree also uses income for some decisions.
    /// 
    /// This is useful for:
    /// - Testing how the model performs with a specific set of features
    /// - Forcing the model to focus on features you believe are important
    /// - Comparing different feature sets without retraining the model
    /// - Implementing manual feature selection techniques
    /// </para>
    /// </remarks>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices), "Feature indices cannot be null.");
        }

        // Initialize the hash set if it doesn't exist
        _explicitlySetActiveFeatures ??= [];

        // Clear existing explicitly set features
        _explicitlySetActiveFeatures.Clear();

        // Add the new feature indices
        foreach (var index in featureIndices)
        {
            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} cannot be negative.");
            }

            _explicitlySetActiveFeatures.Add(index);
        }
    }

    #region IInterpretableModel Implementation

    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
    protected Vector<int> _sensitiveFeatures;
    protected readonly List<FairnessMetric> _fairnessMetrics = new();
    protected IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> _baseModel;

    /// <summary>
    /// Gets the global feature importance across all predictions.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await InterpretableModelHelper.GetGlobalFeatureImportanceAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets the local feature importance for a specific input.
    /// </summary>
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Matrix<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Matrix<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Matrix<T> input, int numFeatures = 10)
    {
        return await InterpretableModelHelper.GetLimeExplanationAsync<T>(_enabledMethods, numFeatures);
    }

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    public virtual async Task<PartialDependenceData<T>> GetPartialDependenceAsync(Vector<int> featureIndices, int gridResolution = 20)
    {
        return await InterpretableModelHelper.GetPartialDependenceAsync<T>(_enabledMethods, featureIndices, gridResolution);
    }

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Matrix<T> input, Vector<T> desiredOutput, int maxChanges = 5)
    {
        return await InterpretableModelHelper.GetCounterfactualAsync<T>(_enabledMethods, maxChanges);
    }

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public virtual async Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync()
    {
        return await InterpretableModelHelper.GetModelSpecificInterpretabilityAsync(this);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public virtual async Task<string> GenerateTextExplanationAsync(Matrix<T> input, Vector<T> prediction)
    {
        return await InterpretableModelHelper.GenerateTextExplanationAsync(this, input, prediction);
    }

    /// <summary>
    /// Gets feature interaction effects between two features.
    /// </summary>
    public virtual async Task<T> GetFeatureInteractionAsync(int feature1Index, int feature2Index)
    {
        return await InterpretableModelHelper.GetFeatureInteractionAsync<T>(_enabledMethods, feature1Index, feature2Index);
    }

    /// <summary>
    /// Validates fairness metrics for the given inputs.
    /// </summary>
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Matrix<T> inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Matrix<T> input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IModel<Matrix<T>, Vector<T>, ModelMetadata<T>> model)
    {
        _baseModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Enables specific interpretation methods.
    /// </summary>
    public virtual void EnableMethod(params InterpretationMethod[] methods)
    {
        foreach (var method in methods)
        {
            _enabledMethods.Add(method);
        }
    }

    /// <summary>
    /// Configures fairness evaluation settings.
    /// </summary>
    public virtual void ConfigureFairness(Vector<int> sensitiveFeatures, params FairnessMetric[] fairnessMetrics)
    {
        _sensitiveFeatures = sensitiveFeatures ?? throw new ArgumentNullException(nameof(sensitiveFeatures));
        _fairnessMetrics.Clear();
        _fairnessMetrics.AddRange(fairnessMetrics);
    }

    #endregion
}