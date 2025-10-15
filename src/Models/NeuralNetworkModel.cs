using System.Threading.Tasks;
using AiDotNet.Interpretability;

namespace AiDotNet.Models;

/// <summary>
/// Represents a neural network model that implements the IFullModel interface.
/// </summary>
/// <remarks>
/// <para>
/// This class wraps a neural network implementation to provide a consistent interface with other model types.
/// It handles training, prediction, serialization, and other operations required by the IFullModel interface,
/// delegating to the underlying neural network. This allows neural networks to be used interchangeably with
/// other model types in optimization and model selection processes.
/// </para>
/// <para><b>For Beginners:</b> This is a wrapper that makes neural networks work with the same interface as simpler models.
/// 
/// Neural networks are powerful machine learning models that can:
/// - Learn complex patterns in data that simpler models might miss
/// - Process different types of data like images, text, or tabular data
/// - Automatically extract useful features from raw data
/// 
/// This class allows you to use neural networks anywhere you would use simpler models,
/// making it easy to compare them or use them in the same optimization processes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class NeuralNetworkModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the underlying neural network.
    /// </summary>
    /// <value>A NeuralNetworkBase&lt;T&gt; instance containing the actual neural network.</value>
    /// <remarks>
    /// <para>
    /// This property provides access to the underlying neural network implementation. The network is responsible for
    /// the actual computations, while this class serves as an adapter to the IFullModel interface. This property
    /// can be used to access network-specific features not exposed through the IFullModel interface.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you direct access to the actual neural network.
    /// 
    /// The network:
    /// - Contains all the layers and connections of the neural network
    /// - Handles the actual calculations and learning
    /// - Stores all the learned weights and parameters
    /// 
    /// You can use this property to access neural network-specific features
    /// that aren't available through the standard model interface.
    /// </para>
    /// </remarks>
    public NeuralNetworkBase<T> Network { get; }

    /// <summary>
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores feature indices that have been explicitly set as active through
    /// the SetActiveFeatureIndices method, overriding the automatic determination based
    /// on the neural network's architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks which input features have been manually
    /// selected as important for the neural network model, regardless of what features
    /// the network might actually use in its internal calculations.
    /// 
    /// When set, these manually selected features take precedence over the automatic
    /// feature detection which for neural networks typically includes all input features.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

    /// <summary>
    /// Gets the architecture of the neural network.
    /// </summary>
    /// <value>A NeuralNetworkArchitecture&lt;T&gt; instance defining the structure of the network.</value>
    /// <remarks>
    /// <para>
    /// This property provides access to the architecture that defines the structure of the neural network, including
    /// its layers, input/output dimensions, and task-specific properties. The architecture serves as a blueprint for
    /// the network and contains information about the network's topology and configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This property gives you access to the blueprint of the neural network.
    /// 
    /// The architecture:
    /// - Defines how many layers the network has
    /// - Specifies how many neurons are in each layer
    /// - Determines what kind of data the network can process
    /// - Configures how the network learns and makes predictions
    /// 
    /// Think of it like the plans for a building - it defines the structure
    /// but doesn't contain the actual building materials.
    /// </para>
    /// </remarks>
    public NeuralNetworkArchitecture<T> Architecture { get; }
    
    /// <summary>
    /// The numeric operations provider used for mathematical operations on type T.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field provides access to basic mathematical operations for the generic type T,
    /// allowing the class to perform calculations regardless of the specific numeric type.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a way to do math with different number types.
    /// 
    /// Since neural networks can work with different types of numbers (float, double, etc.),
    /// we need a way to perform math operations like addition and multiplication
    /// without knowing exactly what number type we're using. This helper provides
    /// those operations in a consistent way regardless of the number type.
    /// </para>
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    
    /// <summary>
    /// The learning rate used during training to control the size of weight updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate determines how quickly the model adapts to the problem.
    /// Smaller values mean slower learning but potentially more precision, while 
    /// larger values mean faster learning but risk overshooting the optimal solution.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how big each learning step is during training.
    /// 
    /// Think of it like adjusting the size of steps when walking:
    /// - Small learning rate = small steps (slow progress but less risk of going too far)
    /// - Large learning rate = large steps (faster progress but might overshoot the target)
    /// 
    /// Finding the right learning rate is important - too small and training takes forever,
    /// too large and the model might never find the best solution.
    /// </para>
    /// </remarks>
    private T _learningRate = default!;
    
    /// <summary>
    /// Indicates whether the model is currently in training mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some neural network components behave differently during training versus inference.
    /// This flag enables those components to adjust their behavior accordingly.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the network whether it's learning or making predictions.
    /// 
    /// Some parts of neural networks work differently depending on whether the network is:
    /// - Training (learning from examples)
    /// - Making predictions (using what it learned)
    /// 
    /// For example, a technique called "dropout" randomly turns off some neurons during
    /// training to prevent overfitting, but doesn't do this during prediction.
    /// </para>
    /// </remarks>
    private bool _isTrainingMode = true;

    /// <summary>
    /// Initializes a new instance of the NeuralNetworkModel class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the neural network.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new NeuralNetworkModel instance with the specified architecture. It initializes
    /// the underlying neural network based on the architecture provided. The architecture determines the network's
    /// structure, including the number and type of layers, the input and output dimensions, and the type of task
    /// the network is designed to perform.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new neural network model with the specified design.
    /// 
    /// When creating a NeuralNetworkModel:
    /// - You provide an architecture that defines the network's structure
    /// - The constructor creates the actual neural network based on this design
    /// - The model is ready to be trained or to make predictions
    /// 
    /// The architecture is crucial as it determines what kind of data the network can process
    /// and what kind of problems it can solve. Different architectures work better for
    /// different types of problems.
    /// </para>
    /// </remarks>
    public NeuralNetworkModel(NeuralNetworkArchitecture<T> architecture)
    {
        Architecture = architecture;
        Network = new NeuralNetwork<T>(architecture);
        _learningRate = _numOps.FromDouble(0.01); // Default learning rate
    }

    /// <summary>
    /// Gets the number of features used by the model.
    /// </summary>
    /// <value>An integer representing the number of input features.</value>
    /// <remarks>
    /// <para>
    /// This property returns the number of features that the model uses, which is determined by the input size
    /// of the neural network. For one-dimensional inputs, this is simply the input size. For multi-dimensional
    /// inputs, this is the total number of input elements (calculated as InputHeight * InputWidth * InputDepth).
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many input variables the neural network uses.
    /// 
    /// The feature count:
    /// - For simple data, it's the number of input values (like age, height, weight)
    /// - For image data, it's the total number of pixels times the number of color channels
    /// - For text data, it might be the vocabulary size or embedding dimension
    /// 
    /// This helps you understand how much input information the network is considering,
    /// and it's important for ensuring your input data has the right dimensions.
    /// </para>
    /// </remarks>
    public int FeatureCount => Architecture.CalculatedInputSize;

    /// <summary>
    /// Gets the complexity of the model.
    /// </summary>
    /// <value>An integer representing the model's complexity.</value>
    /// <remarks>
    /// <para>
    /// This property returns a measure of the model's complexity, which is calculated as the total number of
    /// trainable parameters (weights and biases) in the neural network. The complexity of a neural network is
    /// an important factor in understanding its capacity to learn, its potential for overfitting, and its
    /// computational requirements.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how complex the neural network is.
    /// 
    /// The complexity:
    /// - Is measured by the total number of adjustable parameters in the network
    /// - Higher complexity means the network can learn more complex patterns
    /// - But higher complexity also means more training data is needed
    /// - And higher complexity increases the risk of overfitting
    /// 
    /// A simple network might have hundreds of parameters,
    /// while deep networks can have millions or billions.
    /// </para>
    /// </remarks>
    public int Complexity => Network.GetParameterCount();

    /// <summary>
    /// Sets the learning rate for training the model.
    /// </summary>
    /// <param name="learningRate">The learning rate to use during training.</param>
    /// <returns>This model instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method sets the learning rate used during training. The learning rate controls how quickly the model
    /// adapts to the training data. A higher learning rate means faster learning but may cause instability, while
    /// a lower learning rate means slower but more stable learning.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you control how big each learning step is during training.
    /// 
    /// The learning rate:
    /// - Controls how quickly the network adjusts its weights
    /// - Smaller values (like 0.001) make training more stable but slower
    /// - Larger values (like 0.1) make training faster but potentially unstable
    /// 
    /// Finding the right learning rate is often a process of trial and error.
    /// This method lets you set it to the value you want to try.
    /// </para>
    /// </remarks>
    public NeuralNetworkModel<T> SetLearningRate(T learningRate)
    {
        _learningRate = learningRate;
        return this;
    }

    /// <summary>
    /// Sets whether the model is in training mode or prediction mode.
    /// </summary>
    /// <param name="isTraining">True for training mode, false for prediction mode.</param>
    /// <returns>This model instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method sets whether the model is in training mode or prediction mode. Some components of neural networks
    /// behave differently during training versus prediction, such as dropout layers, which randomly disable neurons
    /// during training but not during prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This switches the network between learning mode and prediction mode.
    /// 
    /// The two modes are:
    /// - Training mode: The network is learning and updating its weights
    /// - Prediction mode: The network is using what it learned to make predictions
    /// 
    /// Some special layers like Dropout and BatchNormalization work differently
    /// depending on which mode the network is in. This method lets you switch between them.
    /// </para>
    /// </remarks>
    public NeuralNetworkModel<T> SetTrainingMode(bool isTraining)
    {
        _isTrainingMode = isTraining;
        Network.SetTrainingMode(isTraining);
        return this;
    }

    /// <summary>
    /// Determines whether a specific feature is used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>Always returns true for neural networks, as they typically use all input features.</returns>
    /// <remarks>
    /// <para>
    /// This method determines whether a specific feature is used by the model. For neural networks, all features
    /// are typically used in some capacity, so this method always returns true. Unlike some linear models where
    /// features can have zero coefficients and therefore no impact, neural networks generally incorporate all
    /// input features, though they may learn to assign different importance to different features during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if a particular input variable affects the model's predictions.
    /// 
    /// For neural networks:
    /// - This method always returns true
    /// - Neural networks typically use all input features in some way
    /// - The network learns which features are important during training
    /// - Even if a feature isn't useful, the network will learn to assign it less weight
    /// 
    /// This differs from simpler models like linear regression,
    /// where features can be explicitly excluded with zero coefficients.
    /// </para>
    /// </remarks>
    public bool IsFeatureUsed(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= FeatureCount)
        {
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                $"Feature index must be between 0 and {FeatureCount - 1}");
        }

        // If we have explicitly set active features, check those
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0)
        {
            return _explicitlySetActiveFeatures.Contains(featureIndex);
        }

        // Otherwise, neural networks typically use all input features in some capacity
        return true;
    }

    /// <summary>
    /// Trains the model with the provided input and expected output.
    /// </summary>
    /// <param name="input">The input tensor to train with.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// This method trains the neural network with the provided input and expected output tensors.
    /// It sets the network to training mode, performs a forward pass through the network, calculates
    /// the error between the predicted output and the expected output, and backpropagates the error
    /// to update the network's weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the neural network using an example.
    /// 
    /// During training:
    /// 1. The input data is sent through the network (forward pass)
    /// 2. The network makes a prediction
    /// 3. The prediction is compared to the expected output
    /// 4. The error is calculated
    /// 5. The network adjusts its weights to reduce the error
    /// 
    /// This process is repeated with many examples to gradually improve the network's performance.
    /// Each example helps the network learn a little more about the patterns in your data.
    /// </para>
    /// </remarks>
    public void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!Network.SupportsTraining)
        {
            throw new InvalidOperationException("This neural network does not support training.");
        }
        
        // Ensure the network is in training mode
        Network.SetTrainingMode(true);
        
        // Convert tensors to the format expected by the network
        Vector<T> expectedOutputVector = expectedOutput.ToVector();
        
        // Forward pass with memory to store intermediate values for backpropagation
        var output = Network.ForwardWithMemory(input);
        
        // Calculate error gradient
        Vector<T> error = CalculateError(output.ToVector(), expectedOutputVector);
        
        // Backpropagate error
        Network.Backpropagate(Tensor<T>.FromVector(error, expectedOutput.Shape));
        
        // Update weights using the calculated gradients
        Vector<T> gradients = Network.GetParameterGradients();
        Vector<T> currentParams = Network.GetParameters();
        Vector<T> newParams = new Vector<T>(currentParams.Length);
        
        for (int i = 0; i < currentParams.Length; i++)
        {
            // Simple gradient descent: param = param - learningRate * gradient
            T update = _numOps.Multiply(_learningRate, gradients[i]);
            newParams[i] = _numOps.Subtract(currentParams[i], update);
        }
        
        Network.UpdateParameters(newParams);
    }

    /// <summary>
    /// Uses the model to make a prediction for the given input.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method uses the trained neural network to make a prediction for the given input tensor.
    /// It sets the network to prediction mode (not training mode), performs a forward pass through
    /// the network, and returns the output as a tensor with the appropriate shape.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes predictions using what the neural network has learned.
    /// 
    /// When making a prediction:
    /// 1. The input data is sent through the network
    /// 2. Each layer processes the data based on its learned weights
    /// 3. The final layer produces the output (prediction)
    /// 
    /// Unlike training, no weights are updated during prediction - the network
    /// is simply using what it already knows to make its best guess.
    /// </para>
    /// </remarks>
    public Tensor<T> Predict(Tensor<T> input)
    {
        // Set to prediction mode (not training)
        Network.SetTrainingMode(false);
    
        // Forward pass through the network
        return Network.Predict(input);
    }

    /// <summary>
    /// Calculates the error between predicted and expected outputs.
    /// </summary>
    /// <param name="predicted">The predicted output values.</param>
    /// <param name="expected">The expected output values.</param>
    /// <returns>A vector containing the error for each output.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the error between the predicted output values and the expected output values.
    /// The error is calculated using a loss function appropriate for the network's task type (e.g., mean squared error
    /// for regression tasks, cross-entropy for classification tasks). The resulting error vector is used during
    /// backpropagation to update the network's weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method measures how wrong each prediction is compared to 
    /// the expected value. These error values are used to adjust the network's weights during training.
    /// 
    /// Different types of problems use different ways to measure error:
    /// - For predicting numeric values (regression), we often use squared differences
    /// - For classifying into categories, we often use cross-entropy
    /// 
    /// This method automatically chooses the right error measure based on what
    /// kind of problem your network is solving.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateError(Vector<T> predicted, Vector<T> expected)
    {
        // Check if vectors have the same length
        if (predicted.Length != expected.Length)
        {
            throw new ArgumentException("Predicted and expected vectors must have the same length.");
        }

        // Get appropriate loss function based on the task type
        var lossFunction = NeuralNetworkHelper<T>.GetDefaultLossFunction(Architecture.TaskType);
    
        // Calculate gradients based on the loss function
        Vector<T> error = lossFunction.CalculateDerivative(predicted, expected);
    
        return error;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata about the model, including its type, feature count, complexity, and additional
    /// information about the neural network. The metadata includes the model type (Neural Network), the number of
    /// features, the complexity (total parameter count), a description, and additional information such as the
    /// architecture details, layer counts, and activation functions used. This metadata is useful for model selection,
    /// analysis, and visualization.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns detailed information about the neural network model.
    /// 
    /// The metadata includes:
    /// - Basic properties like model type, feature count, and complexity
    /// - Architecture details like layer counts and types
    /// - Statistics about the model's parameters
    /// 
    /// This information is useful for:
    /// - Understanding the model's structure
    /// - Comparing different models
    /// - Analyzing the model's capabilities
    /// - Documenting the model for future reference
    /// </para>
    /// </remarks>
    public ModelMetadata<T> GetModelMetadata()
    {
        int[] layerSizes = Architecture.GetLayerSizes();
        
        return new ModelMetadata<T>
        {
            FeatureCount = FeatureCount,
            Complexity = Complexity,
            Description = $"Neural Network model with {layerSizes.Length} layers",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "LayerSizes", layerSizes },
                { "InputShape", Architecture.GetInputShape() },
                { "OutputShape", Architecture.GetOutputShape() },
                { "TaskType", Architecture.TaskType.ToString() },
                { "InputType", Architecture.InputType.ToString() },
                { "HiddenLayerCount", Architecture.GetHiddenLayerSizes().Length },
                { "ParameterCount", Network.GetParameterCount() },
                { "SupportsTraining", Network.SupportsTraining }
            }
        };
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method serializes the model to a byte array by writing the architecture details and the network parameters.
    /// The serialization format includes the architecture information followed by the network parameters. This allows
    /// the model to be stored or transmitted and later reconstructed using the Deserialize method.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts the neural network model to a byte array that can be saved or transmitted.
    /// 
    /// When serializing the model:
    /// - Both the architecture (structure) and parameters (weights) are saved
    /// - The data is formatted in a way that can be efficiently stored
    /// - The resulting byte array contains everything needed to reconstruct the model
    /// 
    /// This is useful for:
    /// - Saving trained models to disk
    /// - Sharing models with others
    /// - Deploying models to production systems
    /// - Creating model checkpoints during long training processes
    /// </para>
    /// </remarks>
    public byte[] Serialize()
    {
        using MemoryStream ms = new MemoryStream();
        using BinaryWriter writer = new BinaryWriter(ms);
        
        // Write a version number for forward compatibility
        writer.Write(1); // Version 1
        
        // Write the architecture type
        writer.Write(Architecture.GetType().FullName ?? "Unknown");
        
        // Serialize the architecture
        // In a real implementation, we would need a more sophisticated approach
        // Here we just write key architecture properties
        writer.Write((int)Architecture.InputType);
        writer.Write((int)Architecture.TaskType);
        writer.Write((int)Architecture.Complexity);
        writer.Write(Architecture.InputSize);
        writer.Write(Architecture.OutputSize);
        writer.Write(Architecture.InputHeight);
        writer.Write(Architecture.InputWidth);
        writer.Write(Architecture.InputDepth);
        
        // Serialize the network parameters
        var serializedNetwork = Network.Serialize();
        writer.Write(serializedNetwork.Length);
        writer.Write(serializedNetwork);
        
        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the model from a byte array by reading the architecture details and the network parameters.
    /// It expects the same format as produced by the Serialize method: the architecture information followed by the network
    /// parameters. This allows a model that was previously serialized to be reconstructed.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs a neural network model from a byte array created by Serialize.
    /// 
    /// When deserializing the model:
    /// - The architecture is read first to recreate the structure
    /// - Then the parameters (weights) are loaded into that structure
    /// - The resulting model is identical to the one that was serialized
    /// 
    /// This is used when:
    /// - Loading a previously saved model
    /// - Receiving a model from another system
    /// - Resuming training from a checkpoint
    /// 
    /// After deserialization, the model can be used for predictions or further training
    /// just as if it had never been serialized.
    /// </para>
    /// </remarks>
    public void Deserialize(byte[] data)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Serialized data cannot be null or empty.", nameof(data));
        }
        
        using MemoryStream ms = new MemoryStream(data);
        using BinaryReader reader = new BinaryReader(ms);
            
        // Read version number
        int version = reader.ReadInt32();
            
        // Read architecture type
        string architectureType = reader.ReadString();
            
        // Read architecture properties
        InputType inputType = (InputType)reader.ReadInt32();
        NeuralNetworkTaskType taskType = (NeuralNetworkTaskType)reader.ReadInt32();
        NetworkComplexity complexity = (NetworkComplexity)reader.ReadInt32();
        int inputSize = reader.ReadInt32();
        int outputSize = reader.ReadInt32();
        int inputHeight = reader.ReadInt32();
        int inputWidth = reader.ReadInt32();
        int inputDepth = reader.ReadInt32();
            
        // Check if the architecture matches
        if (Architecture.InputType != inputType ||
            Architecture.TaskType != taskType ||
            Architecture.InputSize != inputSize ||
            Architecture.OutputSize != outputSize)
        {
            throw new InvalidOperationException(
                "Serialized network architecture doesn't match this model's architecture.");
        }
        
        var length = reader.ReadInt32();
        var bytes = reader.ReadBytes(length);
        // Deserialize the network parameters
        Network.Deserialize(bytes);
    }

    /// <summary>
    /// Gets all trainable parameters of the neural network as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns all trainable parameters of the neural network as a single vector.
    /// These parameters include weights and biases from all layers that support training.
    /// The vector can be used to save the model's state, apply optimization techniques,
    /// or transfer learning between models.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learned weights and biases from the neural network
    /// into a single list. This is useful for saving the model, optimizing it, or transferring its knowledge.
    /// 
    /// The parameters:
    /// - Are the numbers that the neural network has learned during training
    /// - Include weights (how strongly neurons connect to each other)
    /// - Include biases (baseline activation levels for neurons)
    /// 
    /// A simple network might have hundreds of parameters, while modern deep networks
    /// often have millions or billions of parameters.
    /// </para>
    /// </remarks>
    public Vector<T> GetParameters()
    {
        return Network.GetParameters();
    }

    /// <summary>
    /// Sets the parameters of the neural network.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when parameters has a different length than the network's parameter count.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters of the neural network from a single vector.
    /// These parameters include weights and biases from all layers that support training.
    /// The parameter vector must have the same length as the network's total parameter count.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learned weights and biases in the neural network
    /// from a single list. This is useful for loading saved models or applying optimizations.
    /// 
    /// The parameters:
    /// - Replace all the current weights and biases in the network
    /// - Must be in the same order as returned by GetParameters()
    /// - Must have exactly the right number of values
    /// 
    /// This is commonly used when:
    /// - Loading a saved model
    /// - Applying optimization algorithms
    /// - Transferring knowledge between models
    /// </para>
    /// </remarks>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }
        
        var currentParams = Network.GetParameters();
        if (parameters.Length != currentParams.Length)
        {
            throw new ArgumentException($"Parameters length ({parameters.Length}) must match network's parameter count ({currentParams.Length}).", nameof(parameters));
        }
        
        Network.UpdateParameters(parameters);
    }

    /// <summary>
    /// Updates the model with new parameter values.
    /// </summary>
    /// <param name="parameters">The new parameter values to use.</param>
    /// <returns>The updated model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new model with the same architecture as the current model but with the provided
    /// parameter values. This allows creating a modified version of the model without altering the original.
    /// The new parameters must match the number of parameters in the original model.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you change all the weights and biases in the neural network
    /// at once by providing a list of new values. It's useful when optimizing the model or loading saved weights.
    /// 
    /// When updating parameters:
    /// - A new model is created with the same structure as this one
    /// - The new model's weights and biases are set to the values you provide
    /// - The original model remains unchanged
    /// 
    /// This is useful for:
    /// - Loading pre-trained weights
    /// - Testing different parameter values
    /// - Implementing evolutionary algorithms
    /// - Creating ensemble models with different parameter sets
    /// </para>
    /// </remarks>
    public IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        // Create a new model with the same architecture
        var newModel = new NeuralNetworkModel<T>(Architecture);
    
        // Update the parameters of the new model
        newModel.Network.UpdateParameters(parameters);
    
        return newModel;
    }

    /// <summary>
    /// Gets the indices of all features used by this model.
    /// </summary>
    /// <returns>A collection of feature indices.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the indices of all features that are used by the model. For neural networks,
    /// this typically includes all features from 0 to FeatureCount-1, as neural networks generally use
    /// all input features to some extent.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns a list of which input features the model actually uses.
    /// For neural networks, this typically includes all available features unless specific feature selection has been applied.
    /// 
    /// Unlike some simpler models (like linear regression with feature selection) where
    /// certain inputs might be completely ignored, neural networks typically process
    /// all input features and learn which ones are important during training.
    /// 
    /// This method returns all feature indices from 0 to (FeatureCount-1).
    /// </para>
    /// </remarks>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        // If we have explicitly set active features, return those
        if (_explicitlySetActiveFeatures != null && _explicitlySetActiveFeatures.Count > 0)
        {
            return _explicitlySetActiveFeatures.OrderBy(i => i);
        }

        // Otherwise, neural networks typically use all input features
        // Return indices for all features from 0 to FeatureCount-1
        return Enumerable.Range(0, FeatureCount);
    }

    /// <summary>
    /// Creates a deep copy of this model.
    /// </summary>
    /// <returns>A new instance with the same architecture and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a deep copy of the neural network model, including both its architecture and
    /// learned parameters. The new model is independent of the original, so changes to one will not affect
    /// the other. This is useful for creating variations of a model while preserving the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates an exact duplicate of the neural network,
    /// with the same structure and the same learned weights. This is useful when you need to
    /// make changes to a model without affecting the original.
    /// 
    /// The deep copy:
    /// - Has identical architecture (same layers, neurons, connections)
    /// - Has identical parameters (same weights and biases)
    /// - Is completely independent of the original
    /// 
    /// This is useful for:
    /// - Creating model variants for experimentation
    /// - Saving a checkpoint before making changes
    /// - Creating ensemble models
    /// - Implementing techniques like dropout ensemble
    /// </para>
    /// </remarks>
    public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        // Create a new model with the same architecture
        var copy = new NeuralNetworkModel<T>(Architecture);
        
        // Copy the network parameters
        var parameters = Network.GetParameters();
        copy.Network.UpdateParameters(parameters);
        
        // Copy additional properties
        copy._learningRate = _learningRate;
        copy._isTrainingMode = _isTrainingMode;
        copy.Network.SetTrainingMode(_isTrainingMode);
        
        return copy;
    }

    /// <summary>
    /// Creates a shallow copy of this model.
    /// </summary>
    /// <returns>A new instance with the same architecture and parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a copy of the model that shares the same architecture but has its own set
    /// of parameters. It is equivalent to DeepCopy for this implementation but is provided for compatibility
    /// with the IFullModel interface.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the neural network model.
    /// 
    /// In this implementation, Clone and DeepCopy do the same thing - they
    /// both create a completely independent copy of the model with the same
    /// architecture and parameters. Both methods are provided for compatibility
    /// with the IFullModel interface.
    /// </para>
    /// </remarks>
    public IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        return DeepCopy();
    }

    /// <summary>
    /// Sets which features should be considered active in the model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is negative or greater than the feature count.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which features should be considered active in the neural network model,
    /// overriding the default behavior where all features are considered active. Any features not included
    /// in the provided collection will be considered inactive, even though neural networks typically
    /// use all input features to some extent.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you manually specify which input features
    /// the model should consider important, overriding the default neural network behavior
    /// where all features are typically used.
    /// 
    /// For example, if you have 10 features but want to focus on only features 2, 5, and 7,
    /// you can use this method to specify exactly those features. After setting these features:
    /// - Only these specific features will be reported as active by GetActiveFeatureIndices()
    /// - Only these features will return true when checked with IsFeatureUsed()
    /// 
    /// This can be useful for:
    /// - Feature selection experiments (testing different feature subsets)
    /// - Simplifying model interpretation
    /// - Ensuring consistency across different models
    /// - Highlighting specific features you know are important from domain expertise
    /// 
    /// Note that this doesn't actually modify the neural network's internal calculations -
    /// it just changes what features the model reports as being active.
    /// </para>
    /// </remarks>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
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
            if (index < 0 || index >= FeatureCount)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} must be between 0 and {FeatureCount - 1}.");
            }

            _explicitlySetActiveFeatures.Add(index);
        }
    }

    #region IInterpretableModel Implementation

    protected readonly HashSet<InterpretationMethod> _enabledMethods = new();
    protected Vector<int> _sensitiveFeatures;
    protected readonly List<FairnessMetric> _fairnessMetrics = new();
    protected IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> _baseModel;

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
    public virtual async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(Tensor<T> input)
    {
        return await InterpretableModelHelper.GetLocalFeatureImportanceAsync(this, _enabledMethods, input);
    }

    /// <summary>
    /// Gets SHAP values for the given inputs.
    /// </summary>
    public virtual async Task<Matrix<T>> GetShapValuesAsync(Tensor<T> inputs)
    {
        return await InterpretableModelHelper.GetShapValuesAsync(this, _enabledMethods);
    }

    /// <summary>
    /// Gets LIME explanation for a specific input.
    /// </summary>
    public virtual async Task<LimeExplanation<T>> GetLimeExplanationAsync(Tensor<T> input, int numFeatures = 10)
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
    public virtual async Task<CounterfactualExplanation<T>> GetCounterfactualAsync(Tensor<T> input, Tensor<T> desiredOutput, int maxChanges = 5)
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
    public virtual async Task<string> GenerateTextExplanationAsync(Tensor<T> input, Tensor<T> prediction)
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
    public virtual async Task<FairnessMetrics<T>> ValidateFairnessAsync(Tensor<T> inputs, int sensitiveFeatureIndex)
    {
        return await InterpretableModelHelper.ValidateFairnessAsync<T>(_fairnessMetrics);
    }

    /// <summary>
    /// Gets anchor explanation for a given input.
    /// </summary>
    public virtual async Task<AnchorExplanation<T>> GetAnchorExplanationAsync(Tensor<T> input, T threshold)
    {
        return await InterpretableModelHelper.GetAnchorExplanationAsync(_enabledMethods, threshold);
    }

    /// <summary>
    /// Sets the base model for interpretability analysis.
    /// </summary>
    public virtual void SetBaseModel(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
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