namespace AiDotNet.NeuralNetworks;

using AiDotNet.Interpretability;

/// <summary>
/// Base class for all neural network implementations in AiDotNet.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A neural network is a computing system inspired by the human brain. It consists of 
/// interconnected "layers" of artificial neurons that process information and learn patterns from data.
/// This class provides the foundation for building different types of neural networks.
/// </para>
/// </remarks>
public abstract class NeuralNetworkBase<T> : INeuralNetworkModel<T>
{
    /// <summary>
    /// The collection of layers that make up this neural network.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Layers are the building blocks of neural networks. Each layer contains 
    /// neurons that process information and pass it to the next layer. A typical network has 
    /// an input layer (receives data), hidden layers (process data), and an output layer (produces results).
    /// </remarks>
    protected readonly List<ILayer<T>> Layers;
    
    /// <summary>
    /// The architecture definition for this neural network.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The architecture defines the structure of your neural network - how many layers it has,
    /// how many neurons are in each layer, and how they're connected. Think of it as the blueprint for your network.
    /// </remarks>
    public readonly NeuralNetworkArchitecture<T> Architecture;

    /// <summary>
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This set contains feature indices that have been explicitly set as active through
    /// the SetActiveFeatureIndices method, overriding the automatic determination based
    /// on feature importance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tracks which parts of your input data have been manually
    /// selected as important for the neural network, regardless of what the network would
    /// automatically determine based on weights.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

    /// <summary>
    /// Mathematical operations for the numeric type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;
    
    /// <summary>
    /// Stores the input values for each layer during forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When data flows through the network, we need to remember what values went into each layer.
    /// This is necessary for the learning process (backpropagation).
    /// </remarks>
    protected Dictionary<int, Tensor<T>> _layerInputs = [];
    
    /// <summary>
    /// Stores the output values from each layer during forward pass.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Similar to layer inputs, we also need to remember what values came out of each layer
    /// during the learning process.
    /// </remarks>
    protected Dictionary<int, Tensor<T>> _layerOutputs = [];

    /// <summary>
    /// Random number generator for initialization.
    /// </summary>
    protected Random Random => new();

    /// <summary>
    /// The loss function used to calculate error during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The loss function measures how wrong the network's predictions are.
    /// Different types of problems need different loss functions:
    /// - Classification problems often use Cross Entropy Loss
    /// - Regression problems often use Mean Squared Error
    /// - Ranking problems might use Hinge Loss
    /// 
    /// This is like having different ways to score different games - you wouldn't use the same
    /// scoring system for basketball and golf.
    /// </para>
    /// </remarks>
    protected ILossFunction<T> LossFunction;

    /// <summary>
    /// The last calculated loss value during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The loss value tells you how well your neural network is performing.
    /// A lower loss means better performance. This field stores the most recent loss value
    /// calculated during training, which you can use to track progress.
    /// </para>
    /// </remarks>
    protected T? LastLoss;

    /// <summary>
    /// Indicates whether the network is currently in training mode.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Neural networks behave differently during training versus when they're making predictions.
    /// In training mode, the network keeps track of additional information needed for learning.
    /// </remarks>
    protected bool IsTrainingMode = true;
    
    /// <summary>
    /// Indicates whether this network supports training (learning from data).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Not all neural networks can learn. Some are designed only for making predictions
    /// with pre-set parameters. This property tells you if the network can learn from data.
    /// </remarks>
    public virtual bool SupportsTraining => false;

    /// <summary>
    /// The maximum allowed norm for gradients during training.
    /// </summary>
    protected T MaxGradNorm;

    /// <summary>
    /// Creates a new neural network with the specified architecture.
    /// </summary>
    /// <param name="architecture">The architecture defining the structure of the network.</param>
    protected NeuralNetworkBase(NeuralNetworkArchitecture<T> architecture, ILossFunction<T> lossFunction, double maxGradNorm = 1.0)
    {
        Architecture = architecture;
        Layers = [];
        NumOps = MathHelper.GetNumericOperations<T>();
        MaxGradNorm = NumOps.FromDouble(maxGradNorm);
        LossFunction = lossFunction;
    }

    /// <summary>
    /// Applies gradient clipping to prevent exploding gradients.
    /// </summary>
    /// <param name="gradients">A list of tensors containing the gradients to be clipped.</param>
    /// <remarks>
    /// <para>
    /// This method calculates the total norm of all gradients and scales them down if the norm exceeds
    /// the maximum allowed gradient norm (_maxGradNorm).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as a safety mechanism. Sometimes, the network might try to
    /// make very large adjustments, which can make learning unstable. This method checks if the
    /// adjustments are too big, and if they are, it scales them down to a safe level. It's like
    /// having a speed limiter on a car to prevent it from going too fast and losing control.
    /// </para>
    /// </remarks>
    protected void ClipGradients(List<Tensor<T>> gradients)
    {
        T totalNorm = NumOps.Zero;

        // Calculate total norm
        foreach (var gradient in gradients)
        {
            for (int i = 0; i < gradient.Length; i++)
            {
                totalNorm = NumOps.Add(totalNorm, NumOps.Multiply(gradient[i], gradient[i]));
            }
        }

        totalNorm = NumOps.Sqrt(totalNorm);

        // If total norm exceeds MaxGradNorm, clip each gradient tensor
        if (NumOps.GreaterThan(totalNorm, MaxGradNorm))
        {
            T scalingFactor = NumOps.Divide(MaxGradNorm, totalNorm);
            for (int i = 0; i < gradients.Count; i++)
            {
                gradients[i] = ClipGradient(gradients[i], scalingFactor);
            }
        }
    }

    /// <summary>
    /// Clips the gradient tensor by scaling it with a given factor.
    /// </summary>
    /// <param name="gradient">The gradient tensor to be clipped.</param>
    /// <param name="scalingFactor">The factor by which to scale the gradient.</param>
    /// <returns>The clipped gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method adjusts the gradient by multiplying each of its values by a scaling factor.
    /// It's used as part of the gradient clipping process to prevent the gradients from becoming too large,
    /// which can cause instability in training.
    /// </para>
    /// </remarks>
    private Tensor<T> ClipGradient(Tensor<T> gradient, T scalingFactor)
    {
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = NumOps.Multiply(gradient[i], scalingFactor);
        }

        return gradient;
    }

    /// <summary>
    /// Clips the gradient tensor if its norm exceeds the maximum allowed gradient norm.
    /// </summary>
    /// <param name="gradient">The gradient tensor to be clipped.</param>
    /// <returns>The clipped gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the total norm of the gradient and scales it down if it exceeds
    /// the maximum allowed gradient norm (MaxGradNorm).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a safety mechanism to prevent the "exploding gradient" problem.
    /// If the gradient (which represents how much to change the network's parameters) becomes too large,
    /// it can cause the training to become unstable. This method checks if the gradient is too big,
    /// and if so, it scales it down to a safe level.
    /// </para>
    /// <para>
    /// Think of it like having a speed limiter on a car. If the car (gradient) tries to go too fast,
    /// this method slows it down to a safe speed to prevent losing control during training.
    /// </para>
    /// </remarks>
    protected Tensor<T> ClipGradient(Tensor<T> gradient)
    {
        T totalNorm = NumOps.Zero;

        for (int i = 0; i < gradient.Length; i++)
        {
            totalNorm = NumOps.Add(totalNorm, NumOps.Multiply(gradient[i], gradient[i]));
        }

        totalNorm = NumOps.Sqrt(totalNorm);

        if (NumOps.GreaterThan(totalNorm, MaxGradNorm))
        {
            T scalingFactor = NumOps.Divide(MaxGradNorm, totalNorm);
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] = NumOps.Multiply(gradient[i], scalingFactor);
            }
        }

        return gradient;
    }

    /// <summary>
    /// Clips the gradient vector if its norm exceeds the maximum allowed gradient norm.
    /// </summary>
    /// <param name="gradient">The gradient vector to be clipped.</param>
    /// <returns>The clipped gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the total norm of the gradient vector and scales it down if it exceeds
    /// the maximum allowed gradient norm (MaxGradNorm). It uses the tensor-based ClipGradient method
    /// internally, converting the vector to a tensor and back.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is another safety mechanism to prevent the "exploding gradient" problem,
    /// but specifically for vector inputs. If the gradient (which represents how much to change the 
    /// network's parameters) becomes too large, it can cause the training to become unstable. 
    /// This method checks if the gradient is too big, and if so, it scales it down to a safe level.
    /// </para>
    /// <para>
    /// Think of it like having a volume control on a speaker. If the sound (gradient) gets too loud,
    /// this method turns it down to a comfortable level to prevent distortion (instability in training).
    /// </para>
    /// </remarks>
    protected Vector<T> ClipGradient(Vector<T> gradient)
    {
        return ClipGradient(Tensor<T>.FromVector(gradient)).ToVector();
    }

    /// <summary>
    /// Gets all trainable parameters of the network as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters of the network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks learn by adjusting their "parameters" (also called weights and biases).
    /// This method collects all those adjustable values into a single list so they can be updated during training.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameters()
    {
        int totalParameterCount = GetParameterCount();
        var parameters = new Vector<T>(totalParameterCount);
    
        int currentIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                var layerParameters = layer.GetParameters();
                for (int i = 0; i < layerParameterCount; i++)
                {
                    parameters[currentIndex + i] = layerParameters[i];
                }

                currentIndex += layerParameterCount;
            }
        }
    
        return parameters;
    }

    /// <summary>
    /// Performs backpropagation to compute gradients for network parameters.
    /// </summary>
    /// <param name="outputGradients">The gradients of the loss with respect to the network outputs.</param>
    /// <returns>The gradients of the loss with respect to the network inputs.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation is how neural networks learn. After making a prediction, the network
    /// calculates how wrong it was (the error). Then it works backward through the layers to figure out
    /// how each parameter contributed to that error. This method handles that backward flow of information.
    /// </para>
    /// <para>
    /// The "gradients" are numbers that tell us how to adjust each parameter to reduce the error.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the network is not in training mode or doesn't support training.</exception>
    public virtual Tensor<T> Backpropagate(Tensor<T> outputGradients)
    {
        if (!IsTrainingMode)
        {
            throw new InvalidOperationException("Cannot backpropagate when network is not in training mode");
        }

        if (!SupportsTraining)
        {
            throw new InvalidOperationException("This network does not support backpropagation");
        }

        // Backpropagate through layers in reverse order
        var gradientTensor = outputGradients;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        return gradientTensor;
    }

    /// <summary>
    /// Extracts a single example from a batch tensor and formats it as a tensor with shape [1, features].
    /// </summary>
    /// <param name="batchTensor">The batch tensor to extract from.</param>
    /// <param name="index">The index of the example to extract.</param>
    /// <returns>A tensor containing a single example with shape [1, features].</returns>
    protected Tensor<T> ExtractSingleExample(Tensor<T> batchTensor, int index)
    {
        // Get the vector for this example
        Vector<T> row = batchTensor.GetRow(index);

        // Create a tensor with shape [1, features]
        return new Tensor<T>([1, row.Length], row);
    }

    /// <summary>
    /// Performs a forward pass through the network while storing intermediate values for backpropagation.
    /// </summary>
    /// <param name="input">The input data to the network.</param>
    /// <returns>The output of the network.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method passes data through the network from input to output, but also
    /// remembers all the intermediate values. This is necessary for the learning process, as the network
    /// needs to know these values when figuring out how to improve.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the network doesn't support training.</exception>
    public virtual Tensor<T> ForwardWithMemory(Tensor<T> input)
    {
        if (!SupportsTraining)
        {
            throw new InvalidOperationException("This network does not support training mode");
        }

        Tensor<T> current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            // Store input to each layer for backpropagation
            _layerInputs[i] = current;

            // Forward pass through layer
            current = Layers[i].Forward(current);

            // Store output from each layer for backpropagation
            _layerOutputs[i] = current;
        }

        return current;
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the network.
    /// </summary>
    /// <returns>The total parameter count.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you how many adjustable values (weights and biases) your neural network has.
    /// More complex networks typically have more parameters and can learn more complex patterns, but also
    /// require more data to train effectively.
    /// </remarks>
    public virtual int GetParameterCount()
    {
        return Layers.Sum(layer => layer.ParameterCount);
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This property provides the same information as GetParameterCount(),
    /// showing how many adjustable values (weights and biases) your neural network has.
    /// This is part of the IFullModel interface for consistency with other model types.
    /// </remarks>
    public virtual int ParameterCount => GetParameterCount();

    /// <summary>
    /// Validates that the provided layers form a valid neural network architecture.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Not all combinations of layers make a valid neural network. This method checks that
    /// the layers can properly connect to each other (like making sure puzzle pieces fit together).
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when the layer configuration is invalid.</exception>
    protected virtual void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        if (layers == null || layers.Count < 2)
        {
            throw new ArgumentException("Neural network must have at least 2 layers (1 input layer and 1 output layer).");
        }

        var errors = new List<string>();

        // Check input layer
        if (!IsValidInputLayer(layers[0]))
        {
            errors.Add("The first layer must be a valid input layer.");
        }

        // Check layer connections
        for (int i = 1; i < layers.Count; i++)
        {
            var prevLayer = layers[i - 1];
            var currentLayer = layers[i];

            if (!AreLayersCompatible(prevLayer, currentLayer))
            {
                errors.Add($"Layer {i - 1} is not compatible with Layer {i}.");
            }
        }

        // Check output layer
        if (!IsValidOutputLayer(layers[layers.Count - 1]))
        {
            errors.Add("The last layer must be a valid output layer.");
        }

        // Throw exception if any errors were found
        if (errors.Count > 0)
        {
            throw new ArgumentException($"Invalid layer configuration:\n{string.Join("\n", errors)}");
        }
    }

    /// <summary>
    /// Determines if a layer can serve as a valid input layer for the neural network.
    /// </summary>
    /// <param name="layer">The layer to check.</param>
    /// <returns>True if the layer can be used as an input layer; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The input layer is the first layer of your neural network. It receives the raw data 
    /// you want to process (like image pixels or text features). This method checks if a layer is suitable 
    /// to be the first layer in your network.
    /// </para>
    /// </remarks>
    protected virtual bool IsValidInputLayer(ILayer<T> layer)
    {
        // Check if the layer is specifically designed as an input layer
        if (layer is InputLayer<T>)
            return true;

        // For convolutional networks, the first layer is often a ConvolutionalLayer
        if (layer is ConvolutionalLayer<T>)
            return true;

        // For simple feedforward networks, the first layer might be Dense
        if (layer is DenseLayer<T> denseLayer)
        {
            // Ensure the dense layer doesn't have any inputs (it's the first layer)
            return denseLayer.GetInputShape().Length == 1 && denseLayer.GetInputShape()[0] > 0;
        }

        // For recurrent networks, the first layer might be LSTM or GRU
        if (layer is LSTMLayer<T> || layer is GRULayer<T>)
            return true;

        // If none of the above, it's not a valid input layer
        return false;
    }

    /// <summary>
    /// Determines if a layer can serve as a valid output layer for the neural network.
    /// </summary>
    /// <param name="layer">The layer to check.</param>
    /// <returns>True if the layer can be used as an output layer; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The output layer is the last layer of your neural network. It produces the final result 
    /// (like a prediction or classification). This method checks if a layer is suitable to be the final layer 
    /// in your network. Different tasks need different types of output layers - for example, image classification 
    /// might use a Softmax activation, while regression might use a linear activation.
    /// </para>
    /// </remarks>
    protected virtual bool IsValidOutputLayer(ILayer<T> layer)
    {
        // Most commonly, the output layer is a Dense layer
        if (layer is DenseLayer<T> denseLayer)
        {
            // Ensure the dense layer has an output (it's not empty)
            return denseLayer.GetOutputShape().Length == 1 && denseLayer.GetOutputShape()[0] > 0;
        }

        // For some specific tasks, the output might be from other layer types
        // For example, in sequence-to-sequence models, it could be LSTM or GRU
        if (layer is LSTMLayer<T> || layer is GRULayer<T>)
            return true;

        // For image segmentation tasks, it might be a Convolutional layer
        if (layer is ConvolutionalLayer<T>)
            return true;

        // Check if the layer has an activation function typically used in output layers
        if (layer is ActivationLayer<T> activationLayer)
        {
            // Check if the layer has an activation function typically used in output layers
            var activationTypes = layer.GetActivationTypes();
            return activationTypes.Any(type => type == ActivationFunction.Softmax || type == ActivationFunction.Sigmoid);
        }

        // If none of the above, it's not a valid output layer
        return false;
    }

    /// <summary>
    /// Checks if two consecutive layers can be connected in a neural network.
    /// </summary>
    /// <param name="prevLayer">The preceding layer.</param>
    /// <param name="currentLayer">The current layer to check compatibility with.</param>
    /// <returns>True if the layers can be connected; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks work by connecting layers in sequence. For two layers to connect properly, 
    /// the output of one layer must match what the next layer expects as input. This is like making sure puzzle 
    /// pieces fit together. This method checks if two layers can be properly connected.
    /// </para>
    /// <para>
    /// For example, if a layer outputs 100 values, the next layer should expect 100 values as input. Some layer 
    /// combinations also have special rules - like needing a "Flatten" layer between image processing layers and 
    /// regular dense layers.
    /// </para>
    /// </remarks>
    protected virtual bool AreLayersCompatible(ILayer<T> prevLayer, ILayer<T> currentLayer)
    {
        // Check if the output shape of the previous layer matches the input shape of the current layer
        if (!Enumerable.SequenceEqual(prevLayer.GetOutputShape(), currentLayer.GetInputShape()))
            return false;

        // Special checks for specific layer combinations
        if (prevLayer is ConvolutionalLayer<T> && currentLayer is DenseLayer<T>)
        {
            // Ensure there's a Flatten layer between Conv and Dense
            return false;
        }

        if (prevLayer is PoolingLayer<T> && currentLayer is LSTMLayer<T>)
        {
            // Pooling directly to LSTM is usually not valid
            return false;
        }

        // Check for dimension compatibility in case of Reshape or Flatten layers
        if (prevLayer is ReshapeLayer<T> reshapeLayer)
        {
            return reshapeLayer.GetOutputShape().Aggregate((a, b) => a * b) == 
                   currentLayer.GetInputShape().Aggregate((a, b) => a * b);
        }

        // If no incompatibilities found, layers are considered compatible
        return true;
    }

    /// <summary>
    /// Retrieves the gradients for all trainable parameters in the network.
    /// </summary>
    /// <returns>A vector containing all parameter gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When a neural network learns, it needs to know how to adjust each of its internal values 
    /// (parameters). These adjustments are called "gradients" - they tell the network which direction and how much 
    /// to change each parameter. This method collects all those adjustment values into a single list.
    /// </para>
    /// <para>
    /// Think of gradients as a recipe for improvement: "increase this weight by 0.01, decrease that one by 0.03," etc.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetParameterGradients()
    {
        // Collect gradients from all layers
        List<Vector<T>> allGradients = [];
    
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                allGradients.Add(layer.GetParameterGradients());
            }
        }
    
        // Concatenate all gradients into a single vector
        if (allGradients.Count == 0)
        {
            return new Vector<T>(0);
        }
    
        return Vector<T>.Concatenate(allGradients.ToArray());
    }

    /// <summary>
    /// Ensures the architecture is initialized before training begins.
    /// </summary>
    protected void EnsureArchitectureInitialized()
    {
        if (!Architecture.IsInitialized)
        {
            // Initialize from cached data
            Architecture.InitializeFromCachedData<Tensor<T>, Tensor<T>>();

            // Initialize network-specific layers
            InitializeLayers();
        }
    }

    /// <summary>
    /// Initializes the layers of the neural network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method sets up all the layers in your neural network according to the architecture 
    /// you've defined. It's like assembling the parts of your network before you can use it.
    /// </remarks>
    protected abstract void InitializeLayers();

    /// <summary>
    /// Makes a prediction using the neural network.
    /// </summary>
    /// <param name="input">The input data to process.</param>
    /// <returns>The network's prediction.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is the main method you'll use to get results from your trained neural network. 
    /// You provide some input data (like an image or text), and the network processes it through all its 
    /// layers to produce an output (like a classification or prediction).
    /// </remarks>
    public abstract Tensor<T> Predict(Tensor<T> input);

    /// <summary>
    /// Updates the network's parameters with new values.
    /// </summary>
    /// <param name="parameters">The new parameter values to set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, a neural network's internal values (parameters) get adjusted to improve 
    /// its performance. This method allows you to update all those values at once by providing a complete set 
    /// of new parameters.
    /// </para>
    /// <para>
    /// This is typically used by optimization algorithms that calculate better parameter values based on 
    /// training data.
    /// </para>
    /// </remarks>
    public abstract void UpdateParameters(Vector<T> parameters);

    /// <summary>
    /// Sets the neural network to either training or inference mode.
    /// </summary>
    /// <param name="isTraining">True to enable training mode; false to enable inference mode.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks behave differently during training versus when making predictions.
    /// </para>
    /// <para>
    /// When in training mode (isTraining = true):
    /// - The network keeps track of intermediate calculations needed for learning
    /// - Certain layers like Dropout and BatchNormalization behave differently
    /// - The network uses more memory but can learn from its mistakes
    /// </para>
    /// <para>
    /// When in inference/prediction mode (isTraining = false):
    /// - The network only performs forward calculations
    /// - It uses less memory and runs faster
    /// - It cannot learn or update its parameters
    /// </para>
    /// <para>
    /// Think of it like the difference between taking a practice test (training mode) where you 
    /// can check your answers and learn from mistakes, versus taking the actual exam (inference mode)
    /// where you just give your best answers based on what you've already learned.
    /// </para>
    /// </remarks>
    public virtual void SetTrainingMode(bool isTraining)
    {
        if (SupportsTraining)
        {
            IsTrainingMode = isTraining;
        }
    }

    /// <summary>
    /// Gets the loss value from the most recent training iteration.
    /// </summary>
    /// <returns>The loss value from the last training iteration, or zero if no training has occurred.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the error/loss value calculated during the most recent call to the Train method.
    /// It's useful for monitoring the training progress and implementing early stopping.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how well your network is learning.
    /// 
    /// The loss value is a measure of how far off your network's predictions are from the correct answers.
    /// - A high loss means the network is making big mistakes
    /// - A low loss means the network is getting closer to the right answers
    /// 
    /// By tracking this value over time, you can:
    /// - See if your network is improving
    /// - Decide when to stop training (when the loss stops decreasing)
    /// - Compare different network designs to see which learns better
    /// 
    /// Think of it like a score in a game - the lower the score, the better your network is performing.
    /// </para>
    /// </remarks>
    public virtual T GetLastLoss()
    {
        // If we haven't calculated a loss yet, return a default value
        if (LastLoss == null || NumOps.IsNaN(LastLoss))
        {
            return NumOps.Zero;
        }

        return LastLoss;
    }

    /// <summary>
    /// Trains the neural network on a single input-output pair.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="expectedOutput">The expected output for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method performs one training step on the neural network using the provided input and expected output.
    /// It updates the network's parameters to reduce the error between the network's prediction and the expected output.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is how your neural network learns. You provide:
    /// - An input (what the network should process)
    /// - The expected output (what the correct answer should be)
    /// 
    /// The network then:
    /// 1. Makes a prediction based on the input
    /// 2. Compares its prediction to the expected output
    /// 3. Calculates how wrong it was (the loss)
    /// 4. Adjusts its internal values to do better next time
    /// 
    /// After training, you can get the loss value using the GetLastLoss() method to see how well the network is learning.
    /// </para>
    /// </remarks>
    public abstract void Train(Tensor<T> input, Tensor<T> expectedOutput);

    /// <summary>
    /// Gets the metadata for this neural network model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <summary>
    /// Resets the internal state of the different layers, clearing any remembered information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state (hidden state and cell state) of all layers in the network.
    /// This is useful when starting to process a new, unrelated sequence or when the network's memory
    /// should be cleared before making new predictions.
    /// </para>
    /// <para><b>For Beginners:</b> This clears the neural network's memory to start fresh.
    /// 
    /// Think of this like:
    /// - Wiping the slate clean before starting a new task
    /// - Erasing the neural network's "memory" so past inputs don't influence new predictions
    /// - Starting fresh when processing a completely new sequence
    /// 
    /// For example, if you've been using an neural network to analyze one document and now want to
    /// analyze a completely different document, you would reset the state first to avoid
    /// having the first document influence the analysis of the second one.
    /// </para>
    /// </remarks>
    public virtual void ResetState()
    {
        foreach (var layer in Layers)
        {
            layer.ResetState();
        }
    }

    /// <summary>
    /// Saves the model to a file.
    /// </summary>
    /// <param name="filePath">The path where the model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the entire neural network, including all layers and parameters,
    /// and saves it to the specified file path.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This saves your trained neural network to a file on your computer.
    ///
    /// Think of it like saving a document - you can later load the model back from the file
    /// and use it to make predictions without having to retrain it from scratch.
    ///
    /// This is useful when:
    /// - You've finished training and want to save your model
    /// - You want to use the model in a different application
    /// - You need to share the model with others
    /// - You want to deploy the model to production
    /// </para>
    /// </remarks>
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Serializes the neural network to a byte array.
    /// </summary>
    /// <returns>A byte array representing the serialized neural network.</returns>
    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Write the number of layers
        writer.Write(Layers.Count);

        // Write each layer's type and shape
        foreach (var layer in Layers)
        {
            // Write layer type
            writer.Write(layer.GetType().Name);

            // Write input shape
            var inputShape = layer.GetInputShape();
            writer.Write(inputShape.Length);
            foreach (var dim in inputShape)
            {
                writer.Write(dim);
            }

            // Write output shape
            var outputShape = layer.GetOutputShape();
            writer.Write(outputShape.Length);
            foreach (var dim in outputShape)
            {
                writer.Write(dim);
            }

            // Write parameter count
            writer.Write(layer.ParameterCount);

            // Write parameters if any
            if (layer.ParameterCount > 0)
            {
                var parameters = layer.GetParameters();
                foreach (var param in parameters)
                {
                    writer.Write(Convert.ToDouble(param));
                }
            }
        }

        // Write network-specific data
        SerializeNetworkSpecificData(writer);

        return ms.ToArray();
    }

    /// <summary>
    /// Deserializes the neural network from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized neural network data.</param>
    /// <summary>
    /// Deserializes the neural network from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized neural network data.</param>
    public virtual void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Clear existing layers
        Layers.Clear();

        // Read the number of layers
        int layerCount = reader.ReadInt32();

        // Read and recreate each layer
        for (int i = 0; i < layerCount; i++)
        {
            // Read layer type
            string layerType = reader.ReadString();

            // Read input shape
            int inputShapeLength = reader.ReadInt32();
            int[] inputShape = new int[inputShapeLength];
            for (int j = 0; j < inputShapeLength; j++)
            {
                inputShape[j] = reader.ReadInt32();
            }

            // Read output shape
            int outputShapeLength = reader.ReadInt32();
            int[] outputShape = new int[outputShapeLength];
            for (int j = 0; j < outputShapeLength; j++)
            {
                outputShape[j] = reader.ReadInt32();
            }

            // Read parameter count
            int paramCount = reader.ReadInt32();

            // Create the layer (without checking for additional params)
            var layer = DeserializationHelper.CreateLayerFromType<T>(layerType, inputShape, outputShape, null);

            // Read and set parameters if any
            if (paramCount > 0)
            {
                var parameters = new Vector<T>(paramCount);
                for (int j = 0; j < paramCount; j++)
                {
                    parameters[j] = NumOps.FromDouble(reader.ReadDouble());
                }
                // Update layer parameters
                layer.UpdateParameters(parameters);
            }

            // Add the layer to the network
            Layers.Add(layer);
        }

        // Read network-specific data
        DeserializeNetworkSpecificData(reader);
    }

    /// <summary>
    /// Serializes network-specific data that is not covered by the general serialization process.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method is called at the end of the general serialization process to allow derived classes
    /// to write any additional data specific to their implementation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as packing a special compartment in your suitcase. 
    /// While the main serialization method packs the common items (layers, parameters), 
    /// this method allows each specific type of neural network to pack its own unique items 
    /// that other networks might not have.
    /// </para>
    /// </remarks>
    protected abstract void SerializeNetworkSpecificData(BinaryWriter writer);

    /// <summary>
    /// Deserializes network-specific data that was not covered by the general deserialization process.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method is called at the end of the general deserialization process to allow derived classes
    /// to read any additional data specific to their implementation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Continuing the suitcase analogy, this is like unpacking that special 
    /// compartment. After the main deserialization method has unpacked the common items (layers, parameters), 
    /// this method allows each specific type of neural network to unpack its own unique items 
    /// that were stored during serialization.
    /// </para>
    /// </remarks>
    protected abstract void DeserializeNetworkSpecificData(BinaryReader reader);

    /// <summary>
    /// Creates a new neural network with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use for the new network.</param>
    /// <returns>A new neural network with the specified parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new neural network that is a copy of this one, but with different parameter values.
    /// It's useful for creating variations of a model without retraining or for ensemble methods.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of this as creating a copy of your neural network but with different
    /// internal settings. It's like having the same blueprint for a house but using different materials.
    /// 
    /// This is useful when you want to:
    /// - Try different variations of a trained model
    /// - Create an ensemble of similar models with different parameters
    /// - Manually adjust model parameters without retraining
    /// 
    /// The new model will have the same structure but different parameter values.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        // Create a deep copy of the current network
        var newNetwork = (NeuralNetworkBase<T>)DeepCopy();
    
        // Update the parameters of the new network
        newNetwork.UpdateParameters(parameters);
    
        return newNetwork;
    }

    /// <summary>
    /// Gets the indices of input features that are actively used by the network.
    /// </summary>
    /// <returns>A collection of indices representing the active features.</returns>
    /// <remarks>
    /// <para>
    /// This method determines which input features have the most influence on the network's output
    /// by analyzing the weights of the first layer. Features with larger absolute weights are
    /// considered more active or important.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This helps you understand which parts of your input data the network
    /// considers most important for making predictions.
    /// 
    /// For example, if your inputs are:
    /// - Age (index 0)
    /// - Income (index 1)
    /// - Education level (index 2)
    /// 
    /// And this method returns [0, 2], it means the network relies heavily on age and education level,
    /// but not so much on income when making its predictions.
    /// 
    /// This can help you:
    /// - Understand what your model is paying attention to
    /// - Potentially simplify your model by removing unused features
    /// - Gain insights about the problem you're solving
    /// </para>
    /// </remarks>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // If the network has no layers, return an empty list
        if (Layers.Count == 0)
            return Array.Empty<int>();
    
        // Get the first layer for analysis
        var firstLayer = Layers[0];
    
        // If the first layer is not a dense or convolutional layer, we can't easily determine active features
        if (!(firstLayer is DenseLayer<T> || firstLayer is ConvolutionalLayer<T>))
        {
            // Return all indices as potentially active (conservative approach)
            return Enumerable.Range(0, firstLayer.GetInputShape()[0]);
        }
    
        // Get the weights from the first layer
        Vector<T> weights = firstLayer.GetParameters();
        int inputSize = firstLayer.GetInputShape()[0];
        int outputSize = firstLayer.GetOutputShape()[0];
    
        // Calculate feature importance by summing absolute weights per input feature
        var featureImportance = new Dictionary<int, T>();
    
        for (int i = 0; i < inputSize; i++)
        {
            T importance = NumOps.Zero;
        
            // For each neuron in the first layer, add the absolute weight for this feature
            for (int j = 0; j < outputSize; j++)
            {
                // In most layers, weights are organized as [input1-neuron1, input2-neuron1, ..., input1-neuron2, ...]
                int weightIndex = j * inputSize + i;
            
                if (weightIndex < weights.Length)
                {
                    importance = NumOps.Add(importance, NumOps.Abs(weights[weightIndex]));
                }
            }
        
            featureImportance[i] = importance;
        }
    
        // Sort features by importance and get the top 50% (or at least 1)
        int featuresCount = Math.Max(1, inputSize / 2);
    
        return featureImportance
            .OrderByDescending(pair => Convert.ToDouble(pair.Value))
            .Take(featuresCount)
            .Select(pair => pair.Key);
    }

    /// <summary>
    /// Determines if a specific input feature is actively used by the network.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is actively used; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if a specific input feature has a significant influence on the network's
    /// output based on the weights in the first layer. A feature is considered used if its
    /// associated weights have non-negligible magnitudes.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you whether a specific piece of your input data matters
    /// to the neural network's decisions.
    /// 
    /// For example, if your inputs include age, income, and education level, this method can
    /// tell you whether the network is actually using age (or any other specific feature) when
    /// making predictions.
    /// 
    /// This is useful for:
    /// - Understanding what information your model uses
    /// - Simplifying your inputs by removing unused features
    /// - Debugging models that ignore features you think should be important
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

        // If feature index is out of range, it's not used
        if (Layers.Count == 0 || featureIndex < 0 || featureIndex >= Layers[0].GetInputShape()[0])
            return false;
    
        // Get active feature indices
        var activeIndices = GetActiveFeatureIndices().ToList();
    
        // Check if the specified index is in the active indices
        return activeIndices.Contains(featureIndex);
    }

    /// <summary>
    /// Creates a deep copy of the neural network.
    /// </summary>
    /// <returns>A new instance that is a deep copy of this neural network.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a complete independent copy of the network, including all layers
    /// and their parameters. It uses serialization and deserialization to ensure a true deep copy.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a completely independent duplicate of your neural network.
    /// 
    /// Think of it like creating an exact clone of your network where:
    /// - The copy has the same structure (layers, connections)
    /// - The copy has the same learned parameters (weights, biases)
    /// - Changes to one network don't affect the other
    /// 
    /// This is useful when you want to:
    /// - Experiment with modifications without risking your original network
    /// - Create multiple variations of a model
    /// - Save a snapshot of your model at a particular point in training
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        // The most reliable way to create a deep copy is through serialization/deserialization
        byte[] serialized = Serialize();
    
        // Create a new instance of the same type as this network
        var copy = CreateNewInstance();
    
        // Load the serialized data into the new instance
        copy.Deserialize(serialized);
    
        return copy;
    }

    /// <summary>
    /// Creates a clone of the neural network.
    /// </summary>
    /// <returns>A new instance that is a clone of this neural network.</returns>
    /// <remarks>
    /// <para>
    /// For most neural networks, Clone and DeepCopy perform the same function - creating a complete
    /// independent copy of the network. Some specialized networks might implement this differently.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates an identical copy of your neural network.
    /// 
    /// In most cases, this works the same as DeepCopy and creates a completely independent
    /// duplicate of your network. The duplicate will have the same structure and the same
    /// learned parameters, but changes to one won't affect the other.
    /// </para>
    /// </remarks>
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        // By default, Clone behaves the same as DeepCopy
        return DeepCopy();
    }

    /// <summary>
    /// Creates a new instance of the same type as this neural network.
    /// </summary>
    /// <returns>A new instance of the same neural network type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a blank version of the same type of neural network.
    /// 
    /// It's used internally by methods like DeepCopy and Clone to create the right type of
    /// network before copying the data into it.
    /// </para>
    /// </remarks>
    protected abstract IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance();

    /// <summary>
    /// Sets which input features should be considered active in the neural network.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is negative or exceeds the input dimension.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which input features should be considered active
    /// in the neural network, overriding the automatic determination based on weights.
    /// Any features not included in the provided collection will be considered inactive,
    /// regardless of their weights in the network.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method lets you manually select which parts of your input data
    /// the neural network should pay attention to. For example, if your inputs include various
    /// measurements or features, you can tell the network to focus only on specific ones
    /// that you know are important based on your domain knowledge.
    ///
    /// This can be useful for:
    /// - Forcing the network to use features you know are important
    /// - Ignoring features you know are irrelevant or noisy
    /// - Testing how the network performs with different feature subsets
    /// - Implementing feature selection techniques
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

        // Get the input dimension to validate feature indices
        int inputDimension = 0;
        if (Layers.Count > 0)
        {
            inputDimension = Layers[0].GetInputShape()[0];
        }

        // Add the new feature indices
        foreach (var index in featureIndices)
        {
            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} cannot be negative.");
            }

            if (inputDimension > 0 && index >= inputDimension)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} exceeds the input dimension {inputDimension}.");
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

    #region INeuralNetworkModel Implementation

    /// <summary>
    /// Gets the intermediate activations from each layer when processing the given input.
    /// </summary>
    public virtual Dictionary<string, Tensor<T>> GetLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        var current = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <summary>
    /// Gets the architectural structure of the neural network.
    /// </summary>
    public virtual NeuralNetworkArchitecture<T> GetArchitecture()
    {
        return Architecture;
    }

    #endregion

    /// <summary>
    /// Gets the feature importance scores for the model.
    /// </summary>
    /// <returns>A dictionary mapping feature names to their importance scores.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the importance of each input feature by analyzing the weights
    /// in the first layer of the neural network. Features with larger absolute weights are
    /// considered more important to the model's predictions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you which parts of your input data are most important
    /// for the neural network's decisions.
    ///
    /// For example, if you're predicting house prices with features like size, location, and age,
    /// this method might tell you that "location" has an importance of 0.8, "size" has 0.6,
    /// and "age" has 0.2 - meaning the network relies heavily on location and size, but less on age.
    ///
    /// This is useful for:
    /// - Understanding what your model pays attention to
    /// - Explaining model decisions to others
    /// - Identifying which features matter most
    /// - Simplifying your model by removing unimportant features
    /// </para>
    /// </remarks>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();

        // If the network has no layers, return an empty dictionary
        if (Layers.Count == 0)
            return importance;

        // Get the first layer for analysis
        var firstLayer = Layers[0];

        // If the first layer is not a dense or convolutional layer, we can't easily determine importance
        if (!(firstLayer is DenseLayer<T> || firstLayer is ConvolutionalLayer<T>))
        {
            // Return uniform importance for all features (conservative approach)
            int inputSize = firstLayer.GetInputShape()[0];
            T uniformImportance = NumOps.FromDouble(1.0 / inputSize);

            for (int i = 0; i < inputSize; i++)
            {
                importance[$"Feature_{i}"] = uniformImportance;
            }

            return importance;
        }

        // Get the weights from the first layer
        Vector<T> weights = firstLayer.GetParameters();
        int featureCount = firstLayer.GetInputShape()[0];
        int outputSize = firstLayer.GetOutputShape()[0];

        // Calculate feature importance by summing absolute weights per input feature
        var featureScores = new Dictionary<int, T>();

        for (int i = 0; i < featureCount; i++)
        {
            T score = NumOps.Zero;

            // For each neuron in the first layer, add the absolute weight for this feature
            for (int j = 0; j < outputSize; j++)
            {
                // In most layers, weights are organized as [input1-neuron1, input2-neuron1, ..., input1-neuron2, ...]
                int weightIndex = j * featureCount + i;

                if (weightIndex < weights.Length)
                {
                    score = NumOps.Add(score, NumOps.Abs(weights[weightIndex]));
                }
            }

            featureScores[i] = score;
        }

        // Normalize the scores to sum to 1
        T totalScore = featureScores.Values.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));

        if (NumOps.GreaterThan(totalScore, NumOps.Zero))
        {
            foreach (var kvp in featureScores)
            {
                importance[$"Feature_{kvp.Key}"] = NumOps.Divide(kvp.Value, totalScore);
            }
        }
        else
        {
            // If all scores are zero, use uniform importance
            T uniformImportance = NumOps.FromDouble(1.0 / featureCount);
            for (int i = 0; i < featureCount; i++)
            {
                importance[$"Feature_{i}"] = uniformImportance;
            }
        }

        return importance;
    }

    /// <summary>
    /// Sets the parameters of the neural network.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to all layers in the network.
    /// The parameters should be in the same format as returned by GetParameters.
    /// </para>
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        int totalParameterCount = GetParameterCount();
        if (parameters.Length != totalParameterCount)
        {
            throw new ArgumentException($"Expected {totalParameterCount} parameters, got {parameters.Length}");
        }

        int currentIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                // Extract parameters for this layer
                var layerParameters = new Vector<T>(layerParameterCount);
                for (int i = 0; i < layerParameterCount; i++)
                {
                    layerParameters[i] = parameters[currentIndex + i];
                }

                // Set the layer's parameters
                layer.SetParameters(layerParameters);
                currentIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Adds a layer to the neural network.
    /// </summary>
    /// <param name="layerType">The type of layer to add.</param>
    /// <param name="units">The number of units/neurons in the layer.</param>
    /// <param name="activation">The activation function to use.</param>
    public virtual void AddLayer(LayerType layerType, int units, ActivationFunction activation)
    {
        // Get input size from previous layer or use units as default
        int inputSize = Layers.Count > 0 ? Layers[Layers.Count - 1].GetOutputShape()[0] : units;

        // Create activation function from enum
        var activationFunc = ActivationFunctionFactory<T>.CreateActivationFunction(activation);

        ILayer<T> layer = layerType switch
        {
            LayerType.Dense => new DenseLayer<T>(inputSize, units, activationFunc),
            _ => throw new NotSupportedException($"Layer type {layerType} not supported in AddLayer method")
        };
        Layers.Add(layer);
    }

    /// <summary>
    /// Adds a convolutional layer to the neural network.
    /// </summary>
    public virtual void AddConvolutionalLayer(int filters, int kernelSize, int stride, ActivationFunction activation)
    {
        throw new NotImplementedException(
            "AddConvolutionalLayer requires additional parameters that are not provided in this method signature. " +
            "Use ConvolutionalLayer.Configure() with the full input shape, or create the layer directly with " +
            "new ConvolutionalLayer<T>(inputDepth, outputDepth, kernelSize, inputHeight, inputWidth, stride, padding, activation) " +
            "and add it to Layers manually.");
    }

    /// <summary>
    /// Adds an LSTM layer to the neural network.
    /// </summary>
    public virtual void AddLSTMLayer(int units, bool returnSequences = false)
    {
        throw new NotImplementedException(
            "AddLSTMLayer requires additional parameters that are not provided in this method signature. " +
            "Create the layer directly with new LSTMLayer<T>(inputSize, hiddenSize, inputShape, activation, recurrentActivation) " +
            "and add it to Layers manually.");
    }

    /// <summary>
    /// Adds a dropout layer to the neural network.
    /// </summary>
    public virtual void AddDropoutLayer(double dropoutRate)
    {
        var layer = new DropoutLayer<T>(dropoutRate);
        Layers.Add(layer);
    }

    /// <summary>
    /// Adds a batch normalization layer to the neural network.
    /// </summary>
    /// <param name="featureSize">The number of features to normalize.</param>
    /// <param name="epsilon">A small constant for numerical stability (default: 1e-5).</param>
    /// <param name="momentum">The momentum for running statistics (default: 0.9).</param>
    public virtual void AddBatchNormalizationLayer(int featureSize, double epsilon = 1e-5, double momentum = 0.9)
    {
        var layer = new BatchNormalizationLayer<T>(featureSize, epsilon, momentum);
        Layers.Add(layer);
    }

    /// <summary>
    /// Adds a pooling layer to the neural network.
    /// </summary>
    /// <param name="inputShape">The input shape (channels, height, width).</param>
    /// <param name="poolingType">The type of pooling operation.</param>
    /// <param name="poolSize">The size of the pooling window.</param>
    /// <param name="strides">The step size when moving the pooling window (default: same as poolSize).</param>
    public virtual void AddPoolingLayer(int[] inputShape, PoolingType poolingType, int poolSize, int? strides = null)
    {
        var layer = new MaxPoolingLayer<T>(inputShape, poolSize, strides ?? poolSize);
        Layers.Add(layer);
    }

    /// <summary>
    /// Gets the gradients from all layers in the neural network.
    /// </summary>
    /// <returns>A vector containing all gradients from all layers concatenated together.</returns>
    /// <remarks>
    /// <para>
    /// This method collects the gradients from every layer in the network and combines them
    /// into a single vector. This is useful for optimization algorithms that need access to
    /// all gradients at once.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> During training, each layer calculates how its parameters should change
    /// (the gradients). This method gathers all those gradients from every layer and puts them
    /// into one long list.
    ///
    /// Think of it like:
    /// - Each layer has notes about how to improve (gradients)
    /// - This method collects all those notes into one document
    /// - The optimizer can then use this document to update the entire network
    ///
    /// This is essential for the learning process, as it tells the optimizer how to adjust
    /// all the network's parameters to improve performance.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetGradients()
    {
        var allGradients = new List<T>();

        foreach (var layer in Layers)
        {
            var layerGradients = layer.GetParameterGradients();
            if (layerGradients != null && layerGradients.Length > 0)
            {
                for (int i = 0; i < layerGradients.Length; i++)
                {
                    allGradients.Add(layerGradients[i]);
                }
            }
        }

        return new Vector<T>(allGradients.ToArray());
    }

    /// <summary>
    /// Gets the architecture of the neural network.
    /// </summary>
    /// <returns>The neural network architecture defining the structure and configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the architecture definition for this neural network, which includes
    /// information about the layers, their configurations, and how they are connected.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The architecture is like a blueprint of your neural network. It describes
    /// the structure - how many layers there are, what type each layer is, and how they connect.
    /// This is useful for understanding, saving, or recreating the network structure.
    /// </para>
    /// </remarks>
    public virtual NeuralNetworkArchitecture<T> GetArchitecture()
    {
        return Architecture;
    }

    /// <summary>
    /// Gets the input shape expected by the neural network.
    /// </summary>
    /// <returns>An array representing the dimensions of the input.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the shape of input data that the network expects. For example,
    /// if the network expects images of size 28x28 pixels, this might return [28, 28].
    /// If it expects a vector of 100 features, it would return [100].
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you what size and shape of data the network needs as input.
    /// Think of it like knowing what size batteries a device needs - you need to provide the right
    /// dimensions of data for the network to work properly.
    /// </para>
    /// </remarks>
    public virtual int[] GetInputShape()
    {
        if (Layers.Count == 0)
        {
            return Array.Empty<int>();
        }

        return Layers[0].GetInputShape();
    }

    /// <summary>
    /// Gets the activations (outputs) from each layer for a given input.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>A dictionary mapping layer index to layer activation tensors.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input through the network and captures the output of each layer.
    /// This is useful for visualizing what each layer is detecting, debugging the network, or
    /// implementing techniques like feature extraction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This shows you what each layer in your neural network "sees" or produces
    /// when given an input. It's like following a signal through a circuit and measuring the output
    /// at each component. This helps you understand what patterns each layer is detecting.
    ///
    /// For example, in an image recognition network:
    /// - Early layers might detect edges and simple shapes
    /// - Middle layers might detect parts of objects (like eyes or wheels)
    /// - Later layers might detect whole objects
    ///
    /// This method lets you see all of these intermediate representations.
    /// </para>
    /// </remarks>
    public virtual Dictionary<int, Tensor<T>> GetLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<int, Tensor<T>>();

        if (Layers.Count == 0)
        {
            return activations;
        }

        var currentInput = input;

        for (int i = 0; i < Layers.Count; i++)
        {
            var layer = Layers[i];
            var output = layer.Forward(currentInput);
            activations[i] = output;
            currentInput = output;
        }

        return activations;
    }
}