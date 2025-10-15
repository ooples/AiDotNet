namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Residual Neural Network, which is a type of neural network that uses skip connections to address the vanishing gradient problem in deep networks.
/// </summary>
/// <remarks>
/// <para>
/// A Residual Neural Network (ResNet) is an advanced neural network architecture that introduces "skip connections" or "shortcuts"
/// that allow information to bypass one or more layers. These residual connections help address the vanishing gradient problem
/// that occurs in very deep networks, enabling the training of networks with many more layers than previously possible.
/// ResNets were a breakthrough in deep learning that significantly improved performance on image recognition and other tasks.
/// </para>
/// <para><b>For Beginners:</b> A Residual Neural Network is like a highway system for information in a neural network.
/// 
/// Think of it like this:
/// - In a traditional neural network, information must pass through every layer sequentially
/// - In a ResNet, there are "shortcut paths" or "highways" that let information skip ahead
/// 
/// For example, imagine trying to pass a message through a line of 100 people:
/// - In a regular network, each person must whisper to the next person in line
/// - In a ResNet, some people can also shout directly to someone 5 positions ahead
/// 
/// This design solves a major problem: in very deep networks (many layers), information and learning signals
/// tend to fade away or "vanish" as they travel through many layers. The shortcuts in ResNets help information
/// flow more easily through the network, allowing for much deeper networks (some with over 100 layers!)
/// that can learn more complex patterns.
/// 
/// ResNets revolutionized image recognition and are now used in many AI systems that need to identify
/// complex patterns in data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ResidualNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the learning rate for parameter updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate controls how quickly the model adapts to the training data.
    /// It determines the size of the steps taken during gradient descent optimization.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate is like the size of steps when learning.
    /// 
    /// Think of it as:
    /// - Large steps (high learning rate): Move quickly toward the goal but might overshoot
    /// - Small steps (low learning rate): Move carefully but might take a long time
    /// 
    /// Finding the right balance is important:
    /// - Too high: learning becomes unstable, weights oscillate wildly
    /// - Too low: learning takes very long, might get stuck in suboptimal solutions
    /// 
    /// Typical values range from 0.0001 to 0.1, with 0.01 being a common starting point.
    /// </para>
    /// </remarks>
    private T _learningRate = default!;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An epoch represents one complete pass through the entire training dataset.
    /// This property defines how many times the network will iterate through the training dataset.
    /// </para>
    /// <para><b>For Beginners:</b> Epochs are like complete study sessions with your training data.
    /// 
    /// Each epoch:
    /// - Processes every example in your training dataset once
    /// - Updates the network's understanding based on all examples
    /// - Helps the network get incrementally better at its task
    /// 
    /// More epochs generally lead to better learning, but too many can cause the network
    /// to memorize the training data rather than learning general patterns (overfitting).
    /// </para>
    /// </remarks>
    private int _epochs;

    /// <summary>
    /// Gets or sets the batch size for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The batch size determines how many training examples are processed before updating the model parameters.
    /// Smaller batches provide more frequent updates but with higher variance, while larger batches
    /// provide more stable but less frequent updates.
    /// </para>
    /// <para><b>For Beginners:</b> Batch size is like how many examples you study at once before updating your knowledge.
    /// 
    /// When training the network:
    /// - Small batch size (e.g., 16-32): More frequent but noisier updates
    /// - Large batch size (e.g., 128-256): Less frequent but more stable updates
    /// 
    /// The benefits of batching:
    /// - More efficient than processing one example at a time
    /// - Provides a balance between update frequency and stability
    /// - Helps avoid getting stuck in poor solutions
    /// 
    /// Common batch sizes range from 16 to 256, with 32 or 64 being popular choices.
    /// </para>
    /// </remarks>
    private int _batchSize;

    /// <summary>
    /// Indicates whether this network supports training (learning from data).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates whether the network is capable of learning from data through training.
    /// For ResidualNeuralNetwork, this property always returns true since the network is designed for training.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the network can learn from data.
    /// 
    /// The Residual Neural Network supports training, which means:
    /// - It can adjust its internal values based on examples
    /// - It can improve its performance over time
    /// - It can learn to recognize patterns in data
    /// 
    /// This property always returns true because ResNets are specifically designed
    /// to be trainable, even when they're very deep (many layers).
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ResidualNeuralNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the ResNet.</param>
    /// <param name="learningRate">The learning rate for training. Default is 0.01 converted to type T.</param>
    /// <param name="epochs">The number of training epochs. Default is 10.</param>
    /// <param name="batchSize">The batch size for training. Default is 32.</param>
    /// <param name="lossFunction">Optional custom loss function. If null, a default will be chosen based on task type.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Residual Neural Network with the specified architecture.
    /// It initializes the network layers based on the architecture, or creates default ResNet layers if
    /// no specific layers are provided.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the Residual Neural Network with its basic structure.
    /// 
    /// When creating a new ResNet:
    /// - The architecture defines what the network looks like - how many layers it has, how they're connected, etc.
    /// - The constructor prepares the network by either:
    ///   * Using the specific layers provided in the architecture, or
    ///   * Creating default layers designed for ResNets if none are specified
    /// 
    /// The default ResNet layers include special residual blocks that have both:
    /// - A main path where information is processed through multiple layers
    /// - A shortcut path that allows information to skip these layers
    /// 
    /// This combination of paths is what gives ResNets their special ability to train very deep networks.
    /// </para>
    /// </remarks>
    public ResidualNeuralNetwork(
        NeuralNetworkArchitecture<T> architecture, 
        T? learningRate = default, 
        int epochs = 10, 
        int batchSize = 32,
        ILossFunction<T>? lossFunction = null) 
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _learningRate = learningRate ?? NumOps.FromDouble(0.01);
        _epochs = epochs;
        _batchSize = batchSize;
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the neural network layers for the Residual Neural Network. If the architecture
    /// provides specific layers, those are used. Otherwise, a default configuration optimized for ResNets
    /// is created. In a typical ResNet, this involves creating residual blocks that combine a main path
    /// with a shortcut path, allowing information to either pass through layers or bypass them.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the neural network.
    /// 
    /// When initializing layers:
    /// - If the user provided specific layers, those are used
    /// - Otherwise, default layers suitable for ResNets are created automatically
    /// - The system checks that any custom layers will work properly with the ResNet
    /// 
    /// A typical ResNet has specialized building blocks called "residual blocks" that contain:
    /// - Convolutional layers that process the input
    /// - Batch normalization layers that stabilize learning
    /// - Activation layers that introduce non-linearity
    /// - Shortcut connections that allow information to bypass these layers
    /// 
    /// These blocks are then stacked together, often with increasing complexity as you go deeper into the network.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultResNetLayers(Architecture));
        }
    }

    /// <summary>
    /// Updates the parameters of the residual neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the residual neural network based on the provided parameter
    /// updates. The parameters vector is divided into segments corresponding to each layer's parameter count,
    /// and each segment is applied to its respective layer. In a ResNet, these parameters typically include weights
    /// for convolutional layers, as well as parameters for batch normalization and other operations within residual blocks.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates how the ResNet makes decisions based on training.
    /// 
    /// During training:
    /// - The network learns by adjusting its internal parameters
    /// - This method applies those adjustments
    /// - Each layer gets the portion of updates meant specifically for it
    /// 
    /// For a ResNet, these adjustments might include:
    /// - How each convolutional filter detects patterns
    /// - How the batch normalization layers stabilize learning
    /// - How information should flow through both the main and shortcut paths
    /// 
    /// The residual connections (shortcuts) make it easier for these updates to flow backward through the network
    /// during training, which helps very deep networks learn effectively.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.GetSubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the Residual Neural Network.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass through the network to generate a prediction based on the input tensor.
    /// The input flows through all layers sequentially, with residual connections allowing information to bypass
    /// certain layers where applicable. The output represents the network's prediction, which depends on the task
    /// (e.g., class probabilities for classification or continuous values for regression).
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the network to make a prediction based on input data.
    /// 
    /// The prediction process works like this:
    /// - Input data enters the network at the first layer
    /// - The data passes through each layer in sequence
    /// - At residual blocks, there are two paths:
    ///   * A main path through multiple processing layers
    ///   * A shortcut path that bypasses these layers
    /// - The outputs from both paths are combined at the end of each block
    /// - The final layer produces the prediction result
    /// 
    /// For example, in an image recognition task:
    /// - The input might be an image
    /// - Each layer detects increasingly complex patterns
    /// - The shortcuts help information flow through the entire network
    /// - The output tells you what the image contains
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Perform forward pass through all layers sequentially
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        
        return current;
    }

    /// <summary>
    /// Trains the Residual Neural Network on the provided data.
    /// </summary>
    /// <param name="input">The input training data.</param>
    /// <param name="expectedOutput">The expected output for the given input.</param>
    /// <remarks>
    /// <para>
    /// This method trains the Residual Neural Network on the provided data for the specified number of epochs.
    /// It divides the data into batches and trains on each batch using backpropagation and gradient descent.
    /// The method tracks and reports the average loss for each epoch to monitor training progress.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the ResNet to recognize patterns in your data.
    /// 
    /// The training process works like this:
    /// 1. Divides your data into smaller batches for efficient processing
    /// 2. For each batch:
    ///    - Feeds the input data through the network
    ///    - Compares the prediction with the expected output
    ///    - Calculates how wrong the prediction was (the "loss")
    ///    - Adjusts the network's parameters to reduce errors
    /// 3. Repeats this process for multiple epochs (complete passes through the data)
    /// 
    /// The special residual connections in the ResNet help the error signals flow backward
    /// through the network more effectively, making it possible to train very deep networks
    /// that would otherwise suffer from the vanishing gradient problem.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Make sure we're in training mode
        SetTrainingMode(true);
        
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            T totalLoss = NumOps.Zero;
            
            // Process data in batches
            for (int batchStart = 0; batchStart < input.Shape[0]; batchStart += _batchSize)
            {
                // Get a batch of data
                int batchEnd = Math.Min(batchStart + _batchSize, input.Shape[0]);
                int actualBatchSize = batchEnd - batchStart;
                var batchX = input.Slice(batchStart, 0, batchEnd, input.Shape[1]);
                var batchY = expectedOutput.Slice(batchStart, 0, batchEnd, expectedOutput.Shape[1]);
                
                // Reset gradients at the start of each batch
                var totalGradient = new Tensor<T>([GetParameterCount()], Vector<T>.CreateDefault(GetParameterCount(), NumOps.Zero));
                
                // Accumulate gradients for each example in the batch
                for (int i = 0; i < actualBatchSize; i++)
                {
                    var x = ExtractSingleExample(batchX, i);
                    var y = ExtractSingleExample(batchY, i).ToVector();

                    // Forward pass with memory to save intermediate states
                    var prediction = ForwardWithMemory(x).ToVector();
                    
                    // Calculate loss and gradients for this example
                    T loss = LossFunction.CalculateLoss(prediction, y);
                    totalLoss = NumOps.Add(totalLoss, loss);
                    
                    // Calculate output gradients
                    Vector<T> outputGradients = LossFunction.CalculateDerivative(prediction, y);
                    
                    // Backpropagate to compute gradients for all parameters
                    Backpropagate(Tensor<T>.FromVector(outputGradients, expectedOutput.Shape));
                    
                    // Accumulate gradients
                    var gradients = GetParameterGradients();
                    totalGradient = totalGradient.Add(Tensor<T>.FromVector(gradients));
                }
                
                // Average the gradients across the batch
                totalGradient = new Tensor<T>(totalGradient.Shape, totalGradient.ToVector().Divide(NumOps.FromDouble(actualBatchSize)));
                
                // Update parameters with averaged gradients
                var currentParams = GetParameters();
                var updatedParams = new Vector<T>(currentParams.Length);
                for (int j = 0; j < currentParams.Length; j++)
                {
                    updatedParams[j] = NumOps.Subtract(
                        currentParams[j], 
                        NumOps.Multiply(_learningRate, totalGradient.ToVector()[j]));
                }
                
                UpdateParameters(updatedParams);
            }
            
            // Calculate average loss for the epoch
            T avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(input.Shape[0]));
        }
        
        // Set back to inference mode after training
        SetTrainingMode(false);
    }
   
    /// <summary>
    /// Gets metadata about the Residual Neural Network model.
    /// </summary>
    /// <returns>A ModelMetadata object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata that describes the Residual Neural Network, including its type,
    /// architecture details, and training parameters. This information can be useful for model
    /// management, documentation, and versioning.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your network's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (Residual Neural Network)
    /// - The number of layers in the network
    /// - Information about the network's structure
    /// - Training parameters like learning rate and epochs
    /// 
    /// This is useful for:
    /// - Documenting your model
    /// - Comparing different model configurations
    /// - Reproducing your model setup later
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var layerSizes = Layers.Select(layer => layer.GetOutputShape()[0]).ToList();
        
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ResidualNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfLayers", Layers.Count },
                { "LayerSizes", layerSizes },
                { "Epochs", _epochs },
                { "LearningRate", Convert.ToDouble(_learningRate) },
                { "BatchSize", _batchSize },
                { "InputSize", Architecture.CalculatedInputSize },
                { "OutputSize", Architecture.CalculateOutputSize() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data for the Residual Neural Network.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the training parameters specific to the Residual Neural Network to the provided BinaryWriter.
    /// These parameters include the number of epochs, learning rate, and batch size, which are crucial for
    /// reconstructing the network's training configuration during deserialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the special settings for training this ResNet.
    /// 
    /// It writes:
    /// - The number of times to train on the entire dataset (epochs)
    /// - How quickly the network learns from its mistakes (learning rate)
    /// - How many examples the network looks at before updating (batch size)
    /// 
    /// These settings are important because they affect how the network learns and performs.
    /// Saving them allows you to recreate the exact same training setup later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write training parameters
        writer.Write(_epochs);
        writer.Write(Convert.ToDouble(_learningRate));
        writer.Write(_batchSize);
    }

    /// <summary>
    /// Deserializes network-specific data for the Residual Neural Network.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the training parameters specific to the Residual Neural Network from the provided BinaryReader.
    /// It restores the number of epochs, learning rate, and batch size, ensuring that the network's training
    /// configuration is accurately reconstructed during deserialization.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the special settings for training this ResNet.
    /// 
    /// It reads:
    /// - The number of times to train on the entire dataset (epochs)
    /// - How quickly the network learns from its mistakes (learning rate)
    /// - How many examples the network looks at before updating (batch size)
    /// 
    /// Loading these settings ensures that you can continue training or use the network
    /// with the exact same configuration it had when it was saved.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read training parameters
        _epochs = reader.ReadInt32();
        _learningRate = NumOps.FromDouble(reader.ReadDouble());
        _batchSize = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of the residual neural network with the same configuration.
    /// </summary>
    /// <returns>
    /// A new instance of <see cref="ResidualNeuralNetwork{T}"/> with the same configuration as the current instance.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method creates a new residual neural network that has the same configuration as the current instance.
    /// It's used for model persistence, cloning, and transferring the model's configuration to new instances.
    /// The new instance will have the same architecture, learning rate, epochs, batch size, and loss function
    /// as the original, but will not share parameter values unless they are explicitly copied after creation.
    /// </para>
    /// <para><b>For Beginners:</b> This method makes a fresh copy of the current model with the same settings.
    /// 
    /// It's like creating a blueprint copy of your network that can be used to:
    /// - Save your model's settings
    /// - Create a new identical model
    /// - Transfer your model's configuration to another system
    /// 
    /// This is useful when you want to:
    /// - Create multiple similar residual neural networks
    /// - Save a model's configuration for later use
    /// - Reset a model while keeping its settings
    /// 
    /// Note that while the settings are copied, the learned parameters (like the weights for detecting features)
    /// are not automatically transferred, so the new instance will need training or parameter copying
    /// to match the performance of the original.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create a new instance with the cloned architecture and the same parameters
        return new ResidualNeuralNetwork<T>(
            Architecture,
            _learningRate,
            _epochs,
            _batchSize,
            LossFunction
        );
    }
}