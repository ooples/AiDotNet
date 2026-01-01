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
public class ResidualNeuralNetwork<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (deep supervision) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Deep supervision adds auxiliary classifiers at intermediate layers to help gradient flow in very deep networks.
    /// This is particularly useful for ResNets with 100+ layers to prevent vanishing gradients.
    /// </para>
    /// <para><b>For Beginners:</b> Deep supervision is like having multiple teachers check your work at different stages.
    ///
    /// In very deep networks:
    /// - Gradients (learning signals) can become very weak by the time they reach early layers
    /// - Adding intermediate classifiers helps maintain strong gradients throughout the network
    /// - Each intermediate classifier provides additional supervision to guide learning
    ///
    /// When enabled, the network learns from both:
    /// - The final output (main loss)
    /// - Intermediate predictions (auxiliary losses)
    ///
    /// This is optional and most useful for very deep networks (100+ layers).
    /// For shallower networks (< 50 layers), it may not provide significant benefits.
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the deep supervision auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much the intermediate auxiliary classifiers contribute to the total loss.
    /// The total loss is: main_loss + (auxiliary_weight * auxiliary_loss).
    /// Typical values range from 0.1 to 0.5.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the network should care about intermediate predictions.
    ///
    /// The weight determines the balance between:
    /// - Final output accuracy (main loss)
    /// - Intermediate prediction accuracy (auxiliary loss)
    ///
    /// Common values:
    /// - 0.3 (default): Balanced contribution from intermediate classifiers
    /// - 0.1-0.2: Less emphasis on intermediate predictions
    /// - 0.4-0.5: More emphasis on intermediate predictions
    ///
    /// Higher values make the network focus more on getting intermediate predictions correct,
    /// which can help with gradient flow but may slow convergence.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastDeepSupervisionLoss;
    private List<ILayer<T>> _auxiliaryClassifiers = new();
    private readonly List<int> _auxiliaryClassifierPositions = new();
    private List<List<ILayer<T>>> _auxiliaryClassifierLayers = new();
    private Vector<T>? _lastExpectedOutput;

    /// <summary>
    /// Adds an auxiliary classifier at the specified layer position for deep supervision.
    /// </summary>
    /// <param name="classifier">The classifier layer to add for intermediate predictions.</param>
    /// <param name="layerPosition">The layer index where this classifier should be applied.</param>
    /// <remarks>
    /// <para>
    /// Auxiliary classifiers enable deep supervision by providing additional training signals
    /// at intermediate layers. This helps with gradient flow and can improve training stability.
    /// </para>
    /// <para><b>For Beginners:</b> Think of auxiliary classifiers as "checkpoints" in your network.
    /// They make predictions at intermediate stages, helping the network learn better representations
    /// at each layer rather than only at the final output.
    /// </para>
    /// </remarks>
    public void AddAuxiliaryClassifier(ILayer<T> classifier, int layerPosition)
    {
        if (classifier == null)
        {
            throw new ArgumentNullException(nameof(classifier));
        }

        if (layerPosition < 0 || layerPosition >= Layers.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(layerPosition),
                $"Layer position must be between 0 and {Layers.Count - 1}");
        }

        _auxiliaryClassifiers.Add(classifier);
        _auxiliaryClassifierPositions.Add(layerPosition);
    }

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
    private T _learningRate;

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

        // Initialize NumOps-based fields
        AuxiliaryLossWeight = NumOps.FromDouble(0.3);
        _lastDeepSupervisionLoss = NumOps.Zero;

        InitializeResidualLayers();
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
        InitializeResidualLayers();
    }

    private void InitializeResidualLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayersInternal(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultResNetLayers(Architecture));
        }

        // Automatically add auxiliary classifiers for deep supervision if enabled
        if (UseAuxiliaryLoss && Layers.Count > 3)
        {
            InitializeAuxiliaryClassifiers();
        }
    }

    /// <summary>
    /// Automatically initializes auxiliary classifiers at strategic positions based on network depth.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adds auxiliary classifiers at evenly-spaced positions throughout the network
    /// to enable deep supervision. The number and positions are determined automatically based on
    /// the total network depth. Each auxiliary classifier consists of a 2-layer fully connected
    /// network that maps from intermediate representations to the final output space.
    /// </para>
    /// <para><b>For Beginners:</b> This automatically adds "checkpoints" throughout your network
    /// that help it learn better by providing training signals at multiple depths, not just at the end.
    /// </para>
    /// </remarks>
    private void InitializeAuxiliaryClassifiers()
    {
        // Determine how many auxiliary classifiers to add based on network depth
        // Add one auxiliary classifier for every 10-15 layers, but at least 1 and at most 3
        int totalLayers = Layers.Count;
        int numAuxClassifiers = Math.Max(1, Math.Min(3, totalLayers / 12));

        // Calculate positions evenly spaced through the network, avoiding first and last layers
        int spacing = totalLayers / (numAuxClassifiers + 1);

        for (int i = 1; i <= numAuxClassifiers; i++)
        {
            int position = spacing * i;

            // Ensure position is valid and not too close to the end
            if (position < totalLayers - 2 && position > 0)
            {
                // Get the output shape at this position
                var layerOutputShape = Layers[position].GetOutputShape();
                int intermediateSize = layerOutputShape[0];

                // Create a simple 2-layer auxiliary classifier: intermediate -> hidden -> output
                int hiddenSize = Math.Max(128, Architecture.OutputSize * 2);

                // Create the auxiliary classifier layers using existing helpers
                IActivationFunction<T> hiddenActivation = new ReLUActivation<T>();
                IActivationFunction<T> outputActivation = NeuralNetworkHelper<T>.GetDefaultActivationFunction(Architecture.TaskType);

                // First layer: intermediate representation -> hidden layer
                var hiddenLayer = new DenseLayer<T>(intermediateSize, hiddenSize, hiddenActivation);

                // Second layer: hidden -> output
                var outputLayer = new DenseLayer<T>(hiddenSize, Architecture.OutputSize, outputActivation);

                // Store layers as a list that will be executed sequentially
                var classifierLayers = new List<ILayer<T>> { hiddenLayer, outputLayer };
                _auxiliaryClassifierLayers.Add(classifierLayers);
                _auxiliaryClassifierPositions.Add(position);
            }
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
    /// Computes the auxiliary loss for deep supervision from intermediate auxiliary classifiers.
    /// </summary>
    /// <returns>The computed deep supervision auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the auxiliary loss from intermediate classifiers placed at strategic
    /// positions in the network. For very deep ResNets, these intermediate classifiers help
    /// maintain strong gradient signals throughout the network during backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how well the network's intermediate layers are learning.
    ///
    /// Deep supervision works by:
    /// 1. Adding small classifiers at intermediate points in the network
    /// 2. Each classifier tries to predict the final output from intermediate features
    /// 3. Computing loss for each intermediate prediction
    /// 4. Averaging these losses to get the auxiliary loss
    ///
    /// This helps because:
    /// - It provides learning signals to earlier layers
    /// - It prevents gradients from becoming too weak in deep networks
    /// - It encourages intermediate layers to learn meaningful features
    ///
    /// The auxiliary loss is combined with the main loss during training to guide learning.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _auxiliaryClassifierLayers.Count == 0)
        {
            _lastDeepSupervisionLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Ensure we have intermediate activations and expected output
        if (_layerOutputs.Count == 0 || _lastExpectedOutput == null)
        {
            _lastDeepSupervisionLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Compute auxiliary loss from each intermediate classifier
        T totalAuxiliaryLoss = NumOps.Zero;
        int validClassifiers = 0;

        for (int i = 0; i < _auxiliaryClassifierLayers.Count; i++)
        {
            int layerPosition = _auxiliaryClassifierPositions[i];

            // Check if we have the intermediate activation at this position
            if (!_layerOutputs.ContainsKey(layerPosition))
            {
                continue;
            }

            var intermediateActivation = _layerOutputs[layerPosition];
            var classifierLayers = _auxiliaryClassifierLayers[i];

            // Pass intermediate activation through auxiliary classifier layers sequentially
            var current = intermediateActivation;
            foreach (var layer in classifierLayers)
            {
                current = layer.Forward(current);
            }

            // Compute loss for this auxiliary prediction
            Vector<T> auxPredictionVector = current.ToVector();
            T classifierLoss = LossFunction.CalculateLoss(auxPredictionVector, _lastExpectedOutput);

            totalAuxiliaryLoss = NumOps.Add(totalAuxiliaryLoss, classifierLoss);
            validClassifiers++;
        }

        // Average the losses across all auxiliary classifiers
        if (validClassifiers > 0)
        {
            _lastDeepSupervisionLoss = NumOps.Divide(totalAuxiliaryLoss, NumOps.FromDouble(validClassifiers));
        }
        else
        {
            _lastDeepSupervisionLoss = NumOps.Zero;
        }

        return _lastDeepSupervisionLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the deep supervision auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about auxiliary losses.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about the deep supervision system, including
    /// the number of auxiliary classifiers, their positions in the network, and the computed losses.
    /// This information is useful for monitoring training progress and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how deep supervision is working.
    ///
    /// The diagnostics include:
    /// - Total auxiliary loss from all intermediate classifiers
    /// - Weight applied to the auxiliary loss
    /// - Number of auxiliary classifiers in the network
    /// - Whether deep supervision is enabled
    ///
    /// This helps you:
    /// - Monitor if auxiliary classifiers are contributing to training
    /// - Debug issues with deep supervision
    /// - Understand the impact of intermediate supervision on learning
    ///
    /// You can use this information to adjust the auxiliary loss weight or
    /// the placement of auxiliary classifiers for better training results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalDeepSupervisionLoss", _lastDeepSupervisionLoss?.ToString() ?? "0" },
            { "AuxiliaryWeight", AuxiliaryLossWeight?.ToString() ?? "0.3" },
            { "UseDeepSupervision", UseAuxiliaryLoss.ToString() },
            { "NumberOfAuxiliaryClassifiers", _auxiliaryClassifierLayers.Count.ToString() },
            { "AuxiliaryClassifierPositions", string.Join(", ", _auxiliaryClassifierPositions) }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Includes auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including auxiliary loss diagnostics.
    /// </returns>
    public Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
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
    /// If deep supervision is enabled and auxiliary classifiers are configured, auxiliary losses from intermediate classifiers are included.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the ResNet to recognize patterns in your data.
    ///
    /// The training process works like this:
    /// 1. Divides your data into smaller batches for efficient processing
    /// 2. For each batch:
    ///    - Feeds the input data through the network
    ///    - Compares the prediction with the expected output
    ///    - Calculates how wrong the prediction was (the "loss")
    ///    - If deep supervision is enabled, also computes losses from intermediate classifiers
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
                    var x = batchX.GetRow(i);
                    var y = batchY.GetRow(i);

                    // Convert input vector to tensor once before forward pass
                    var xTensor = Tensor<T>.FromVector(x);

                    // Forward pass with memory to save intermediate states
                    var prediction = ForwardWithMemory(xTensor);

                    // Cache prediction vector to avoid repeated conversions
                    Vector<T> predictionVector = prediction.ToVector();

                    // Calculate main loss
                    T loss = LossFunction.CalculateLoss(predictionVector, y);

                    // Add auxiliary loss if enabled
                    if (UseAuxiliaryLoss)
                    {
                        // Cache expected output for auxiliary classifiers
                        _lastExpectedOutput = y;
                        T auxLoss = ComputeAuxiliaryLoss();
                        T weightedAuxLoss = NumOps.Multiply(AuxiliaryLossWeight, auxLoss);
                        loss = NumOps.Add(loss, weightedAuxLoss);
                    }

                    totalLoss = NumOps.Add(totalLoss, loss);

                    // Calculate output gradients
                    Vector<T> outputGradients = LossFunction.CalculateDerivative(predictionVector, y);

                    // Convert output gradients to tensor once before backpropagation
                    var outputGradientsTensor = Tensor<T>.FromVector(outputGradients);

                    // Backpropagate to compute gradients for all parameters
                    Backpropagate(outputGradientsTensor);

                    // Accumulate gradients - convert once before adding
                    var gradients = GetParameterGradients();
                    var gradientsTensor = Tensor<T>.FromVector(gradients);
                    totalGradient = totalGradient.Add(gradientsTensor);
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
    /// <returns>A ModelMetaData object containing information about the model.</returns>
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
