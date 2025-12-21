namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Deep Belief Network, a generative graphical model composed of multiple layers of Restricted Boltzmann Machines.
/// </summary>
/// <remarks>
/// <para>
/// A Deep Belief Network (DBN) is a probabilistic, generative model composed of multiple layers of stochastic 
/// latent variables. It is built by stacking multiple Restricted Boltzmann Machines (RBMs), where each RBM's 
/// hidden layer serves as the input layer for the next RBM. DBNs are trained using a two-phase approach: 
/// an unsupervised pre-training phase followed by a supervised fine-tuning phase. This allows them to learn 
/// complex patterns in data even with limited labeled examples.
/// </para>
/// <para><b>For Beginners:</b> A Deep Belief Network is like a tower of pattern-recognizing layers.
/// 
/// Imagine building a tower where:
/// - Each floor of the tower is a Restricted Boltzmann Machine (RBM)
/// - The bottom floor learns simple patterns from the raw data
/// - Each higher floor learns more complex patterns based on what the floor below it discovered
/// - The tower is built and trained one floor at a time, from bottom to top
/// 
/// For example, if analyzing images of faces:
/// - The first floor might learn to detect edges and basic shapes
/// - The middle floors might recognize features like eyes, noses, and mouths
/// - The top floors might identify complete facial expressions or identities
/// 
/// This layer-by-layer approach helps the network discover meaningful patterns even when you don't have a lot of labeled examples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DeepBeliefNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the list of RBM layers for greedy layer-wise pre-training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property stores the Restricted Boltzmann Machine (RBM) layers that form the core of the DBN.
    /// These layers are used during the pre-training phase, where each layer is trained greedily as a separate RBM.
    /// </para>
    /// <para><b>For Beginners:</b> These are the individual "floors" of our pattern-recognition tower.
    /// 
    /// Each RBM layer:
    /// - Learns patterns independently during pre-training
    /// - Takes input from the layer below and provides output to the layer above
    /// - Has its own set of weights and biases that capture patterns at different levels of abstraction
    /// 
    /// During pre-training, we train each floor separately, then combine them into a complete tower.
    /// </para>
    /// </remarks>
    private List<RBMLayer<T>> _rbmLayers;

    /// <summary>
    /// Gets or sets the learning rate for parameter updates during fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate controls how quickly the model adapts to the training data during fine-tuning.
    /// A higher learning rate means bigger parameter updates, which can lead to faster convergence but risks overshooting.
    /// A lower learning rate means smaller updates, which can be more precise but may take longer to converge.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate is like the size of steps when fine-tuning the network.
    /// 
    /// Think of it as:
    /// - Large steps (high learning rate): Move quickly toward the goal but might overshoot
    /// - Small steps (low learning rate): Move carefully but might take a long time
    /// 
    /// Finding the right balance is important for effective training.
    /// Typical values range from 0.0001 to 0.1, with 0.01 being a common starting point.
    /// </para>
    /// </remarks>
    private T _learningRate;

    /// <summary>
    /// Gets or sets the number of epochs for fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An epoch represents one complete pass through the entire training dataset.
    /// This property defines how many times the network will iterate through the training dataset during fine-tuning.
    /// </para>
    /// <para><b>For Beginners:</b> Epochs are like complete study sessions with your training data.
    /// 
    /// Each epoch:
    /// - Processes every example in your training dataset once
    /// - Updates the network's understanding based on all examples
    /// - Helps the network get incrementally better at its task
    /// 
    /// More epochs generally lead to better learning, but too many can cause the network to memorize
    /// the training data rather than learning general patterns (overfitting).
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
    /// Gets or sets the loss function used for fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The loss function measures how well the network's predictions match the expected outputs.
    /// It provides a signal that guides the network's learning during the fine-tuning phase.
    /// Different tasks require different loss functions, so this property allows you to specify
    /// the appropriate loss function for your specific task.
    /// </para>
    /// <para><b>For Beginners:</b> The loss function is like a scorecard that tells the network how well it's doing.
    /// 
    /// Think of it as:
    /// - A way to measure the difference between the network's predictions and the correct answers
    /// - A signal that guides the network's learning process
    /// - Different tasks need different ways of measuring performance
    /// 
    /// If no loss function is provided, the network automatically selects one based on the task type:
    /// - For classification tasks: Cross-entropy loss
    /// - For regression tasks: Mean squared error loss
    /// </para>
    /// </remarks>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Indicates whether the network supports training (learning from data).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This tells you if the network can learn from data.
    /// 
    /// The Deep Belief Network supports both:
    /// - Unsupervised pre-training (learning patterns without labels)
    /// - Supervised fine-tuning (improving performance for specific tasks)
    /// 
    /// This property always returns true because DBNs are designed to learn and improve with training.
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="DeepBeliefNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration, which must include RBM layers.</param>
    /// <param name="learningRate">The learning rate for fine-tuning. Default is 0.01 converted to type T.</param>
    /// <param name="epochs">The number of epochs for fine-tuning. Default is 10.</param>
    /// <param name="batchSize">The batch size for training. Default is 32.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Deep Belief Network with the specified architecture. The architecture
    /// must include a collection of Restricted Boltzmann Machines (RBMs) that will form the pre-training layers
    /// of the network. The constructor initializes the network by setting up these RBM layers.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new Deep Belief Network with your chosen settings.
    /// 
    /// When creating a Deep Belief Network:
    /// - You provide an "architecture" that defines how the network is structured
    /// - The architecture must include a set of RBM layers (the floors of our tower)
    /// - The constructor sets up the initial structure, but doesn't train the network yet
    /// 
    /// Think of it like designing a blueprint for the tower before construction begins.
    /// </para>
    /// </remarks>
    public DeepBeliefNetwork(
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
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _rbmLayers = [];

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the Deep Belief Network based on the architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method sets up the layers of the Deep Belief Network. If custom layers are provided in the architecture,
    /// those layers are used. Otherwise, default layers are created based on the architecture's specifications.
    /// After setting up the regular layers, the method validates the RBM layers to ensure they have compatible
    /// dimensions and are properly configured.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the actual structure of the network.
    /// 
    /// When initializing the layers:
    /// - If you've specified your own custom layers, the network will use those
    /// - If not, the network will create a standard set of layers
    /// - The method also checks that the RBM layers (our tower floors) are properly designed
    /// - Each floor must connect properly to the floors above and below it
    /// 
    /// This is like making sure all the pieces of your tower will fit together properly
    /// before you start building it.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepBeliefNetworkLayers(Architecture));
        }

        _rbmLayers.AddRange(Layers.OfType<RBMLayer<T>>());

        ValidateRbmLayers();
    }

    /// <summary>
    /// Validates that the RBM layers form a valid sequence for a Deep Belief Network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method checks that the RBM layers are properly configured for a Deep Belief Network.
    /// It ensures that the input size of each RBM matches the output size of the previous RBM,
    /// creating a chain where each layer's output becomes the next layer's input.
    /// </para>
    /// <para><b>For Beginners:</b> This makes sure all the RBM layers connect properly to each other.
    /// 
    /// For a valid tower of RBMs:
    /// - The first RBM's input layer size must match the network's input size
    /// - Each RBM's output size must match the next RBM's input size
    /// - This creates a chain where information flows smoothly up the tower
    /// 
    /// If these connections don't match up, the network can't function properly,
    /// similar to how misaligned floors would make a building structurally unsound.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when RBM layers are not properly configured.</exception>
    private void ValidateRbmLayers()
    {
        if (_rbmLayers.Count == 0)
        {
            throw new ArgumentException("Deep Belief Network requires at least one RBM layer.");
        }

        // Check that the first RBM's input size matches the network's input size
        var firstRbmInputShape = _rbmLayers[0].GetInputShape();
        var firstRbmInputSize = firstRbmInputShape[0];

        if (firstRbmInputSize != Architecture.CalculatedInputSize)
        {
            throw new ArgumentException($"The first RBM layer's input size ({firstRbmInputSize}) must match the network's input size ({Architecture.CalculatedInputSize}).");
        }

        // Check that each RBM's output size matches the next RBM's input size
        for (int i = 0; i < _rbmLayers.Count - 1; i++)
        {
            var currentRbmOutputShape = _rbmLayers[i].GetOutputShape();
            var nextRbmInputShape = _rbmLayers[i + 1].GetInputShape();

            var currentRbmOutputSize = currentRbmOutputShape[0];
            var nextRbmInputSize = nextRbmInputShape[0];

            if (currentRbmOutputSize != nextRbmInputSize)
            {
                throw new ArgumentException($"RBM layer {i}'s output size ({currentRbmOutputSize}) must match RBM layer {i + 1}'s input size ({nextRbmInputSize}).");
            }
        }
    }

    /// <summary>
    /// Performs unsupervised pre-training of the DBN using greedy layer-wise approach.
    /// </summary>
    /// <param name="trainingData">The training data tensor.</param>
    /// <param name="pretrainingEpochs">The number of epochs for pre-training each RBM layer. Default is 10.</param>
    /// <param name="pretrainingLearningRate">The learning rate for pre-training. Default is 0.1 converted to type T.</param>
    /// <param name="cdSteps">The number of contrastive divergence steps. Default is 1.</param>
    /// <remarks>
    /// <para>
    /// This method implements the greedy layer-wise pre-training algorithm for Deep Belief Networks.
    /// Each RBM layer is trained separately, starting from the bottom layer and moving up. After a layer
    /// is trained, the training data is transformed through that layer to create the training data for the next layer.
    /// This bottom-up approach helps the network learn a hierarchical representation of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches each floor of the tower one at a time, from bottom to top.
    /// 
    /// The pre-training process works like this:
    /// 1. Start with the raw input data and train the bottom RBM layer
    /// 2. Use the bottom layer to transform the data and train the second layer
    /// 3. Continue this process, training each layer with data transformed by all layers below it
    /// 
    /// This step-by-step approach:
    /// - Helps the network learn increasingly abstract patterns
    /// - Makes training more stable and effective
    /// - Allows the network to learn useful features even without labeled data
    /// 
    /// After pre-training, the network has learned general patterns in your data and is ready
    /// for fine-tuning on specific tasks.
    /// </para>
    /// </remarks>
    public void PretrainGreedyLayerwise(
        Tensor<T> trainingData,
        int pretrainingEpochs = 10,
        T? pretrainingLearningRate = default,
        int cdSteps = 1)
    {
        var learningRate = pretrainingLearningRate ?? NumOps.FromDouble(0.1);

        Console.WriteLine("Starting greedy layer-wise pre-training...");

        // Initialize variable to hold input for the current layer
        var layerInput = trainingData;

        // Train each RBM layer one by one, from bottom to top
        for (int layerIdx = 0; layerIdx < _rbmLayers.Count; layerIdx++)
        {
            var rbm = _rbmLayers[layerIdx];
            Console.WriteLine($"Pre-training layer {layerIdx + 1}/{_rbmLayers.Count}...");

            // Train the current RBM layer
            for (int epoch = 0; epoch < pretrainingEpochs; epoch++)
            {
                T totalLoss = NumOps.Zero;

                // Process data in batches
                for (int batchStart = 0; batchStart < layerInput.Shape[0]; batchStart += _batchSize)
                {
                    // Get a batch of data
                    int batchEnd = Math.Min(batchStart + _batchSize, layerInput.Shape[0]);
                    int actualBatchSize = batchEnd - batchStart;
                    var batch = layerInput.Slice(batchStart, 0, batchEnd, layerInput.Shape[1]);

                    // Train the RBM on the current batch using contrastive divergence
                    for (int i = 0; i < actualBatchSize; i++)
                    {
                        var example = batch.GetRow(i);
                        rbm.TrainWithContrastiveDivergence(example, learningRate, cdSteps);
                    }

                    // Calculate reconstruction error for monitoring
                    for (int i = 0; i < actualBatchSize; i++)
                    {
                        var example = batch.GetRow(i);
                        var hidden = rbm.Forward(Tensor<T>.FromVector(example));
                        // Use RBMLayer Backward method to reconstruct the input
                        var reconstruction = rbm.Backward(hidden);

                        T sampleLoss = CalculateReconstructionError(
                            Tensor<T>.FromVector(example),
                            reconstruction);
                        totalLoss = NumOps.Add(totalLoss, sampleLoss);
                    }
                }

                // Calculate average loss for the epoch
                T avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(layerInput.Shape[0]));
                Console.WriteLine($"Layer {layerIdx + 1}, Epoch {epoch + 1}/{pretrainingEpochs}, Average Loss: {avgLoss}");
            }

            // If not the last layer, transform data for the next layer
            if (layerIdx < _rbmLayers.Count - 1)
            {
                var transformedData = new Tensor<T>(new[] { layerInput.Shape[0], rbm.GetOutputShape()[0] });

                for (int i = 0; i < layerInput.Shape[0]; i++)
                {
                    var example = layerInput.GetRow(i);
                    var hidden = rbm.Forward(Tensor<T>.FromVector(example));
                    transformedData.SetRow(i, hidden.ToVector());
                }

                layerInput = transformedData;
            }
        }

        Console.WriteLine("Greedy layer-wise pre-training complete.");
    }

    /// <summary>
    /// Calculates the reconstruction error between original and reconstructed data.
    /// </summary>
    /// <param name="original">The original input data.</param>
    /// <param name="reconstruction">The reconstructed data after passing through the network.</param>
    /// <returns>The mean squared error between the original and reconstructed data.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the mean squared error (MSE) between the original input and its reconstruction.
    /// The MSE is computed by taking the average of the squared differences between the original and reconstructed values.
    /// This provides a measure of how well the network can reconstruct its input, which is a key aspect of RBM and DBN training.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how well the network can recreate what it was shown.
    /// 
    /// The reconstruction error:
    /// - Compares the original input with the network's attempt to recreate it
    /// - Calculates the average squared difference between original and reconstructed values
    /// - Lower values mean better reconstruction and better pattern learning
    /// 
    /// During training, we aim to minimize this error, meaning the network gets better at
    /// recognizing and recreating patterns from the input data.
    /// </para>
    /// </remarks>
    private T CalculateReconstructionError(Tensor<T> original, Tensor<T> reconstruction)
    {
        // Check that shapes match
        if (!Enumerable.SequenceEqual(original.Shape, reconstruction.Shape))
        {
            throw new ArgumentException("Original and reconstruction tensors must have the same shape.");
        }

        // Calculate mean squared error
        T sumSquaredError = NumOps.Zero;
        for (int i = 0; i < original.Length; i++)
        {
            T diff = NumOps.Subtract(original.ToArray()[i], reconstruction.ToArray()[i]);
            T squaredDiff = NumOps.Multiply(diff, diff);
            sumSquaredError = NumOps.Add(sumSquaredError, squaredDiff);
        }

        return NumOps.Divide(sumSquaredError, NumOps.FromDouble(original.Length));
    }

    /// <summary>
    /// Makes a prediction using the current state of the Deep Belief Network.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The predicted output tensor after passing through all layers of the network.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction by passing the input tensor through each layer of the Deep Belief Network
    /// in sequence. Each layer processes the output of the previous layer, transforming the data until it reaches
    /// the final output layer. The result is a tensor representing the network's prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This method uses the network to make a prediction based on input data.
    /// 
    /// The prediction process works like this:
    /// - The input data enters the first layer of the network
    /// - Each layer processes the data and passes it to the next layer
    /// - The data is transformed as it flows up through the tower
    /// - The final layer produces the prediction result
    /// 
    /// Once the network is trained, this is how you use it to recognize patterns,
    /// classify new data, or make predictions.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Updates the parameters of all layers in the Deep Belief Network.
    /// </summary>
    /// <param name="parameters">A vector containing the parameters to update all layers with.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the provided parameter vector among all the layers in the network.
    /// Each layer receives a portion of the parameter vector corresponding to its number of parameters.
    /// The method keeps track of the starting index for each layer's parameters in the input vector.
    /// This is typically used during the supervised fine-tuning phase that follows pre-training.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the network's internal values during fine-tuning.
    /// 
    /// When updating parameters:
    /// - The input is a long list of numbers representing all values in the entire network
    /// - The method divides this list into smaller chunks
    /// - Each layer gets its own chunk of values
    /// - The layers use these values to adjust their internal settings
    /// 
    /// After pre-training the individual RBM layers, this method helps fine-tune
    /// the entire network to improve its performance on specific tasks.
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
    /// Performs supervised fine-tuning of the Deep Belief Network after pre-training.
    /// </summary>
    /// <param name="input">The input training data.</param>
    /// <param name="expectedOutput">The expected output for the given input data.</param>
    /// <remarks>
    /// <para>
    /// This method implements the supervised fine-tuning phase of training a Deep Belief Network.
    /// After the unsupervised pre-training of individual RBM layers, this method uses backpropagation
    /// and gradient descent to fine-tune the entire network for a specific task. This phase helps the
    /// network adapt its pre-trained features to perform well on the specific supervised learning task.
    /// </para>
    /// <para><b>For Beginners:</b> This method fine-tunes the entire network for a specific task.
    /// 
    /// After pre-training each layer individually:
    /// - We now train the entire network end-to-end
    /// - We use labeled data (inputs with known correct outputs)
    /// - The network compares its predictions with the expected outputs
    /// - It adjusts its parameters to make its predictions more accurate
    /// 
    /// Think of it like:
    /// - Pre-training: Teaching general pattern recognition (like learning to see)
    /// - Fine-tuning: Teaching a specific task using those patterns (like identifying specific objects)
    /// 
    /// This two-phase approach often works better than training from scratch, especially
    /// when you don't have a lot of labeled examples.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Make sure we're in training mode
        SetTrainingMode(true);

        Console.WriteLine("Starting supervised fine-tuning...");

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
                int[] gradientShape = GetGradientShape();
                Tensor<T> totalGradient = Tensor<T>.CreateDefault(gradientShape, NumOps.Zero);

                // Accumulate gradients for each example in the batch
                for (int i = 0; i < actualBatchSize; i++)
                {
                    var x = batchX.GetRow(i);
                    var y = batchY.GetRow(i);

                    // Forward pass with memory to save intermediate states
                    // NOTE: This optimization uses the prediction tensor directly instead of converting to Vector<T> and back.
                    // This is the recommended pattern for consistency across all neural network implementations.
                    var prediction = ForwardWithMemory(Tensor<T>.FromVector(x));

                    // Calculate loss and gradients for this example
                    T loss = CalculateLoss(prediction, Tensor<T>.FromVector(y));
                    totalLoss = NumOps.Add(totalLoss, loss);

                    // Calculate output gradients
                    Vector<T> outputGradients = CalculateOutputGradients(prediction.ToVector(), y);

                    // Backpropagate to compute gradients for all parameters
                    Backpropagate(Tensor<T>.FromVector(outputGradients));

                    // Accumulate gradients
                    var gradients = GetParameterGradients();
                    for (int j = 0; j < gradients.Length; j++)
                    {
                        totalGradient[j] = NumOps.Add(totalGradient[j], gradients[j]);
                    }
                }

                // Average the gradients across the batch
                for (int j = 0; j < totalGradient.Length; j++)
                {
                    totalGradient[j] = NumOps.Divide(totalGradient[j], NumOps.FromDouble(actualBatchSize));
                }

                // Update parameters with averaged gradients
                var currentParams = GetParameters();
                var updatedParams = new Vector<T>(currentParams.Length);
                for (int j = 0; j < currentParams.Length; j++)
                {
                    updatedParams[j] = NumOps.Subtract(
                        currentParams[j],
                        NumOps.Multiply(_learningRate, totalGradient[j]));
                }

                UpdateParameters(updatedParams);
            }

            // Calculate average loss for the epoch
            T avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(input.Shape[0]));
            Console.WriteLine($"Epoch {epoch + 1}/{_epochs}, Average Loss: {avgLoss}");

            // Store the current loss
            LastLoss = avgLoss;
        }

        // Set back to inference mode after training
        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets the shape of the gradient tensor for all layers in the Deep Belief Network.
    /// </summary>
    /// <returns>An array of integers representing the shape of the gradient tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the shape of the gradient tensor by iterating through all layers in the network.
    /// For each layer, it adds the length of its parameter gradients to the shape array. The resulting array
    /// represents the structure of the gradient tensor, which is used for accumulating gradients during training.
    /// </para>
    /// <para><b>For Beginners:</b> This method determines the size and structure of the network's gradients.
    /// 
    /// The gradient shape:
    /// - Represents how many adjustable values (parameters) are in each layer
    /// - Helps organize the gradients for all layers into a single structure
    /// - Is used to create a tensor that can hold all gradients for the entire network
    /// 
    /// Think of it like creating a blueprint for a container that can hold all the
    /// network's learning information in an organized way.
    /// </para>
    /// </remarks>
    private int[] GetGradientShape()
    {
        List<int> shape = [];
        foreach (var layer in Layers)
        {
            var layerGradients = layer.GetParameterGradients();
            if (layerGradients != null)
            {
                shape.Add(layerGradients.Length);
            }
        }

        return [.. shape];
    }

    /// <summary>
    /// Calculates the loss between predicted and expected outputs using the appropriate loss function.
    /// </summary>
    /// <param name="predicted">The predicted output from the network.</param>
    /// <param name="expected">The expected (ground truth) output.</param>
    /// <returns>The loss value based on the selected loss function.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the loss between the network's prediction and the expected output
    /// using the loss function specified during initialization or a default one based on the task type.
    /// The loss value provides a measure of how well the network is performing on the task.
    /// </para>
    /// <para><b>For Beginners:</b> This measures how wrong the network's predictions are.
    /// 
    /// The loss function:
    /// - Compares the network's prediction with the correct answer
    /// - Produces a number that's higher when predictions are worse
    /// - Uses the appropriate calculation based on your task type
    /// 
    /// During training, we aim to minimize this loss, meaning the network gets better at
    /// making accurate predictions for the specific task.
    /// </para>
    /// </remarks>
    private T CalculateLoss(Tensor<T> predicted, Tensor<T> expected)
    {
        return _lossFunction.CalculateLoss(predicted.ToVector(), expected.ToVector());
    }

    /// <summary>
    /// Calculates the gradients of the loss with respect to the network outputs.
    /// </summary>
    /// <param name="predicted">The predicted values from the network.</param>
    /// <param name="expected">The expected (target) values.</param>
    /// <returns>A vector of gradients for the output layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates how the loss changes with respect to changes in the network's outputs.
    /// These gradients are used during backpropagation to update the network's parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how to adjust the network's outputs to reduce errors.
    /// 
    /// The gradient tells the network:
    /// - How much each output value contributes to the overall error
    /// - Which direction to adjust each output to reduce the error
    /// - The size of adjustment needed for each output
    /// 
    /// This information flows backward through the network during training,
    /// helping all parts of the network learn from its mistakes.
    /// </para>
    /// </remarks>
    private Vector<T> CalculateOutputGradients(Vector<T> predicted, Vector<T> expected)
    {
        return _lossFunction.CalculateDerivative(predicted, expected);
    }

    /// <summary>
    /// Gets metadata about the Deep Belief Network model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// <para>
    /// This method returns metadata that describes the Deep Belief Network, including its type,
    /// architecture details, and training parameters. This information can be useful for model
    /// management, documentation, and versioning.
    /// </para>
    /// <para><b>For Beginners:</b> This provides a summary of your network's configuration.
    /// 
    /// The metadata includes:
    /// - The type of model (Deep Belief Network)
    /// - The number of RBM layers in the network
    /// - The size of each layer
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
        var layerSizes = new List<int>();

        // Collect layer sizes from RBM layers
        foreach (var rbm in _rbmLayers)
        {
            layerSizes.Add(rbm.GetInputShape()[0]);
        }

        // Add the size of the final hidden layer
        if (_rbmLayers.Count > 0)
        {
            layerSizes.Add(_rbmLayers[_rbmLayers.Count - 1].GetOutputShape()[0]);
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.DeepBeliefNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfLayers", layerSizes.Count },
                { "LayerSizes", layerSizes },
                { "Epochs", _epochs },
                { "LearningRate", Convert.ToDouble(_learningRate) },
                { "BatchSize", _batchSize }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data for the Deep Belief Network.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific configuration and state of the Deep Belief Network
    /// to a binary stream. This includes training parameters and RBM layer configurations
    /// that need to be preserved for later reconstruction of the network.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the unique settings of your Deep Belief Network.
    /// 
    /// It writes:
    /// - The number of RBM layers
    /// - The configuration of each RBM layer
    /// - Training parameters like learning rate, epochs, and batch size
    /// 
    /// Saving these details allows you to recreate the exact same network structure and state later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write training parameters
        writer.Write(_epochs);
        writer.Write(Convert.ToDouble(_learningRate));
        writer.Write(_batchSize);

        // Serialize the loss function using the helper method
        SerializationHelper<T>.SerializeInterface(writer, _lossFunction);
    }

    /// <summary>
    /// Deserializes network-specific data for the Deep Belief Network.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific configuration and state of the Deep Belief Network from a binary stream.
    /// It reconstructs the network's structure, including RBM layers and training parameters, to match
    /// the state of the network when it was serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the unique settings of your Deep Belief Network.
    /// 
    /// It reads:
    /// - The number of RBM layers
    /// - The configuration of each RBM layer
    /// - Training parameters like learning rate, epochs, and batch size
    /// 
    /// Loading these details allows you to recreate the exact same network structure and state that was previously saved.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read training parameters
        _epochs = reader.ReadInt32();
        _learningRate = NumOps.FromDouble(reader.ReadDouble());
        _batchSize = reader.ReadInt32();

        // Read and set the loss function if a custom one was used
        var lossFunction = DeserializationHelper.DeserializeInterface<ILossFunction<T>>(reader) ??
            throw new InvalidOperationException("Failed to deserialize the loss function. The loss function cannot be null.");
        _lossFunction = lossFunction;
    }

    /// <summary>
    /// Creates a new instance of the deep belief network model.
    /// </summary>
    /// <returns>A new instance of the deep belief network model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the deep belief network model with the same configuration as the current instance.
    /// It is used internally during serialization/deserialization processes to create a fresh instance that can be populated
    /// with the serialized data. The new instance will have the same architecture, learning rate, epochs, batch size,
    /// and loss function as the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the network structure without copying the learned data.
    /// 
    /// Think of it like making a blueprint copy of the tower:
    /// - It copies the same overall design (architecture)
    /// - It preserves settings like learning rate and batch size
    /// - It maintains the same RBM layer structure
    /// - But it doesn't copy any of the learned patterns and weights
    /// 
    /// This is primarily used when saving or loading models, creating an empty framework that the saved parameters
    /// can be loaded into later.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DeepBeliefNetwork<T>(
            Architecture,
            _learningRate,
            _epochs,
            _batchSize,
            _lossFunction
        );
    }
}
