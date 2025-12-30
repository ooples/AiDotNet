namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Deep Boltzmann Machine (DBM), a hierarchical generative model consisting of multiple layers of stochastic neurons.
/// </summary>
/// <remarks>
/// <para>
/// A Deep Boltzmann Machine is an extension of the Restricted Boltzmann Machine to multiple hidden layers.
/// It consists of a visible layer and multiple hidden layers with connections between adjacent layers but no connections
/// within the same layer. DBMs are used for unsupervised learning, feature extraction, and generative modeling.
/// </para>
/// <para><b>For Beginners:</b> A Deep Boltzmann Machine is like a multi-story pattern detector.
/// 
/// Think of it as a series of layers, each learning increasingly abstract patterns:
/// - The visible layer represents the raw data (e.g., pixel values in an image)
/// - The first hidden layer might learn simple patterns (e.g., edges, corners)
/// - Higher hidden layers learn more complex patterns (e.g., shapes, objects)
/// - The deeper the network, the more abstract the patterns it can learn
/// 
/// For example, in an image recognition system:
/// - Layer 1 might detect edges and basic textures
/// - Layer 2 might combine these into simple shapes
/// - Layer 3 might recognize more complex objects
/// 
/// DBMs can both recognize patterns in data and generate new data with similar patterns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DeepBoltzmannMachine<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the bias vectors for each layer in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list contains tensors representing the bias values for each layer in the DBM.
    /// Each tensor has shape [1, layerSize], where layerSize is the number of units in that layer.
    /// </para>
    /// <para><b>For Beginners:</b> Biases are like the default tendencies of units to activate.
    /// 
    /// Think of biases as the baseline sensitivity of each unit:
    /// - A positive bias means the unit tends to be active by default
    /// - A negative bias means the unit tends to be inactive by default
    /// - Zero bias means no inherent preference
    /// 
    /// Each layer has its own set of biases that get adjusted during training to capture
    /// the statistical patterns in your data.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> _layerBiases;

    /// <summary>
    /// Gets or sets the weight matrices connecting adjacent layers in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This list contains tensors representing the connection weights between adjacent layers.
    /// Each tensor has shape [sizeOfLayer_i, sizeOfLayer_i+1], representing connections from
    /// layer i to layer i+1. The list has length one less than the total number of layers.
    /// </para>
    /// <para><b>For Beginners:</b> Weights are like the strengths of connections between units in adjacent layers.
    /// 
    /// Think of weights as determining how strongly units influence each other:
    /// - A positive weight means activating one unit tends to activate the connected unit
    /// - A negative weight means activating one unit tends to deactivate the connected unit
    /// - A zero weight means no influence
    /// 
    /// For example, if a DBM is learning to recognize faces:
    /// - A "nose detector" unit might have strong positive weights to "face detector" units
    /// - It might have negative weights to "car detector" units
    /// 
    /// These weights form the "knowledge" of the DBM, capturing the patterns and relationships in your data.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> _layerWeights;

    /// <summary>
    /// Gets or sets the number of units in each layer of the network.
    /// </summary>
    /// <remarks>
    /// This list contains the size (number of units) for each layer in the DBM, from the visible
    /// layer to the deepest hidden layer.
    /// </remarks>
    private List<int> _layerSizes;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <remarks>
    /// The number of complete passes through the training dataset during model training.
    /// </remarks>
    private int _epochs;

    /// <summary>
    /// Gets or sets the learning rate for parameter updates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The learning rate controls the size of parameter updates during training.
    /// It determines how quickly the model adapts to the training data.
    /// </para>
    /// <para><b>For Beginners:</b> The learning rate is like the step size for learning.
    /// 
    /// Think of learning rate as how boldly the DBM adjusts its understanding:
    /// - High learning rate: large, confident updates (may overshoot optimal values)
    /// - Low learning rate: small, cautious updates (may learn very slowly)
    /// 
    /// Finding the right balance is important:
    /// - Too high: training becomes unstable, weights oscillate wildly
    /// - Too low: training takes very long, might get stuck in suboptimal solutions
    /// 
    /// Typical values range from 0.001 to 0.1, with 0.01 being a common starting point.
    /// </para>
    /// </remarks>
    private T _learningRate;

    /// <summary>
    /// Gets or sets the number of examples processed in each training batch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The batch size determines how many examples are processed before updating the model parameters.
    /// Smaller batches provide more frequent updates but with higher variance, while larger batches
    /// provide more stable but less frequent updates.
    /// </para>
    /// <para><b>For Beginners:</b> Batch size is like the number of examples studied before updating notes.
    /// 
    /// When training the DBM:
    /// - Small batch size (e.g., 16-32): more frequent but noisier updates
    /// - Large batch size (e.g., 128-256): less frequent but more stable updates
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
    /// Gets or sets the number of contrastive divergence steps in training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of steps in the contrastive divergence algorithm for approximating the gradient.
    /// More steps provide a better approximation but require more computation. CD-1 (one step)
    /// is commonly used as a practical compromise.
    /// </para>
    /// <para><b>For Beginners:</b> CD steps determine how thoroughly the DBM "daydreams" during training.
    /// 
    /// Contrastive divergence works by comparing:
    /// - How the network responds to real data
    /// - How the network responds to its own generated data ("daydreams")
    /// 
    /// The number of CD steps controls:
    /// - How many cycles of back-and-forth processing occur to generate the "daydream"
    /// - More steps: better approximation, but slower training
    /// - Fewer steps: faster training, but rougher approximation
    /// 
    /// CD-1 (one step) works surprisingly well in practice, while CD-10 or higher
    /// might be used for more precise training when computational resources allow.
    /// </para>
    /// </remarks>
    private int _cdSteps;

    /// <summary>
    /// Gets or sets the scalar activation function for the network.
    /// </summary>
    /// <remarks>
    /// The activation function that processes individual values in the network,
    /// typically sigmoid for Boltzmann Machines. This function is applied element-wise
    /// to unit activations.
    /// </remarks>
    private IActivationFunction<T>? _activationFunction;

    /// <summary>
    /// Gets or sets the vector activation function for the network.
    /// </summary>
    /// <remarks>
    /// The activation function that processes entire vectors at once. This is an
    /// alternative to scalar activation that can be more computationally efficient.
    /// </remarks>
    private IVectorActivationFunction<T>? _vectorActivationFunction;

    /// <summary>
    /// Gets or sets the learning rate decay factor per epoch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This factor determines how quickly the learning rate decreases over epochs.
    /// A value of 1.0 means no decay, while values between 0 and 1 cause gradual decay.
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as gradually reducing the step size during training.
    /// 
    /// Learning rate decay:
    /// - Starts with larger steps for quick initial progress
    /// - Gradually reduces to smaller steps for fine-tuning
    /// - Helps avoid overshooting the optimal solution
    /// 
    /// For example, with a decay of 0.95:
    /// - Initial learning rate: 0.01
    /// - After 10 epochs: ~0.006
    /// - After 50 epochs: ~0.001
    /// 
    /// This approach often leads to better final results than using a fixed learning rate.
    /// </para>
    /// </remarks>
    private T _learningRateDecay;

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the DeepBoltzmannMachine class with scalar activation.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="epochs">The number of training epochs.</param>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <param name="learningRateDecay">The learning rate decay factor per epoch. Default is 1.0 (no decay).</param>
    /// <param name="activationFunction">The scalar activation function to use. Default is sigmoid.</param>
    /// <param name="batchSize">The number of examples in each training batch. Default is 32.</param>
    /// <param name="cdSteps">The number of contrastive divergence steps. Default is 1.</param>
    /// <remarks>
    /// This constructor creates a Deep Boltzmann Machine with the specified architecture and training parameters,
    /// using a scalar activation function that is applied element-wise to unit activations.
    /// </remarks>
    public DeepBoltzmannMachine(
        NeuralNetworkArchitecture<T> architecture,
        int epochs,
        T learningRate,
        double learningRateDecay = 1.0,
        ILossFunction<T>? lossFunction = null,
        IActivationFunction<T>? activationFunction = null,
        int batchSize = 32,
        int cdSteps = 1)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _epochs = epochs;
        _learningRate = learningRate;
        _learningRateDecay = NumOps.FromDouble(learningRateDecay);
        _batchSize = batchSize;
        _cdSteps = cdSteps;
        _activationFunction = activationFunction ?? new SigmoidActivation<T>();
        _layerBiases = new List<Tensor<T>>();
        _layerWeights = new List<Tensor<T>>();
        _layerSizes = new List<int>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the DeepBoltzmannMachine class with vector activation.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="epochs">The number of training epochs.</param>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <param name="learningRateDecay">The learning rate decay factor per epoch. Default is 1.0 (no decay).</param>
    /// <param name="vectorActivationFunction">The vector activation function to use. Default is sigmoid.</param>
    /// <param name="batchSize">The number of examples in each training batch. Default is 32.</param>
    /// <param name="cdSteps">The number of contrastive divergence steps. Default is 1.</param>
    /// <remarks>
    /// This constructor creates a Deep Boltzmann Machine with the specified architecture and training parameters,
    /// using a vector activation function that processes entire tensors at once for improved performance.
    /// </remarks>
    public DeepBoltzmannMachine(
        NeuralNetworkArchitecture<T> architecture,
        int epochs,
        T learningRate,
        double learningRateDecay = 1.0,
        ILossFunction<T>? lossFunction = null,
        IVectorActivationFunction<T>? vectorActivationFunction = null,
        int batchSize = 32,
        int cdSteps = 1)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _epochs = epochs;
        _learningRate = learningRate;
        _learningRateDecay = NumOps.FromDouble(learningRateDecay);
        _batchSize = batchSize;
        _cdSteps = cdSteps;
        _vectorActivationFunction = vectorActivationFunction ?? new SigmoidActivation<T>();
        _layerBiases = new List<Tensor<T>>();
        _layerWeights = new List<Tensor<T>>();
        _layerSizes = new List<int>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Performs layer-wise pretraining of the DBM using a greedy approach.
    /// </summary>
    /// <param name="input">The input training data.</param>
    /// <param name="pretrainingEpochs">The number of epochs for pretraining each layer.</param>
    /// <param name="pretrainingLearningRate">The learning rate for pretraining.</param>
    /// <remarks>
    /// <para>
    /// This method pretrains the DBM layer by layer, treating each adjacent pair of layers
    /// as a separate RBM. This greedy approach often leads to better final results.
    /// </para>
    /// <para><b>For Beginners:</b> This is like training the network one layer at a time.
    /// 
    /// Instead of training the whole network at once:
    /// 1. First train the bottom two layers (visible and first hidden)
    /// 2. Then freeze the first hidden layer and train it with the second hidden layer
    /// 3. Continue this process up the network
    /// 
    /// This step-by-step approach:
    /// - Makes training more stable
    /// - Often leads to better final results
    /// - Can be thought of as "teaching the basics before the advanced concepts"
    /// </para>
    /// </remarks>
    public void PretrainLayerwise(Tensor<T> input, int pretrainingEpochs, T pretrainingLearningRate)
    {
        // Create temporary tensors to hold activations between layers
        Tensor<T> layerInput = input;

        // Train each layer pair as an RBM
        for (int layer = 0; layer < _layerSizes.Count - 1; layer++)
        {
            Console.WriteLine($"Pretraining layer {layer + 1}/{_layerSizes.Count - 1}");

            // Create a temporary RBM for this layer pair
            var tmpRBM = _activationFunction != null ? new RBMLayer<T>(
                _layerSizes[layer],
                _layerSizes[layer + 1],
                _activationFunction) : _vectorActivationFunction != null ? new RBMLayer<T>(
                _layerSizes[layer],
                _layerSizes[layer + 1],
                _vectorActivationFunction) : new RBMLayer<T>(
                _layerSizes[layer],
                _layerSizes[layer + 1], null as IActivationFunction<T>);

            // Train the RBM on the current layer activations
            for (int epoch = 0; epoch < pretrainingEpochs; epoch++)
            {
                T epochLoss = NumOps.Zero;
                for (int i = 0; i < layerInput.Shape[0]; i += _batchSize)
                {
                    var batchSize = Math.Min(_batchSize, layerInput.Shape[0] - i);
                    var batch = layerInput.Slice(i, 0, i + batchSize, layerInput.Shape[1]);
                    tmpRBM.TrainWithContrastiveDivergence(batch.ToVector(), pretrainingLearningRate);

                    // Calculate loss (simplified)
                    var reconstructed = tmpRBM.Forward(Tensor<T>.FromVector(batch.ToVector()));
                    var loss = CalculateLoss(batch, reconstructed);
                    epochLoss = NumOps.Add(epochLoss, loss);
                }

                epochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(layerInput.Shape[0]));
                Console.WriteLine($"Layer {layer + 1}, Epoch {epoch + 1}/{pretrainingEpochs}, Loss: {epochLoss}");
            }

            // Copy the trained weights and biases to our DBM
            _layerWeights[layer] = new Tensor<T>(
                _layerWeights[layer].Shape,
                tmpRBM.GetParameters().GetSubVector(0, _layerSizes[layer] * _layerSizes[layer + 1]));

            _layerBiases[layer] = new Tensor<T>(
                _layerBiases[layer].Shape,
                tmpRBM.GetParameters().GetSubVector(
                    _layerSizes[layer] * _layerSizes[layer + 1],
                    _layerSizes[layer]));

            // Generate activations for the next layer
            if (layer < _layerSizes.Count - 2)
            {
                layerInput = PropagateToLayer(input, layer + 1);
            }
        }

        Console.WriteLine("Layer-wise pretraining complete.");
    }

    /// <summary>
    /// Propagates the input to a specific layer in the network.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="targetLayer">The target layer index to propagate to.</param>
    /// <returns>The activation of the target layer.</returns>
    /// <remarks>
    /// This helper method propagates the input data through the network up to a specified layer.
    /// It applies the weights, biases, and activation functions of each layer in sequence.
    /// </remarks>
    private Tensor<T> PropagateToLayer(Tensor<T> input, int targetLayer)
    {
        var activation = input;
        for (int layer = 0; layer < targetLayer; layer++)
        {
            activation = ActivationFunction(
                activation.Multiply(_layerWeights[layer]).Add(_layerBiases[layer + 1]));
        }

        return activation;
    }

    /// <summary>
    /// Initializes the layers of the neural network.
    /// </summary>
    /// <remarks>
    /// This method sets up the layer structure of the DBM based on the provided architecture.
    /// It either uses user-specified layers or creates default layers if none are provided.
    /// After initializing the layers, it extracts the layer sizes and initializes the parameters.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepBoltzmannMachineLayers(Architecture));
        }

        // Extract layer sizes starting with input size, then output sizes of RBM layers only
        // (skip BatchNorm and other non-RBM layers to get the DBM structure)
        var inputSize = Architecture.GetInputShape()[0];
        var rbmOutputSizes = Layers
            .Where(l => l is RBMLayer<T>)
            .Select(l => l.GetOutputShape()[0])
            .ToList();

        _layerSizes = [inputSize, .. rbmOutputSizes];
        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weights and biases of the DBM.
    /// </summary>
    /// <remarks>
    /// This method initializes all weight matrices and bias vectors in the DBM with small random values.
    /// For each layer, it creates a bias tensor, and for each pair of adjacent layers, it creates a
    /// weight tensor connecting them.
    /// </remarks>
    private void InitializeParameters()
    {
        for (int i = 0; i < _layerSizes.Count; i++)
        {
            var biasShape = new[] { 1, _layerSizes[i] };
            var biasVector = Vector<T>.CreateRandom(biasShape[0] * biasShape[1], -0.1, 0.1);
            _layerBiases.Add(new Tensor<T>(biasShape, biasVector));

            if (i < _layerSizes.Count - 1)
            {
                var weightShape = new[] { _layerSizes[i], _layerSizes[i + 1] };
                var weightVector = Vector<T>.CreateRandom(weightShape[0] * weightShape[1], -0.1, 0.1);
                _layerWeights.Add(new Tensor<T>(weightShape, weightVector));
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the Deep Boltzmann Machine.
    /// </summary>
    /// <param name="input">The input tensor to make predictions for.</param>
    /// <returns>The predicted reconstruction of the input.</returns>
    /// <remarks>
    /// This method makes a prediction by reconstructing the input through the DBM.
    /// It propagates the input up through all hidden layers and then back down to generate
    /// a reconstruction of the original input.
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var reconstructed = Reconstruct(input);
        return reconstructed;
    }

    /// <summary>
    /// Reconstructs the input by propagating it up through the hidden layers and back down.
    /// </summary>
    /// <param name="input">The input tensor to reconstruct.</param>
    /// <returns>The reconstructed tensor.</returns>
    /// <remarks>
    /// This method reconstructs the input by first propagating it up through all hidden layers
    /// and then propagating the result back down to the visible layer. This is equivalent to
    /// using the DBM as an autoencoder.
    /// </remarks>
    private Tensor<T> Reconstruct(Tensor<T> input)
    {
        // Remember original shape
        var originalShape = input.Shape;
        var was1D = originalShape.Length == 1;

        var hidden = PropagateUp(input);
        var result = PropagateDown(hidden);

        // Restore original shape if input was 1D
        if (was1D && result.Shape.Length == 2 && result.Shape[0] == 1)
        {
            result = result.Reshape([result.Shape[1]]);
        }

        return result;
    }

    /// <summary>
    /// Propagates the input upward through the network from visible to hidden layers.
    /// </summary>
    /// <param name="visible">The visible layer activation.</param>
    /// <returns>The activation of the deepest hidden layer.</returns>
    /// <remarks>
    /// This method propagates the input data from the visible layer through all hidden layers,
    /// applying weights, biases, and activation functions at each step. The result is the
    /// activation of the deepest hidden layer.
    /// </remarks>
    private Tensor<T> PropagateUp(Tensor<T> visible)
    {
        // Ensure input is 2D for matrix multiplication
        // 1D [features] -> 2D [1, features]
        var hidden = visible.Shape.Length == 1
            ? visible.Reshape([1, visible.Shape[0]])
            : visible;

        for (int layer = 0; layer < _layerSizes.Count - 1; layer++)
        {
            hidden = ActivationFunction(hidden.Multiply(_layerWeights[layer]).Add(_layerBiases[layer + 1]));
        }

        return hidden;
    }

    /// <summary>
    /// Propagates the deepest hidden layer activation downward through the network to the visible layer.
    /// </summary>
    /// <param name="hidden">The activation of the deepest hidden layer.</param>
    /// <returns>The reconstructed visible layer activation.</returns>
    /// <remarks>
    /// This method propagates the activation from the deepest hidden layer back down through
    /// all layers to the visible layer, applying transposed weights, biases, and activation
    /// functions at each step. The result is a reconstruction of the original input.
    /// </remarks>
    private Tensor<T> PropagateDown(Tensor<T> hidden)
    {
        var visible = hidden;
        for (int layer = _layerSizes.Count - 2; layer >= 0; layer--)
        {
            // Use the correct overload of Transpose
            visible = ActivationFunction(visible.Multiply(_layerWeights[layer].Transpose([1, 0])).Add(_layerBiases[layer]));
        }

        return visible;
    }

    /// <summary>
    /// Applies the activation function to a tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after applying the activation function.</returns>
    /// <remarks>
    /// This method applies the configured activation function to each element in the input tensor.
    /// It uses either the scalar or vector activation function depending on which was provided.
    /// </remarks>
    private Tensor<T> ActivationFunction(Tensor<T> input)
    {
        return _activationFunction != null ? input.Transform((x, _) => _activationFunction.Activate(x)) : _vectorActivationFunction != null ? _vectorActivationFunction.Activate(input) : input;
    }

    /// <summary>
    /// Calculates the reconstruction error between the original input and its reconstruction.
    /// </summary>
    /// <param name="original">The original input tensor.</param>
    /// <param name="reconstructed">The reconstructed tensor.</param>
    /// <returns>The mean squared error between the original and reconstructed tensors.</returns>
    /// <remarks>
    /// This method calculates the mean squared error (MSE) between the original input and
    /// its reconstruction. This is used as a measure of how well the DBM is reconstructing
    /// the input data, with lower values indicating better performance.
    /// </remarks>
    private T CalculateLoss(Tensor<T> original, Tensor<T> reconstructed)
    {
        var squaredDifferences = original.Subtract(reconstructed).Transform((x, _) => NumOps.Multiply(x, x));
        var sumOfSquaredDifferences = squaredDifferences.Sum();
        T scalarSum = sumOfSquaredDifferences[0];

        return NumOps.Divide(scalarSum, NumOps.FromDouble(original.Length));
    }

    /// <summary>
    /// Trains the Deep Boltzmann Machine on the provided data.
    /// </summary>
    /// <param name="input">The input training data.</param>
    /// <param name="expectedOutput">The expected output (unused in DBMs, as they are self-supervised).</param>
    /// <remarks>
    /// <para>
    /// This method trains the DBM on the provided data for the specified number of epochs.
    /// It divides the data into batches and trains on each batch, tracking and reporting
    /// the average loss for each epoch. The learning rate decays according to the specified
    /// learning rate decay factor.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the DBM to recognize patterns in your data.
    /// 
    /// The training process:
    /// 1. Divides your data into smaller batches for efficient processing
    /// 2. Processes each batch through the DBM
    /// 3. Updates the weights and biases to better reconstruct the input
    /// 4. Repeats this for the specified number of epochs
    /// 5. Tracks and reports the average error for each epoch
    /// 
    /// You should see the error decrease over time as the DBM learns.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        T currentLearningRate = _learningRate;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            T epochLoss = NumOps.Zero;
            for (int i = 0; i < input.Shape[0]; i += _batchSize)
            {
                var batchSize = Math.Min(_batchSize, input.Shape[0] - i);
                var batch = input.Slice(i, 0, i + batchSize, input.Shape[1]);

                var (_, batchLoss) = TrainOnBatch(batch, currentLearningRate);
                epochLoss = NumOps.Add(epochLoss, batchLoss);
            }
            epochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(input.Shape[0]));

            Console.WriteLine($"Epoch {epoch + 1}/{_epochs}, Loss: {epochLoss}, Learning Rate: {currentLearningRate}");

            // Store the current loss
            LastLoss = epochLoss;

            // Decay learning rate for next epoch
            currentLearningRate = NumOps.Multiply(currentLearningRate, _learningRateDecay);
        }
    }

    /// <summary>
    /// Trains the DBM on a single batch of data using contrastive divergence.
    /// </summary>
    /// <param name="batch">The batch of training data.</param>
    /// <param name="learningRate">The current learning rate.</param>
    /// <returns>A tuple containing the reconstructed data and the reconstruction loss.</returns>
    /// <remarks>
    /// <para>
    /// This method implements contrastive divergence (CD) training for a single batch of data.
    /// It performs the positive phase (data-driven) and negative phase (model-driven) of CD,
    /// then updates the weights and biases based on the difference between these phases.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the DBM's internal connections based on a mini-batch.
    /// 
    /// Contrastive divergence works by:
    /// 1. Positive phase: Recording how units activate with real data
    /// 2. Negative phase: Recording how units activate with the DBM's own generated data
    /// 3. Updating weights to make the DBM's "imagination" more like reality
    /// </para>
    /// </remarks>
    private (Tensor<T> reconstructed, T loss) TrainOnBatch(Tensor<T> batch, T learningRate)
    {
        var positivePhase = batch;
        var negativePhase = batch;

        // Contrastive Divergence
        for (int step = 0; step < _cdSteps; step++)
        {
            negativePhase = Reconstruct(negativePhase);
        }

        for (int layer = 0; layer < _layerSizes.Count - 1; layer++)
        {
            var positiveAssociations = positivePhase.Transpose([1, 0]).Multiply(PropagateUp(positivePhase));
            var negativeAssociations = negativePhase.Transpose([1, 0]).Multiply(PropagateUp(negativePhase));

            var weightGradient = positiveAssociations.Subtract(negativeAssociations).Transform((x, _) => NumOps.Divide(x, NumOps.FromDouble(batch.Shape[0])));
            _layerWeights[layer] = _layerWeights[layer].Add(weightGradient.Multiply(learningRate));

            var biasGradient = positivePhase.Sum([0]).Subtract(negativePhase.Sum([0])).Transform((x, _) => NumOps.Divide(x, NumOps.FromDouble(batch.Shape[0])));
            _layerBiases[layer] = _layerBiases[layer].Add(biasGradient.Multiply(learningRate));
        }

        var reconstructed = Reconstruct(batch);
        var loss = CalculateLoss(batch, reconstructed);

        return (reconstructed, loss);
    }

    /// <summary>
    /// Updates the parameters of the DBM with the given vector of parameter values.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <remarks>
    /// This method updates all the parameters of the DBM (weights and biases) from a single vector.
    /// It expects the parameters to be arranged in the same order as they are returned by GetParameters.
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        for (int i = 0; i < _layerWeights.Count; i++)
        {
            int weightCount = _layerWeights[i].Length;
            var weightVector = parameters.GetSubVector(index, weightCount);
            _layerWeights[i] = new Tensor<T>(_layerWeights[i].Shape, weightVector);
            index += weightCount;

            int biasCount = _layerBiases[i].Length;
            var biasVector = parameters.GetSubVector(index, biasCount);
            _layerBiases[i] = new Tensor<T>(_layerBiases[i].Shape, biasVector);
            index += biasCount;
        }
    }

    /// <summary>
    /// Gets metadata about the Deep Boltzmann Machine model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// This method returns metadata about the DBM, including the model type, number of layers,
    /// layer sizes, and training parameters. This information can be useful for model management
    /// and serialization.
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.DeepBoltzmannMachine,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfLayers", _layerSizes.Count },
                { "LayerSizes", _layerSizes },
                { "Epochs", _epochs },
                { "LearningRate", Convert.ToDouble(_learningRate) }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes Deep Boltzmann Machine-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the specific parameters and state of the Deep Boltzmann Machine to a binary stream.
    /// It includes layer sizes, training parameters, activation functions, weights, and biases.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves all the important details of the DBM to a file.
    /// 
    /// Think of it like packing a suitcase for your neural network:
    /// - We pack the number and sizes of layers (the network's structure)
    /// - We include training settings like learning rate and epochs (how the network learns)
    /// - We save the activation function (how neurons in the network activate)
    /// - We carefully pack all the weights and biases (what the network has learned)
    /// 
    /// This allows us to later "unpack" the network exactly as it was, preserving all its learned knowledge.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write the number of layers
        writer.Write(_layerSizes.Count);

        // Write layer sizes
        foreach (var size in _layerSizes)
        {
            writer.Write(size);
        }

        // Write epochs
        writer.Write(_epochs);

        // Write learning rate
        writer.Write(Convert.ToDouble(_learningRate));

        // Write learning rate decay
        writer.Write(Convert.ToDouble(_learningRateDecay));

        // Write batch size
        writer.Write(_batchSize);

        // Write CD steps
        writer.Write(_cdSteps);

        // Write activation function type
        writer.Write(_activationFunction != null ? 0 : (_vectorActivationFunction != null ? 1 : -1));

        // Serialize activation function if present
        if (_activationFunction != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _activationFunction);
        }
        else if (_vectorActivationFunction != null)
        {
            SerializationHelper<T>.SerializeInterface(writer, _vectorActivationFunction);
        }

        // Write layer weights
        foreach (var weights in _layerWeights)
        {
            SerializationHelper<T>.SerializeTensor(writer, weights);
        }

        // Write layer biases
        foreach (var biases in _layerBiases)
        {
            SerializationHelper<T>.SerializeTensor(writer, biases);
        }
    }

    /// <summary>
    /// Deserializes Deep Boltzmann Machine-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the specific parameters and state of the Deep Boltzmann Machine from a binary stream.
    /// It reconstructs the layer sizes, training parameters, activation functions, weights, and biases.
    /// After reading all data, it reinitializes the layers to ensure the network structure is properly set up.
    /// </para>
    /// <para><b>For Beginners:</b> This method rebuilds the DBM from saved data.
    /// 
    /// Imagine "unpacking" the neural network suitcase we packed earlier:
    /// - We unpack the network's structure (number and sizes of layers)
    /// - We set up the learning settings (learning rate, epochs, etc.)
    /// - We restore the activation function
    /// - We carefully place all the weights and biases back where they belong
    /// 
    /// After unpacking, we make sure everything is connected properly (reinitialize layers).
    /// This allows us to continue using the network exactly where we left off, with all its learned knowledge intact.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read the number of layers
        int layerCount = reader.ReadInt32();

        // Read layer sizes
        _layerSizes = new List<int>();
        for (int i = 0; i < layerCount; i++)
        {
            _layerSizes.Add(reader.ReadInt32());
        }

        // Read epochs
        _epochs = reader.ReadInt32();

        // Read learning rate
        _learningRate = NumOps.FromDouble(reader.ReadDouble());

        // Read learning rate decay
        _learningRateDecay = NumOps.FromDouble(reader.ReadDouble());

        // Read batch size
        _batchSize = reader.ReadInt32();

        // Read CD steps
        _cdSteps = reader.ReadInt32();

        // Read activation function type
        int activationType = reader.ReadInt32();

        // Deserialize activation function
        if (activationType == 0)
        {
            _activationFunction = DeserializationHelper.DeserializeInterface<IActivationFunction<T>>(reader);
        }
        else if (activationType == 1)
        {
            _vectorActivationFunction = DeserializationHelper.DeserializeInterface<IVectorActivationFunction<T>>(reader);
        }

        // Read layer weights
        _layerWeights = new List<Tensor<T>>();
        for (int i = 0; i < layerCount - 1; i++)
        {
            _layerWeights.Add(SerializationHelper<T>.DeserializeTensor(reader));
        }

        // Read layer biases
        _layerBiases = new List<Tensor<T>>();
        for (int i = 0; i < layerCount; i++)
        {
            _layerBiases.Add(SerializationHelper<T>.DeserializeTensor(reader));
        }
    }

    /// <summary>
    /// Creates a new instance of the deep boltzmann machine model.
    /// </summary>
    /// <returns>A new instance of the deep boltzmann machine model with the same configuration.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of the deep boltzmann machine model with the same configuration 
    /// as the current instance. It is used internally during serialization/deserialization processes to 
    /// create a fresh instance that can be populated with the serialized data. The new instance will have 
    /// the same architecture, training parameters, and activation function type as the original.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the network structure without copying the learned data.
    /// 
    /// Think of it like making a blueprint copy of the DBM:
    /// - It copies the same multi-layer structure (architecture)
    /// - It uses the same learning settings (learning rate, epochs, etc.)
    /// - It keeps the same activation function (how neurons respond to input)
    /// - But it doesn't copy any of the weights and biases (the learned knowledge)
    /// 
    /// This is primarily used when saving or loading models, creating an empty framework
    /// that the saved parameters can be loaded into later.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Choose the appropriate constructor based on which activation function is used
        if (_activationFunction != null)
        {
            return new DeepBoltzmannMachine<T>(
                Architecture,
                _epochs,
                _learningRate,
                Convert.ToDouble(_learningRateDecay),
                _lossFunction,
                _activationFunction,
                _batchSize,
                _cdSteps
            );
        }
        else
        {
            return new DeepBoltzmannMachine<T>(
                Architecture,
                _epochs,
                _learningRate,
                Convert.ToDouble(_learningRateDecay),
                _lossFunction,
                _vectorActivationFunction,
                _batchSize,
                _cdSteps
            );
        }
    }
}
