global using AiDotNet.NeuralNetworks.Layers;

using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Siamese Neural Network for comparing pairs of inputs and determining their similarity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Siamese Network is a special type of neural network designed to compare two inputs
/// and determine how similar they are to each other.
/// 
/// Imagine you have two photos and want to know if they show the same person. A Siamese Network
/// processes both photos through identical neural networks (like twins, hence the name "Siamese"),
/// creates a compact representation (called an "embedding") of each photo, and then compares these
/// representations to determine similarity.
/// 
/// Common applications include:
/// - Face recognition (are these two faces the same person?)
/// - Signature verification (is this signature authentic?)
/// - Document similarity (how similar are these two texts?)
/// - Product recommendations (finding similar products)
/// 
/// The key advantage of Siamese Networks is that they can learn to recognize similarity even for
/// inputs they've never seen before during training.
/// </para>
/// </remarks>
public class SiameseNetwork<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    private readonly SiameseNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets or sets whether auxiliary loss (contrastive/triplet loss) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Contrastive loss encourages similar pairs to have small distances and dissimilar pairs to have large distances.
    /// Triplet loss ensures that an anchor is closer to positive examples than negative examples by a margin.
    /// </para>
    /// <para><b>For Beginners:</b> This helps the Siamese network learn better similarity representations.
    ///
    /// Contrastive loss works like this:
    /// - Similar pairs should have embeddings close together
    /// - Dissimilar pairs should have embeddings far apart
    /// - Formula: L = (1-Y) * 0.5 * D² + Y * 0.5 * max(0, margin - D)²
    ///   where Y=1 for similar, Y=0 for dissimilar, D=distance
    ///
    /// This helps the network:
    /// - Learn meaningful similarity measures
    /// - Create well-separated embedding spaces
    /// - Improve discrimination between similar/dissimilar pairs
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the contrastive auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much contrastive loss contributes to the total loss.
    /// Typical values range from 0.1 to 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much we encourage good similarity learning.
    ///
    /// Common values:
    /// - 0.5 (default): Balanced contribution
    /// - 0.1-0.3: Light contrastive emphasis
    /// - 0.7-1.0: Strong contrastive emphasis
    ///
    /// Higher values make the network focus more on learning good embeddings.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Gets or sets the margin for contrastive loss.
    /// </summary>
    public T ContrastiveMargin { get; set; }

    private T _lastContrastiveLoss;

    /// <summary>
    /// Cache for embedding pairs and their similarity labels during training.
    /// Used to compute contrastive auxiliary loss.
    /// </summary>
    private List<(Vector<T> embedding1, Vector<T> embedding2, T label)> _cachedEmbeddingPairs;

    /// <summary>
    /// The shared neural network that processes each input independently.
    /// </summary>
    /// <remarks>
    /// This network creates the embeddings (compact representations) for each input.
    /// </remarks>
    private ConvolutionalNeuralNetwork<T> _subnetwork;

    /// <summary>
    /// The final layer that compares the embeddings and produces a similarity score.
    /// </summary>
    private DenseLayer<T> _outputLayer;

    /// <summary>
    /// Initializes a new instance of the SiameseNetwork class.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining the structure of the shared subnetwork.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up your Siamese Network with the specified architecture.
    /// 
    /// The architecture defines the structure of the shared subnetwork that will process each input.
    /// The constructor creates:
    /// 1. A shared subnetwork (the identical twin networks)
    /// 2. An output layer that takes the embeddings from both inputs and produces a similarity score
    /// 
    /// The "embedding size" refers to how many numbers are used to represent each processed input.
    /// For example, a face might be represented by 128 numbers that capture its key features.
    /// 
    /// The sigmoid activation function at the end ensures the output is between 0 and 1,
    /// where 0 means "completely different" and 1 means "identical".
    /// </para>
    /// </remarks>
    public SiameseNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null,
        SiameseNetworkOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new SiameseNetworkOptions();
        Options = _options;
        _subnetwork = new ConvolutionalNeuralNetwork<T>(architecture);
        int embeddingSize = architecture.GetOutputShape()[0];
        _outputLayer = new DenseLayer<T>(embeddingSize * 2, 1, new SigmoidActivation<T>() as IActivationFunction<T>);

        // Initialize NumOps-based fields
        AuxiliaryLossWeight = NumOps.FromDouble(0.5);
        ContrastiveMargin = NumOps.FromDouble(1.0);
        _lastContrastiveLoss = NumOps.Zero;
        _cachedEmbeddingPairs = new List<(Vector<T>, Vector<T>, T)>();
    }

    /// <summary>
    /// Initializes the layers of the neural network.
    /// </summary>
    /// <remarks>
    /// This method is overridden but empty because the layers are initialized in the constructor.
    /// </remarks>
    protected override void InitializeLayers()
    {
        // The layers are initialized in the subnetwork constructor
    }

    /// <summary>
    /// Combines two embedding vectors into a single vector for comparison.
    /// </summary>
    /// <param name="embedding1">The first embedding vector.</param>
    /// <param name="embedding2">The second embedding vector.</param>
    /// <returns>A combined vector containing both embeddings.</returns>
    private Vector<T> CombineEmbeddings(Vector<T> embedding1, Vector<T> embedding2)
    {
        var combined = new Vector<T>(embedding1.Length * 2);
        for (int i = 0; i < embedding1.Length; i++)
        {
            combined[i] = embedding1[i];
            combined[i + embedding1.Length] = embedding2[i];
        }

        return combined;
    }

    /// <summary>
    /// Computes the auxiliary loss (contrastive loss) for similarity learning.
    /// </summary>
    /// <returns>The computed contrastive auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes contrastive loss to improve embedding quality.
    /// Formula: L = (1-Y) * 0.5 * D² + Y * 0.5 * max(0, margin - D)²
    /// where Y=1 for similar pairs, Y=0 for dissimilar, D=Euclidean distance
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how well the network separates similar from dissimilar pairs.
    ///
    /// Contrastive loss works by:
    /// 1. For similar pairs: Penalize large distances (pull them together)
    /// 2. For dissimilar pairs: Penalize small distances (push them apart)
    /// 3. Use a margin to define "far enough" for dissimilar pairs
    ///
    /// This helps because:
    /// - Creates well-organized embedding spaces
    /// - Similar items cluster together
    /// - Dissimilar items stay separated
    /// - Improves the network's ability to judge similarity
    ///
    /// The auxiliary loss is combined with the main loss during training.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss)
        {
            _lastContrastiveLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // Return zero if no cached pairs (e.g., before first training step)
        if (_cachedEmbeddingPairs.Count == 0)
        {
            _lastContrastiveLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        T totalLoss = NumOps.Zero;
        T half = NumOps.FromDouble(0.5);

        // Compute contrastive loss for each cached pair
        foreach (var (embedding1, embedding2, label) in _cachedEmbeddingPairs)
        {
            // Compute Euclidean distance using vectorized operations
            var diff = (Vector<T>)Engine.Subtract(embedding1, embedding2);
            T distanceSquared = Engine.DotProduct(diff, diff);
            T distance = NumOps.Sqrt(distanceSquared);

            // Compute contrastive loss based on label
            // For similar pairs (label close to 1): minimize distance
            // For dissimilar pairs (label close to 0): maximize distance up to margin
            T one = NumOps.FromDouble(1.0);
            T zero = NumOps.Zero;

            // Check if similar (label close to 1) or dissimilar (label close to 0)
            // Using threshold of 0.5 to classify
            bool isSimilar = NumOps.GreaterThan(label, half);

            T pairLoss;
            if (isSimilar)
            {
                // Similar: loss = 0.5 * D²
                pairLoss = NumOps.Multiply(half, distanceSquared);
            }
            else
            {
                // Dissimilar: loss = 0.5 * max(0, margin - D)²
                T marginMinusDistance = NumOps.Subtract(ContrastiveMargin, distance);
                // Clamp to max(0, marginMinusDistance)
                T clamped = NumOps.GreaterThan(marginMinusDistance, zero) ? marginMinusDistance : zero;
                T clampedSquared = NumOps.Multiply(clamped, clamped);
                pairLoss = NumOps.Multiply(half, clampedSquared);
            }

            totalLoss = NumOps.Add(totalLoss, pairLoss);
        }

        // Average over all pairs
        T averageLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(_cachedEmbeddingPairs.Count));

        _lastContrastiveLoss = averageLoss;
        return _lastContrastiveLoss;
    }

    /// <summary>
    /// Gets diagnostic information about the contrastive auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about contrastive learning.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about contrastive loss, including
    /// the computed loss value, margin, weight, and configuration parameters.
    /// This information is useful for monitoring similarity learning and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how well the network learns similarity.
    ///
    /// The diagnostics include:
    /// - Total contrastive loss (how well embeddings are organized)
    /// - Contrastive margin (minimum distance for dissimilar pairs)
    /// - Weight applied to the contrastive loss
    /// - Whether contrastive learning is enabled
    ///
    /// This helps you:
    /// - Monitor embedding quality during training
    /// - Debug issues with similarity learning
    /// - Understand the impact of contrastive loss on performance
    ///
    /// You can use this information to adjust margin and weight for better results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalContrastiveLoss", _lastContrastiveLoss?.ToString() ?? "0" },
            { "ContrastiveMargin", ContrastiveMargin?.ToString() ?? "1.0" },
            { "ContrastiveWeight", AuxiliaryLossWeight?.ToString() ?? "0.5" },
            { "UseContrastiveLoss", UseAuxiliaryLoss.ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
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
    /// Updates the network parameters with new values.
    /// </summary>
    /// <param name="parameters">The vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method updates the internal values (weights and biases) of the neural network
    /// during training.
    /// 
    /// The parameters vector contains all the numbers that define how the network processes inputs.
    /// These parameters are split into two parts:
    /// 1. Parameters for the shared subnetwork (which processes each input)
    /// 2. Parameters for the output layer (which compares the embeddings)
    /// 
    /// During training, these parameters are gradually adjusted to make the network better at
    /// determining whether two inputs are similar or different.
    /// 
    /// You typically won't call this method directly - it's used by the training algorithms
    /// that optimize the network.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int subnetworkParameterCount = _subnetwork.GetParameterCount();
        Vector<T> subnetworkParameters = parameters.SubVector(0, subnetworkParameterCount);
        _subnetwork.UpdateParameters(subnetworkParameters);

        Vector<T> outputLayerParameters = parameters.SubVector(subnetworkParameterCount, _outputLayer.ParameterCount);
        _outputLayer.UpdateParameters(outputLayerParameters);
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the Siamese network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property tells you how many numbers (parameters) define your neural network.
    ///
    /// Neural networks learn by adjusting these parameters during training. The parameter count gives you
    /// an idea of how complex your model is:
    ///
    /// - A network with more parameters can potentially learn more complex patterns
    /// - A network with too many parameters might "memorize" the training data instead of learning general patterns
    /// - More parameters require more training data and computational resources
    ///
    /// For example, a Siamese network for face recognition might have millions of parameters to capture
    /// all the subtle features that distinguish different faces.
    ///
    /// This property adds together:
    /// 1. The number of parameters in the shared subnetwork (which processes each input)
    /// 2. The number of parameters in the output layer (which compares the embeddings)
    ///
    /// You might use this information to:
    /// - Estimate how much memory your model will need
    /// - Compare the complexity of different network architectures
    /// - Determine if you have enough training data (typically you want many times more examples than parameters)
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            return _subnetwork.ParameterCount + _outputLayer.ParameterCount;
        }
    }

    /// <summary>
    /// Makes a prediction using the Siamese network to compare the similarity between inputs.
    /// </summary>
    /// <param name="input">The input tensor containing pairs to compare. Expected shape: [batchSize, 2, ...dimensions]</param>
    /// <returns>The similarity scores between each pair as a tensor with shape [batchSize, 1].</returns>
    /// <remarks>
    /// <para>
    /// The prediction process involves passing each input through the shared subnetwork to generate
    /// embeddings, then comparing these embeddings using the output layer to produce similarity scores.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method takes pairs of inputs and tells you how similar they are to each other.
    /// Each input (like an image or text) is processed through the same network to create a compact
    /// representation (embedding). These representations are then compared to produce a similarity score
    /// between 0 (completely different) and 1 (identical).
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Make sure we're in inference mode
        bool originalTrainingMode = IsTrainingMode;
        SetTrainingMode(false);

        try
        {
            // Validate input shape - should have at least 2 dimensions with second dimension = 2
            if (input.Shape.Length < 2 || input.Shape[1] != 2)
            {
                throw new ArgumentException(
                    $"Input tensor must have shape [batchSize, 2, ...dimensions] for Siamese comparison. Got shape: {string.Join(",", input.Shape)}");
            }

            int batchSize = input.Shape[0];
            var output = new Tensor<T>(new[] { batchSize, 1 });

            // Process each pair in the batch
            for (int b = 0; b < batchSize; b++)
            {
                // Extract the pair of inputs - GetSlice returns a tensor
                var input1 = input.GetSlice(b).GetSlice(0);
                var input2 = input.GetSlice(b).GetSlice(1);

                // Process each input through the shared subnetwork
                var embedding1 = _subnetwork.Predict(input1).ToVector();
                var embedding2 = _subnetwork.Predict(input2).ToVector();

                // Combine embeddings and compute similarity
                var combinedEmbedding = CombineEmbeddings(embedding1, embedding2);
                var similarityScore = _outputLayer.Forward(Tensor<T>.FromVector(combinedEmbedding)).ToVector();

                output[b, 0] = similarityScore[0];
            }

            return output;
        }
        finally
        {
            // Restore original training mode
            SetTrainingMode(originalTrainingMode);
        }
    }

    /// <summary>
    /// Trains the Siamese network on pairs of inputs with their expected similarity.
    /// </summary>
    /// <param name="input">The input tensor containing pairs of items. Expected shape: [batchSize, 2, ...dimensions]</param>
    /// <param name="expectedOutput">The expected similarity scores. Shape: [batchSize, 1]</param>
    /// <remarks>
    /// <para>
    /// This method trains the Siamese network by processing pairs through the shared subnetwork,
    /// calculating the similarity between their embeddings, and updating the network parameters
    /// based on the difference between predicted and expected similarity scores.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the network to recognize when two inputs are similar.
    /// You provide pairs of inputs along with how similar they should be (0 to 1). The network learns
    /// to produce embeddings that are close together for similar inputs and far apart for different inputs.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Ensure we're in training mode
        SetTrainingMode(true);

        // Validate input shape and expected output shape
        if (input.Shape.Length < 2 || input.Shape[1] != 2)
        {
            throw new ArgumentException(
                $"Input tensor must have shape [batchSize, 2, ...dimensions] for Siamese training. Got shape: {string.Join(",", input.Shape)}");
        }

        if (expectedOutput.Shape.Length != 2 || expectedOutput.Shape[0] != input.Shape[0] || expectedOutput.Shape[1] != 1)
        {
            throw new ArgumentException(
                $"Expected output tensor must have shape [batchSize, 1]. Got shape: {string.Join(",", expectedOutput.Shape)}");
        }

        int batchSize = input.Shape[0];

        // Clear cached embedding pairs for this training batch
        if (UseAuxiliaryLoss)
        {
            _cachedEmbeddingPairs.Clear();
        }

        // Forward pass to get predictions
        var predictions = Predict(input);

        // Calculate loss using the loss function
        Vector<T> predictedVector = predictions.ToVector();
        Vector<T> expectedVector = expectedOutput.ToVector();
        LastLoss = LossFunction.CalculateLoss(predictedVector, expectedVector);

        // Calculate error gradients
        var outputGradients = new Tensor<T>(predictions.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            // Error = expected - predicted
            outputGradients[b, 0] = NumOps.Subtract(expectedOutput[b, 0], predictions[b, 0]);
        }

        // Process each pair in the batch for the backward pass
        _subnetwork.SetTrainingMode(true);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract the pair - GetSlice returns a tensor
            var input1 = input.GetSlice(b).GetSlice(0);
            var input2 = input.GetSlice(b).GetSlice(1);

            // Forward pass through the subnetwork
            var embedding1 = _subnetwork.Forward(input1).ToVector();
            var embedding2 = _subnetwork.Forward(input2).ToVector();

            // Cache embeddings and label for contrastive loss computation
            if (UseAuxiliaryLoss)
            {
                T label = expectedOutput[b, 0];
                _cachedEmbeddingPairs.Add((embedding1, embedding2, label));
            }

            // Combine embeddings
            var combinedEmbedding = CombineEmbeddings(embedding1, embedding2);

            // Backpropagate through output layer
            var outputGradient = new Tensor<T>(new[] { 1, 1 });
            outputGradient[0, 0] = outputGradients[b, 0];
            var embeddingGradients = _outputLayer.Backward(outputGradient).ToVector();

            // Split gradients for each embedding
            int embeddingSize = embedding1.Length;
            var embedding1Gradients = new Vector<T>(embeddingSize);
            var embedding2Gradients = new Vector<T>(embeddingSize);

            for (int i = 0; i < embeddingSize; i++)
            {
                embedding1Gradients[i] = embeddingGradients[i];
                embedding2Gradients[i] = embeddingGradients[i + embeddingSize];
            }

            // Backpropagate through the subnetwork for each input
            _subnetwork.Backward(Tensor<T>.FromVector(embedding1Gradients).Reshape(input1.Shape));
            _subnetwork.Backward(Tensor<T>.FromVector(embedding2Gradients).Reshape(input2.Shape));
        }

        // Get the learning rate from the architecture or use default
        T learningRate = NumOps.FromDouble(0.001);

        // Update parameters for the subnetwork
        // Assume the subnetwork has internal logic to update all its layers
        Vector<T> subnetworkGradients = _subnetwork.GetParameterGradients();
        Vector<T> subnetworkParameters = _subnetwork.GetParameters();

        // Apply learning rate to gradients using vectorized operations
        var gradientStep = (Vector<T>)Engine.Multiply(subnetworkGradients, learningRate);
        var updatedParameters = (Vector<T>)Engine.Add(subnetworkParameters, gradientStep);

        // Update the subnetwork with new parameters
        _subnetwork.UpdateParameters(updatedParameters);

        // Update output layer parameters directly using the learning rate
        _outputLayer.UpdateParameters(learningRate);

        // Reset training mode
        _subnetwork.SetTrainingMode(false);
    }

    /// <summary>
    /// Serializes Siamese network-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Siamese network to a binary stream, including
    /// the shared subnetwork and the output layer parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method saves your trained Siamese network to a file,
    /// allowing you to load it later without having to retrain it.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Serialize the subnetwork
        var subNetworkData = _subnetwork.Serialize();
        writer.Write(subNetworkData.Length);
        writer.Write(subNetworkData);

        // Serialize the output layer parameters
        Vector<T> outputLayerParams = _outputLayer.GetParameters();
        writer.Write(outputLayerParams.Length);

        for (int i = 0; i < outputLayerParams.Length; i++)
        {
            writer.Write(Convert.ToDouble(outputLayerParams[i]));
        }
    }

    /// <summary>
    /// Deserializes Siamese network-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads the state of a previously saved Siamese network from a binary stream,
    /// reconstructing both the shared subnetwork and the output layer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method loads a previously saved Siamese network from a file,
    /// restoring all its learned parameters so you can use it without retraining.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Deserialize the subnetwork
        _subnetwork = new ConvolutionalNeuralNetwork<T>(Architecture);
        var subNetworkCount = reader.ReadInt32();
        _subnetwork.Deserialize(reader.ReadBytes(subNetworkCount));

        // Deserialize the output layer parameters
        int paramCount = reader.ReadInt32();
        Vector<T> outputLayerParams = new Vector<T>(paramCount);

        for (int i = 0; i < paramCount; i++)
        {
            outputLayerParams[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Initialize the output layer with the correct dimensions
        int embeddingSize = Architecture.GetOutputShape()[0];
        _outputLayer = new DenseLayer<T>(embeddingSize * 2, 1, new SigmoidActivation<T>() as IActivationFunction<T>);
        _outputLayer.SetParameters(outputLayerParams);
    }

    /// <summary>
    /// Gets metadata about the Siamese Network.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the Siamese network, including information
    /// about its architecture, embedding size, and other relevant parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This provides detailed information about your Siamese network,
    /// such as the size of embeddings and the structure of the subnetwork. This information
    /// is useful for documentation, debugging, and understanding the network's configuration.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Prepare Siamese-specific information
        var metadata = new ModelMetadata<T>
        {
            ModelType = ModelType.SiameseNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "EmbeddingSize", Architecture.GetOutputShape()[0] },
                { "SubnetworkType", _subnetwork.GetType().Name },
                { "TotalParameters", GetParameterCount() },
                { "InputShape", string.Join(",", Architecture.GetInputShape()) }
            },
            ModelData = this.Serialize()
        };

        return metadata;
    }

    /// <summary>
    /// Creates a new instance of the Siamese network with the same architecture.
    /// </summary>
    /// <returns>A new instance of the Siamese network.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new Siamese network with the same architecture as the current instance.
    /// The new instance has freshly initialized parameters and is ready for training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a brand new Siamese network with the same structure.
    /// 
    /// Think of it like creating a copy of your current network's blueprint:
    /// - It has the same subnetwork structure for processing inputs
    /// - It processes the same types of inputs (like images of the same size)
    /// - But it starts with fresh, untrained parameters
    /// 
    /// This is useful when you want to:
    /// - Start over with a fresh network but keep the same design
    /// - Create multiple networks with identical structures for comparison
    /// - Train networks with different data but the same architecture
    /// 
    /// The new network will need to be trained from scratch, as it doesn't
    /// inherit any of the "knowledge" from the original network.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SiameseNetwork<T>(Architecture, LossFunction);
    }
}
