global using AiDotNet.NeuralNetworks.Layers;

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
public class SiameseNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// The shared neural network that processes each input independently.
    /// </summary>
    /// <remarks>
    /// This network creates the embeddings (compact representations) for each input.
    /// </remarks>
    private ConvolutionalNeuralNetwork<T> _subnetwork = default!;
    
    /// <summary>
    /// The final layer that compares the embeddings and produces a similarity score.
    /// </summary>
    private DenseLayer<T> _outputLayer = default!;

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
    public SiameseNetwork(NeuralNetworkArchitecture<T> architecture, ILossFunction<T>? lossFunction = null) : 
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _subnetwork = new ConvolutionalNeuralNetwork<T>(architecture);
        int embeddingSize = architecture.GetOutputShape()[0];
        _outputLayer = new DenseLayer<T>(embeddingSize * 2, 1, new SigmoidActivation<T>() as IActivationFunction<T>);
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
    /// <returns>The total count of parameters in both the subnetwork and output layer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you how many numbers (parameters) define your neural network.
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
    /// This method adds together:
    /// 1. The number of parameters in the shared subnetwork (which processes each input)
    /// 2. The number of parameters in the output layer (which compares the embeddings)
    /// 
    /// You might use this information to:
    /// - Estimate how much memory your model will need
    /// - Compare the complexity of different network architectures
    /// - Determine if you have enough training data (typically you want many times more examples than parameters)
    /// </para>
    /// </remarks>
    public override int GetParameterCount()
    {
        return _subnetwork.GetParameterCount() + _outputLayer.ParameterCount;
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
        Vector<T> updatedParameters = new Vector<T>(subnetworkParameters.Length);
    
        // Apply learning rate to gradients
        for (int i = 0; i < subnetworkParameters.Length; i++)
        {
            T gradientStep = NumOps.Multiply(subnetworkGradients[i], learningRate);
            updatedParameters[i] = NumOps.Add(subnetworkParameters[i], gradientStep);
        }
    
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
    /// <returns>A ModelMetadata object containing information about the network.</returns>
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