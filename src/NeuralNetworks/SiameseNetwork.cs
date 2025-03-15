global using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Siamese Neural Network for comparing pairs of inputs and determining their similarity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// For Beginners: A Siamese Network is a special type of neural network designed to compare two inputs
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
    /// For Beginners: This constructor sets up your Siamese Network with the specified architecture.
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
    public SiameseNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
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
    /// Predicts the similarity between two input vectors.
    /// </summary>
    /// <param name="input1">The first input vector to compare.</param>
    /// <param name="input2">The second input vector to compare.</param>
    /// <returns>A vector containing a single value between 0 and 1, where higher values indicate greater similarity.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This method takes two inputs (like two images) and tells you how similar they are.
    /// 
    /// The process works in three steps:
    /// 1. Each input is processed through the same neural network to create an "embedding"
    ///    (think of this as extracting the key features of each input)
    /// 2. These embeddings are combined into a single vector
    /// 3. The combined vector is passed through a final layer that outputs a similarity score
    ///    between 0 and 1
    /// 
    /// A score close to 1 means the inputs are very similar, while a score close to 0 means
    /// they are very different.
    /// 
    /// For example, if comparing two face images, a score of 0.95 would suggest they're likely
    /// the same person, while a score of 0.2 would suggest they're different people.
    /// </para>
    /// </remarks>
    public Vector<T> PredictPair(Vector<T> input1, Vector<T> input2)
    {
        var embedding1 = GetEmbedding(input1);
        var embedding2 = GetEmbedding(input2);
        var combinedEmbedding = CombineEmbeddings(embedding1, embedding2);

        return _outputLayer.Forward(Tensor<T>.FromVector(combinedEmbedding)).ToVector();
    }

    /// <summary>
    /// Processes a single input and returns its embedding representation.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The embedding vector representing the input's features.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This method processes a single input (like an image) and returns its
    /// "embedding" - a compact representation of its key features.
    /// 
    /// Unlike the PredictPair method which compares two inputs, this method just processes
    /// one input. This is useful when you want to:
    /// 
    /// 1. Store embeddings for later comparison (like saving face embeddings in a database)
    /// 2. Compare one input against many others efficiently
    /// 3. Visualize or analyze the features the network has learned
    /// 
    /// The embedding is what makes Siamese Networks efficient - once you have the embeddings,
    /// comparing them is much faster than processing the original inputs again.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        return GetEmbedding(input);
    }

    /// <summary>
    /// Gets the embedding representation of an input by processing it through the shared subnetwork.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The embedding vector representing the input's features.</returns>
    private Vector<T> GetEmbedding(Vector<T> input)
    {
        return _subnetwork.Predict(input);
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
    /// For Beginners: This method updates the internal values (weights and biases) of the neural network
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
    /// Saves the network's state to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write the network state to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <remarks>
    /// <para>
    /// For Beginners: This method saves your trained network to a file so you can use it later
    /// without having to train it again.
    /// 
    /// It saves all the important information about your network:
    /// - The structure of the shared subnetwork
    /// - The learned parameters (weights and biases) of both the subnetwork and output layer
    /// 
    /// This is useful when:
    /// - You've spent time training a network and want to save your progress
    /// - You want to deploy your trained network to a different application or device
    /// - You want to share your trained network with others
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        _subnetwork.Serialize(writer);
        _outputLayer.Serialize(writer);
    }

    /// <summary>
    /// Loads the network's state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read the network state from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <remarks>
    /// <para>
    /// For Beginners: This method loads a previously saved network from a file.
    /// 
    /// Instead of training a new network from scratch (which can take a lot of time),
    /// you can load a network that was already trained and saved.
    /// 
    /// This is the counterpart to the Serialize method - it reads all the network information
    /// that was previously saved:
    /// - The structure of the shared subnetwork
    /// - The learned parameters (weights and biases) of both the subnetwork and output layer
    /// 
    /// After loading, the network is ready to make predictions immediately.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        _subnetwork.Deserialize(reader);
        _outputLayer.Deserialize(reader);
    }

    /// <summary>
    /// Gets the total number of trainable parameters in the Siamese network.
    /// </summary>
    /// <returns>The total count of parameters in both the subnetwork and output layer.</returns>
    /// <remarks>
    /// <para>
    /// For Beginners: This method tells you how many numbers (parameters) define your neural network.
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
}