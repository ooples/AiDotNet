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
    /// Gets or sets the list of Restricted Boltzmann Machines that form the layers of the Deep Belief Network.
    /// </summary>
    /// <value>A list of Restricted Boltzmann Machine layers.</value>
    /// <remarks>
    /// <para>
    /// This property contains the Restricted Boltzmann Machines (RBMs) that make up the Deep Belief Network.
    /// Each RBM is trained to model the distribution of its input data, and the hidden layer of one RBM serves
    /// as the visible layer for the next RBM in the stack. This greedy layer-wise training allows the network
    /// to learn increasingly abstract representations of the data.
    /// </para>
    /// <para><b>For Beginners:</b> These are the floors of our pattern-recognition tower.
    /// 
    /// Think of RBMLayers as:
    /// - The individual floors in our tower (the Deep Belief Network)
    /// - Each floor (RBM) learns patterns from the floor below it
    /// - The floors are trained one at a time, starting from the bottom
    /// - Each floor transforms the data into a more abstract representation before passing it up
    /// 
    /// This approach is powerful because each layer can focus on learning specific patterns
    /// without worrying about the entire network at once.
    /// </para>
    /// </remarks>
    private List<RestrictedBoltzmannMachine<T>> _rbmLayers { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="DeepBeliefNetwork{T}"/> class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration, which must include RBM layers.</param>
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
    public DeepBeliefNetwork(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        _rbmLayers = architecture.RbmLayers;
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

        ValidateAndInitializeRbmLayers();
    }

    /// <summary>
    /// Validates and initializes the RBM layers of the Deep Belief Network.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no RBM layers are provided, when an RBM layer is null, or when there is a dimension mismatch between layers.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method validates the Restricted Boltzmann Machine (RBM) layers to ensure they form a valid Deep Belief Network.
    /// It checks that RBM layers are provided, that none of them is null, and that the dimensions of consecutive layers
    /// are compatible (the hidden size of one layer matches the visible size of the next). If the validation passes,
    /// it initializes the RBM layers for use in pre-training.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks that all the floors of our tower will fit together properly.
    /// 
    /// During validation:
    /// - It makes sure you've included RBM layers (you can't build a tower with no floors!)
    /// - It checks that each floor is properly defined
    /// - It ensures that each floor connects correctly to the floor above it
    ///   (the output size of one floor must match the input size of the next floor)
    /// - It makes sure the bottom floor fits the size of your input data
    /// 
    /// If any of these checks fail, the method will stop and tell you what's wrong,
    /// like a building inspector finding problems with your blueprint.
    /// </para>
    /// </remarks>
    private void ValidateAndInitializeRbmLayers()
    {
        if (Architecture.RbmLayers == null || Architecture.RbmLayers.Count == 0)
        {
            throw new InvalidOperationException("RBM layers are required for a Deep Belief Network but none were provided.");
        }

        // Validate RBM layers
        for (int i = 0; i < Architecture.RbmLayers.Count; i++)
        {
            var rbm = Architecture.RbmLayers[i];
            if (rbm == null)
            {
                throw new InvalidOperationException($"RBM layer at index {i} is null.");
            }

            if (i > 0)
            {
                var prevRbm = Architecture.RbmLayers[i - 1];
                if (rbm.VisibleSize != prevRbm.HiddenSize)
                {
                    throw new InvalidOperationException($"Mismatch in RBM layer dimensions. Layer {i-1} hidden size ({prevRbm.HiddenSize}) " +
                        $"do not match layer {i} visible size ({rbm.VisibleSize}).");
                }
            }
            else
            {
                // Check if the first RBM layer matches the input dimension
                if (rbm.VisibleSize != Architecture.CalculatedInputSize)
                {
                    throw new InvalidOperationException($"The first RBM layer's visible units ({rbm.VisibleSize}) " +
                        $"do not match the network's calculated input size ({Architecture.CalculatedInputSize}).");
                }
            }
        }

        // If validation passes, initialize RBMLayers
        _rbmLayers = [.. Architecture.RbmLayers];
    }

    /// <summary>
    /// Pretrains the Restricted Boltzmann Machine layers of the Deep Belief Network.
    /// </summary>
    /// <param name="trainingData">The input training data tensor.</param>
    /// <param name="epochs">The number of training epochs for each RBM layer.</param>
    /// <param name="learningRate">The learning rate for the training process.</param>
    /// <remarks>
    /// <para>
    /// This method performs the unsupervised pre-training phase of the Deep Belief Network. It trains each RBM layer
    /// in sequence, from the bottom up. After training an RBM layer, the method transforms the training data through
    /// that layer to create the input for the next layer. This greedy layer-wise pre-training helps the network
    /// learn meaningful representations of the data before fine-tuning.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds and trains our tower one floor at a time.
    /// 
    /// The pre-training process works like this:
    /// - We start with the bottom floor (the first RBM)
    /// - We train it for a number of rounds (epochs) using our original data
    /// - Once trained, we use this floor to transform our data
    /// - We then use the transformed data to train the next floor up
    /// - We repeat this process for each floor, working our way up the tower
    /// 
    /// This step-by-step approach is what makes Deep Belief Networks special. Each floor
    /// learns patterns based on what the floor below it has already discovered,
    /// creating increasingly sophisticated representations of the data.
    /// </para>
    /// </remarks>
    public void PretrainRBMs(Tensor<T> trainingData, int epochs, T learningRate)
    {
        for (int i = 0; i < _rbmLayers.Count; i++)
        {
            Console.WriteLine($"Pretraining RBM layer {i + 1}");
            var rbm = _rbmLayers[i];

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                rbm.Train(trainingData, 1, learningRate);
            }

            // Transform the data through this RBM for the next layer
            trainingData = rbm.GetHiddenLayerActivation(trainingData);
        }
    }

    /// <summary>
    /// Makes a prediction using the current state of the Deep Belief Network.
    /// </summary>
    /// <param name="input">The input vector to make a prediction for.</param>
    /// <returns>The predicted output vector after passing through all layers of the network.</returns>
    /// <remarks>
    /// <para>
    /// This method generates a prediction by passing the input vector through each layer of the Deep Belief Network
    /// in sequence. Each layer processes the output of the previous layer, transforming the data until it reaches
    /// the final output layer. The result is a vector representing the network's prediction.
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
    public override Vector<T> Predict(Vector<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(Tensor<T>.FromVector(current)).ToVector();
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
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Serializes the Deep Belief Network to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when a null layer is encountered or a layer type name cannot be determined.</exception>
    /// <remarks>
    /// <para>
    /// This method saves the state of the Deep Belief Network to a binary stream. It writes the number of layers,
    /// followed by the type name and serialized state of each layer. This allows the Deep Belief Network
    /// to be saved to disk and later restored with its trained parameters intact.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the network to a file.
    /// 
    /// When saving the Deep Belief Network:
    /// - First, it saves how many layers the network has
    /// - Then, for each layer, it saves:
    ///   - What type of layer it is
    ///   - All the values and settings for that layer
    /// 
    /// This is like taking a complete snapshot of the network so you can reload it later
    /// without having to train it all over again.
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        writer.Write(Layers.Count);
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Encountered a null layer during serialization.");

            string? fullName = layer.GetType().FullName;
            if (string.IsNullOrEmpty(fullName))
                throw new InvalidOperationException($"Unable to get full name for layer type {layer.GetType()}");

            writer.Write(fullName);
            layer.Serialize(writer);
        }
    }

    /// <summary>
    /// Deserializes the Deep Belief Network from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when layer type information is invalid or instance creation fails.</exception>
    /// <remarks>
    /// <para>
    /// This method restores the state of the Deep Belief Network from a binary stream. It reads the number of layers,
    /// followed by the type name and serialized state of each layer. This allows a previously saved
    /// Deep Belief Network to be restored from disk with all its trained parameters. It also reconstructs
    /// the list of RBM layers by identifying RestrictedBoltzmannMachine instances among the deserialized layers.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved network from a file.
    /// 
    /// When loading the Deep Belief Network:
    /// - First, it reads how many layers the network had
    /// - Then, for each layer, it:
    ///   - Reads what type of layer it was
    ///   - Creates a new layer of that type
    ///   - Loads all the values and settings for that layer
    ///   - Adds the layer to the network
    /// - It also identifies which layers are RBM layers and rebuilds the tower structure
    /// 
    /// This is like restoring a complete snapshot of your network, bringing back
    /// all the patterns it had learned before.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();
        _rbmLayers.Clear();

        for (int i = 0; i < layerCount; i++)
        {
            string layerTypeName = reader.ReadString();
            if (string.IsNullOrEmpty(layerTypeName))
                throw new InvalidOperationException("Encountered an empty layer type name during deserialization.");

            Type? layerType = Type.GetType(layerTypeName);
            if (layerType == null)
                throw new InvalidOperationException($"Cannot find type {layerTypeName}");

            if (!typeof(ILayer<T>).IsAssignableFrom(layerType))
                throw new InvalidOperationException($"Type {layerTypeName} does not implement ILayer<T>");

            object? instance = Activator.CreateInstance(layerType);
            if (instance == null)
                throw new InvalidOperationException($"Failed to create an instance of {layerTypeName}");

            var layer = (ILayer<T>)instance;
            layer.Deserialize(reader);
            Layers.Add(layer);

            // Reconstruct RBM layers
            if (layer is RestrictedBoltzmannMachine<T> rbm)
            {
                _rbmLayers.Add(rbm);
            }
        }
    }
}