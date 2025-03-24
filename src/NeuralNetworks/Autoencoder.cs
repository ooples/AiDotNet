namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an autoencoder neural network that can compress data into a lower-dimensional representation and reconstruct it.
/// </summary>
/// <remarks>
/// <para>
/// An autoencoder is a type of neural network designed to learn efficient data encodings in an unsupervised manner.
/// It consists of an encoder that compresses the input data into a lower-dimensional representation (the latent space)
/// and a decoder that reconstructs the original data from this representation. Autoencoders are trained to minimize
/// the difference between the original input and the reconstructed output.
/// </para>
/// <para><b>For Beginners:</b> An autoencoder is like a sophisticated compression and decompression system.
/// 
/// Think of it like this:
/// - The encoder part takes your original data (like an image) and compresses it into a smaller representation
/// - The middle layer (latent space) holds this compressed version of your data
/// - The decoder part takes this compressed version and tries to recreate the original data
/// 
/// For example, with images:
/// - You might compress a 256×256 pixel image (65,536 values) into just 100 numbers
/// - The network learns which features are most important to preserve
/// - It then learns to reconstruct the image from only those 100 numbers
/// 
/// This is useful for:
/// - Data compression
/// - Noise reduction (by removing noise during reconstruction)
/// - Feature learning (the compressed representation often contains meaningful features)
/// - Anomaly detection (unusual data is reconstructed poorly)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class Autoencoder<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the size of the encoded representation (latent space).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates the dimensionality of the encoded representation, which is the size of the output from
    /// the middle (bottleneck) layer of the autoencoder. This is typically smaller than the input size, forcing the
    /// autoencoder to learn a compressed representation of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This is the size of the compressed representation.
    /// 
    /// For example:
    /// - If your input data has 1000 dimensions (like a 1000-pixel image)
    /// - And EncodedSize is 50
    /// - Then your data is being compressed to 5% of its original size
    /// 
    /// The smaller this value:
    /// - The more compression is happening
    /// - The more the network has to be selective about what information to keep
    /// - The more efficient but potentially less accurate the reconstruction becomes
    /// </para>
    /// </remarks>
    public int EncodedSize { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Autoencoder{T}"/> class.
    /// </summary>
    /// <param name="architecture">The architecture specification for the autoencoder.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates an autoencoder with the specified architecture. It initializes the layers according to
    /// the architecture specification or uses default layers if none are provided. It also sets the EncodedSize property
    /// based on the size of the middle layer.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new autoencoder with the specified settings.
    /// 
    /// The architecture parameter determines:
    /// - How many layers the autoencoder has
    /// - How many neurons are in each layer
    /// - What activation functions are used
    /// - How the data flows through the network
    /// 
    /// A typical autoencoder architecture might look like:
    /// - Input: 784 neurons (for a 28×28 image)
    /// - Encoder: 500 neurons, then 250 neurons
    /// - Latent space: 50 neurons
    /// - Decoder: 250 neurons, then 500 neurons
    /// - Output: 784 neurons (same as input)
    /// 
    /// The architecture is symmetric, with the decoder mirroring the encoder.
    /// </para>
    /// </remarks>
    public Autoencoder(NeuralNetworkArchitecture<T> architecture) : base(architecture)
    {
        EncodedSize = 0;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the layers of the autoencoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the layers of the autoencoder either by using the layers provided by the user in the
    /// architecture specification or by creating default autoencoder layers if none are provided. It also sets the
    /// EncodedSize property based on the size of the middle layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the building blocks of the autoencoder.
    /// 
    /// It does one of two things:
    /// 1. If you provided specific layers in the architecture, it uses those
    /// 2. If you didn't provide layers, it creates a default set of layers appropriate for an autoencoder
    /// 
    /// The default layers typically create a "bottleneck" shape:
    /// - Starting with larger layers (the input)
    /// - Getting progressively smaller (the encoder)
    /// - Reaching a small middle layer (the latent space)
    /// - Getting progressively larger again (the decoder)
    /// - Ending with a layer the same size as the input (the output)
    /// 
    /// It also sets the EncodedSize property to the size of the middle layer.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAutoEncoderLayers(Architecture));
        }

        // Set EncodedSize based on the middle layer
        EncodedSize = Layers[Layers.Count / 2].GetOutputShape()[0];
    }

    /// <summary>
    /// Validates that the custom layers conform to autoencoder requirements.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <exception cref="ArgumentException">Thrown when the layers do not conform to autoencoder requirements.</exception>
    /// <remarks>
    /// <para>
    /// This method validates that the provided layers conform to the requirements of an autoencoder:
    /// 1. There must be at least 3 layers (input, encoded, and output).
    /// 2. The input and output layers must have the same size.
    /// 3. The architecture must be symmetric.
    /// 4. The activation functions must be symmetric.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if the layers you provided will work for an autoencoder.
    /// 
    /// It makes sure:
    /// - You have at least 3 layers (input → encoded → output)
    /// - The input and output layers are the same size (since an autoencoder reconstructs its input)
    /// - The network is symmetric (decoder mirrors encoder)
    /// - The activation functions are symmetric (same functions used in corresponding encoder/decoder layers)
    /// 
    /// If any of these requirements aren't met, it shows an error explaining what's wrong.
    /// This helps ensure your autoencoder is structured correctly before you start training it.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
        {
            throw new ArgumentException("Autoencoder must have at least 3 layers (input, encoded, and output).");
        }

        // Check if input and output layers have the same size
        if (!Enumerable.SequenceEqual(layers[0].GetInputShape(), layers[layers.Count - 1].GetOutputShape()))
        {
            throw new ArgumentException("Input and output layer sizes must be the same for an autoencoder.");
        }

        // Ensure the architecture is symmetric
        for (int i = 0; i < layers.Count / 2; i++)
        {
            if (!Enumerable.SequenceEqual(layers[i].GetOutputShape(), layers[layers.Count - 1 - i].GetInputShape()))
            {
                throw new ArgumentException($"Layer sizes must be symmetric. Mismatch at position {i} and {layers.Count - i - 1}");
            }
        }

        // Validate activation functions
        for (int i = 0; i < layers.Count / 2; i++)
        {
            var leftActivation = layers[i].GetActivationTypes();
            var rightActivation = layers[layers.Count - 1 - i].GetActivationTypes();

            if (!Enumerable.SequenceEqual(leftActivation, rightActivation))
            {
                throw new ArgumentException($"Activation functions must be symmetric. Mismatch at position {i} and {layers.Count - i - 1}");
            }
        }
    }

    /// <summary>
    /// Encodes the input data into the latent space representation.
    /// </summary>
    /// <param name="input">The input vector to encode.</param>
    /// <returns>The encoded vector (latent space representation).</returns>
    /// <remarks>
    /// <para>
    /// This method performs the encoding part of the autoencoder, passing the input data through the encoder layers
    /// to produce the latent space representation. It stops at the middle layer of the autoencoder.
    /// </para>
    /// <para><b>For Beginners:</b> This method compresses your data into the smaller representation.
    /// 
    /// When you call Encode:
    /// - Your data is processed by the first half of the network (the encoder part)
    /// - It passes through progressively smaller layers
    /// - It stops at the middle layer (the bottleneck)
    /// - You get back the compressed representation
    /// 
    /// This is useful when you:
    /// - Want the compressed representation for other tasks
    /// - Need to reduce the dimensionality of your data
    /// - Want to extract the key features learned by the autoencoder
    /// 
    /// For example, you might encode images into a small vector to cluster similar images together.
    /// </para>
    /// </remarks>
    public Vector<T> Encode(Vector<T> input)
    {
        var current = input;
        for (int i = 0; i < Layers.Count / 2; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    /// <summary>
    /// Decodes the latent space representation back to the original space.
    /// </summary>
    /// <param name="encodedInput">The encoded vector (latent space representation) to decode.</param>
    /// <returns>The reconstructed vector in the original space.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the decoding part of the autoencoder, passing the latent space representation through
    /// the decoder layers to reconstruct the original data. It starts from the middle layer of the autoencoder.
    /// </para>
    /// <para><b>For Beginners:</b> This method reconstructs your original data from the compressed representation.
    /// 
    /// When you call Decode:
    /// - Your compressed data is processed by the second half of the network (the decoder part)
    /// - It passes through progressively larger layers
    /// - It expands the compressed representation back to the original size
    /// - You get back a reconstruction of the original data
    /// 
    /// This is useful when you:
    /// - Want to see what information was preserved during compression
    /// - Need to generate new data from the latent space
    /// - Want to remove noise or fill in missing data
    /// 
    /// For example, you might create a new face image by decoding a point in the latent space.
    /// </para>
    /// </remarks>
    public Vector<T> Decode(Vector<T> encodedInput)
    {
        var current = encodedInput;
        for (int i = Layers.Count / 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    /// <summary>
    /// Makes a prediction using the autoencoder by encoding and then decoding the input.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The reconstructed output vector.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a full forward pass through the autoencoder, encoding the input into the latent space
    /// and then decoding it back to the original space. The goal of training is to make this reconstructed output
    /// as close as possible to the original input.
    /// </para>
    /// <para><b>For Beginners:</b> This method runs your data through the entire autoencoder (compress then reconstruct).
    /// 
    /// It's essentially a shortcut that:
    /// 1. Calls Encode() to compress your data
    /// 2. Then calls Decode() to reconstruct it
    /// 
    /// When the autoencoder is well-trained:
    /// - The output should closely resemble the input
    /// - But with some differences based on what the network learned was important
    /// 
    /// For example, if you feed in a noisy image, you might get back a cleaner version
    /// as the autoencoder learns to focus on the important features and ignore the noise.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        return Decode(Encode(input));
    }

    /// <summary>
    /// Updates the parameters of the autoencoder.
    /// </summary>
    /// <param name="parameters">The parameters to update the network with.</param>
    /// <remarks>
    /// <para>
    /// This method updates the parameters of each layer in the autoencoder with the provided parameter values.
    /// It distributes the parameters to each layer based on the number of parameters in each layer.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the network's internal values to improve its performance.
    /// 
    /// During training:
    /// - The learning algorithm calculates how the parameters should change
    /// - This method applies those changes to the actual parameters
    /// - Each layer gets its own portion of the parameter updates
    /// 
    /// Think of it like fine-tuning all the components of the autoencoder based on feedback:
    /// - The encoder learns better ways to compress the data
    /// - The decoder learns better ways to reconstruct the data
    /// - Together they improve at preserving the most important information
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
    /// Serializes the autoencoder to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to serialize to.</param>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when serialization encounters an error.</exception>
    /// <remarks>
    /// <para>
    /// This method serializes the autoencoder by writing the number of layers and then serializing each layer
    /// in sequence. For each layer, it writes the full type name followed by the layer's serialized data.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the autoencoder to a file or stream so it can be used later.
    /// 
    /// Serialization is like taking a snapshot of the autoencoder:
    /// - It saves the structure of the network (number and types of layers)
    /// - It saves all the learned parameters (weights, biases, etc.)
    /// - It ensures everything can be reconstructed exactly as it was
    /// 
    /// This is useful for:
    /// - Saving a trained model for later use
    /// - Sharing a model with others
    /// - Creating backups during long training processes
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
    /// Deserializes the autoencoder from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize from.</param>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when deserialization encounters an error.</exception>
    /// <remarks>
    /// <para>
    /// This method deserializes the autoencoder by reading the number of layers and then deserializing each layer
    /// in sequence. For each layer, it reads the full type name, creates an instance of that type, and then deserializes
    /// the layer's data.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved autoencoder from a file or stream.
    /// 
    /// Deserialization is like reconstructing the autoencoder from a snapshot:
    /// - It reads the structure of the network (number and types of layers)
    /// - It loads all the learned parameters (weights, biases, etc.)
    /// - It recreates the autoencoder exactly as it was when saved
    /// 
    /// This allows you to:
    /// - Use a previously trained model without retraining it
    /// - Continue training from where you left off
    /// - Deploy the same model across different systems
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        int layerCount = reader.ReadInt32();
        Layers.Clear();

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
        }
    }
}