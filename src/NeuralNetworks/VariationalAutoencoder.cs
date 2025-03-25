namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Variational Autoencoder (VAE) neural network architecture, which is used for 
/// generating new data similar to the training data and learning compressed representations.
/// </summary>
/// <remarks>
/// <para>
/// A Variational Autoencoder is a type of generative model that learns to encode input data into a 
/// probabilistic latent space and then decode samples from that space back into the original data space.
/// Unlike standard autoencoders, VAEs ensure the latent space has good properties for generating new samples 
/// by learning a distribution rather than a fixed encoding.
/// </para>
/// <para>
/// VAEs consist of:
/// - An encoder network that maps input data to a probability distribution in latent space
/// - A sampling mechanism that draws samples from this distribution
/// - A decoder network that maps samples from latent space back to the original data space
/// </para>
/// <para><b>For Beginners:</b> A Variational Autoencoder is like a creative compression system.
/// 
/// Imagine you have a folder full of photos of cats:
/// - The encoder is like a person who studies all these photos and learns to describe any cat using just a few key attributes (like fur color, ear shape, size)
/// - These few attributes are the "latent space" - a much smaller representation of the data
/// - The special thing about a VAE is that instead of exact values, it describes each attribute as a range of possible values (a probability distribution)
/// - The decoder is like an artist who can take these attribute descriptions and draw a new cat based on them
/// 
/// This ability to work with probability distributions means:
/// - You can generate new, never-before-seen cats by sampling from these distributions
/// - The generated cats will look realistic because they follow the patterns learned from real cats
/// - You can smoothly transition between different types of cats by moving through the latent space
/// 
/// VAEs are used for image generation, data compression, anomaly detection, and other creative applications.
/// </para>
/// </remarks>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
public class VariationalAutoencoder<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the size of the latent space dimension in the Variational Autoencoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The latent size determines the dimensionality of the compressed representation that the VAE learns.
    /// Smaller values create more compressed representations but may lose more information, while larger 
    /// values preserve more details but may be less efficient for compression or generation.
    /// </para>
    /// <para><b>For Beginners:</b> The latent size is like the number of describing words you can use.
    /// 
    /// Think of it as how many attributes you can use to describe the data:
    /// - A small latent size (e.g., 2-10) means using very few attributes, creating a highly compressed but possibly less detailed representation
    /// - A larger latent size (e.g., 32-256) allows for more detailed representations but requires more computation
    /// 
    /// For example, if you're working with face images:
    /// - A small latent size might only capture basic features like hair color and face shape
    /// - A larger latent size could capture more subtle details like wrinkles, lighting, and expressions
    /// 
    /// The right latent size depends on your specific task - smaller for simple datasets, larger for complex ones.
    /// </para>
    /// </remarks>
    public int LatentSize { get; private set; }

    /// <summary>
    /// Gets or sets the layer that computes the mean parameters of the latent distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This layer computes the mean values for each dimension of the latent probability distribution.
    /// These means represent the center points of the Gaussian distributions in the latent space.
    /// </para>
    /// <para><b>For Beginners:</b> The mean layer determines the "average value" for each feature in the compressed representation.
    /// 
    /// For example, if one dimension of your latent space represents hair length:
    /// - The mean would be the most likely hair length value for a particular input
    /// - This layer calculates the best "guess" for each attribute in the latent space
    /// 
    /// Think of it as finding the most representative value for each feature based on the input data.
    /// </para>
    /// </remarks>
    private MeanLayer<T>? _meanLayer { get; set; }

    /// <summary>
    /// Gets or sets the layer that computes the log variance parameters of the latent distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This layer computes the log variance values for each dimension of the latent probability distribution.
    /// Log variance is used instead of variance for numerical stability in training. The variance represents 
    /// the spread or uncertainty in each dimension of the latent space.
    /// </para>
    /// <para><b>For Beginners:</b> The log variance layer determines how much flexibility or uncertainty there is around each feature value.
    /// 
    /// Continuing the hair length example:
    /// - The log variance would indicate how much the hair length could vary
    /// - A small variance means the model is very certain about the hair length
    /// - A large variance means there's more flexibility or uncertainty
    /// 
    /// We use the logarithm of variance (log variance) because it's more stable for calculations,
    /// but you can think of it as controlling the "wiggle room" for each attribute.
    /// </para>
    /// </remarks>
    private LogVarianceLayer<T>? _logVarianceLayer { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="VariationalAutoencoder{T}"/> class with the 
    /// specified architecture and latent space size.
    /// </summary>
    /// <param name="architecture">The neural network architecture that defines the overall structure.</param>
    /// <param name="latentSize">The size of the latent space dimension.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new VAE with the provided architecture and latent size. The architecture
    /// defines the general structure of the network, while the latent size specifically controls the
    /// dimensionality of the compressed representation that the VAE will learn.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you set up your VAE with basic settings.
    /// 
    /// When creating a VAE, you need to specify:
    /// - The overall architecture (like how many layers, their sizes, etc.)
    /// - The latent size (how many attributes or features to use in the compressed representation)
    /// 
    /// It's like configuring a new camera - you choose the general model (architecture) and then 
    /// specify how much compression you want for the images (latent size).
    /// </para>
    /// </remarks>
    public VariationalAutoencoder(NeuralNetworkArchitecture<T> architecture, int latentSize) : base(architecture)
    {
        LatentSize = latentSize;
    }

    /// <summary>
    /// Sets up the layers of the Variational Autoencoder based on the provided architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method either uses custom layers provided by the user or creates default VAE layers.
    /// It then sets up specific references to key layers like the mean and log variance layers that
    /// are essential for the VAE's functioning.
    /// </para>
    /// <para><b>For Beginners:</b> This method builds the structure of your VAE.
    /// 
    /// It works in one of two ways:
    /// - If you've provided your own custom layers, it uses those
    /// - Otherwise, it creates a standard set of VAE layers based on your settings
    /// 
    /// Then it identifies and sets up the special layers that make a VAE work:
    /// - The mean layer (for calculating the average values in the latent space)
    /// - The log variance layer (for calculating the uncertainty ranges)
    /// 
    /// It's like assembling a machine based on either your custom blueprint or a standard design.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultVAELayers(Architecture, LatentSize));
        }

        SetSpecificLayers();
    }

    /// <summary>
    /// Ensures that custom layers provided for the VAE meet the minimum requirements.
    /// </summary>
    /// <param name="layers">The list of layers to validate.</param>
    /// <remarks>
    /// <para>
    /// A valid VAE must include a mean layer, a log variance layer, and a pooling layer for the
    /// reparameterization trick. This method checks for these required components and throws
    /// an exception if any are missing.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if your custom layers will actually work as a VAE.
    /// 
    /// For a VAE to function properly, it needs at minimum:
    /// - A mean layer (to calculate the central values in the latent space)
    /// - A log variance layer (to calculate the uncertainty ranges)
    /// - A pooling layer (for the "reparameterization trick" - a special technique that makes training possible)
    /// 
    /// If any of these essential components are missing, it's like trying to build a car without wheels or an engine - it won't work!
    /// 
    /// This method checks for these essential components and raises an error if they're missing.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the custom layers don't include required layer types.
    /// </exception>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        bool hasMeanLayer = false;
        bool hasLogVarianceLayer = false;
        bool hasPoolingLayer = false;

        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is MeanLayer<T>)
            {
                hasMeanLayer = true;
            }
            else if (layers[i] is LogVarianceLayer<T>)
            {
                hasLogVarianceLayer = true;
            }
            else if (layers[i] is PoolingLayer<T>)
            {
                hasPoolingLayer = true;
            }
        }

        if (!hasMeanLayer)
        {
            throw new InvalidOperationException("Custom VAE layers must include a MeanLayer.");
        }

        if (!hasLogVarianceLayer)
        {
            throw new InvalidOperationException("Custom VAE layers must include a LogVarianceLayer.");
        }

        if (!hasPoolingLayer)
        {
            throw new InvalidOperationException("Custom VAE layers must include a PoolingLayer for the reparameterization trick.");
        }
    }

    /// <summary>
    /// Sets up references to the specific layers needed for the VAE's functionality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method finds and sets references to the mean layer and log variance layer, which are
    /// essential components of a VAE. These layers are needed for computing the parameters of the
    /// latent space distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This method identifies and saves references to the special layers that make a VAE work.
    /// 
    /// It searches through all the layers to find:
    /// - The mean layer (that calculates average values for the latent space)
    /// - The log variance layer (that calculates uncertainty ranges for the latent space)
    /// 
    /// These references are saved so that other methods can easily access these special layers
    /// when encoding inputs or generating new outputs.
    /// 
    /// It's like labeling the special components in a machine so you can find them quickly later.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the mean layer or log variance layer cannot be found.
    /// </exception>
    private void SetSpecificLayers()
    {
        _meanLayer = Layers.OfType<MeanLayer<T>>().FirstOrDefault();
        _logVarianceLayer = Layers.OfType<LogVarianceLayer<T>>().FirstOrDefault();

        if (_meanLayer == null || _logVarianceLayer == null)
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer must be present in the network.");
        }
    }

    /// <summary>
    /// Encodes an input vector into mean and log variance parameters in the latent space.
    /// </summary>
    /// <param name="input">The input vector to encode.</param>
    /// <returns>A tuple containing the mean and log variance vectors of the latent distribution.</returns>
    /// <remarks>
    /// <para>
    /// This method passes the input through the encoder portion of the VAE to produce the parameters
    /// of the latent distribution (mean and log variance). These parameters define a probability
    /// distribution in the latent space from which samples can be drawn.
    /// </para>
    /// <para><b>For Beginners:</b> This method compresses your input data into a compact representation.
    /// 
    /// When you encode data:
    /// - The input passes through the first half of the network (the encoder)
    /// - The encoder produces two sets of values for each dimension in the latent space:
    ///   * Mean values (the central or most likely value for each feature)
    ///   * Log variance values (how much uncertainty or flexibility there is around each feature)
    /// 
    /// For example, if encoding a face image:
    /// - The means might represent the most likely values for features like hair color, face shape, etc.
    /// - The log variances represent how certain the model is about these values
    /// 
    /// This compressed representation captures the essential information about the input in a much smaller form.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the mean layer or log variance layer have not been properly initialized.
    /// </exception>
    public (Vector<T> Mean, Vector<T> LogVariance) Encode(Vector<T> input)
    {
        if (_meanLayer == null || _logVarianceLayer == null)
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer have not been properly initialized.");
        }

        var current = input;
        for (int i = 0; i < Layers.Count / 2; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        var mean = _meanLayer.Forward(Tensor<T>.FromVector(current)).ToVector();
        var logVariance = _logVarianceLayer.Forward(Tensor<T>.FromVector(current)).ToVector();

        return (mean, logVariance);
    }

    /// <summary>
    /// Implements the reparameterization trick to sample from the latent distribution in a way that allows gradient flow.
    /// </summary>
    /// <param name="mean">The mean vector of the latent distribution.</param>
    /// <param name="logVariance">The log variance vector of the latent distribution.</param>
    /// <returns>A sampled vector from the latent distribution.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the reparameterization trick, which is a key innovation in VAEs. It allows
    /// the model to sample from the latent distribution while still enabling gradient flow during training.
    /// The trick works by sampling from a standard normal distribution and then transforming those samples
    /// using the mean and variance parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method generates a random sample from your compressed representation.
    /// 
    /// The "reparameterization trick" is a clever technique that:
    /// - Takes the mean and log variance values from the encoder
    /// - Adds the right amount of randomness to create a sample point in the latent space
    /// - Does this in a way that still allows the network to learn effectively
    /// 
    /// It's like having a recipe (the mean) and some flexibility (the variance):
    /// - If you're very certain about a feature (low variance), the sample will be close to the mean
    /// - If you're less certain (high variance), the sample could be further from the mean
    /// 
    /// The randomness is important because:
    /// - It lets the VAE generate different outputs even for the same input
    /// - It forces the VAE to learn a smooth, continuous latent space
    /// - It allows for creative generation of new, unique examples
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when the mean and log variance vectors don't have the same length.
    /// </exception>
    public Vector<T> Reparameterize(Vector<T> mean, Vector<T> logVariance)
    {
        if (mean.Length != logVariance.Length)
            throw new ArgumentException("Mean and log variance vectors must have the same length.");

        var result = new T[mean.Length];

        for (int i = 0; i < mean.Length; i++)
        {
            // Generate two random numbers from a uniform distribution
            double u1 = 1.0 - Random.NextDouble(); // Uniform(0,1] random number
            double u2 = 1.0 - Random.NextDouble(); // Uniform(0,1] random number

            // Box-Muller transform to generate a sample from a standard normal distribution
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

            // Convert z to T type
            T zT = NumOps.FromDouble(z);

            // Reparameterization trick: sample = mean + exp(0.5 * logVariance) * z
            T halfLogVariance = NumOps.Multiply(NumOps.FromDouble(0.5), logVariance[i]);
            T stdDev = NumOps.Exp(halfLogVariance);
            result[i] = NumOps.Add(mean[i], NumOps.Multiply(stdDev, zT));
        }

        return new Vector<T>(result);
    }

    /// <summary>
    /// Decodes a vector from the latent space back to the original data space.
    /// </summary>
    /// <param name="latentVector">The vector in latent space to decode.</param>
    /// <returns>The reconstructed vector in the original data space.</returns>
    /// <remarks>
    /// <para>
    /// This method passes a latent vector through the decoder portion of the VAE to generate a
    /// reconstruction in the original data space. The decoder learns to map points in the latent
    /// space back to the format of the original input data.
    /// </para>
    /// <para><b>For Beginners:</b> This method recreates the original-style data from the compressed representation.
    /// 
    /// When you decode a latent vector:
    /// - The vector passes through the second half of the network (the decoder)
    /// - The decoder transforms the compact representation back to the original data format
    /// 
    /// For example, with a face image:
    /// - The latent vector might contain compressed information about features like hair color, face shape, etc.
    /// - The decoder uses this information to generate a complete face image
    /// 
    /// The amazing thing is that you can:
    /// - Decode latent vectors that didn't come from real inputs
    /// - Generate new, never-before-seen but realistic-looking data
    /// - Smoothly transition between different types of outputs by moving through the latent space
    /// </para>
    /// </remarks>
    public Vector<T> Decode(Vector<T> latentVector)
    {
        var current = latentVector;
        for (int i = Layers.Count / 2; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
    }

    /// <summary>
    /// Processes an input vector through the full VAE pipeline to produce a reconstruction.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The reconstructed output vector.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the full VAE pipeline: encoding the input to latent distribution parameters,
    /// sampling from the latent distribution using the reparameterization trick, and decoding the sample
    /// back to the original data space. During training, this process helps the model learn a useful latent
    /// representation while ensuring good reconstruction quality.
    /// </para>
    /// <para><b>For Beginners:</b> This method runs the complete process of compressing and then reconstructing your data.
    /// 
    /// The complete VAE process:
    /// 1. The encoder compresses your input into mean and variance values (the encode step)
    /// 2. The reparameterization trick adds some randomness to create a latent vector
    /// 3. The decoder reconstructs a new version of the input from this latent vector
    /// 
    /// What makes VAEs special is that:
    /// - The reconstruction won't be a perfect copy, but a similar, plausible version
    /// - The randomness forces the model to learn a meaningful latent space
    /// - This means the latent space can be used to generate new examples or smoothly transition between examples
    /// 
    /// For instance, with face images, the complete process takes an image, compresses it into
    /// a compact representation with some controlled randomness, and then generates a
    /// new face image that looks similar but might have slightly different details.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Vector<T> input)
    {
        var (mean, logVariance) = Encode(input);
        var latentVector = Reparameterize(mean, logVariance);

        return Decode(latentVector);
    }

    /// <summary>
    /// Updates the parameters of all layers in the VAE network.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for the network.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the parameters to each layer based on their parameter counts.
    /// It updates both the standard network layers and the specialized mean and log variance layers.
    /// This is typically used during training when applying gradient updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the internal values of the VAE during training.
    /// 
    /// Think of parameters as the "settings" of the VAE:
    /// - Each layer needs a certain number of parameters to function
    /// - During training, these parameters are constantly adjusted to improve performance
    /// - This method takes a big list of new parameter values and gives each layer its share
    /// 
    /// It makes sure to update:
    /// - All the regular layers in the network
    /// - The special mean and log variance layers that make the VAE work
    /// 
    /// It's like distributing updated parts to each section of a machine so it works better.
    /// Each layer gets exactly the number of parameters it needs.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the mean layer or log variance layer have not been properly initialized.
    /// </exception>
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

        // Update mean and log variance layers
        if (_meanLayer != null && _logVarianceLayer != null)
        {
            int meanParameterCount = _meanLayer.ParameterCount;
            _meanLayer.UpdateParameters(parameters.SubVector(startIndex, meanParameterCount));
            startIndex += meanParameterCount;

            int logVarianceParameterCount = _logVarianceLayer.ParameterCount;
            _logVarianceLayer.UpdateParameters(parameters.SubVector(startIndex, logVarianceParameterCount));
        }
        else
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer have not been properly initialized.");
        }
    }

    /// <summary>
    /// Saves the VAE network structure and parameters to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to save the network to.</param>
    /// <remarks>
    /// <para>
    /// This method saves the type information and parameters of each layer, including the specialized
    /// mean and log variance layers. This allows the network to be reconstructed later using the 
    /// Deserialize method. Serialization is useful for saving trained models to disk or transferring
    /// them between applications.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the VAE to a file so you can use it later.
    /// 
    /// When saving the VAE:
    /// - Information about each layer's type is saved
    /// - All the learned parameter values are saved
    /// - The entire structure of the network is preserved
    /// - The special mean and log variance layers are saved with their configuration
    /// 
    /// This is useful for:
    /// - Saving a trained model after spending time and resources on training
    /// - Sharing your model with others
    /// - Using your model in a different application
    /// 
    /// It's like taking a snapshot of the entire VAE that can be restored later.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when the writer is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when serialization encounters issues with layer information or when the mean layer 
    /// or log variance layer have not been properly initialized.
    /// </exception>
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

        // Serialize mean and log variance layers
        if (_meanLayer != null && _logVarianceLayer != null)
        {
            writer.WriteInt32Array(_meanLayer.GetInputShape());
            writer.Write(_meanLayer.Axis);
            _meanLayer.Serialize(writer);

            writer.WriteInt32Array(_logVarianceLayer.GetInputShape());
            writer.Write(_logVarianceLayer.Axis);
            _logVarianceLayer.Serialize(writer);
        }
        else
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer have not been properly initialized.");
        }
    }

    /// <summary>
    /// Loads a VAE network structure and parameters from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to load the network from.</param>
    /// <remarks>
    /// <para>
    /// This method reconstructs the network by reading the type information and parameters of each layer,
    /// including the specialized mean and log variance layers. It's used to load previously saved models
    /// for inference or continued training.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads a previously saved VAE from a file.
    /// 
    /// When loading the VAE:
    /// - The number and types of layers are read from the file
    /// - Each layer is created with the correct type
    /// - The parameter values are loaded into each layer
    /// - The special mean and log variance layers are reconstructed with their configuration
    /// 
    /// This allows you to:
    /// - Use a model that was trained earlier
    /// - Continue training a model from where you left off
    /// - Use models created by others
    /// 
    /// It's like reassembling the VAE from a blueprint and parts list that was saved earlier.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when the reader is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when deserialization encounters issues with layer information.
    /// </exception>
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

        // Deserialize mean and log variance layers
        int[] meanInputShape = reader.ReadInt32Array();
        int meanAxis = reader.ReadInt32();
        _meanLayer = new MeanLayer<T>(meanInputShape, meanAxis);
        _meanLayer.Deserialize(reader);

        int[] logVarianceInputShape = reader.ReadInt32Array();
        int logVarianceAxis = reader.ReadInt32();
        _logVarianceLayer = new LogVarianceLayer<T>(logVarianceInputShape, logVarianceAxis);
        _logVarianceLayer.Deserialize(reader);
    }
}