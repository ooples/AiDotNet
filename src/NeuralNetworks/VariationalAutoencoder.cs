using AiDotNet.Extensions;
using AiDotNet.NeuralNetworks.Options;

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
public class VariationalAutoencoder<T> : NeuralNetworkBase<T>, IAuxiliaryLossLayer<T>
{
    private readonly VariationalAutoencoderOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// Gets or sets the gradient optimizer used for training the VAE.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer { get; set; }

    /// <summary>
    /// Stores the last computed mean vector from the encoder for auxiliary loss computation.
    /// </summary>
    private Vector<T>? _lastMean;

    /// <summary>
    /// Stores the last computed log variance vector from the encoder for auxiliary loss computation.
    /// </summary>
    private Vector<T>? _lastLogVariance;

    /// <summary>
    /// Stores the last computed KL divergence value for diagnostics.
    /// </summary>
    private T _lastKLDivergence;

    /// <summary>
    /// Gets or sets whether to use auxiliary loss (KL divergence) during training.
    /// For VAEs, this should always be true as KL divergence is required for proper functioning.
    /// </summary>
    public bool UseAuxiliaryLoss { get; set; } = true;

    /// <summary>
    /// Gets or sets the weight (beta parameter) for the KL divergence auxiliary loss.
    /// Default is 1.0. Can be adjusted for beta-VAE variants.
    /// </summary>
    public T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="VariationalAutoencoder{T}"/> class with the 
    /// specified architecture, latent space size, and optional optimizer and loss function.
    /// </summary>
    /// <param name="architecture">The neural network architecture that defines the overall structure.</param>
    /// <param name="latentSize">The size of the latent space dimension.</param>
    /// <param name="optimizer">The gradient optimizer to use for training (optional).</param>
    /// <param name="lossFunction">The loss function to use for reconstruction loss (optional).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new VAE with the provided architecture, latent size, and optional
    /// optimizer and loss function. If no optimizer or loss function is provided, default ones will be used.
    /// </para>
    /// <para><b>For Beginners:</b> This is where you set up your VAE with basic settings and optional advanced configurations.
    /// 
    /// When creating a VAE, you need to specify:
    /// - The overall architecture (like how many layers, their sizes, etc.)
    /// - The latent size (how many attributes or features to use in the compressed representation)
    /// 
    /// You can also optionally specify:
    /// - An optimizer (a method for adjusting the network's internal values during training)
    /// - A loss function (a way to measure how well the VAE is performing)
    /// 
    /// If you don't specify an optimizer or loss function, the VAE will use default options that work well in most cases.
    /// </para>
    /// </remarks>
    public VariationalAutoencoder(
        NeuralNetworkArchitecture<T> architecture,
        int latentSize,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0,
        VariationalAutoencoderOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new VariationalAutoencoderOptions();
        Options = _options;

        LatentSize = latentSize;

        // Initialize NumOps-based fields
        _lastKLDivergence = NumOps.Zero;
        AuxiliaryLossWeight = NumOps.One;

        // Initialize layers first so the model is fully constructed
        InitializeLayers();

        // Now bind optimizer to this fully-initialized model instance
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
        // Process encoder layers, skipping MeanLayer and LogVarianceLayer
        // (they are processed separately after the encoder loop)
        for (int i = 0; i < Layers.Count / 2; i++)
        {
            if (Layers[i] is MeanLayer<T> || Layers[i] is LogVarianceLayer<T>)
            {
                continue; // Skip special layers - they're processed separately
            }
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        // Use MeanLayer/LogVarianceLayer to preserve gradient flow
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
            // Convert z to T type
            T zT = NumOps.FromDouble(Random.NextGaussian());

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
        // Process decoder layers, skipping MeanLayer and LogVarianceLayer
        // (they are encoder-specific and shouldn't be in the decoder path)
        for (int i = Layers.Count / 2; i < Layers.Count; i++)
        {
            if (Layers[i] is MeanLayer<T> || Layers[i] is LogVarianceLayer<T>)
            {
                continue; // Skip special layers - they're not part of the decoder
            }
            current = Layers[i].Forward(Tensor<T>.FromVector(current)).ToVector();
        }

        return current;
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
    /// Makes a prediction using the Variational Autoencoder by encoding the input, sampling from the latent space, and decoding.
    /// </summary>
    /// <param name="input">The input tensor to make a prediction for.</param>
    /// <returns>The reconstructed output tensor after passing through the VAE.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a full forward pass through the VAE:
    /// 1. Encodes the input to get mean and log variance of the latent distribution.
    /// 2. Samples a point from this distribution using the reparameterization trick.
    /// 3. Decodes the sampled point to produce a reconstruction of the input.
    /// </para>
    /// <para><b>For Beginners:</b> This method takes your input data, compresses it, and then tries to recreate it.
    /// 
    /// The process:
    /// 1. The input is compressed into a small representation (encoding)
    /// 2. A random point is chosen from this compressed space (sampling)
    /// 3. This point is then expanded back into the original data format (decoding)
    /// 
    /// The output is the VAE's attempt to recreate the input. It won't be exactly the same,
    /// but it should capture the important features of the original input.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Flatten the input tensor to a vector
        var inputVector = input.ToVector();

        // Encode the input
        var (mean, logVariance) = Encode(inputVector);
        _lastMean = mean;
        _lastLogVariance = logVariance;

        // Sample from the latent space
        var latentSample = Reparameterize(mean, logVariance);

        // Decode the sample
        var reconstructed = Decode(latentSample);

        // Reshape the output to match the input shape
        return new Tensor<T>(input.Shape, reconstructed);
    }

    /// <summary>
    /// Trains the Variational Autoencoder using the provided input data.
    /// </summary>
    /// <param name="input">The input tensor used for training.</param>
    /// <param name="expectedOutput">The expected output tensor (typically the same as the input for VAEs).</param>
    /// <remarks>
    /// <para>
    /// This method implements the training process for the VAE:
    /// 1. Performs a forward pass to get the reconstructed output.
    /// 2. Calculates the reconstruction loss and the KL divergence.
    /// 3. Computes the total loss (reconstruction loss + KL divergence).
    /// 4. Backpropagates the error and updates the network parameters using the specified optimizer.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the VAE to compress and reconstruct data effectively.
    /// 
    /// The training process:
    /// 1. The VAE tries to reconstruct the input
    /// 2. It measures how well it did (reconstruction error) using the specified loss function
    /// 3. It also measures how well it's using the latent space (KL divergence)
    /// 4. It combines these measurements into a total score
    /// 5. It then adjusts its internal settings to do better next time, using the specified optimizer
    /// 
    /// This process is repeated many times with different inputs to improve the VAE's performance.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        IsTrainingMode = true;

        // Flatten the input tensor to a vector
        var inputVector = input.ToVector();

        // Forward pass
        var (mean, logVariance) = Encode(inputVector);
        _lastMean = mean;
        _lastLogVariance = logVariance;

        var latentSample = Reparameterize(mean, logVariance);
        var reconstructed = Decode(latentSample);

        // Calculate reconstruction loss
        var reconstructionLoss = LossFunction.CalculateLoss(inputVector, reconstructed);

        // Calculate auxiliary loss (KL divergence) using the interface
        T auxiliaryLoss = NumOps.Zero;
        if (UseAuxiliaryLoss)
        {
            var klDivergence = ComputeAuxiliaryLoss();
            auxiliaryLoss = NumOps.Multiply(klDivergence, AuxiliaryLossWeight);
        }

        // Calculate total loss
        var totalLoss = NumOps.Add(reconstructionLoss, auxiliaryLoss);
        LastLoss = totalLoss;

        // Backpropagation
        var gradient = CalculateGradient(totalLoss);

        // Update parameters using the optimizer
        _optimizer.UpdateParameters(Layers);

        IsTrainingMode = false;
    }

    /// <summary>
    /// Calculates the gradient for the entire Variational Autoencoder network.
    /// </summary>
    /// <param name="totalLoss">The total loss value used to initiate the gradient calculation.</param>
    /// <returns>A list of tensors representing the gradients for each layer.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a full backward pass through the VAE network:
    /// 1. Calculates gradients for the decoder layers.
    /// 2. Computes gradients for the latent space (mean and log variance).
    /// 3. Calculates gradients for the encoder layers.
    /// 4. Applies gradient clipping to prevent exploding gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out how to adjust each part of the VAE to improve its performance.
    /// 
    /// It works backwards through the network:
    /// 1. Starting from the output, it calculates how to change the decoder
    /// 2. Then it figures out how to adjust the middle (latent) part
    /// 3. Finally, it calculates changes for the encoder
    /// 4. It also makes sure the changes aren't too big, which could cause problems
    /// 
    /// This process helps the VAE learn from its mistakes and get better over time.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> CalculateGradient(T totalLoss)
    {
        var gradients = new List<Tensor<T>>();

        // Backward pass through the decoder layers
        Tensor<T> currentGradient = new Tensor<T>([1], new Vector<T>(Enumerable.Repeat(totalLoss, 1)));
        for (int i = Layers.Count - 1; i >= Layers.Count / 2; i--)
        {
            var layerGradient = Layers[i].Backward(currentGradient);
            gradients.Insert(0, layerGradient);
            currentGradient = layerGradient;
        }

        // Backward pass through the latent space
        var (meanGradient, logVarianceGradient) = CalculateLatentGradients(currentGradient);

        // Backward pass through the encoder layers
        for (int i = (Layers.Count / 2) - 1; i >= 0; i--)
        {
            if (Layers[i] is MeanLayer<T>)
            {
                currentGradient = Layers[i].Backward(meanGradient);
            }
            else if (Layers[i] is LogVarianceLayer<T>)
            {
                currentGradient = Layers[i].Backward(logVarianceGradient);
            }
            else
            {
                currentGradient = Layers[i].Backward(currentGradient);
            }
            gradients.Insert(0, currentGradient);
        }

        // Apply gradient clipping to prevent exploding gradients
        ClipGradients(gradients);

        return gradients;
    }

    /// <summary>
    /// Calculates the gradients for the latent space (mean and log variance) of the VAE.
    /// </summary>
    /// <param name="upstreamGradient">The gradient flowing back from the decoder.</param>
    /// <returns>A tuple containing the gradients for the mean and log variance.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the gradients for the latent space parameters:
    /// 1. It calculates the gradient for the mean, considering both the upstream gradient and the KL divergence.
    /// 2. It calculates the gradient for the log variance, incorporating the reparameterization trick.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out how to adjust the "compression" part of the VAE.
    /// 
    /// The VAE compresses data into two parts: a "best guess" (mean) and an "uncertainty range" (variance):
    /// - This method calculates how to change both parts to make the VAE work better
    /// - It considers both how well the VAE is reconstructing data and how well it's compressing information
    /// - The calculations are complex because they need to balance good reconstruction with efficient compression
    /// 
    /// This process is key to making the VAE learn a useful compressed representation of the data.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the mean layer or log variance layer have not been properly initialized.
    /// </exception>
    private (Tensor<T> MeanGradient, Tensor<T> LogVarianceGradient) CalculateLatentGradients(Tensor<T> upstreamGradient)
    {
        if (_meanLayer == null || _logVarianceLayer == null)
        {
            throw new InvalidOperationException("MeanLayer and LogVarianceLayer have not been properly initialized.");
        }

        var meanGradient = new Tensor<T>(upstreamGradient.Shape);
        var logVarianceGradient = new Tensor<T>(upstreamGradient.Shape);

        // Get the outputs of the mean and log variance layers
        var meanOutput = _meanLayer.Forward(upstreamGradient);
        var logVarianceOutput = _logVarianceLayer.Forward(upstreamGradient);

        for (int i = 0; i < upstreamGradient.Length; i++)
        {
            // Gradient for mean: upstream gradient + KL divergence gradient
            meanGradient[i] = NumOps.Add(upstreamGradient[i], meanOutput[i]);

            // Gradient for log variance: 0.5 * (exp(log_var) - 1) + upstream gradient * 0.5 * exp(log_var) * epsilon
            var expLogVar = NumOps.Exp(logVarianceOutput[i]);
            var epsilon = NumOps.Divide(NumOps.Subtract(meanOutput[i], upstreamGradient[i]), expLogVar);
            logVarianceGradient[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Subtract(expLogVar, NumOps.One)),
                NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(0.5), upstreamGradient[i]), NumOps.Multiply(expLogVar, epsilon))
            );
        }

        return (meanGradient, logVarianceGradient);
    }

    /// <summary>
    /// Computes the auxiliary loss for the VAE, which is the KL divergence between the learned
    /// latent distribution and a standard normal distribution.
    /// </summary>
    /// <returns>The KL divergence loss value.</returns>
    /// <remarks>
    /// <para>
    /// The KL divergence is computed as: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    /// This regularizes the latent space to follow a standard normal distribution, which is
    /// essential for VAEs to generate new samples and ensure a smooth, continuous latent space.
    /// </para>
    /// <para><b>For Beginners:</b> This computes how different the VAE's compression is from an ideal "standard" compression.
    ///
    /// The KL divergence measures:
    /// - How much the learned latent space differs from a standard normal distribution
    /// - This difference acts as a penalty to encourage the VAE to organize its latent space properly
    ///
    /// Without this loss:
    /// - The VAE might create "holes" in the latent space where nothing meaningful exists
    /// - Generated samples might not look realistic
    /// - The latent space might not be smooth or continuous
    ///
    /// The KL divergence ensures the latent space has good properties for generation and interpolation.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastMean == null || _lastLogVariance == null)
        {
            return NumOps.Zero;
        }

        _lastKLDivergence = CalculateKLDivergence(_lastMean, _lastLogVariance);
        return _lastKLDivergence;
    }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about KL divergence and latent space statistics.</returns>
    /// <remarks>
    /// <para>
    /// This method provides insights into the VAE's latent space behavior, including:
    /// - The current KL divergence value
    /// - The beta weight parameter
    /// - Statistics about the mean and variance of the latent distribution
    /// </para>
    /// <para><b>For Beginners:</b> This gives you information to help understand and debug your VAE.
    ///
    /// The diagnostics include:
    /// - KL Divergence: How much the latent space differs from ideal (lower is more "standard")
    /// - Beta Weight: How much the KL divergence is weighted in training
    /// - Latent Mean Norm: How far the average latent values are from zero
    /// - Latent Std Mean: The average uncertainty in the latent space
    ///
    /// These values help you:
    /// - Understand if training is progressing well
    /// - Detect problems like "posterior collapse" (when the VAE ignores the latent space)
    /// - Tune hyperparameters like the beta weight
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>
        {
            { "KLDivergence", _lastKLDivergence?.ToString() ?? "0" },
            { "Beta", AuxiliaryLossWeight?.ToString() ?? "1.0" },
            { "UseAuxiliaryLoss", UseAuxiliaryLoss.ToString() }
        };

        if (_lastMean != null)
        {
            // Calculate L2 norm of mean vector
            T meanNormSquared = NumOps.Zero;
            for (int i = 0; i < _lastMean.Length; i++)
            {
                meanNormSquared = NumOps.Add(meanNormSquared, NumOps.Multiply(_lastMean[i], _lastMean[i]));
            }
            var meanNorm = NumOps.Sqrt(meanNormSquared);
            diagnostics["LatentMeanNorm"] = meanNorm?.ToString() ?? "0";
        }

        if (_lastLogVariance != null)
        {
            // Calculate mean of standard deviations
            T stdSum = NumOps.Zero;
            for (int i = 0; i < _lastLogVariance.Length; i++)
            {
                var halfLogVar = NumOps.Multiply(NumOps.FromDouble(0.5), _lastLogVariance[i]);
                stdSum = NumOps.Add(stdSum, NumOps.Exp(halfLogVar));
            }
            var stdMean = NumOps.Divide(stdSum, NumOps.FromDouble(_lastLogVariance.Length));
            diagnostics["LatentStdMean"] = stdMean?.ToString() ?? "0";
        }

        return diagnostics;
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
    /// Calculates the KL divergence between the learned distribution and a standard normal distribution.
    /// </summary>
    private T CalculateKLDivergence(Vector<T> mean, Vector<T> logVariance)
    {
        // === Vectorized KL divergence using IEngine (Phase B: US-GPU-015) ===
        // KL = 0.5 * sum(exp(logVar) + mean^2 - 1 - logVar)
        // Breaking into: sum(exp(logVar)) + sum(mean^2) - n - sum(logVar)
        var expLogVar = Engine.Exp(logVariance);
        var meanSquared = Engine.Multiply(mean, mean);

        T sumExpLogVar = Engine.Sum(expLogVar);
        T sumMeanSquared = Engine.Sum(meanSquared);
        T sumLogVar = Engine.Sum(logVariance);
        T n = NumOps.FromDouble(mean.Length);

        // sum(exp(logVar) + mean^2 - 1 - logVar)
        T sum = NumOps.Subtract(
            NumOps.Add(sumExpLogVar, sumMeanSquared),
            NumOps.Add(n, sumLogVar));

        return NumOps.Multiply(NumOps.FromDouble(0.5), sum);
    }

    /// <summary>
    /// Gets metadata about the Variational Autoencoder model.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the model.</returns>
    /// <remarks>
    /// This method returns metadata about the VAE, including the model type, input/output dimensions,
    /// latent size, and layer configuration. This information is useful for model management, serialization,
    /// and transfer learning.
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.VariationalAutoencoder,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", Layers[0].GetInputShape()[0] },
                { "LatentSize", LatentSize },
                { "LayerCount", Layers.Count },
                { "EncoderLayers", Layers.Take(Layers.Count / 2).Select(l => l.GetType().Name).ToArray() },
                { "DecoderLayers", Layers.Skip(Layers.Count / 2).Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes network-specific data for the Variational Autoencoder.
    /// </summary>
    /// <param name="writer">The BinaryWriter to write the data to.</param>
    /// <remarks>
    /// This method writes the specific configuration and state of the VAE to a binary stream.
    /// It includes network-specific parameters that are essential for later reconstruction of the network.
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(LatentSize);
        SerializationHelper<T>.SerializeInterface(writer, _optimizer);
    }

    /// <summary>
    /// Deserializes network-specific data for the Variational Autoencoder.
    /// </summary>
    /// <param name="reader">The BinaryReader to read the data from.</param>
    /// <remarks>
    /// This method reads the specific configuration and state of the VAE from a binary stream.
    /// It reconstructs the network-specific parameters to match the state of the network when it was serialized.
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        LatentSize = reader.ReadInt32();
        _optimizer = DeserializationHelper.DeserializeInterface<IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>>(reader) ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    /// <summary>
    /// Creates a new instance of the Variational Autoencoder with the same architecture and configuration.
    /// </summary>
    /// <returns>A new instance of the Variational Autoencoder with the same configuration as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new Variational Autoencoder with the same architecture, latent size,
    /// optimizer, loss function, and gradient clipping settings as the current instance. The new
    /// instance has freshly initialized parameters, making it useful for creating separate instances
    /// with identical configurations or for resetting the network while preserving its structure.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a brand new VAE with the same setup as the current one.
    /// 
    /// Think of it like creating a copy of your VAE's blueprint:
    /// - It has the same overall structure
    /// - It uses the same latent size (compression level)
    /// - It has the same optimizer (learning method)
    /// - It uses the same loss function (way of measuring performance)
    /// - But it starts with fresh parameters (internal values)
    /// 
    /// This is useful when you want to:
    /// - Start over with a fresh network but keep the same design
    /// - Create multiple networks with identical settings for comparison
    /// - Reset a network to its initial state
    /// 
    /// The new VAE will need to be trained from scratch, as it doesn't inherit any
    /// of the learned knowledge from the original network.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VariationalAutoencoder<T>(
            Architecture,
            LatentSize,
            _optimizer,
            LossFunction,
            Convert.ToDouble(MaxGradNorm));
    }
}
