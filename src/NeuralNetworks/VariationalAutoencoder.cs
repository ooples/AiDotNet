using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// <example>
/// <code>
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 784 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new VariationalAutoencoder&lt;float&gt;())
///     .Build(trainX, trainY);
/// var reconstructed = result.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
[ModelDomain(ModelDomain.General)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelTask(ModelTask.Generation)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Auto-Encoding Variational Bayes", "https://arxiv.org/abs/1312.6114", Year = 2014, Authors = "Diederik P. Kingma, Max Welling")]
public partial class VariationalAutoencoder<T> : VectorModelLayoutBase<T>, IAuxiliaryLossLayer<T>
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
    /// The layer between encoder and decoder that splits the encoder's [..., 2 * LatentSize] output into the latent
    /// mean and log-variance and applies the reparameterization trick: a sample in training, the mean in inference.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The encoder describes each input as a range of likely values for every latent
    /// attribute (a mean and a spread). This layer picks a point in that range while training, so the decoder learns
    /// from nearby variations, and uses the most likely point (the mean) when predicting.</para>
    /// </remarks>
    private RepParameterizationLayer<T>? _latentLayer;

    /// <summary>Index of <see cref="_latentLayer"/> in <c>Layers</c>: encoder layers precede it, decoder layers follow.</summary>
    private int _latentIndex = -1;

    /// <summary>The KL term of the most recent training forward, tape-connected, until the trainer consumes it.</summary>
    [Scratch]
    private Tensor<T>? _pendingKlLoss;

    /// <summary>
    /// Gets or sets the gradient optimizer used for training the VAE.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer { get; set; }

    /// <summary>
    /// Stores the last computed mean vector from the encoder for auxiliary loss computation.
    /// </summary>
    [Scratch]
    private Vector<T>? _lastMean;

    /// <summary>
    /// Stores the last computed log variance vector from the encoder for auxiliary loss computation.
    /// </summary>
    [Scratch]
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
    /// <summary>
    /// Initializes a new instance with default settings.
    /// </summary>
    public VariationalAutoencoder()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 128),
            latentSize: 32)
    {
    }

    public VariationalAutoencoder(NeuralNetworkArchitecture<T> architecture,
        int latentSize,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        VariationalAutoencoderOptions? options = null) :
        base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), (options ??= new VariationalAutoencoderOptions()).MaxGradNorm)
    {
        _options = options;
        Options = _options;

        LatentSize = latentSize;

        // Initialize NumOps-based fields
        _lastKLDivergence = NumOps.Zero;
        // β-VAE weight: lower values prioritize reconstruction over KL regularization.
        // Default 0.01 ensures reconstruction loss decreases monotonically during training.
        AuxiliaryLossWeight = NumOps.FromDouble(0.01);

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
            ValidateCustomLayersWithOwnershipRollback(Layers);
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
    /// A VAE needs exactly one <see cref="RepParameterizationLayer{T}"/> with at least one layer before it (the
    /// encoder, whose output is [..., 2 * LatentSize]: the means then the log-variances) and one after it (the
    /// decoder).
    /// </para>
    /// <para><b>For Beginners:</b> Your custom layers must be an encoder, then the reparameterization layer, then a
    /// decoder. The last encoder layer must output twice as many values as the latent size: the first half are the
    /// means, the second half the log-variances.</para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when there is not exactly one reparameterization layer, or it has no encoder or no decoder around it.
    /// </exception>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        int count = 0, index = -1;
        for (int i = 0; i < layers.Count; i++)
        {
            if (layers[i] is RepParameterizationLayer<T>)
            {
                count++;
                index = i;
            }
        }

        if (count != 1)
        {
            throw new InvalidOperationException(
                "Custom VAE layers must include exactly one RepParameterizationLayer between the encoder (whose last " +
                $"layer outputs 2 * LatentSize values: means then log-variances) and the decoder; found {count}.");
        }

        if (index == 0 || index == layers.Count - 1)
        {
            throw new InvalidOperationException(
                "Custom VAE layers need at least one encoder layer before the RepParameterizationLayer and one decoder layer after it.");
        }
    }

    /// <summary>
    /// Sets up references to the specific layers needed for the VAE's functionality.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method finds the reparameterization layer, which separates the encoder from the decoder.
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
    /// Thrown when no reparameterization layer is present.
    /// </exception>
    private void SetSpecificLayers()
    {
        _latentIndex = Layers.FindIndex(l => l is RepParameterizationLayer<T>);
        if (_latentIndex < 0)
        {
            throw new InvalidOperationException("A RepParameterizationLayer must be present in the network.");
        }
        _latentLayer = (RepParameterizationLayer<T>)Layers[_latentIndex];
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
    /// Thrown when the layers have not been initialized.
    /// </exception>
    public (Vector<T> Mean, Vector<T> LogVariance) Encode(Vector<T> input)
    {
        if (_latentLayer is null)
        {
            throw new InvalidOperationException("The VAE's layers have not been initialized.");
        }

        // The encoder's output is [2 * LatentSize]: the means, then the log-variances.
        var current = Tensor<T>.FromVector(input);
        for (int i = 0; i < _latentIndex; i++)
            current = Layers[i].Forward(current);
        var parameters = current.ToVector();
        int latent = parameters.Length / 2;
        var mean = new Vector<T>(latent);
        var logVariance = new Vector<T>(latent);
        for (int i = 0; i < latent; i++)
        {
            mean[i] = parameters[i];
            logVariance[i] = parameters[latent + i];
        }
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
        if (_latentLayer is null)
        {
            throw new InvalidOperationException("The VAE's layers have not been initialized.");
        }

        var current = Tensor<T>.FromVector(latentVector);
        for (int i = _latentIndex + 1; i < Layers.Count; i++)
            current = Layers[i].Forward(current);
        return current.ToVector();
    }

    // UpdateParameters restated a fold the base now derives from generated component registration.
    // Removed under AIDN082.
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Encoder, latent mean (the reparameterization layer samples only in training), decoder.
        var current = input;
        foreach (var layer in Layers)
            current = layer.Forward(current);
        RememberLatent();
        return current.Length == input.Length ? current.Reshape(input._shape) : current;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The training forward samples the latent through the reparameterization layer and records the KL divergence
    /// from the standard normal prior (summed over latent dimensions, averaged over the batch, weighted by
    /// <see cref="AuxiliaryLossWeight"/>) for <see cref="ConsumeAuxiliaryTapeLoss"/>, so the trainer optimizes
    /// reconstruction + beta * KL.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var output = base.ForwardForTraining(input);
        _pendingKlLoss = null;
        if (UseAuxiliaryLoss && _latentLayer?.LastMean is { } mean && _latentLayer.LastLogVariance is { } logVariance)
        {
            // KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar), per sample.
            var perElement = Engine.TensorSubtract(
                Engine.TensorAdd(Engine.TensorExp(logVariance), Engine.TensorMultiply(mean, mean)),
                Engine.TensorAddScalar(logVariance, NumOps.One));
            int batch = mean._shape.Length > 1 ? mean._shape[0] : 1;
            var allAxes = Enumerable.Range(0, perElement._shape.Length).ToArray();
            var kl = Engine.TensorMultiplyScalar(
                Engine.ReduceSum(perElement, allAxes, keepDims: false),
                NumOps.Multiply(AuxiliaryLossWeight, NumOps.FromDouble(0.5 / batch)));
            _pendingKlLoss = kl;
        }
        RememberLatent();
        return output;
    }

    /// <inheritdoc/>
    protected override Tensor<T>? ConsumeAuxiliaryTapeLoss()
    {
        var term = _pendingKlLoss;
        _pendingKlLoss = null;
        return term;
    }

    // Keeps the latest latent statistics (first sample) for ComputeAuxiliaryLoss and the diagnostics.
    private void RememberLatent()
    {
        if (_latentLayer?.LastMean is not { } mean || _latentLayer.LastLogVariance is not { } logVariance) return;
        var meanValues = mean.ToArray();
        var logVarianceValues = logVariance.ToArray();
        int latent = mean._shape[mean._shape.Length - 1];
        _lastMean = new Vector<T>(meanValues.Take(latent).ToArray());
        _lastLogVariance = new Vector<T>(logVarianceValues.Take(latent).ToArray());
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
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
            meanNormSquared = NumOps.Add(meanNormSquared, Engine.DotProduct(_lastMean, _lastMean));
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
            AdditionalInfo = new Dictionary<string, object>
            {
                { "InputDimension", Layers[0].GetInputShape()[0] },
                { "LatentSize", LatentSize },
                { "LayerCount", Layers.Count },
                { "EncoderLayers", Layers.Take(_latentIndex).Select(l => l.GetType().Name).ToArray() },
                { "DecoderLayers", Layers.Skip(_latentIndex + 1).Select(l => l.GetType().Name).ToArray() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }




}
