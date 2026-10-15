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
    /// The number of leading layers that form the encoder. The last of them outputs 2 * <see cref="LatentSize"/>
    /// features: the mean of the approximate posterior q(z|x) in the first half and its log variance in the second
    /// (Kingma &amp; Welling 2014, appendix C.2). The remaining layers form the decoder p(x|z).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> the encoder compresses the input into two lists of numbers per example: where each
    /// latent feature most likely is (the mean) and how unsure the encoder is about it (the log variance). The decoder
    /// turns a latent code back into an input-shaped reconstruction.</para>
    /// </remarks>
    private int _encoderLayerCount;

    /// <summary>The weighted KL term of the most recent training forward, added to the objective once.</summary>
    [Scratch]
    private Tensor<T>? _pendingKlLoss;

    /// <summary>The posterior mean of the most recent forward, kept for diagnostics.</summary>
    [Scratch]
    private Tensor<T>? _lastMeanTensor;

    /// <summary>The posterior log variance of the most recent forward, kept for diagnostics.</summary>
    [Scratch]
    private Tensor<T>? _lastLogVarianceTensor;

    /// <summary>
    /// Gets or sets the gradient optimizer used for training the VAE.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer { get; set; }



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

        _encoderLayerCount = LocateLatentBoundary(Layers, LatentSize);
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
        LocateLatentBoundary(layers, LatentSize);
    }

    /// <summary>
    /// The number of encoder layers: everything up to and including the first dense layer that outputs
    /// 2 * <paramref name="latentSize"/> features (the posterior mean and log variance). At least one decoder layer
    /// must follow it.
    /// </summary>
    private static int LocateLatentBoundary(IReadOnlyList<ILayer<T>> layers, int latentSize)
    {
        for (int i = 0; i < layers.Count; i++)
        {
            var shape = layers[i].GetOutputShape();
            if (layers[i] is DenseLayer<T> && shape.Length == 1 && shape[0] == 2 * latentSize)
            {
                if (i == layers.Count - 1)
                    throw new InvalidOperationException(
                        "A VAE needs decoder layers after the encoder's posterior layer.");
                return i + 1;
            }
        }

        throw new InvalidOperationException(
            $"A VAE's encoder must end in a dense layer with {2 * latentSize} outputs (2 * latentSize): the mean of "
            + "q(z|x) in the first half and its log variance in the second.");
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
        var (mean, logVariance) = EncodePosterior(Tensor<T>.FromVector(input));
        return (mean.ToVector(), logVariance.ToVector());
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

        return SampleLatent(Tensor<T>.FromVector(mean), Tensor<T>.FromVector(logVariance)).ToVector();
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
        => RunLayers(_encoderLayerCount, Layers.Count, Tensor<T>.FromVector(latentVector)).ToVector();

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
        // Inference decodes the posterior mean, without sampling (Kingma & Welling 2014).
        var (mean, _) = EncodePosterior(input);
        var reconstructed = RunLayers(_encoderLayerCount, Layers.Count, mean);
        return reconstructed.Length == input.Length ? Engine.Reshape(reconstructed, input._shape) : reconstructed;
    }

    /// <summary>
    /// The training forward (Kingma &amp; Welling 2014, section 2.4): encode to the posterior q(z|x), draw
    /// z = mu + exp(logvar / 2) * eps with eps ~ N(0, I), and decode. The KL divergence from the prior N(0, I) is
    /// held for <see cref="ConsumeAuxiliaryTapeLoss"/>, which adds it to the reconstruction loss.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var (mean, logVariance) = EncodePosterior(input);
        var latent = SampleLatent(mean, logVariance);
        var reconstructed = RunLayers(_encoderLayerCount, Layers.Count, latent);

        _pendingKlLoss = UseAuxiliaryLoss
            ? Engine.TensorMultiplyScalar(KlDivergence(mean, logVariance), AuxiliaryLossWeight)
            : null;
        return reconstructed.Length == input.Length ? Engine.Reshape(reconstructed, input._shape) : reconstructed;
    }

    /// <inheritdoc />
    protected override Tensor<T>? ConsumeAuxiliaryTapeLoss()
    {
        var kl = _pendingKlLoss;
        _pendingKlLoss = null;
        return kl;
    }

    // The encoder's output split along its last axis: the mean in the first LatentSize features, the log variance
    // in the rest.
    private (Tensor<T> Mean, Tensor<T> LogVariance) EncodePosterior(Tensor<T> input)
    {
        var posterior = RunLayers(0, _encoderLayerCount, input);
        int axis = posterior.Rank - 1;
        var mean = Engine.TensorNarrow(posterior, axis, 0, LatentSize);
        var logVariance = Engine.TensorNarrow(posterior, axis, LatentSize, LatentSize);
        _lastMeanTensor = mean;
        _lastLogVarianceTensor = logVariance;
        return (mean, logVariance);
    }

    // z = mu + exp(logvar / 2) * eps. eps is drawn through CompiledStepRandom, so a fused compiled step redraws it
    // on every replay instead of training on its first draw.
    private Tensor<T> SampleLatent(Tensor<T> mean, Tensor<T> logVariance)
    {
        var epsilon = AiDotNet.Training.CompiledStepRandom<T>.Draw(mean._shape, FillStandardNormal);
        var std = Engine.TensorExp(Engine.TensorMultiplyScalar(logVariance, NumOps.FromDouble(0.5)));
        return Engine.TensorAdd(mean, Engine.TensorMultiply(std, epsilon));
    }

    private void FillStandardNormal(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble(Random.NextGaussian());
    }

    // KL(q(z|x) || N(0, I)) = -1/2 * sum(1 + logvar - mu^2 - exp(logvar)) per example, averaged over the batch.
    private Tensor<T> KlDivergence(Tensor<T> mean, Tensor<T> logVariance)
    {
        var terms = Engine.TensorSubtract(
            Engine.TensorAdd(Engine.TensorMultiply(mean, mean), Engine.TensorExp(logVariance)),
            Engine.TensorAddScalar(logVariance, NumOps.One));
        int examples = Math.Max(1, mean.Length / LatentSize);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(terms, null), NumOps.FromDouble(0.5 / examples));
    }

    private Tensor<T> RunLayers(int start, int end, Tensor<T> input)
    {
        var current = input;
        for (int i = start; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
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
        if (!UseAuxiliaryLoss || _lastMeanTensor == null || _lastLogVarianceTensor == null)
        {
            return NumOps.Zero;
        }

        _lastKLDivergence = CalculateKLDivergence(_lastMeanTensor.ToVector(), _lastLogVarianceTensor.ToVector());
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

        if (_lastMeanTensor?.ToVector() is { } lastMean)
        {
            // Calculate L2 norm of mean vector
            T meanNormSquared = NumOps.Zero;
            meanNormSquared = NumOps.Add(meanNormSquared, Engine.DotProduct(lastMean, lastMean));
            var meanNorm = NumOps.Sqrt(meanNormSquared);
            diagnostics["LatentMeanNorm"] = meanNorm?.ToString() ?? "0";
        }

        if (_lastLogVarianceTensor?.ToVector() is { } lastLogVariance)
        {
            // Calculate mean of standard deviations
            T stdSum = NumOps.Zero;
            for (int i = 0; i < lastLogVariance.Length; i++)
            {
                var halfLogVar = NumOps.Multiply(NumOps.FromDouble(0.5), lastLogVariance[i]);
                stdSum = NumOps.Add(stdSum, NumOps.Exp(halfLogVar));
            }
            var stdMean = NumOps.Divide(stdSum, NumOps.FromDouble(lastLogVariance.Length));
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
                { "EncoderLayers", Layers.Take(_encoderLayerCount).Select(l => l.GetType().Name).ToArray() },
                { "DecoderLayers", Layers.Skip(_encoderLayerCount).Select(l => l.GetType().Name).ToArray() }
            },
            ModelDataProvider = () => SerializeForMetadata()
        };
    }




}
