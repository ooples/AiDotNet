using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a StyleGAN (Style-Based Generator Architecture for GANs) that generates
/// high-quality images with fine-grained control over image style at different levels.
/// </summary>
/// <remarks>
/// <para>
/// StyleGAN introduces several key innovations:
/// - Style-based generator with mapping network and synthesis network
/// - Adaptive Instance Normalization (AdaIN) for style injection
/// - Stochastic variation through noise injection
/// - Style mixing for disentangled control
/// - Progressive growing for high-resolution images
/// - State-of-the-art image quality
/// </para>
/// <para><b>For Beginners:</b> StyleGAN generates incredibly realistic images with fine control.
///
/// Key innovations:
/// - **Mapping Network**: Transforms random noise into style codes
/// - **Style Injection**: Injects style at each layer via AdaIN
/// - **Noise Injection**: Adds stochastic variation (hair, pores, etc.)
/// - **Style Mixing**: Combines styles from different sources
/// - **Progressive Growing**: Starts small, gradually adds detail
///
/// Architecture:
/// 1. Mapping Network (Z → W): Transforms latent code to intermediate space
/// 2. Synthesis Network: Generates image with style injection at each layer
/// 3. Each layer: Upsample → Conv → AdaIN → Noise → Conv → AdaIN → Noise
///
/// Why it's better:
/// - Exceptional image quality
/// - Disentangled style control (separate coarse/fine features)
/// - Style mixing (combine different sources)
/// - Perceptual path length is shorter
///
/// Applications:
/// - High-quality face generation
/// - Style transfer and manipulation
/// - Image editing and synthesis
/// - Creative AI applications
///
/// Reference: Karras et al., "A Style-Based Generator Architecture for
/// Generative Adversarial Networks" (2019)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class StyleGAN<T> : NeuralNetworkBase<T>
{
    // MappingNetwork optimizer state
    private Vector<T> _mappingMomentum;
    private Vector<T> _mappingSecondMoment;

    // SynthesisNetwork optimizer state
    private Vector<T> _synthesisMomentum;
    private Vector<T> _synthesisSecondMoment;

    // Discriminator optimizer state
    private Vector<T> _discMomentum;
    private Vector<T> _discSecondMoment;

    private readonly double _initialLearningRate;

    /// <summary>
    /// The size of the latent code Z.
    /// </summary>
    private int _latentSize;

    /// <summary>
    /// The size of the intermediate latent code W.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mapping network transforms Z to W, which typically has the same dimensionality.
    /// The W space is more disentangled than Z space.
    /// </para>
    /// <para><b>For Beginners:</b> W is a "better organized" version of random noise.
    ///
    /// - Z: Random input (entangled features)
    /// - W: Organized style codes (disentangled features)
    /// - W makes it easier to control specific aspects of the image
    /// </para>
    /// </remarks>
    private int _intermediateLatentSize;

    /// <summary>
    /// Gets the mapping network that transforms Z to W.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The mapping network is typically an 8-layer MLP that learns to map the input
    /// latent space Z to an intermediate latent space W with better disentanglement properties.
    /// </para>
    /// <para><b>For Beginners:</b> The "style organizer" network.
    ///
    /// Takes: Random noise (Z)
    /// Returns: Organized style codes (W)
    /// Why: W space has better separated features
    /// Result: Easier to control individual aspects
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> MappingNetwork { get; private set; }

    /// <summary>
    /// Gets the synthesis network that generates images from styles.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The synthesis network applies styles at multiple resolutions through AdaIN.
    /// It starts from a learned constant and progressively generates higher resolutions.
    /// </para>
    /// <para><b>For Beginners:</b> The "image painter" network.
    ///
    /// Process:
    /// 1. Starts from a learned constant (4x4)
    /// 2. Applies style via AdaIN at each layer
    /// 3. Adds random noise for details
    /// 4. Upsamples to next resolution
    /// 5. Repeats until final resolution
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> SynthesisNetwork { get; private set; }

    /// <summary>
    /// Gets the discriminator network.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    /// <summary>
    /// Enables style mixing during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Style mixing uses two random latent codes and switches between them
    /// at a random layer. This encourages the network to learn localized styles.
    /// </para>
    /// <para><b>For Beginners:</b> Style mixing combines different style sources.
    ///
    /// Example:
    /// - Use Style A for coarse features (face shape, pose)
    /// - Use Style B for fine features (hair texture, skin details)
    /// - Result: Face shape from A, details from B
    ///
    /// Benefits:
    /// - Prevents features from being tied together
    /// - Enables fine-grained control
    /// - Improves disentanglement
    /// </para>
    /// </remarks>
    private bool _enableStyleMixing;

    /// <summary>
    /// Probability of style mixing during training.
    /// </summary>
    private double _styleMixingProbability;

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Creates the combined StyleGAN architecture with correct dimension handling.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateStyleGANArchitecture(
        int latentSize,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType)
    {
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.VeryDeep,
                inputSize: 0,
                inputHeight: discriminatorArchitecture.InputHeight,
                inputWidth: discriminatorArchitecture.InputWidth,
                inputDepth: discriminatorArchitecture.InputDepth,
                outputSize: discriminatorArchitecture.OutputSize,
                layers: null);
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.VeryDeep,
            inputSize: latentSize,
            outputSize: discriminatorArchitecture.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="StyleGAN{T}"/> class.
    /// </summary>
    /// <param name="mappingNetworkArchitecture">Architecture for the mapping network (Z → W).</param>
    /// <param name="synthesisNetworkArchitecture">Architecture for the synthesis network.</param>
    /// <param name="discriminatorArchitecture">Architecture for the discriminator.</param>
    /// <param name="latentSize">Size of input latent code Z.</param>
    /// <param name="intermediateLatentSize">Size of intermediate latent code W.</param>
    /// <param name="inputType">Input type.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">Initial learning rate. Default is 0.001.</param>
    /// <param name="enableStyleMixing">Enable style mixing. Default is true.</param>
    /// <param name="styleMixingProbability">Probability of style mixing. Default is 0.9.</param>
    public StyleGAN(
        NeuralNetworkArchitecture<T> mappingNetworkArchitecture,
        NeuralNetworkArchitecture<T> synthesisNetworkArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int latentSize,
        int intermediateLatentSize,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.001,
        bool enableStyleMixing = true,
        double styleMixingProbability = 0.9)
        : base(CreateStyleGANArchitecture(latentSize, discriminatorArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        // Input validation
        if (mappingNetworkArchitecture is null)
        {
            throw new ArgumentNullException(nameof(mappingNetworkArchitecture), "Mapping network architecture cannot be null.");
        }

        if (synthesisNetworkArchitecture is null)
        {
            throw new ArgumentNullException(nameof(synthesisNetworkArchitecture), "Synthesis network architecture cannot be null.");
        }

        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");
        }

        if (latentSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(latentSize), latentSize, "Latent size must be positive.");
        }

        if (intermediateLatentSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(intermediateLatentSize), intermediateLatentSize, "Intermediate latent size must be positive.");
        }

        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }

        if (styleMixingProbability < 0 || styleMixingProbability > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(styleMixingProbability), styleMixingProbability, "Style mixing probability must be in range [0, 1].");
        }

        _latentSize = latentSize;
        _intermediateLatentSize = intermediateLatentSize;
        _enableStyleMixing = enableStyleMixing;
        _styleMixingProbability = styleMixingProbability;
        _initialLearningRate = initialLearningRate;

        // Initialize MappingNetwork optimizer state
        _mappingMomentum = Vector<T>.Empty();
        _mappingSecondMoment = Vector<T>.Empty();

        // Initialize SynthesisNetwork optimizer state
        _synthesisMomentum = Vector<T>.Empty();
        _synthesisSecondMoment = Vector<T>.Empty();

        // Initialize Discriminator optimizer state
        _discMomentum = Vector<T>.Empty();
        _discSecondMoment = Vector<T>.Empty();

        MappingNetwork = new ConvolutionalNeuralNetwork<T>(mappingNetworkArchitecture);
        SynthesisNetwork = new ConvolutionalNeuralNetwork<T>(synthesisNetworkArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for StyleGAN.
    /// </summary>
    /// <param name="realImages">Real images.</param>
    /// <param name="latentCodes">Random latent codes Z.</param>
    /// <returns>Tuple of (discriminator loss, generator loss).</returns>
    /// <remarks>
    /// <para>
    /// StyleGAN training follows the standard GAN training procedure but with
    /// the style-based generator. Style mixing is applied during training.
    /// </para>
    /// <para><b>For Beginners:</b> One round of StyleGAN training.
    ///
    /// Steps:
    /// 1. Map latent codes Z to style codes W
    /// 2. Optionally apply style mixing
    /// 3. Generate images using styles
    /// 4. Train discriminator on real/fake images
    /// 5. Train generator to fool discriminator
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> latentCodes)
    {
        MappingNetwork.SetTrainingMode(true);
        SynthesisNetwork.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        int batchSize = realImages.Shape[0];

        // ----- Generate Images with Style -----

        // Map Z to W
        var styles = MappingNetwork.Predict(latentCodes);

        // Apply style mixing if enabled
        if (_enableStyleMixing && RandomHelper.ThreadSafeRandom.NextDouble() < _styleMixingProbability)
        {
            var latentCodes2 = GenerateRandomLatentCodes(batchSize);
            var styles2 = MappingNetwork.Predict(latentCodes2);
            styles = MixStyles(styles, styles2);
        }

        // Generate images using styles
        var fakeImages = SynthesisNetwork.Predict(styles);

        // ----- Train Discriminator -----

        var realLabels = CreateLabelTensor(batchSize, NumOps.One);
        var fakeLabels = CreateLabelTensor(batchSize, NumOps.Zero);

        // Train on real images
        var realPredictions = Discriminator.Predict(realImages);
        T realLoss = CalculateBinaryLoss(realPredictions, realLabels, batchSize);
        var realGradients = CalculateBinaryGradients(realPredictions, realLabels, batchSize);
        Discriminator.Backpropagate(realGradients);
        UpdateDiscriminatorParameters();

        // Train on fake images
        var fakePredictions = Discriminator.Predict(fakeImages);
        T fakeLoss = CalculateBinaryLoss(fakePredictions, fakeLabels, batchSize);
        var fakeGradients = CalculateBinaryGradients(fakePredictions, fakeLabels, batchSize);
        Discriminator.Backpropagate(fakeGradients);
        UpdateDiscriminatorParameters();

        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLoss), NumOps.FromDouble(2.0));

        // ----- Train Generator (Mapping + Synthesis) -----

        MappingNetwork.SetTrainingMode(true);
        SynthesisNetwork.SetTrainingMode(true);
        // Keep Discriminator in training mode - required for backpropagation
        // We just don't call UpdateDiscriminatorParameters() during generator training

        // Generate new images
        var newLatentCodes = GenerateRandomLatentCodes(batchSize);
        var newStyles = MappingNetwork.Predict(newLatentCodes);
        var newFakeImages = SynthesisNetwork.Predict(newStyles);

        // Get discriminator predictions
        var genPredictions = Discriminator.Predict(newFakeImages);
        var allRealLabels = CreateLabelTensor(batchSize, NumOps.One);
        T generatorLoss = CalculateBinaryLoss(genPredictions, allRealLabels, batchSize);

        // Backpropagate through discriminator to get input gradients
        var genGradients = CalculateBinaryGradients(genPredictions, allRealLabels, batchSize);
        var discInputGradients = Discriminator.BackwardWithInputGradient(genGradients);

        // Backprop through synthesis network
        var styleGradients = SynthesisNetwork.BackwardWithInputGradient(discInputGradients);

        // Backprop through mapping network
        MappingNetwork.Backward(styleGradients);

        // Update both generator networks
        UpdateSynthesisNetworkParameters();
        UpdateMappingNetworkParameters();

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Mixes two sets of styles at a random layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Style mixing combines coarse and fine features.
    ///
    /// Process:
    /// - Pick a random "mixing point"
    /// - Use Style A before the mixing point (coarse features)
    /// - Use Style B after the mixing point (fine features)
    /// - Result: Face shape from A, details from B
    /// </para>
    /// </remarks>
    private Tensor<T> MixStyles(Tensor<T> styles1, Tensor<T> styles2)
    {
        // Validate inputs
        if (styles1.Shape[0] != styles2.Shape[0])
        {
            throw new ArgumentException(
                $"Batch size mismatch: styles1 has {styles1.Shape[0]} samples, styles2 has {styles2.Shape[0]} samples.");
        }

        var random = RandomHelper.ThreadSafeRandom;
        int styleSize = styles1.Shape[1];

        // Handle edge cases where style size is too small for meaningful mixing
        // Need at least 3 elements to mix (1 for coarse, 1 for mixing point, 1 for fine)
        if (styleSize < 3)
        {
            // If too small to mix, just return styles1
            return styles1;
        }

        // Mix in middle layers: pick a layer between 1 and styleSize/2
        // random.Next(minInclusive, maxExclusive) requires maxExclusive > minInclusive
        int maxMixingLayer = Math.Max(2, styleSize / 2);
        int mixingLayer = random.Next(1, maxMixingLayer);

        var mixedStyles = new Tensor<T>(styles1.Shape);

        for (int b = 0; b < styles1.Shape[0]; b++)
        {
            for (int i = 0; i < styleSize; i++)
            {
                // Use styles1 for coarse features, styles2 for fine features
                mixedStyles[b, i] = i < mixingLayer ? styles1[b, i] : styles2[b, i];
            }
        }

        return mixedStyles;
    }

    /// <summary>
    /// Generates images from latent codes.
    /// </summary>
    /// <param name="latentCodes">Latent codes Z.</param>
    /// <returns>Generated images.</returns>
    public Tensor<T> Generate(Tensor<T> latentCodes)
    {
        MappingNetwork.SetTrainingMode(false);
        SynthesisNetwork.SetTrainingMode(false);

        var styles = MappingNetwork.Predict(latentCodes);
        return SynthesisNetwork.Predict(styles);
    }

    /// <summary>
    /// Generates images with style mixing.
    /// </summary>
    /// <param name="latentCodes1">First set of latent codes (for coarse features).</param>
    /// <param name="latentCodes2">Second set of latent codes (for fine features).</param>
    /// <returns>Generated images with mixed styles.</returns>
    public Tensor<T> GenerateWithStyleMixing(Tensor<T> latentCodes1, Tensor<T> latentCodes2)
    {
        MappingNetwork.SetTrainingMode(false);
        SynthesisNetwork.SetTrainingMode(false);

        var styles1 = MappingNetwork.Predict(latentCodes1);
        var styles2 = MappingNetwork.Predict(latentCodes2);
        var mixedStyles = MixStyles(styles1, styles2);

        return SynthesisNetwork.Predict(mixedStyles);
    }

    /// <summary>
    /// Generates random latent codes using vectorized Gaussian noise generation.
    /// Uses Engine.GenerateGaussianNoise for SIMD/GPU acceleration.
    /// </summary>
    public Tensor<T> GenerateRandomLatentCodes(int batchSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                $"Batch size must be positive, got {batchSize}.");
        }

        var totalElements = batchSize * _latentSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;

        // Use Engine's vectorized Gaussian noise generation
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        // Reshape to [batchSize, latentSize]
        return Tensor<T>.FromVector(noiseVector, [batchSize, _latentSize]);
    }

    private T CalculateBinaryLoss(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);
        T oneMinusEpsilon = NumOps.Subtract(NumOps.One, epsilon);

        for (int i = 0; i < batchSize; i++)
        {
            T prediction = predictions[i, 0];
            T target = targets[i, 0];

            // Clamp prediction to [epsilon, 1-epsilon] to avoid log(0) NaN/Inf
            if (NumOps.LessThan(prediction, epsilon))
            {
                prediction = epsilon;
            }
            else if (NumOps.GreaterThan(prediction, oneMinusEpsilon))
            {
                prediction = oneMinusEpsilon;
            }

            T logP = NumOps.Log(prediction);
            T logOneMinusP = NumOps.Log(NumOps.Subtract(NumOps.One, prediction));

            T loss = NumOps.Negate(NumOps.Add(
                NumOps.Multiply(target, logP),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, target), logOneMinusP)
            ));

            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    private Tensor<T> CalculateBinaryGradients(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
        var gradients = new Tensor<T>(predictions.Shape);
        T epsilon = NumOps.FromDouble(1e-10);
        T oneMinusEpsilon = NumOps.Subtract(NumOps.One, epsilon);

        for (int i = 0; i < batchSize; i++)
        {
            T p = predictions[i, 0];
            T t = targets[i, 0];

            // Clamp p to [epsilon, 1-epsilon] to avoid division by zero
            if (NumOps.LessThan(p, epsilon))
            {
                p = epsilon;
            }
            else if (NumOps.GreaterThan(p, oneMinusEpsilon))
            {
                p = oneMinusEpsilon;
            }

            // BCE gradient w.r.t. probability: dL/dp = (p - t) / (p * (1 - p))
            T oneMinusP = NumOps.Subtract(NumOps.One, p);
            T pTimesOneMinusP = NumOps.Multiply(p, oneMinusP);
            T gradient = NumOps.Divide(
                NumOps.Subtract(p, t),
                NumOps.Add(pTimesOneMinusP, epsilon)
            );

            gradients[i, 0] = NumOps.Divide(gradient, NumOps.FromDouble(batchSize));
        }

        return gradients;
    }

    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        var tensor = new Tensor<T>(new int[] { batchSize, 1 });
        // === Vectorized tensor fill using IEngine (Phase B: US-GPU-015) ===
        Engine.TensorFill(tensor, value);
        return tensor;
    }

    /// <summary>
    /// Updates MappingNetwork parameters using vectorized Adam optimizer.
    /// Uses Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private void UpdateMappingNetworkParameters()
    {
        var parameters = MappingNetwork.GetParameters();
        var gradients = MappingNetwork.GetParameterGradients();

        // Initialize mapping network optimizer state if needed
        if (_mappingMomentum.Length != parameters.Length)
        {
            _mappingMomentum = new Vector<T>(parameters.Length);
            _mappingMomentum.Fill(NumOps.Zero);
        }

        if (_mappingSecondMoment.Length != parameters.Length)
        {
            _mappingSecondMoment = new Vector<T>(parameters.Length);
            _mappingSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer parameters (beta1=0 for StyleGAN)
        var beta1 = NumOps.FromDouble(0.0);
        var beta2 = NumOps.FromDouble(0.99);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        var epsilon = NumOps.FromDouble(1e-8);
        var learningRate = NumOps.FromDouble(_initialLearningRate);

        // Vectorized momentum update: m = beta1 * m + (1 - beta1) * g
        var mScaled = (Vector<T>)Engine.Multiply(_mappingMomentum, beta1);
        var gScaled = (Vector<T>)Engine.Multiply(gradients, oneMinusBeta1);
        _mappingMomentum = (Vector<T>)Engine.Add(mScaled, gScaled);

        // Vectorized second moment update: v = beta2 * v + (1 - beta2) * g^2
        var vScaled = (Vector<T>)Engine.Multiply(_mappingSecondMoment, beta2);
        var gSquared = (Vector<T>)Engine.Multiply(gradients, gradients);
        var gSquaredScaled = (Vector<T>)Engine.Multiply(gSquared, oneMinusBeta2);
        _mappingSecondMoment = (Vector<T>)Engine.Add(vScaled, gSquaredScaled);

        // Vectorized parameter update: p = p - lr * m / (sqrt(v) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(_mappingSecondMoment);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);
        var adaptiveGradient = (Vector<T>)Engine.Divide(_mappingMomentum, sqrtVPlusEps);
        var update = (Vector<T>)Engine.Multiply(adaptiveGradient, learningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        MappingNetwork.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates SynthesisNetwork parameters using vectorized Adam optimizer.
    /// Uses Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private void UpdateSynthesisNetworkParameters()
    {
        var parameters = SynthesisNetwork.GetParameters();
        var gradients = SynthesisNetwork.GetParameterGradients();

        // Initialize synthesis network optimizer state if needed
        if (_synthesisMomentum.Length != parameters.Length)
        {
            _synthesisMomentum = new Vector<T>(parameters.Length);
            _synthesisMomentum.Fill(NumOps.Zero);
        }

        if (_synthesisSecondMoment.Length != parameters.Length)
        {
            _synthesisSecondMoment = new Vector<T>(parameters.Length);
            _synthesisSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer parameters (beta1=0 for StyleGAN)
        var beta1 = NumOps.FromDouble(0.0);
        var beta2 = NumOps.FromDouble(0.99);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        var epsilon = NumOps.FromDouble(1e-8);
        var learningRate = NumOps.FromDouble(_initialLearningRate);

        // Vectorized momentum update: m = beta1 * m + (1 - beta1) * g
        var mScaled = (Vector<T>)Engine.Multiply(_synthesisMomentum, beta1);
        var gScaled = (Vector<T>)Engine.Multiply(gradients, oneMinusBeta1);
        _synthesisMomentum = (Vector<T>)Engine.Add(mScaled, gScaled);

        // Vectorized second moment update: v = beta2 * v + (1 - beta2) * g^2
        var vScaled = (Vector<T>)Engine.Multiply(_synthesisSecondMoment, beta2);
        var gSquared = (Vector<T>)Engine.Multiply(gradients, gradients);
        var gSquaredScaled = (Vector<T>)Engine.Multiply(gSquared, oneMinusBeta2);
        _synthesisSecondMoment = (Vector<T>)Engine.Add(vScaled, gSquaredScaled);

        // Vectorized parameter update: p = p - lr * m / (sqrt(v) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(_synthesisSecondMoment);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);
        var adaptiveGradient = (Vector<T>)Engine.Divide(_synthesisMomentum, sqrtVPlusEps);
        var update = (Vector<T>)Engine.Multiply(adaptiveGradient, learningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        SynthesisNetwork.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates Discriminator parameters using vectorized Adam optimizer.
    /// Uses Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private void UpdateDiscriminatorParameters()
    {
        var parameters = Discriminator.GetParameters();
        var gradients = Discriminator.GetParameterGradients();

        // Initialize discriminator optimizer state if needed
        if (_discMomentum.Length != parameters.Length)
        {
            _discMomentum = new Vector<T>(parameters.Length);
            _discMomentum.Fill(NumOps.Zero);
        }

        if (_discSecondMoment.Length != parameters.Length)
        {
            _discSecondMoment = new Vector<T>(parameters.Length);
            _discSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer parameters (beta1=0 for StyleGAN)
        var beta1 = NumOps.FromDouble(0.0);
        var beta2 = NumOps.FromDouble(0.99);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        var epsilon = NumOps.FromDouble(1e-8);
        var learningRate = NumOps.FromDouble(_initialLearningRate);

        // Vectorized momentum update: m = beta1 * m + (1 - beta1) * g
        var mScaled = (Vector<T>)Engine.Multiply(_discMomentum, beta1);
        var gScaled = (Vector<T>)Engine.Multiply(gradients, oneMinusBeta1);
        _discMomentum = (Vector<T>)Engine.Add(mScaled, gScaled);

        // Vectorized second moment update: v = beta2 * v + (1 - beta2) * g^2
        var vScaled = (Vector<T>)Engine.Multiply(_discSecondMoment, beta2);
        var gSquared = (Vector<T>)Engine.Multiply(gradients, gradients);
        var gSquaredScaled = (Vector<T>)Engine.Multiply(gSquared, oneMinusBeta2);
        _discSecondMoment = (Vector<T>)Engine.Add(vScaled, gSquaredScaled);

        // Vectorized parameter update: p = p - lr * m / (sqrt(v) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(_discSecondMoment);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);
        var adaptiveGradient = (Vector<T>)Engine.Divide(_discMomentum, sqrtVPlusEps);
        var update = (Vector<T>)Engine.Multiply(adaptiveGradient, learningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        Discriminator.UpdateParameters(updatedParameters);
    }

    protected override void InitializeLayers() { }

    public override Tensor<T> Predict(Tensor<T> input) => Generate(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(expectedOutput, input);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.StyleGAN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "MappingNetworkParameters", MappingNetwork.GetParameterCount() },
                { "SynthesisNetworkParameters", SynthesisNetwork.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "LatentSize", _latentSize },
                { "IntermediateLatentSize", _intermediateLatentSize },
                { "StyleMixingEnabled", _enableStyleMixing }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_initialLearningRate);
        writer.Write(_latentSize);
        writer.Write(_intermediateLatentSize);
        writer.Write(_enableStyleMixing);
        writer.Write(_styleMixingProbability);

        var mappingBytes = MappingNetwork.Serialize();
        writer.Write(mappingBytes.Length);
        writer.Write(mappingBytes);

        var synthesisBytes = SynthesisNetwork.Serialize();
        writer.Write(synthesisBytes.Length);
        writer.Write(synthesisBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);

        // Serialize optimizer state for complete training state preservation
        SerializationHelper<T>.SerializeVector(writer, _mappingMomentum);
        SerializationHelper<T>.SerializeVector(writer, _mappingSecondMoment);

        SerializationHelper<T>.SerializeVector(writer, _synthesisMomentum);
        SerializationHelper<T>.SerializeVector(writer, _synthesisSecondMoment);

        SerializationHelper<T>.SerializeVector(writer, _discMomentum);
        SerializationHelper<T>.SerializeVector(writer, _discSecondMoment);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read the learning rate that was written in SerializeNetworkSpecificData
        // Note: _initialLearningRate is readonly so we can't reassign it here
        // The value will be correctly set by the constructor when CreateNewInstance is used
        _ = reader.ReadDouble(); // Consume the stored learning rate value
        _latentSize = reader.ReadInt32();
        _intermediateLatentSize = reader.ReadInt32();
        _enableStyleMixing = reader.ReadBoolean();
        _styleMixingProbability = reader.ReadDouble();

        int mappingLength = reader.ReadInt32();
        MappingNetwork.Deserialize(reader.ReadBytes(mappingLength));

        int synthesisLength = reader.ReadInt32();
        SynthesisNetwork.Deserialize(reader.ReadBytes(synthesisLength));

        int discriminatorLength = reader.ReadInt32();
        Discriminator.Deserialize(reader.ReadBytes(discriminatorLength));

        // Deserialize optimizer state
        _mappingMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _mappingSecondMoment = SerializationHelper<T>.DeserializeVector(reader);

        _synthesisMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _synthesisSecondMoment = SerializationHelper<T>.DeserializeVector(reader);

        _discMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _discSecondMoment = SerializationHelper<T>.DeserializeVector(reader);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new StyleGAN<T>(
            MappingNetwork.Architecture,
            SynthesisNetwork.Architecture,
            Discriminator.Architecture,
            _latentSize,
            _intermediateLatentSize,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate,
            _enableStyleMixing,
            _styleMixingProbability);
    }

    /// <summary>
    /// Updates the parameters of all networks in the StyleGAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <exception cref="ArgumentException">Thrown when parameters length doesn't match expected total.</exception>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int mappingCount = MappingNetwork.GetParameterCount();
        int synthesisCount = SynthesisNetwork.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();
        int expectedTotal = mappingCount + synthesisCount + discriminatorCount;

        if (parameters.Length != expectedTotal)
        {
            throw new ArgumentException(
                $"Expected {expectedTotal} parameters (mapping: {mappingCount}, synthesis: {synthesisCount}, discriminator: {discriminatorCount}), got {parameters.Length}.",
                nameof(parameters));
        }

        int offset = 0;

        // Update MappingNetwork parameters
        var mappingParams = new Vector<T>(mappingCount);
        for (int i = 0; i < mappingCount; i++)
            mappingParams[i] = parameters[offset + i];
        MappingNetwork.UpdateParameters(mappingParams);
        offset += mappingCount;

        // Update SynthesisNetwork parameters
        var synthesisParams = new Vector<T>(synthesisCount);
        for (int i = 0; i < synthesisCount; i++)
            synthesisParams[i] = parameters[offset + i];
        SynthesisNetwork.UpdateParameters(synthesisParams);
        offset += synthesisCount;

        // Update Discriminator parameters
        var discriminatorParams = new Vector<T>(discriminatorCount);
        for (int i = 0; i < discriminatorCount; i++)
            discriminatorParams[i] = parameters[offset + i];
        Discriminator.UpdateParameters(discriminatorParams);
    }
}
