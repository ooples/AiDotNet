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
    private Vector<T> _momentum;
    private Vector<T> _secondMoment;
    private T _beta1Power;
    private T _beta2Power;
    private double _currentLearningRate;
    private double _initialLearningRate;
    private double _learningRateDecay;

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
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.VeryHigh,
            latentSize,
            discriminatorArchitecture.OutputSize,
            0, 0, 0,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        _latentSize = latentSize;
        _intermediateLatentSize = intermediateLatentSize;
        _enableStyleMixing = enableStyleMixing;
        _styleMixingProbability = styleMixingProbability;
        _initialLearningRate = initialLearningRate;
        _currentLearningRate = initialLearningRate;
        _learningRateDecay = 0.9999;

        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;

        MappingNetwork = new ConvolutionalNeuralNetwork<T>(mappingNetworkArchitecture);
        SynthesisNetwork = new ConvolutionalNeuralNetwork<T>(synthesisNetworkArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();
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
        if (_enableStyleMixing && new Random().NextDouble() < _styleMixingProbability)
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
        UpdateNetworkParameters(Discriminator);

        // Train on fake images
        var fakePredictions = Discriminator.Predict(fakeImages);
        T fakeLoss = CalculateBinaryLoss(fakePredictions, fakeLabels, batchSize);
        var fakeGradients = CalculateBinaryGradients(fakePredictions, fakeLabels, batchSize);
        Discriminator.Backpropagate(fakeGradients);
        UpdateNetworkParameters(Discriminator);

        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLoss), NumOps.FromDouble(2.0));

        // ----- Train Generator (Mapping + Synthesis) -----

        MappingNetwork.SetTrainingMode(true);
        SynthesisNetwork.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false);

        // Generate new images
        var newLatentCodes = GenerateRandomLatentCodes(batchSize);
        var newStyles = MappingNetwork.Predict(newLatentCodes);
        var newFakeImages = SynthesisNetwork.Predict(newStyles);

        // Get discriminator predictions
        var genPredictions = Discriminator.Predict(newFakeImages);
        var allRealLabels = CreateLabelTensor(batchSize, NumOps.One);
        T generatorLoss = CalculateBinaryLoss(genPredictions, allRealLabels, batchSize);

        // Backpropagate
        var genGradients = CalculateBinaryGradients(genPredictions, allRealLabels, batchSize);
        var discInputGradients = Discriminator.Backpropagate(genGradients);

        // Backprop through synthesis network
        var styleGradients = SynthesisNetwork.Backpropagate(discInputGradients);

        // Backprop through mapping network
        MappingNetwork.Backpropagate(styleGradients);

        // Update both generator networks
        UpdateNetworkParameters(SynthesisNetwork);
        UpdateNetworkParameters(MappingNetwork);

        Discriminator.SetTrainingMode(true);

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
        var random = new Random();
        int mixingLayer = random.Next(1, styles1.Shape[1] / 2); // Mix in middle layers

        var mixedStyles = new Tensor<T>(styles1.Shape);

        for (int b = 0; b < styles1.Shape[0]; b++)
        {
            for (int i = 0; i < styles1.Shape[1]; i++)
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
    /// Generates random latent codes.
    /// </summary>
    public Tensor<T> GenerateRandomLatentCodes(int batchSize)
    {
        var random = new Random();
        var codes = new Tensor<T>(new int[] { batchSize, _latentSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _latentSize; i += 2)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;

                codes[b, i] = NumOps.FromDouble(radius * Math.Cos(theta));
                if (i + 1 < _latentSize)
                {
                    codes[b, i + 1] = NumOps.FromDouble(radius * Math.Sin(theta));
                }
            }
        }

        return codes;
    }

    private T CalculateBinaryLoss(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < batchSize; i++)
        {
            T prediction = predictions[i, 0];
            T target = targets[i, 0];

            T logP = NumOps.Log(NumOps.Add(prediction, epsilon));
            T logOneMinusP = NumOps.Log(NumOps.Add(NumOps.Subtract(NumOps.One, prediction), epsilon));

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

        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = NumOps.Divide(
                NumOps.Subtract(predictions[i, 0], targets[i, 0]),
                NumOps.FromDouble(batchSize)
            );
        }

        return gradients;
    }

    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        var tensor = new Tensor<T>(new int[] { batchSize, 1 });
        for (int i = 0; i < batchSize; i++)
        {
            tensor[i, 0] = value;
        }
        return tensor;
    }

    private void UpdateNetworkParameters(ConvolutionalNeuralNetwork<T> network)
    {
        var parameters = network.GetParameters();
        var gradients = network.GetParameterGradients();

        if (_momentum == null || _momentum.Length != parameters.Length)
        {
            _momentum = new Vector<T>(parameters.Length);
            _momentum.Fill(NumOps.Zero);
        }

        if (_secondMoment == null || _secondMoment.Length != parameters.Length)
        {
            _secondMoment = new Vector<T>(parameters.Length);
            _secondMoment.Fill(NumOps.Zero);
        }

        var learningRate = NumOps.FromDouble(_currentLearningRate);
        var beta1 = NumOps.FromDouble(0.0);  // StyleGAN uses beta1=0
        var beta2 = NumOps.FromDouble(0.99);
        var epsilon = NumOps.FromDouble(1e-8);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            _momentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _momentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            _secondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _secondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(_secondMoment[i]), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, _momentum[i])
            );
        }

        network.UpdateParameters(updatedParameters);
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
        writer.Write(_currentLearningRate);
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
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _currentLearningRate = reader.ReadDouble();
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
}
