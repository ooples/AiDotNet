namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Information Maximizing Generative Adversarial Network (InfoGAN), which learns
/// disentangled representations in an unsupervised manner by maximizing mutual information
/// between latent codes and generated observations.
/// </summary>
/// <remarks>
/// <para>
/// InfoGAN extends the GAN framework by:
/// - Decomposing the input noise into incompressible noise (z) and latent codes (c)
/// - Maximizing the mutual information I(c; G(z,c)) between codes and generated images
/// - Learning interpretable and disentangled representations automatically
/// - Using an auxiliary network Q to approximate the posterior P(c|x)
/// - Enabling control over semantic features without labeled data
/// </para>
/// <para><b>For Beginners:</b> InfoGAN learns to separate different features automatically.
///
/// Key concept:
/// - Splits random input into two parts:
///   1. Random noise (z): provides variety
///   2. Latent codes (c): control specific features
/// - Learns what each code controls WITHOUT labels
/// - Example: For faces, might learn codes for:
///   * Code 1: controls rotation
///   * Code 2: controls width
///   * Code 3: controls lighting
///
/// How it works:
/// - Generator uses both z and c to create images
/// - Auxiliary network Q tries to predict c from the generated image
/// - If Q can predict c accurately, the codes are meaningful
/// - This forces codes to represent interpretable features
///
/// Use cases:
/// - Discover semantic features in datasets
/// - Disentangled representation learning
/// - Controllable image generation
/// - Feature manipulation (change one aspect, keep others)
///
/// Reference: Chen et al., "InfoGAN: Interpretable Representation Learning by
/// Information Maximizing Generative Adversarial Nets" (2016)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class InfoGAN<T> : NeuralNetworkBase<T>
{
    // Generator optimizer state
    private Vector<T> _genMomentum;
    private Vector<T> _genSecondMoment;
    private T _genBeta1Power;
    private T _genBeta2Power;
    private double _genCurrentLearningRate;

    // Discriminator optimizer state
    private Vector<T> _discMomentum;
    private Vector<T> _discSecondMoment;
    private T _discBeta1Power;
    private T _discBeta2Power;
    private double _discCurrentLearningRate;

    // QNetwork optimizer state
    private Vector<T> _qMomentum;
    private Vector<T> _qSecondMoment;
    private T _qBeta1Power;
    private T _qBeta2Power;
    private double _qCurrentLearningRate;

    private double _initialLearningRate;
    private double _learningRateDecay;
    private readonly List<T> _generatorLosses = [];
    private readonly List<T> _discriminatorLosses = [];

    /// <summary>
    /// The size of the latent code c.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The latent code represents interpretable factors of variation. A typical value is 10-20.
    /// This can include both discrete codes (for categorical features) and continuous codes
    /// (for continuous features like rotation angle).
    /// </para>
    /// <para><b>For Beginners:</b> How many controllable features to learn.
    ///
    /// - Larger values: more features to discover, but may be harder to train
    /// - Smaller values: fewer but potentially clearer features
    /// - Typical: 10 codes can capture many important features
    /// </para>
    /// </remarks>
    private int _latentCodeSize;

    /// <summary>
    /// The coefficient for the mutual information loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the trade-off between the standard GAN objective and the mutual information
    /// objective. A typical value is 1.0. Higher values enforce stronger disentanglement.
    /// </para>
    /// <para><b>For Beginners:</b> How important is the feature learning vs image quality.
    ///
    /// - Higher (e.g., 2.0): prioritize learning clear features
    /// - Lower (e.g., 0.5): prioritize image quality
    /// - Default (1.0): balanced approach
    /// </para>
    /// </remarks>
    private double _mutualInfoCoefficient;

    /// <summary>
    /// Gets the generator network.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    /// <summary>
    /// Gets the auxiliary Q network that predicts latent codes from images.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Q network shares most of its parameters with the discriminator (up to the
    /// last layer). It outputs the predicted latent code distribution given an image.
    /// This network is key to maximizing mutual information.
    /// </para>
    /// <para><b>For Beginners:</b> The Q network is the "feature detector".
    ///
    /// - Takes an image as input
    /// - Outputs: "I think these codes were used to make this"
    /// - Training makes Q better at guessing codes
    /// - This forces generator to use codes meaningfully
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> QNetwork { get; private set; }

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="InfoGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The generator architecture.</param>
    /// <param name="discriminatorArchitecture">The discriminator architecture.</param>
    /// <param name="qNetworkArchitecture">The Q network architecture (should output latentCodeSize values).</param>
    /// <param name="latentCodeSize">The size of the latent code.</param>
    /// <param name="inputType">The input type.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">Initial learning rate. Default is 0.0002.</param>
    /// <param name="mutualInfoCoefficient">Mutual information coefficient. Default is 1.0.</param>
    public InfoGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        NeuralNetworkArchitecture<T> qNetworkArchitecture,
        int latentCodeSize,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0002,
        double mutualInfoCoefficient = 1.0)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Deep,
            generatorArchitecture.InputSize,
            discriminatorArchitecture.OutputSize,
            0, 0, 0,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _latentCodeSize = latentCodeSize;
        _mutualInfoCoefficient = mutualInfoCoefficient;
        _initialLearningRate = initialLearningRate;
        _learningRateDecay = 0.9999;

        // Initialize Generator optimizer state
        _genBeta1Power = NumOps.One;
        _genBeta2Power = NumOps.One;
        _genCurrentLearningRate = initialLearningRate;
        _genMomentum = Vector<T>.Empty();
        _genSecondMoment = Vector<T>.Empty();

        // Initialize Discriminator optimizer state
        _discBeta1Power = NumOps.One;
        _discBeta2Power = NumOps.One;
        _discCurrentLearningRate = initialLearningRate;
        _discMomentum = Vector<T>.Empty();
        _discSecondMoment = Vector<T>.Empty();

        // Initialize QNetwork optimizer state
        _qBeta1Power = NumOps.One;
        _qBeta2Power = NumOps.One;
        _qCurrentLearningRate = initialLearningRate;
        _qMomentum = Vector<T>.Empty();
        _qSecondMoment = Vector<T>.Empty();

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
        QNetwork = new ConvolutionalNeuralNetwork<T>(qNetworkArchitecture);

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for InfoGAN.
    /// </summary>
    /// <param name="realImages">Real images.</param>
    /// <param name="noise">Random noise (z).</param>
    /// <param name="latentCodes">Latent codes (c) to condition generation.</param>
    /// <returns>Tuple of (discriminator loss, generator loss, mutual info loss).</returns>
    /// <remarks>
    /// <para>
    /// InfoGAN training:
    /// 1. Train discriminator (standard GAN objective)
    /// 2. Train generator with GAN loss + mutual information loss
    /// 3. Train Q network to predict latent codes from generated images
    /// </para>
    /// <para><b>For Beginners:</b> One round of InfoGAN training.
    ///
    /// Steps:
    /// 1. Generate images using noise + latent codes
    /// 2. Train discriminator to spot fakes (standard GAN)
    /// 3. Train generator to fool discriminator
    /// 4. Train Q network to guess the codes from images
    /// 5. Make generator use codes that Q can predict
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss, T mutualInfoLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> noise,
        Tensor<T> latentCodes)
    {
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);
        QNetwork.SetTrainingMode(true);

        int batchSize = realImages.Shape[0];

        // ----- Train Discriminator -----

        // Concatenate noise and latent codes for generator
        var generatorInput = ConcatenateTensors(noise, latentCodes);

        // Generate fake images
        var fakeImages = Generator.Predict(generatorInput);

        // Real labels
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

        // ----- Train Generator and Q Network -----

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false);
        QNetwork.SetTrainingMode(true);

        // Generate new fake images
        var newGeneratorInput = ConcatenateTensors(noise, latentCodes);
        var newFakeImages = Generator.Predict(newGeneratorInput);

        // GAN loss: fool the discriminator
        var genPredictions = Discriminator.Predict(newFakeImages);
        var allRealLabels = CreateLabelTensor(batchSize, NumOps.One);
        T ganLoss = CalculateBinaryLoss(genPredictions, allRealLabels, batchSize);

        // Mutual information loss: Q predicts latent codes
        var predictedCodes = QNetwork.Predict(newFakeImages);
        T mutualInfoLoss = CalculateMutualInfoLoss(predictedCodes, latentCodes, batchSize);

        // Total generator loss
        T miCoeff = NumOps.FromDouble(_mutualInfoCoefficient);
        T generatorLoss = NumOps.Add(ganLoss, NumOps.Multiply(miCoeff, mutualInfoLoss));

        // Backpropagate through discriminator (for GAN loss)
        var ganGradients = CalculateBinaryGradients(genPredictions, allRealLabels, batchSize);
        var discInputGradients = Discriminator.Backpropagate(ganGradients);

        // Backpropagate through Q network (for MI loss)
        var miGradients = CalculateMutualInfoGradients(predictedCodes, latentCodes, batchSize);
        var qInputGradients = QNetwork.Backpropagate(miGradients);

        // Combine gradients
        var combinedGradients = new Tensor<T>(discInputGradients.Shape);
        int gradLength = discInputGradients.Shape.Aggregate(1, (a, b) => a * b);
        for (int i = 0; i < gradLength; i++)
        {
            combinedGradients.SetFlat(i, NumOps.Add(
                discInputGradients.GetFlat(i),
                NumOps.Multiply(miCoeff, qInputGradients.GetFlat(i))
            ));
        }

        // Backpropagate through generator
        Generator.Backpropagate(combinedGradients);
        UpdateGeneratorParameters();
        UpdateQNetworkParameters();

        Discriminator.SetTrainingMode(true);

        // Track losses
        _discriminatorLosses.Add(discriminatorLoss);
        _generatorLosses.Add(generatorLoss);

        if (_discriminatorLosses.Count > 100)
        {
            _discriminatorLosses.RemoveAt(0);
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss, mutualInfoLoss);
    }

    /// <summary>
    /// Calculates mutual information loss (MSE between predicted and true codes).
    /// </summary>
    private T CalculateMutualInfoLoss(Tensor<T> predictedCodes, Tensor<T> trueCodes, int batchSize)
    {
        T totalLoss = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _latentCodeSize; c++)
            {
                T diff = NumOps.Subtract(predictedCodes[b, c], trueCodes[b, c]);
                T squaredDiff = NumOps.Multiply(diff, diff);
                totalLoss = NumOps.Add(totalLoss, squaredDiff);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble((double)batchSize * _latentCodeSize));
    }

    /// <summary>
    /// Calculates gradients for mutual information loss.
    /// </summary>
    private Tensor<T> CalculateMutualInfoGradients(Tensor<T> predictedCodes, Tensor<T> trueCodes, int batchSize)
    {
        var gradients = new Tensor<T>(predictedCodes.Shape);
        T scale = NumOps.FromDouble(2.0 / ((double)batchSize * _latentCodeSize));

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _latentCodeSize; c++)
            {
                T diff = NumOps.Subtract(predictedCodes[b, c], trueCodes[b, c]);
                gradients[b, c] = NumOps.Multiply(scale, diff);
            }
        }

        return gradients;
    }

    /// <summary>
    /// Calculates binary cross-entropy loss.
    /// </summary>
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

    /// <summary>
    /// Calculates gradients for binary cross-entropy.
    /// </summary>
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

    /// <summary>
    /// Creates a label tensor.
    /// </summary>
    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        var tensor = new Tensor<T>(new int[] { batchSize, 1 });
        for (int i = 0; i < batchSize; i++)
        {
            tensor[i, 0] = value;
        }
        return tensor;
    }

    /// <summary>
    /// Concatenates noise and latent codes.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> noise, Tensor<T> codes)
    {
        int batchSize = noise.Shape[0];
        int noiseSize = noise.Shape[1];
        int codeSize = codes.Shape[1];

        var result = new Tensor<T>(new int[] { batchSize, noiseSize + codeSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < noiseSize; i++)
            {
                result[b, i] = noise[b, i];
            }
            for (int i = 0; i < codeSize; i++)
            {
                result[b, noiseSize + i] = codes[b, i];
            }
        }

        return result;
    }

    /// <summary>
    /// Generates images with specific latent codes.
    /// </summary>
    /// <param name="noise">Random noise.</param>
    /// <param name="latentCodes">Latent codes to control generation.</param>
    /// <returns>Generated images.</returns>
    public Tensor<T> Generate(Tensor<T> noise, Tensor<T> latentCodes)
    {
        Generator.SetTrainingMode(false);
        var input = ConcatenateTensors(noise, latentCodes);
        return Generator.Predict(input);
    }

    /// <summary>
    /// Generates random noise tensor.
    /// </summary>
    public Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
    {
        var random = new Random();
        var noise = new Tensor<T>(new int[] { batchSize, noiseSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < noiseSize; i += 2)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;

                noise[b, i] = NumOps.FromDouble(radius * Math.Cos(theta));
                if (i + 1 < noiseSize)
                {
                    noise[b, i + 1] = NumOps.FromDouble(radius * Math.Sin(theta));
                }
            }
        }

        return noise;
    }

    /// <summary>
    /// Generates random latent codes (continuous, uniform in [-1, 1]).
    /// </summary>
    public Tensor<T> GenerateRandomLatentCodes(int batchSize)
    {
        var random = new Random();
        var codes = new Tensor<T>(new int[] { batchSize, _latentCodeSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _latentCodeSize; c++)
            {
                codes[b, c] = NumOps.FromDouble(random.NextDouble() * 2.0 - 1.0);
            }
        }

        return codes;
    }

    /// <summary>
    /// Updates Generator parameters using Adam optimizer.
    /// </summary>
    private void UpdateGeneratorParameters()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        if (_genMomentum == null || _genMomentum.Length != parameters.Length)
        {
            _genMomentum = new Vector<T>(parameters.Length);
            _genMomentum.Fill(NumOps.Zero);
        }

        if (_genSecondMoment == null || _genSecondMoment.Length != parameters.Length)
        {
            _genSecondMoment = new Vector<T>(parameters.Length);
            _genSecondMoment.Fill(NumOps.Zero);
        }

        var learningRate = NumOps.FromDouble(_genCurrentLearningRate);
        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
        var epsilon = NumOps.FromDouble(1e-8);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            _genMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _genMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            _genSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _genSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            var momentumCorrected = NumOps.Divide(_genMomentum[i], NumOps.Subtract(NumOps.One, _genBeta1Power));
            var secondMomentCorrected = NumOps.Divide(_genSecondMoment[i], NumOps.Subtract(NumOps.One, _genBeta2Power));

            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(secondMomentCorrected), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, momentumCorrected)
            );
        }

        _genBeta1Power = NumOps.Multiply(_genBeta1Power, beta1);
        _genBeta2Power = NumOps.Multiply(_genBeta2Power, beta2);
        _genCurrentLearningRate *= _learningRateDecay;

        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates Discriminator parameters using Adam optimizer.
    /// </summary>
    private void UpdateDiscriminatorParameters()
    {
        var parameters = Discriminator.GetParameters();
        var gradients = Discriminator.GetParameterGradients();

        if (_discMomentum == null || _discMomentum.Length != parameters.Length)
        {
            _discMomentum = new Vector<T>(parameters.Length);
            _discMomentum.Fill(NumOps.Zero);
        }

        if (_discSecondMoment == null || _discSecondMoment.Length != parameters.Length)
        {
            _discSecondMoment = new Vector<T>(parameters.Length);
            _discSecondMoment.Fill(NumOps.Zero);
        }

        var learningRate = NumOps.FromDouble(_discCurrentLearningRate);
        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
        var epsilon = NumOps.FromDouble(1e-8);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            _discMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _discMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            _discSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _discSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            var momentumCorrected = NumOps.Divide(_discMomentum[i], NumOps.Subtract(NumOps.One, _discBeta1Power));
            var secondMomentCorrected = NumOps.Divide(_discSecondMoment[i], NumOps.Subtract(NumOps.One, _discBeta2Power));

            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(secondMomentCorrected), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, momentumCorrected)
            );
        }

        _discBeta1Power = NumOps.Multiply(_discBeta1Power, beta1);
        _discBeta2Power = NumOps.Multiply(_discBeta2Power, beta2);
        _discCurrentLearningRate *= _learningRateDecay;

        Discriminator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates QNetwork parameters using Adam optimizer.
    /// </summary>
    private void UpdateQNetworkParameters()
    {
        var parameters = QNetwork.GetParameters();
        var gradients = QNetwork.GetParameterGradients();

        if (_qMomentum == null || _qMomentum.Length != parameters.Length)
        {
            _qMomentum = new Vector<T>(parameters.Length);
            _qMomentum.Fill(NumOps.Zero);
        }

        if (_qSecondMoment == null || _qSecondMoment.Length != parameters.Length)
        {
            _qSecondMoment = new Vector<T>(parameters.Length);
            _qSecondMoment.Fill(NumOps.Zero);
        }

        var learningRate = NumOps.FromDouble(_qCurrentLearningRate);
        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
        var epsilon = NumOps.FromDouble(1e-8);

        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            _qMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _qMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i])
            );

            _qSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _qSecondMoment[i]),
                NumOps.Multiply(
                    NumOps.Subtract(NumOps.One, beta2),
                    NumOps.Multiply(gradients[i], gradients[i])
                )
            );

            var momentumCorrected = NumOps.Divide(_qMomentum[i], NumOps.Subtract(NumOps.One, _qBeta1Power));
            var secondMomentCorrected = NumOps.Divide(_qSecondMoment[i], NumOps.Subtract(NumOps.One, _qBeta2Power));

            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(secondMomentCorrected), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, momentumCorrected)
            );
        }

        _qBeta1Power = NumOps.Multiply(_qBeta1Power, beta1);
        _qBeta2Power = NumOps.Multiply(_qBeta2Power, beta2);
        _qCurrentLearningRate *= _learningRateDecay;

        QNetwork.UpdateParameters(updatedParameters);
    }

    protected override void InitializeLayers()
    {
        // InfoGAN doesn't use layers directly
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        int batchSize = expectedOutput.Shape[0];
        var codes = GenerateRandomLatentCodes(batchSize);
        TrainStep(expectedOutput, input, codes);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.InfoGAN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "QNetworkParameters", QNetwork.GetParameterCount() },
                { "LatentCodeSize", _latentCodeSize },
                { "MutualInfoCoefficient", _mutualInfoCoefficient }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Save all three learning rates
        writer.Write(_genCurrentLearningRate);
        writer.Write(_discCurrentLearningRate);
        writer.Write(_qCurrentLearningRate);
        writer.Write(_latentCodeSize);
        writer.Write(_mutualInfoCoefficient);

        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);

        var qNetworkBytes = QNetwork.Serialize();
        writer.Write(qNetworkBytes.Length);
        writer.Write(qNetworkBytes);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read all three learning rates
        _genCurrentLearningRate = reader.ReadDouble();
        _discCurrentLearningRate = reader.ReadDouble();
        _qCurrentLearningRate = reader.ReadDouble();
        _latentCodeSize = reader.ReadInt32();
        _mutualInfoCoefficient = reader.ReadDouble();

        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int discriminatorDataLength = reader.ReadInt32();
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);

        int qNetworkDataLength = reader.ReadInt32();
        byte[] qNetworkData = reader.ReadBytes(qNetworkDataLength);
        QNetwork.Deserialize(qNetworkData);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new InfoGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            QNetwork.Architecture,
            _latentCodeSize,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate,
            _mutualInfoCoefficient);
    }

    /// <summary>
    /// Updates the parameters of all networks in the InfoGAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();
        int qNetworkCount = QNetwork.GetParameterCount();

        int offset = 0;

        // Update Generator parameters
        var generatorParams = new Vector<T>(generatorCount);
        for (int i = 0; i < generatorCount; i++)
            generatorParams[i] = parameters[offset + i];
        Generator.UpdateParameters(generatorParams);
        offset += generatorCount;

        // Update Discriminator parameters
        var discriminatorParams = new Vector<T>(discriminatorCount);
        for (int i = 0; i < discriminatorCount; i++)
            discriminatorParams[i] = parameters[offset + i];
        Discriminator.UpdateParameters(discriminatorParams);
        offset += discriminatorCount;

        // Update QNetwork parameters
        var qNetworkParams = new Vector<T>(qNetworkCount);
        for (int i = 0; i < qNetworkCount; i++)
            qNetworkParams[i] = parameters[offset + i];
        QNetwork.UpdateParameters(qNetworkParams);
    }
}
