namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents an Auxiliary Classifier Generative Adversarial Network (AC-GAN), which extends
/// conditional GANs by having the discriminator also predict the class label of the input.
/// </summary>
/// <remarks>
/// <para>
/// AC-GAN improves upon conditional GANs by:
/// - Making the discriminator predict both authenticity AND class label
/// - Providing stronger gradient signals for class-conditional generation
/// - Improving image quality and class separability
/// - Enabling better control over generated samples
/// - Training more stable than basic conditional GANs
/// </para>
/// <para><b>For Beginners:</b> AC-GAN generates specific types of images with better quality.
///
/// Key improvements over cGAN:
/// - Discriminator has two tasks: "Is it real?" AND "What class is it?"
/// - This dual task helps the discriminator learn better features
/// - Generator must create images that fool both checks
/// - Results in higher quality and more class-consistent images
///
/// Example use case:
/// - Generate digit "7" that looks very realistic
/// - Discriminator checks: 1) Is it real? 2) Is it a "7"?
/// - This forces the generator to make better "7"s
///
/// Reference: Odena et al., "Conditional Image Synthesis with Auxiliary Classifier GANs" (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ACGAN<T> : NeuralNetworkBase<T>
{
    private Vector<T> _momentum;
    private Vector<T> _secondMoment;
    private T _beta1Power;
    private T _beta2Power;
    private double _currentLearningRate;
    private double _initialLearningRate;
    private double _learningRateDecay;
    private List<T> _generatorLosses = [];
    private List<T> _discriminatorLosses = [];

    /// <summary>
    /// The number of classes for classification.
    /// </summary>
    private int _numClasses;

    /// <summary>
    /// Gets the generator network that creates class-conditional synthetic data.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network that predicts both authenticity and class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Unlike standard GANs, the AC-GAN discriminator has two outputs:
    /// 1. Authenticity score (real vs fake) - 1 output
    /// 2. Class probability distribution - numClasses outputs
    /// </para>
    /// <para><b>For Beginners:</b> The discriminator is a multi-task network.
    ///
    /// Two outputs:
    /// - "Is this real or fake?" (1 number: 0-1)
    /// - "What class is this?" (probability for each class)
    ///
    /// This dual purpose makes it a better feature learner.
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="ACGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.
    /// Note: Output size should be 1 + numClasses (authenticity + class logits).</param>
    /// <param name="numClasses">The number of classes.</param>
    /// <param name="inputType">The type of input.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">The initial learning rate. Default is 0.0002.</param>
    public ACGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numClasses,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0002)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            generatorArchitecture.InputSize + numClasses,
            discriminatorArchitecture.OutputSize,
            0, 0, 0,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _numClasses = numClasses;
        _initialLearningRate = initialLearningRate;
        _currentLearningRate = initialLearningRate;
        _learningRateDecay = 0.9999;

        // Initialize optimizer parameters
        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for the AC-GAN.
    /// </summary>
    /// <param name="realImages">Real images tensor.</param>
    /// <param name="realLabels">Real image class labels (one-hot encoded).</param>
    /// <param name="noise">Random noise for generator.</param>
    /// <param name="fakeLabels">Class labels for images to generate (one-hot encoded).</param>
    /// <returns>Tuple of (discriminator loss, generator loss).</returns>
    /// <remarks>
    /// <para>
    /// AC-GAN training involves:
    /// 1. Train discriminator on real images (authentic + correct class)
    /// 2. Train discriminator on fake images (fake + should match requested class)
    /// 3. Train generator to fool discriminator and match the requested class
    /// </para>
    /// <para><b>For Beginners:</b> One round of AC-GAN training.
    ///
    /// Steps:
    /// 1. Show discriminator real images with their true labels
    /// 2. Generate fake images with specific class labels
    /// 3. Train discriminator to spot fakes AND identify classes
    /// 4. Train generator to create realistic images of the requested class
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> realLabels,
        Tensor<T> noise,
        Tensor<T> fakeLabels)
    {
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        int batchSize = realImages.Shape[0];

        // ----- Train Discriminator -----

        // Concatenate noise with class labels for generator
        var generatorInput = ConcatenateTensors(noise, fakeLabels);

        // Generate fake images
        var fakeImages = Generator.Predict(generatorInput);

        // Train discriminator on real images
        var realDiscOutput = Discriminator.Predict(realImages);
        T realAuthLoss = CalculateAuthenticityLoss(realDiscOutput, isReal: true, batchSize);
        T realClassLoss = CalculateClassificationLoss(realDiscOutput, realLabels, batchSize);
        T realLoss = NumOps.Add(realAuthLoss, realClassLoss);

        // Backpropagate for real images
        var realGradients = CalculateDiscriminatorGradients(realDiscOutput, realLabels, isReal: true, batchSize);
        Discriminator.Backpropagate(realGradients);
        UpdateNetworkParameters(Discriminator);

        // Train discriminator on fake images
        var fakeDiscOutput = Discriminator.Predict(fakeImages);
        T fakeAuthLoss = CalculateAuthenticityLoss(fakeDiscOutput, isReal: false, batchSize);
        T fakeClassLoss = CalculateClassificationLoss(fakeDiscOutput, fakeLabels, batchSize);
        T fakeLoss = NumOps.Add(fakeAuthLoss, fakeClassLoss);

        // Backpropagate for fake images
        var fakeGradients = CalculateDiscriminatorGradients(fakeDiscOutput, fakeLabels, isReal: false, batchSize);
        Discriminator.Backpropagate(fakeGradients);
        UpdateNetworkParameters(Discriminator);

        // Total discriminator loss
        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLoss), NumOps.FromDouble(2.0));

        // ----- Train Generator -----

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false);

        // Generate new fake images
        var newGeneratorInput = ConcatenateTensors(noise, fakeLabels);
        var newFakeImages = Generator.Predict(newGeneratorInput);

        // Get discriminator output
        var genDiscOutput = Discriminator.Predict(newFakeImages);

        // For generator, we want discriminator to think images are real AND match the class
        T genAuthLoss = CalculateAuthenticityLoss(genDiscOutput, isReal: true, batchSize);
        T genClassLoss = CalculateClassificationLoss(genDiscOutput, fakeLabels, batchSize);
        T generatorLoss = NumOps.Add(genAuthLoss, genClassLoss);

        // Backpropagate through discriminator and generator
        var genGradients = CalculateDiscriminatorGradients(genDiscOutput, fakeLabels, isReal: true, batchSize);
        var discInputGradients = Discriminator.Backpropagate(genGradients);
        Generator.Backpropagate(discInputGradients);
        UpdateNetworkParameters(Generator);

        Discriminator.SetTrainingMode(true);

        // Track losses
        _discriminatorLosses.Add(discriminatorLoss);
        _generatorLosses.Add(generatorLoss);

        if (_discriminatorLosses.Count > 100)
        {
            _discriminatorLosses.RemoveAt(0);
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Calculates the authenticity loss (real vs fake).
    /// </summary>
    private T CalculateAuthenticityLoss(Tensor<T> discOutput, bool isReal, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);
        T target = isReal ? NumOps.One : NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // First output is authenticity score
            T prediction = discOutput[i, 0];

            // Binary cross-entropy
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
    /// Calculates the classification loss for the class predictions.
    /// </summary>
    private T CalculateClassificationLoss(Tensor<T> discOutput, Tensor<T> labels, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < batchSize; i++)
        {
            // Class outputs start from index 1 (index 0 is authenticity)
            for (int c = 0; c < _numClasses; c++)
            {
                T prediction = discOutput[i, 1 + c];
                T target = labels[i, c];

                // Cross-entropy for class prediction
                T logP = NumOps.Log(NumOps.Add(prediction, epsilon));
                T loss = NumOps.Multiply(target, logP);
                totalLoss = NumOps.Subtract(totalLoss, loss);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates gradients for discriminator backpropagation.
    /// </summary>
    private Tensor<T> CalculateDiscriminatorGradients(
        Tensor<T> discOutput,
        Tensor<T> labels,
        bool isReal,
        int batchSize)
    {
        var gradients = new Tensor<T>(discOutput.Shape);
        T authTarget = isReal ? NumOps.One : NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // Gradient for authenticity output
            gradients[i, 0] = NumOps.Divide(
                NumOps.Subtract(discOutput[i, 0], authTarget),
                NumOps.FromDouble(batchSize)
            );

            // Gradients for class outputs
            for (int c = 0; c < _numClasses; c++)
            {
                gradients[i, 1 + c] = NumOps.Divide(
                    NumOps.Subtract(discOutput[i, 1 + c], labels[i, c]),
                    NumOps.FromDouble(batchSize)
                );
            }
        }

        return gradients;
    }

    /// <summary>
    /// Concatenates noise and class labels for generator input.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> noise, Tensor<T> labels)
    {
        int batchSize = noise.Shape[0];
        int noiseSize = noise.Shape[1];
        int labelSize = labels.Shape[1];

        var result = new Tensor<T>(new int[] { batchSize, noiseSize + labelSize });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < noiseSize; i++)
            {
                result[b, i] = noise[b, i];
            }
            for (int i = 0; i < labelSize; i++)
            {
                result[b, noiseSize + i] = labels[b, i];
            }
        }

        return result;
    }

    /// <summary>
    /// Generates class-conditional images.
    /// </summary>
    /// <param name="noise">Random noise tensor.</param>
    /// <param name="classLabels">One-hot encoded class labels.</param>
    /// <returns>Generated images.</returns>
    public Tensor<T> GenerateConditional(Tensor<T> noise, Tensor<T> classLabels)
    {
        Generator.SetTrainingMode(false);
        var input = ConcatenateTensors(noise, classLabels);
        return Generator.Predict(input);
    }

    /// <summary>
    /// Creates one-hot encoded class labels.
    /// </summary>
    public Tensor<T> CreateOneHotLabels(int batchSize, int classIndex)
    {
        var labels = new Tensor<T>(new int[] { batchSize, _numClasses });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                labels[b, c] = c == classIndex ? NumOps.One : NumOps.Zero;
            }
        }

        return labels;
    }

    /// <summary>
    /// Generates random noise tensor.
    /// </summary>
    public Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
    {
        var random = new Random();
        var shape = new int[] { batchSize, noiseSize };
        var noise = new Tensor<T>(shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < noiseSize; i += 2)
            {
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();

                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double theta = 2.0 * Math.PI * u2;

                double z1 = radius * Math.Cos(theta);
                noise[b, i] = NumOps.FromDouble(z1);

                if (i + 1 < noiseSize)
                {
                    double z2 = radius * Math.Sin(theta);
                    noise[b, i + 1] = NumOps.FromDouble(z2);
                }
            }
        }

        return noise;
    }

    /// <summary>
    /// Updates network parameters using Adam optimizer.
    /// </summary>
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
        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
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

            var momentumCorrected = NumOps.Divide(_momentum[i], NumOps.Subtract(NumOps.One, _beta1Power));
            var secondMomentCorrected = NumOps.Divide(_secondMoment[i], NumOps.Subtract(NumOps.One, _beta2Power));

            var adaptiveLR = NumOps.Divide(
                learningRate,
                NumOps.Add(NumOps.Sqrt(secondMomentCorrected), epsilon)
            );

            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(adaptiveLR, momentumCorrected)
            );
        }

        _beta1Power = NumOps.Multiply(_beta1Power, beta1);
        _beta2Power = NumOps.Multiply(_beta2Power, beta2);
        _currentLearningRate *= _learningRateDecay;

        network.UpdateParameters(updatedParameters);
    }

    protected override void InitializeLayers()
    {
        // AC-GAN doesn't use layers directly
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Simplified training interface
        int batchSize = expectedOutput.Shape[0];
        var labels = CreateOneHotLabels(batchSize, 0);
        TrainStep(expectedOutput, labels, input, labels);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.AuxiliaryClassifierGAN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "NumClasses", _numClasses }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_currentLearningRate);
        writer.Write(_numClasses);

        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _currentLearningRate = reader.ReadDouble();
        _numClasses = reader.ReadInt32();

        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int discriminatorDataLength = reader.ReadInt32();
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ACGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            _numClasses,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate);
    }
}
