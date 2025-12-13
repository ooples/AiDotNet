using System.IO;
using AiDotNet.Helpers;

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

    private readonly double _initialLearningRate;
    private double _learningRateDecay;
    private readonly List<T> _generatorLosses = [];
    private readonly List<T> _discriminatorLosses = [];

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

    private readonly ILossFunction<T> _lossFunction;

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
            InputType.OneDimensional,  // Base GAN takes latent vector input (1D)
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            generatorArchitecture.InputSize + numClasses,
            0, 0, 1,  // inputHeight, inputWidth=0 for 1D, inputDepth=1 required
            discriminatorArchitecture.OutputSize,  // outputSize
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        // Input validation
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }

        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");
        }

        if (numClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), numClasses, "Number of classes must be positive.");
        }

        if (discriminatorArchitecture.OutputSize != 1 + numClasses)
        {
            throw new ArgumentException(
                $"Discriminator output size must be 1 + numClasses ({1 + numClasses}), but was {discriminatorArchitecture.OutputSize}.",
                nameof(discriminatorArchitecture));
        }

        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }

        _numClasses = numClasses;
        _initialLearningRate = initialLearningRate;
        _genCurrentLearningRate = initialLearningRate;
        _discCurrentLearningRate = initialLearningRate;
        _learningRateDecay = 0.9999;

        // Initialize generator optimizer state
        _genBeta1Power = NumOps.One;
        _genBeta2Power = NumOps.One;
        _genMomentum = Vector<T>.Empty();
        _genSecondMoment = Vector<T>.Empty();

        // Initialize discriminator optimizer state
        _discBeta1Power = NumOps.One;
        _discBeta2Power = NumOps.One;
        _discMomentum = Vector<T>.Empty();
        _discSecondMoment = Vector<T>.Empty();

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
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
    /// <exception cref="ArgumentNullException">Thrown when any input tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when tensor shapes are incompatible.</exception>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> realLabels,
        Tensor<T> noise,
        Tensor<T> fakeLabels)
    {
        // Input validation
        if (realImages is null)
        {
            throw new ArgumentNullException(nameof(realImages), "Real images tensor cannot be null.");
        }

        if (realLabels is null)
        {
            throw new ArgumentNullException(nameof(realLabels), "Real labels tensor cannot be null.");
        }

        if (noise is null)
        {
            throw new ArgumentNullException(nameof(noise), "Noise tensor cannot be null.");
        }

        if (fakeLabels is null)
        {
            throw new ArgumentNullException(nameof(fakeLabels), "Fake labels tensor cannot be null.");
        }

        int batchSize = realImages.Shape[0];

        if (realLabels.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Real labels batch size ({realLabels.Shape[0]}) must match real images batch size ({batchSize}).",
                nameof(realLabels));
        }

        if (noise.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Noise batch size ({noise.Shape[0]}) must match real images batch size ({batchSize}).",
                nameof(noise));
        }

        if (fakeLabels.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Fake labels batch size ({fakeLabels.Shape[0]}) must match real images batch size ({batchSize}).",
                nameof(fakeLabels));
        }

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

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
        UpdateDiscriminatorParameters();

        // Train discriminator on fake images
        var fakeDiscOutput = Discriminator.Predict(fakeImages);
        T fakeAuthLoss = CalculateAuthenticityLoss(fakeDiscOutput, isReal: false, batchSize);
        T fakeClassLoss = CalculateClassificationLoss(fakeDiscOutput, fakeLabels, batchSize);
        T fakeLoss = NumOps.Add(fakeAuthLoss, fakeClassLoss);

        // Backpropagate for fake images
        var fakeGradients = CalculateDiscriminatorGradients(fakeDiscOutput, fakeLabels, isReal: false, batchSize);
        Discriminator.Backpropagate(fakeGradients);
        UpdateDiscriminatorParameters();

        // Total discriminator loss
        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLoss), NumOps.FromDouble(2.0));

        // ----- Train Generator -----

        Generator.SetTrainingMode(true);
        // Keep Discriminator in training mode - required for backpropagation
        // We just don't call UpdateDiscriminatorParameters() during generator training

        // Generate new fake images
        var newGeneratorInput = ConcatenateTensors(noise, fakeLabels);
        var newFakeImages = Generator.Predict(newGeneratorInput);

        // Get discriminator output
        var genDiscOutput = Discriminator.Predict(newFakeImages);

        // For generator, we want discriminator to think images are real AND match the class
        T genAuthLoss = CalculateAuthenticityLoss(genDiscOutput, isReal: true, batchSize);
        T genClassLoss = CalculateClassificationLoss(genDiscOutput, fakeLabels, batchSize);
        T generatorLoss = NumOps.Add(genAuthLoss, genClassLoss);

        // Backpropagate through discriminator to get input gradients, then through generator
        var genGradients = CalculateDiscriminatorGradients(genDiscOutput, fakeLabels, isReal: true, batchSize);
        var discInputGradients = Discriminator.BackwardWithInputGradient(genGradients);
        Generator.Backward(discInputGradients);
        UpdateGeneratorParameters();

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
    /// <remarks>
    /// This method assumes the discriminator outputs probabilities (after sigmoid activation),
    /// not raw logits. If the discriminator outputs logits, apply sigmoid activation first
    /// or use a numerically stable BCEWithLogits loss function.
    /// </remarks>
    private T CalculateAuthenticityLoss(Tensor<T> discOutput, bool isReal, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);
        T target = isReal ? NumOps.One : NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            // First output is authenticity probability (should be in [0,1] range after sigmoid)
            // Clamp prediction to valid probability range for numerical stability
            T prediction = MathHelper.Clamp(discOutput[i, 0], epsilon, NumOps.Subtract(NumOps.One, epsilon));

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
    /// <remarks>
    /// This method assumes the discriminator outputs probabilities (after softmax activation)
    /// for class predictions, not raw logits. Values should be in [0,1] range.
    /// </remarks>
    private T CalculateClassificationLoss(Tensor<T> discOutput, Tensor<T> labels, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < batchSize; i++)
        {
            // Class outputs start from index 1 (index 0 is authenticity)
            for (int c = 0; c < _numClasses; c++)
            {
                // Clamp prediction to valid probability range for numerical stability
                T prediction = MathHelper.Clamp(discOutput[i, 1 + c], epsilon, NumOps.Subtract(NumOps.One, epsilon));
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
        if (noise.Shape.Length != 2 || labels.Shape.Length != 2)
            throw new ArgumentException("noise and labels must be rank-2 tensors [batch, features].");

        int batchSize = noise.Shape[0];
        if (labels.Shape[0] != batchSize)
            throw new ArgumentException($"noise and labels must have the same batch size. noise: {batchSize}, labels: {labels.Shape[0]}");

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
    /// Generates random noise tensor using vectorized Gaussian noise generation.
    /// Uses Engine.GenerateGaussianNoise for SIMD/GPU acceleration.
    /// </summary>
    /// <param name="batchSize">Number of noise samples in the batch.</param>
    /// <param name="noiseSize">Dimension of each noise vector.</param>
    /// <returns>Tensor of shape [batchSize, noiseSize] filled with standard Gaussian noise.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when batchSize or noiseSize is not positive.</exception>
    public Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be positive.");
        }

        if (noiseSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseSize), noiseSize, "Noise size must be positive.");
        }

        var totalElements = batchSize * noiseSize;
        var mean = NumOps.Zero;
        var stddev = NumOps.One;

        // Use Engine's vectorized Gaussian noise generation for SIMD/GPU acceleration
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        // Reshape to [batchSize, noiseSize]
        return Tensor<T>.FromVector(noiseVector, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Updates generator parameters using vectorized Adam optimizer.
    /// Uses Engine operations for SIMD/GPU acceleration.
    /// </summary>
    private void UpdateGeneratorParameters()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Initialize generator optimizer state if needed
        if (_genMomentum.Length != parameters.Length)
        {
            _genMomentum = new Vector<T>(parameters.Length);
            _genMomentum.Fill(NumOps.Zero);
        }

        if (_genSecondMoment.Length != parameters.Length)
        {
            _genSecondMoment = new Vector<T>(parameters.Length);
            _genSecondMoment.Fill(NumOps.Zero);
        }

        // Adam optimizer parameters (beta1=0.5 for AC-GAN as per typical practice)
        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        var epsilon = NumOps.FromDouble(1e-8);
        var learningRate = NumOps.FromDouble(_genCurrentLearningRate);

        // Update beta powers for bias correction
        _genBeta1Power = NumOps.Multiply(_genBeta1Power, beta1);
        _genBeta2Power = NumOps.Multiply(_genBeta2Power, beta2);

        // Vectorized momentum update: m = beta1 * m + (1 - beta1) * g
        var mScaled = (Vector<T>)Engine.Multiply(_genMomentum, beta1);
        var gScaled = (Vector<T>)Engine.Multiply(gradients, oneMinusBeta1);
        _genMomentum = (Vector<T>)Engine.Add(mScaled, gScaled);

        // Vectorized second moment update: v = beta2 * v + (1 - beta2) * g^2
        var vScaled = (Vector<T>)Engine.Multiply(_genSecondMoment, beta2);
        var gSquared = (Vector<T>)Engine.Multiply(gradients, gradients);
        var gSquaredScaled = (Vector<T>)Engine.Multiply(gSquared, oneMinusBeta2);
        _genSecondMoment = (Vector<T>)Engine.Add(vScaled, gSquaredScaled);

        // Bias correction
        var biasCorrection1 = NumOps.Subtract(NumOps.One, _genBeta1Power);
        var biasCorrection2 = NumOps.Subtract(NumOps.One, _genBeta2Power);
        var mCorrected = (Vector<T>)Engine.Divide(_genMomentum, biasCorrection1);
        var vCorrected = (Vector<T>)Engine.Divide(_genSecondMoment, biasCorrection2);

        // Vectorized parameter update: p = p - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(vCorrected);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);
        var adaptiveGradient = (Vector<T>)Engine.Divide(mCorrected, sqrtVPlusEps);
        var update = (Vector<T>)Engine.Multiply(adaptiveGradient, learningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        _genCurrentLearningRate *= _learningRateDecay;

        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates discriminator parameters using vectorized Adam optimizer.
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

        // Adam optimizer parameters (beta1=0.5 for AC-GAN as per typical practice)
        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);
        var epsilon = NumOps.FromDouble(1e-8);
        var learningRate = NumOps.FromDouble(_discCurrentLearningRate);

        // Update beta powers for bias correction
        _discBeta1Power = NumOps.Multiply(_discBeta1Power, beta1);
        _discBeta2Power = NumOps.Multiply(_discBeta2Power, beta2);

        // Vectorized momentum update: m = beta1 * m + (1 - beta1) * g
        var mScaled = (Vector<T>)Engine.Multiply(_discMomentum, beta1);
        var gScaled = (Vector<T>)Engine.Multiply(gradients, oneMinusBeta1);
        _discMomentum = (Vector<T>)Engine.Add(mScaled, gScaled);

        // Vectorized second moment update: v = beta2 * v + (1 - beta2) * g^2
        var vScaled = (Vector<T>)Engine.Multiply(_discSecondMoment, beta2);
        var gSquared = (Vector<T>)Engine.Multiply(gradients, gradients);
        var gSquaredScaled = (Vector<T>)Engine.Multiply(gSquared, oneMinusBeta2);
        _discSecondMoment = (Vector<T>)Engine.Add(vScaled, gSquaredScaled);

        // Bias correction
        var biasCorrection1 = NumOps.Subtract(NumOps.One, _discBeta1Power);
        var biasCorrection2 = NumOps.Subtract(NumOps.One, _discBeta2Power);
        var mCorrected = (Vector<T>)Engine.Divide(_discMomentum, biasCorrection1);
        var vCorrected = (Vector<T>)Engine.Divide(_discSecondMoment, biasCorrection2);

        // Vectorized parameter update: p = p - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(vCorrected);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);
        var adaptiveGradient = (Vector<T>)Engine.Divide(mCorrected, sqrtVPlusEps);
        var update = (Vector<T>)Engine.Multiply(adaptiveGradient, learningRate);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        _discCurrentLearningRate *= _learningRateDecay;

        Discriminator.UpdateParameters(updatedParameters);
    }

    protected override void InitializeLayers()
    {
        // AC-GAN doesn't use layers directly
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    /// <summary>
    /// Simplified training interface. For proper GAN training, use TrainStep directly.
    /// </summary>
    /// <remarks>
    /// This method maps inputs to TrainStep but may not provide optimal GAN training semantics.
    /// Parameters are interpreted as: expectedOutput = real images, input = noise.
    /// All samples are assigned to class 0. For multi-class training, use TrainStep.
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Simplified training interface - maps to TrainStep with class 0
        // For proper GAN training, use TrainStep directly with appropriate labels
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
        writer.Write(_genCurrentLearningRate);
        writer.Write(_discCurrentLearningRate);
        writer.Write(_numClasses);
        writer.Write(_learningRateDecay);

        // Serialize generator optimizer state
        SerializeOptimizerState(writer, _genMomentum, _genSecondMoment, _genBeta1Power, _genBeta2Power);

        // Serialize discriminator optimizer state
        SerializeOptimizerState(writer, _discMomentum, _discSecondMoment, _discBeta1Power, _discBeta2Power);

        // Serialize loss histories
        writer.Write(_generatorLosses.Count);
        foreach (var loss in _generatorLosses)
            writer.Write(NumOps.ToDouble(loss));

        writer.Write(_discriminatorLosses.Count);
        foreach (var loss in _discriminatorLosses)
            writer.Write(NumOps.ToDouble(loss));

        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);
    }

    private void SerializeOptimizerState(BinaryWriter writer, Vector<T> momentum, Vector<T> secondMoment, T beta1Power, T beta2Power)
    {
        // Write momentum vector
        bool hasMomentum = momentum.Length > 0;
        writer.Write(hasMomentum);
        if (hasMomentum)
        {
            writer.Write(momentum.Length);
            for (int i = 0; i < momentum.Length; i++)
                writer.Write(NumOps.ToDouble(momentum[i]));
        }

        // Write second moment vector
        bool hasSecondMoment = secondMoment.Length > 0;
        writer.Write(hasSecondMoment);
        if (hasSecondMoment)
        {
            writer.Write(secondMoment.Length);
            for (int i = 0; i < secondMoment.Length; i++)
                writer.Write(NumOps.ToDouble(secondMoment[i]));
        }

        // Write beta powers
        writer.Write(NumOps.ToDouble(beta1Power));
        writer.Write(NumOps.ToDouble(beta2Power));
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _genCurrentLearningRate = reader.ReadDouble();
        _discCurrentLearningRate = reader.ReadDouble();
        _numClasses = reader.ReadInt32();
        _learningRateDecay = reader.ReadDouble();

        // Deserialize generator optimizer state
        (_genMomentum, _genSecondMoment, _genBeta1Power, _genBeta2Power) = DeserializeOptimizerState(reader);

        // Deserialize discriminator optimizer state
        (_discMomentum, _discSecondMoment, _discBeta1Power, _discBeta2Power) = DeserializeOptimizerState(reader);

        // Deserialize loss histories
        _generatorLosses.Clear();
        int genLossCount = reader.ReadInt32();
        for (int i = 0; i < genLossCount; i++)
            _generatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));

        _discriminatorLosses.Clear();
        int discLossCount = reader.ReadInt32();
        for (int i = 0; i < discLossCount; i++)
            _discriminatorLosses.Add(NumOps.FromDouble(reader.ReadDouble()));

        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int discriminatorDataLength = reader.ReadInt32();
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);
    }

    private (Vector<T> momentum, Vector<T> secondMoment, T beta1Power, T beta2Power) DeserializeOptimizerState(BinaryReader reader)
    {
        // Read momentum vector
        Vector<T> momentum;
        bool hasMomentum = reader.ReadBoolean();
        if (hasMomentum)
        {
            int length = reader.ReadInt32();
            momentum = new Vector<T>(length);
            for (int i = 0; i < length; i++)
                momentum[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        else
        {
            momentum = Vector<T>.Empty();
        }

        // Read second moment vector
        Vector<T> secondMoment;
        bool hasSecondMoment = reader.ReadBoolean();
        if (hasSecondMoment)
        {
            int length = reader.ReadInt32();
            secondMoment = new Vector<T>(length);
            for (int i = 0; i < length; i++)
                secondMoment[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        else
        {
            secondMoment = Vector<T>.Empty();
        }

        // Read beta powers
        T beta1Power = NumOps.FromDouble(reader.ReadDouble());
        T beta2Power = NumOps.FromDouble(reader.ReadDouble());

        return (momentum, secondMoment, beta1Power, beta2Power);
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

    /// <summary>
    /// Updates the parameters of all networks in the ACGAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();
        int totalCount = generatorCount + discriminatorCount;

        if (parameters.Length < totalCount)
            throw new ArgumentException($"parameters vector length ({parameters.Length}) must be at least {totalCount} (generator: {generatorCount} + discriminator: {discriminatorCount}).", nameof(parameters));

        // Update Generator parameters
        var generatorParams = new Vector<T>(generatorCount);
        for (int i = 0; i < generatorCount; i++)
            generatorParams[i] = parameters[i];
        Generator.UpdateParameters(generatorParams);

        // Update Discriminator parameters
        var discriminatorParams = new Vector<T>(discriminatorCount);
        for (int i = 0; i < discriminatorCount; i++)
            discriminatorParams[i] = parameters[generatorCount + i];
        Discriminator.UpdateParameters(discriminatorParams);
    }
}
