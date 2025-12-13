using System.IO;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Conditional Generative Adversarial Network (cGAN), which generates data conditioned
/// on additional information such as class labels, attributes, or other contextual data.
/// </summary>
/// <remarks>
/// <para>
/// Conditional GANs extend the basic GAN framework by:
/// - Conditioning both the generator and discriminator on additional information
/// - Allowing controlled generation (e.g., "generate a digit 7")
/// - Enabling class-conditional image synthesis
/// - Providing explicit control over the generated output characteristics
/// </para>
/// <para><b>For Beginners:</b> cGAN lets you control what kind of image is generated.
///
/// Key features:
/// - You can specify what you want to generate (e.g., "cat" vs. "dog")
/// - Both the generator and discriminator see the conditioning information
/// - Generator: "Given this label, create a matching image"
/// - Discriminator: "Is this image both real AND matching the label?"
///
/// Example use cases:
/// - Generate a specific digit (0-9) in MNIST
/// - Create images of specific object classes
/// - Generate faces with specific attributes (smiling, glasses, etc.)
///
/// Reference: Mirza and Osindero, "Conditional Generative Adversarial Nets" (2014)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ConditionalGAN<T> : NeuralNetworkBase<T>
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
    private readonly double _learningRateDecay;
    private readonly List<T> _generatorLosses = [];

    /// <summary>
    /// The number of condition classes/categories.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This represents the number of distinct conditioning values (e.g., 10 for MNIST digits,
    /// 1000 for ImageNet classes). The conditioning information is typically provided as
    /// one-hot encoded vectors of this size.
    /// </para>
    /// <para><b>For Beginners:</b> The number of different types of things you can generate.
    ///
    /// Examples:
    /// - MNIST digits: 10 classes (0-9)
    /// - CIFAR-10: 10 object classes
    /// - ImageNet: 1000 object classes
    /// - Custom dataset: however many categories you have
    /// </para>
    /// </remarks>
    private int _numConditionClasses;

    /// <summary>
    /// Gets the generator network that creates synthetic data conditioned on labels.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the discriminator network that evaluates image-label pairs.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Creates the combined ConditionalGAN architecture with correct dimension handling.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateConditionalGANArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numConditionClasses,
        InputType inputType)
    {
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Medium,
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
            complexity: NetworkComplexity.Medium,
            inputSize: generatorArchitecture.InputSize + numConditionClasses,
            outputSize: discriminatorArchitecture.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.</param>
    /// <param name="numConditionClasses">The number of conditioning classes/categories.</param>
    /// <param name="inputType">The type of input the cGAN will process.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">The initial learning rate. Default is 0.0002.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a conditional GAN where both the generator and discriminator
    /// receive conditioning information. The generator takes noise concatenated with a
    /// condition vector, and the discriminator takes an image concatenated with the same
    /// condition vector.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a GAN that can generate specific types of images.
    ///
    /// Parameters:
    /// - generatorArchitecture: How the generator network is structured
    /// - discriminatorArchitecture: How the discriminator network is structured
    /// - numConditionClasses: How many different types/classes you have
    /// - inputType: What kind of data (usually images)
    /// - initialLearningRate: How fast the networks learn
    /// </para>
    /// </remarks>
    public ConditionalGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numConditionClasses,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0002)
        : base(CreateConditionalGANArchitecture(generatorArchitecture, discriminatorArchitecture, numConditionClasses, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        // Input validation
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture));
        }

        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture));
        }

        if (numConditionClasses <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numConditionClasses), numConditionClasses, "Number of condition classes must be positive.");
        }

        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }

        _numConditionClasses = numConditionClasses;
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
    /// Performs one training step for the conditional GAN.
    /// </summary>
    /// <param name="realImages">A tensor containing real images.</param>
    /// <param name="conditions">A tensor containing conditioning labels (one-hot encoded).</param>
    /// <param name="noise">A tensor containing random noise for the generator.</param>
    /// <returns>A tuple containing the discriminator and generator loss values.</returns>
    /// <remarks>
    /// <para>
    /// This method trains both the generator and discriminator with conditioning information:
    /// 1. Train discriminator on real images with their true labels
    /// 2. Train discriminator on fake images with the generator's conditioning labels
    /// 3. Train generator to create images that fool the discriminator for the given conditions
    /// </para>
    /// <para><b>For Beginners:</b> One training round for conditional GAN.
    ///
    /// The training process:
    /// - Discriminator learns to verify image-label pairs are correct
    /// - Generator learns to create images matching the specified labels
    /// - Both networks use the conditioning information to guide learning
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> conditions,
        Tensor<T> noise)
    {
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        int batchSize = realImages.Shape[0];

        // ----- Train Discriminator -----

        // Concatenate noise with conditions for generator input
        Tensor<T> generatorInput = ConcatenateTensors(noise, conditions);

        // Generate fake images conditioned on the labels
        Tensor<T> fakeImages = Generator.Predict(generatorInput);

        // Create labels
        Tensor<T> realLabels = CreateLabelTensor(batchSize, NumOps.One);
        Tensor<T> fakeLabels = CreateLabelTensor(batchSize, NumOps.Zero);

        // Train discriminator on real images with conditions
        Tensor<T> realImagesWithConditions = ConcatenateImageAndCondition(realImages, conditions);
        T realLoss = TrainDiscriminatorBatch(realImagesWithConditions, realLabels);

        // Train discriminator on fake images with conditions
        Tensor<T> fakeImagesWithConditions = ConcatenateImageAndCondition(fakeImages, conditions);
        T fakeLoss = TrainDiscriminatorBatch(fakeImagesWithConditions, fakeLabels);

        // Total discriminator loss
        T discriminatorLoss = NumOps.Add(realLoss, fakeLoss);
        discriminatorLoss = NumOps.Divide(discriminatorLoss, NumOps.FromDouble(2.0));

        // ----- Train Generator -----

        // Generate new fake images
        Tensor<T> newGeneratorInput = ConcatenateTensors(noise, conditions);
        Tensor<T> newFakeImages = Generator.Predict(newGeneratorInput);

        // For generator training, we want discriminator to think fake images are real
        Tensor<T> allRealLabels = CreateLabelTensor(batchSize, NumOps.One);

        // Concatenate with conditions
        Tensor<T> newFakeImagesWithConditions = ConcatenateImageAndCondition(newFakeImages, conditions);

        // Train generator
        T generatorLoss = TrainGeneratorBatch(newGeneratorInput, newFakeImagesWithConditions, allRealLabels);

        // Track losses
        _generatorLosses.Add(generatorLoss);
        if (_generatorLosses.Count > 100)
        {
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Concatenates noise and condition vectors.
    /// </summary>
    /// <param name="noise">The noise tensor.</param>
    /// <param name="conditions">The condition tensor.</param>
    /// <returns>A tensor with noise and conditions concatenated along the feature dimension.</returns>
    /// <remarks>
    /// <para>
    /// This combines the random noise with the conditioning information to create the
    /// generator's input. For example, if noise is [batch, 100] and conditions is [batch, 10],
    /// the result will be [batch, 110].
    /// </para>
    /// <para><b>For Beginners:</b> This combines random noise with the label information.
    ///
    /// Example:
    /// - Noise: random numbers that provide variety
    /// - Condition: the label for what to generate (e.g., "digit 7")
    /// - Combined: random numbers + "digit 7" → tells generator to create a random digit 7
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenateTensors(Tensor<T> noise, Tensor<T> conditions)
    {
        int batchSize = noise.Shape[0];
        int noiseSize = noise.Shape[1];
        int conditionSize = conditions.Shape[1];

        var result = new Tensor<T>(new int[] { batchSize, noiseSize + conditionSize });

        for (int b = 0; b < batchSize; b++)
        {
            // Copy noise
            for (int i = 0; i < noiseSize; i++)
            {
                result[b, i] = noise[b, i];
            }

            // Copy conditions
            for (int i = 0; i < conditionSize; i++)
            {
                result[b, noiseSize + i] = conditions[b, i];
            }
        }

        return result;
    }

    /// <summary>
    /// Concatenates images with condition vectors for discriminator input.
    /// </summary>
    /// <param name="images">The image tensor.</param>
    /// <param name="conditions">The condition tensor.</param>
    /// <returns>A tensor with images and conditions combined.</returns>
    /// <remarks>
    /// <para>
    /// This combines images with their conditioning labels for the discriminator.
    /// The conditioning information is typically replicated spatially or appended
    /// as additional channels.
    /// </para>
    /// <para><b>For Beginners:</b> This attaches label information to images.
    ///
    /// Why:
    /// - Discriminator needs to know "Is this image real AND does it match the label?"
    /// - By attaching the label, discriminator can check both authenticity and correctness
    /// - Example: Given image of "7" and label "7" → check if real and if it's actually a 7
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenateImageAndCondition(Tensor<T> images, Tensor<T> conditions)
    {
        int batchSize = images.Shape[0];
        int imageSize = images.Length / batchSize;
        int conditionSize = conditions.Shape[1];

        // Create result tensor with space for both image and condition
        var result = new Tensor<T>(new int[] { batchSize, imageSize + conditionSize });

        for (int b = 0; b < batchSize; b++)
        {
            // Copy image data
            for (int i = 0; i < imageSize; i++)
            {
                result[b, i] = images.GetFlatIndexValue(b * imageSize + i);
            }

            // Append condition
            for (int i = 0; i < conditionSize; i++)
            {
                result[b, imageSize + i] = conditions[b, i];
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a label tensor filled with a specified value.
    /// </summary>
    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        var shape = new int[] { batchSize, 1 };
        var tensor = new Tensor<T>(shape);

        for (int i = 0; i < batchSize; i++)
        {
            tensor[i, 0] = value;
        }

        return tensor;
    }

    /// <summary>
    /// Trains the discriminator on a batch of images.
    /// </summary>
    private T TrainDiscriminatorBatch(Tensor<T> images, Tensor<T> labels)
    {
        Discriminator.SetTrainingMode(true);

        // Forward pass
        var predictions = Discriminator.Predict(images);

        // Calculate loss
        var loss = CalculateBatchLoss(predictions, labels);

        // Calculate gradients
        var outputGradients = CalculateBatchGradients(predictions, labels);

        // Backpropagate
        Discriminator.Backpropagate(outputGradients);

        // Update parameters
        UpdateDiscriminatorParameters();

        return loss;
    }

    /// <summary>
    /// Trains the generator on a batch.
    /// </summary>
    private T TrainGeneratorBatch(Tensor<T> generatorInput, Tensor<T> fakeImagesWithConditions, Tensor<T> targetLabels)
    {
        Generator.SetTrainingMode(true);
        // Keep Discriminator in training mode for backpropagation (required for Backpropagate to work)
        // We just won't update its parameters
        Discriminator.SetTrainingMode(true);

        // Get discriminator output
        var discriminatorOutput = Discriminator.Predict(fakeImagesWithConditions);

        // Calculate loss
        var loss = CalculateBatchLoss(discriminatorOutput, targetLabels);

        // Calculate gradients
        var outputGradients = CalculateBatchGradients(discriminatorOutput, targetLabels);

        // Backpropagate through discriminator to get input gradients (but don't update discriminator weights)
        var discriminatorInputGradients = Discriminator.Backpropagate(outputGradients);

        // Extract gradients for the image part (not the condition part)
        int batchSize = generatorInput.Shape[0];
        int imageSize = discriminatorInputGradients.Length / batchSize - _numConditionClasses;

        var generatorGradients = new Tensor<T>(new int[] { batchSize, imageSize });
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < imageSize; i++)
            {
                generatorGradients.SetFlatIndex(b * imageSize + i, discriminatorInputGradients.GetFlatIndexValue(b * (imageSize + _numConditionClasses) + i));
            }
        }

        // Backpropagate through generator
        Generator.Backpropagate(generatorGradients);

        // Update generator
        UpdateGeneratorParameters();

        Discriminator.SetTrainingMode(true);

        return loss;
    }

    /// <summary>
    /// Calculates batch loss using binary cross-entropy.
    /// </summary>
    private T CalculateBatchLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        int batchSize = predictions.Shape[0];
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);

        for (int i = 0; i < batchSize; i++)
        {
            T prediction = predictions[i, 0];
            T target = targets[i, 0];

            T logP = NumOps.Log(NumOps.Add(prediction, epsilon));
            T logOneMinusP = NumOps.Log(NumOps.Add(NumOps.Subtract(NumOps.One, prediction), epsilon));

            T termOne = NumOps.Multiply(target, logP);
            T termTwo = NumOps.Multiply(NumOps.Subtract(NumOps.One, target), logOneMinusP);

            T sampleLoss = NumOps.Negate(NumOps.Add(termOne, termTwo));
            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates gradients for backpropagation.
    /// </summary>
    private Tensor<T> CalculateBatchGradients(Tensor<T> predictions, Tensor<T> targets)
    {
        int batchSize = predictions.Shape[0];
        var gradients = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            gradients[i, 0] = NumOps.Subtract(predictions[i, 0], targets[i, 0]);
        }

        return gradients;
    }

    /// <summary>
    /// Updates generator parameters using vectorized Adam optimizer with generator-specific state.
    /// Uses SIMD-accelerated Engine operations for CPU/GPU acceleration.
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

        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
        var epsilon = NumOps.FromDouble(1e-8);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);

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
        var oneMinusBeta1Power = NumOps.Subtract(NumOps.One, _genBeta1Power);
        var oneMinusBeta2Power = NumOps.Subtract(NumOps.One, _genBeta2Power);
        var mCorrected = (Vector<T>)Engine.Divide(_genMomentum, oneMinusBeta1Power);
        var vCorrected = (Vector<T>)Engine.Divide(_genSecondMoment, oneMinusBeta2Power);

        // Vectorized parameter update: p = p - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(vCorrected);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);

        var learningRate = NumOps.FromDouble(_genCurrentLearningRate);
        var lrTimesM = (Vector<T>)Engine.Multiply(mCorrected, learningRate);
        var update = (Vector<T>)Engine.Divide(lrTimesM, sqrtVPlusEps);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        // Update beta powers for next iteration
        _genBeta1Power = NumOps.Multiply(_genBeta1Power, beta1);
        _genBeta2Power = NumOps.Multiply(_genBeta2Power, beta2);
        _genCurrentLearningRate *= _learningRateDecay;

        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates discriminator parameters using vectorized Adam optimizer with discriminator-specific state.
    /// Uses SIMD-accelerated Engine operations for CPU/GPU acceleration.
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

        var beta1 = NumOps.FromDouble(0.5);
        var beta2 = NumOps.FromDouble(0.999);
        var epsilon = NumOps.FromDouble(1e-8);
        var oneMinusBeta1 = NumOps.Subtract(NumOps.One, beta1);
        var oneMinusBeta2 = NumOps.Subtract(NumOps.One, beta2);

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
        var oneMinusBeta1Power = NumOps.Subtract(NumOps.One, _discBeta1Power);
        var oneMinusBeta2Power = NumOps.Subtract(NumOps.One, _discBeta2Power);
        var mCorrected = (Vector<T>)Engine.Divide(_discMomentum, oneMinusBeta1Power);
        var vCorrected = (Vector<T>)Engine.Divide(_discSecondMoment, oneMinusBeta2Power);

        // Vectorized parameter update: p = p - lr * m_corrected / (sqrt(v_corrected) + epsilon)
        var sqrtV = (Vector<T>)Engine.Sqrt(vCorrected);
        var epsilonVec = Vector<T>.CreateDefault(sqrtV.Length, epsilon);
        var sqrtVPlusEps = (Vector<T>)Engine.Add(sqrtV, epsilonVec);

        var learningRate = NumOps.FromDouble(_discCurrentLearningRate);
        var lrTimesM = (Vector<T>)Engine.Multiply(mCorrected, learningRate);
        var update = (Vector<T>)Engine.Divide(lrTimesM, sqrtVPlusEps);
        var updatedParameters = (Vector<T>)Engine.Subtract(parameters, update);

        // Update beta powers for next iteration
        _discBeta1Power = NumOps.Multiply(_discBeta1Power, beta1);
        _discBeta2Power = NumOps.Multiply(_discBeta2Power, beta2);
        _discCurrentLearningRate *= _learningRateDecay;

        Discriminator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Generates images conditioned on specific labels.
    /// </summary>
    /// <param name="noise">Random noise tensor.</param>
    /// <param name="conditions">Conditioning labels (one-hot encoded).</param>
    /// <returns>Generated images matching the conditioning labels.</returns>
    /// <remarks>
    /// <para>
    /// This method generates images based on the provided conditioning information.
    /// The conditions are typically one-hot encoded class labels.
    /// </para>
    /// <para><b>For Beginners:</b> Create images of a specific type.
    ///
    /// Example usage:
    /// - noise: random starting point
    /// - conditions: [0,0,0,0,0,0,0,1,0,0] (means "generate a 7")
    /// - result: image of a digit 7
    /// </para>
    /// </remarks>
    public Tensor<T> GenerateConditional(Tensor<T> noise, Tensor<T> conditions)
    {
        Generator.SetTrainingMode(false);
        var input = ConcatenateTensors(noise, conditions);
        return Generator.Predict(input);
    }

    /// <summary>
    /// Generates random noise tensor using vectorized Gaussian noise generation.
    /// Uses SIMD-accelerated Engine operations for CPU/GPU acceleration.
    /// </summary>
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

        // Use vectorized Gaussian noise generation
        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        return Tensor<T>.FromVector(noiseVector, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Creates a one-hot encoded condition tensor.
    /// </summary>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="classIndex">The class index to encode.</param>
    /// <returns>A one-hot encoded tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates label vectors for conditioning.
    ///
    /// Example (with 10 classes):
    /// - classIndex = 7
    /// - result = [0,0,0,0,0,0,0,1,0,0]
    /// - The "1" is at position 7, indicating class 7
    /// </para>
    /// </remarks>
    public Tensor<T> CreateOneHotCondition(int batchSize, int classIndex)
    {
        var conditions = new Tensor<T>(new int[] { batchSize, _numConditionClasses });

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < _numConditionClasses; c++)
            {
                conditions[b, c] = c == classIndex ? NumOps.One : NumOps.Zero;
            }
        }

        return conditions;
    }

    protected override void InitializeLayers()
    {
        // cGAN doesn't use layers directly
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For cGAN, input should be noise + conditions concatenated
        return Generator.Predict(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // This simplified version assumes input contains noise and expectedOutput contains real images
        // In practice, you'd need to extract conditions from somewhere
        int batchSize = expectedOutput.Shape[0];
        var conditions = CreateOneHotCondition(batchSize, 0); // Placeholder
        TrainStep(expectedOutput, conditions, input);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ConditionalGAN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "NumConditionClasses", _numConditionClasses }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_genCurrentLearningRate);
        writer.Write(_discCurrentLearningRate);
        writer.Write(_numConditionClasses);

        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);

        // Serialize generator optimizer state
        SerializationHelper<T>.SerializeVector(writer, _genMomentum);
        SerializationHelper<T>.SerializeVector(writer, _genSecondMoment);

        // Serialize discriminator optimizer state
        SerializationHelper<T>.SerializeVector(writer, _discMomentum);
        SerializationHelper<T>.SerializeVector(writer, _discSecondMoment);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _genCurrentLearningRate = reader.ReadDouble();
        _discCurrentLearningRate = reader.ReadDouble();
        _numConditionClasses = reader.ReadInt32();

        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int discriminatorDataLength = reader.ReadInt32();
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);

        // Deserialize generator optimizer state
        _genMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _genSecondMoment = SerializationHelper<T>.DeserializeVector(reader);

        // Deserialize discriminator optimizer state
        _discMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _discSecondMoment = SerializationHelper<T>.DeserializeVector(reader);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ConditionalGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            _numConditionClasses,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate);
    }

    /// <summary>
    /// Updates the parameters of all networks in the ConditionalGAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();

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
