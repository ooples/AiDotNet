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
    private readonly List<T> _generatorLosses = [];
    private readonly List<T> _discriminatorLosses = [];

    /// <summary>
    /// The number of classes for classification.
    /// </summary>
    private int _numClasses;

    /// <summary>
    /// The optimizer for the generator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;

    /// <summary>
    /// The optimizer for the discriminator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

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
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="discriminatorOptimizer">Optional optimizer for the discriminator. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    public ACGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numClasses,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            generatorArchitecture.InputSize + numClasses,
            0, 0, 1,
            discriminatorArchitecture.OutputSize,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
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

        _numClasses = numClasses;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType);

        // Initialize optimizers (default to Adam if not provided)
        _generatorOptimizer = generatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator);
        _discriminatorOptimizer = discriminatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator);

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
    public (T discriminatorLoss, T generatorLoss) TrainStep(
        Tensor<T> realImages,
        Tensor<T> realLabels,
        Tensor<T> noise,
        Tensor<T> fakeLabels)
    {
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
        UpdateDiscriminatorWithOptimizer();

        // Train discriminator on fake images
        var fakeDiscOutput = Discriminator.Predict(fakeImages);
        T fakeAuthLoss = CalculateAuthenticityLoss(fakeDiscOutput, isReal: false, batchSize);
        T fakeClassLoss = CalculateClassificationLoss(fakeDiscOutput, fakeLabels, batchSize);
        T fakeLoss = NumOps.Add(fakeAuthLoss, fakeClassLoss);

        // Backpropagate for fake images
        var fakeGradients = CalculateDiscriminatorGradients(fakeDiscOutput, fakeLabels, isReal: false, batchSize);
        Discriminator.Backpropagate(fakeGradients);
        UpdateDiscriminatorWithOptimizer();

        // Total discriminator loss
        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLoss), NumOps.FromDouble(2.0));

        // ----- Train Generator -----

        Generator.SetTrainingMode(true);

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
        UpdateGeneratorWithOptimizer();

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
    /// Updates generator parameters using the configured optimizer.
    /// </summary>
    private void UpdateGeneratorWithOptimizer()
    {
        var parameters = Generator.GetParameters();
        var gradients = Generator.GetParameterGradients();

        // Gradient clipping
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = NumOps.Multiply(gradients[i], scaleFactor);
            }
        }

        var updatedParameters = _generatorOptimizer.UpdateParameters(parameters, gradients);
        Generator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Updates discriminator parameters using the configured optimizer.
    /// </summary>
    private void UpdateDiscriminatorWithOptimizer()
    {
        var parameters = Discriminator.GetParameters();
        var gradients = Discriminator.GetParameterGradients();

        // Gradient clipping
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = NumOps.Multiply(gradients[i], scaleFactor);
            }
        }

        var updatedParameters = _discriminatorOptimizer.UpdateParameters(parameters, gradients);
        Discriminator.UpdateParameters(updatedParameters);
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
            T prediction = MathHelper.Clamp(discOutput[i, 0], epsilon, NumOps.Subtract(NumOps.One, epsilon));

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
            for (int c = 0; c < _numClasses; c++)
            {
                T prediction = MathHelper.Clamp(discOutput[i, 1 + c], epsilon, NumOps.Subtract(NumOps.One, epsilon));
                T target = labels[i, c];

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
            gradients[i, 0] = NumOps.Divide(
                NumOps.Subtract(discOutput[i, 0], authTarget),
                NumOps.FromDouble(batchSize)
            );

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
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                "Batch size must be positive.");
        }

        if (classIndex < 0 || classIndex >= _numClasses)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex), classIndex,
                $"Class index must be between 0 and {_numClasses - 1} (inclusive).");
        }

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

        var noiseVector = Engine.GenerateGaussianNoise<T>(totalElements, mean, stddev);

        return Tensor<T>.FromVector(noiseVector, [batchSize, noiseSize]);
    }

    /// <summary>
    /// Resets both optimizer states for a fresh training run.
    /// </summary>
    public void ResetOptimizerState()
    {
        _generatorOptimizer.Reset();
        _discriminatorOptimizer.Reset();
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
        writer.Write(_numClasses);

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

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numClasses = reader.ReadInt32();

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

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ACGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            _numClasses,
            Architecture.InputType,
            null, // Use default optimizer
            null, // Use default optimizer
            _lossFunction);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();
        int totalCount = generatorCount + discriminatorCount;

        if (parameters.Length != totalCount)
        {
            throw new ArgumentException(
                $"Parameters vector length ({parameters.Length}) must equal {totalCount} " +
                $"(generator: {generatorCount} + discriminator: {discriminatorCount}).",
                nameof(parameters));
        }

        var generatorParams = new Vector<T>(generatorCount);
        for (int i = 0; i < generatorCount; i++)
            generatorParams[i] = parameters[i];
        Generator.UpdateParameters(generatorParams);

        var discriminatorParams = new Vector<T>(discriminatorCount);
        for (int i = 0; i < discriminatorCount; i++)
            discriminatorParams[i] = parameters[generatorCount + i];
        Discriminator.UpdateParameters(discriminatorParams);
    }
}
