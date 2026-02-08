using System.IO;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

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
    private readonly ACGANOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly List<T> _generatorLosses = new List<T>();
    private readonly List<T> _discriminatorLosses = new List<T>();

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

    /// <summary>
    /// Gets the total number of trainable parameters in the ACGAN.
    /// </summary>
    public override int ParameterCount => Generator.GetParameterCount() + Discriminator.GetParameterCount();

    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="ACGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.
    /// Note: Output size should be 1 + numClasses (authenticity probability + class probabilities).
    /// All outputs must be in range (0, 1) - use sigmoid/softmax activations in the final layer.</param>
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
        ILossFunction<T>? lossFunction = null,
        ACGANOptions? options = null)
        : base(new NeuralNetworkArchitecture<T>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Medium,
            generatorArchitecture.InputSize + numClasses,
            0, 0, 1,
            discriminatorArchitecture.OutputSize,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(generatorArchitecture.TaskType))
    {
        _options = options ?? new ACGANOptions();
        Options = _options;

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

        // Validate label shape dimensions - must be 2D with correct class count
        if (realLabels.Shape.Length != 2 || realLabels.Shape[1] != _numClasses)
        {
            throw new ArgumentException(
                $"realLabels must be [batch,{_numClasses}], got [{string.Join(",", realLabels.Shape)}].",
                nameof(realLabels));
        }

        if (fakeLabels.Shape.Length != 2 || fakeLabels.Shape[1] != _numClasses)
        {
            throw new ArgumentException(
                $"fakeLabels must be [batch,{_numClasses}], got [{string.Join(",", fakeLabels.Shape)}].",
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
        var realDiscOutputRaw = Discriminator.Predict(realImages);
        var realDiscOutput = NormalizeToProbabilities(realDiscOutputRaw);
        T realAuthLoss = CalculateAuthenticityLoss(realDiscOutput, isReal: true, batchSize);
        T realClassLoss = CalculateClassificationLoss(realDiscOutput, realLabels, batchSize);
        T realLoss = NumOps.Add(realAuthLoss, realClassLoss);

        // Backpropagate for real images
        var realGradients = CalculateDiscriminatorGradients(realDiscOutput, realLabels, isReal: true, batchSize);
        Discriminator.Backpropagate(realGradients);
        UpdateDiscriminatorWithOptimizer();

        // Train discriminator on fake images
        var fakeDiscOutputRaw = Discriminator.Predict(fakeImages);
        var fakeDiscOutput = NormalizeToProbabilities(fakeDiscOutputRaw);
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
        var genDiscOutputRaw = Discriminator.Predict(newFakeImages);
        var genDiscOutput = NormalizeToProbabilities(genDiscOutputRaw);

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

        // Gradient clipping using vectorized operations
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = Engine.Multiply(gradients, scaleFactor);
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

        // Gradient clipping using vectorized operations
        var gradientNorm = gradients.L2Norm();
        var clipThreshold = NumOps.FromDouble(5.0);

        if (NumOps.GreaterThan(gradientNorm, clipThreshold))
        {
            var scaleFactor = NumOps.Divide(clipThreshold, gradientNorm);
            gradients = Engine.Multiply(gradients, scaleFactor);
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
    /// Calculates the classification loss for the class predictions using binary cross-entropy.
    /// Uses full BCE formula: -[target*log(p) + (1-target)*log(1-p)]
    /// This is consistent with the gradient formula: (p - target) / (p * (1 - p))
    /// </summary>
    private T CalculateClassificationLoss(Tensor<T> discOutput, Tensor<T> labels, int batchSize)
    {
        T totalLoss = NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);
        T one = NumOps.One;

        for (int i = 0; i < batchSize; i++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                T prediction = MathHelper.Clamp(discOutput[i, 1 + c], epsilon, NumOps.Subtract(one, epsilon));
                T target = labels[i, c];

                // Full BCE loss: -[target*log(p) + (1-target)*log(1-p)]
                T logP = NumOps.Log(NumOps.Add(prediction, epsilon));
                T oneMinusP = NumOps.Subtract(one, prediction);
                T logOneMinusP = NumOps.Log(NumOps.Add(oneMinusP, epsilon));
                T oneMinusTarget = NumOps.Subtract(one, target);

                T loss = NumOps.Negate(NumOps.Add(
                    NumOps.Multiply(target, logP),
                    NumOps.Multiply(oneMinusTarget, logOneMinusP)
                ));
                totalLoss = NumOps.Add(totalLoss, loss);
            }
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates gradients for discriminator backpropagation.
    /// Uses the correct gradient formula for probability-based BCE:
    /// dL/dp = (p - target) / (p * (1 - p))
    /// </summary>
    private Tensor<T> CalculateDiscriminatorGradients(
        Tensor<T> discOutput,
        Tensor<T> labels,
        bool isReal,
        int batchSize)
    {
        var gradients = new Tensor<T>(discOutput.Shape);
        T authTarget = isReal ? NumOps.One : NumOps.Zero;
        T epsilon = NumOps.FromDouble(1e-10);
        T batchSizeT = NumOps.FromDouble(batchSize);

        for (int i = 0; i < batchSize; i++)
        {
            // Clamp prediction to avoid division by zero
            T p = MathHelper.Clamp(discOutput[i, 0], epsilon, NumOps.Subtract(NumOps.One, epsilon));

            // Gradient for probability-based BCE: (p - target) / (p * (1 - p)) / batchSize
            T pTimesOneMinusP = NumOps.Multiply(p, NumOps.Subtract(NumOps.One, p));
            T authGrad = NumOps.Divide(
                NumOps.Divide(NumOps.Subtract(p, authTarget), pTimesOneMinusP),
                batchSizeT
            );
            gradients[i, 0] = authGrad;

            for (int c = 0; c < _numClasses; c++)
            {
                // Clamp class prediction to avoid division by zero
                T classP = MathHelper.Clamp(discOutput[i, 1 + c], epsilon, NumOps.Subtract(NumOps.One, epsilon));
                T classPTimesOneMinusP = NumOps.Multiply(classP, NumOps.Subtract(NumOps.One, classP));

                // Gradient for probability-based BCE
                T classGrad = NumOps.Divide(
                    NumOps.Divide(NumOps.Subtract(classP, labels[i, c]), classPTimesOneMinusP),
                    batchSizeT
                );
                gradients[i, 1 + c] = classGrad;
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
    /// Normalizes discriminator output to valid probability range [0, 1].
    /// If values appear to be logits (outside [0,1]), applies sigmoid transformation.
    /// </summary>
    private Tensor<T> NormalizeToProbabilities(Tensor<T> discOutput)
    {
        int batchSize = discOutput.Shape[0];
        int outputSize = discOutput.Shape[1];
        bool hasLogits = false;
        T zero = NumOps.Zero;
        T one = NumOps.One;

        // Check if any values are outside [0, 1] range (indicating logits)
        for (int i = 0; i < batchSize && !hasLogits; i++)
        {
            for (int j = 0; j < outputSize && !hasLogits; j++)
            {
                T val = discOutput[i, j];
                if (NumOps.LessThan(val, zero) || NumOps.GreaterThan(val, one))
                {
                    hasLogits = true;
                }
            }
        }

        if (!hasLogits)
        {
            return discOutput;
        }

        // === Vectorized sigmoid using IEngine (Phase B: US-GPU-015) ===
        return Engine.Sigmoid(discOutput);
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
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return Generator.Predict(input);
    }

    /// <summary>
    /// Performs a single training iteration using the standard neural network interface.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method adapts the AC-GAN's specialized training to the standard <see cref="NeuralNetworkBase{T}.Train"/>
    /// interface by automatically generating random class labels for both real and fake samples.
    /// </para>
    /// <para>
    /// The AC-GAN training process differs from standard neural networks because it requires:
    /// <list type="bullet">
    /// <item><description>Real images with their class labels</description></item>
    /// <item><description>Noise vectors for generating fake images</description></item>
    /// <item><description>Target class labels for the generated images</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// When using this simplified interface, random class labels are generated using
    /// <see cref="RandomHelper.ThreadSafeRandom"/> for thread-safe, cryptographically-seeded
    /// random number generation. For more control over class labels, use the
    /// <see cref="TrainStep"/> method directly.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you train an AC-GAN using the same
    /// interface as other neural networks. Just provide:
    /// <list type="bullet">
    /// <item><description><paramref name="input"/>: Random noise vectors (like random seeds for image generation)</description></item>
    /// <item><description><paramref name="expectedOutput"/>: Real images to learn from</description></item>
    /// </list>
    ///
    /// The method automatically assigns random class labels (like "digit 3", "digit 7", etc.)
    /// to both the real images and the images to generate. While this is convenient,
    /// for best results you should use <see cref="TrainStep"/> with actual class labels
    /// from your dataset.
    /// </para>
    /// </remarks>
    /// <param name="input">The noise tensor used as input to the generator network.
    /// Shape should be [batchSize, noiseSize] where noiseSize matches the generator's expected input.</param>
    /// <param name="expectedOutput">The real images tensor used for discriminator training.
    /// Shape should be [batchSize, height, width, channels] or equivalent flattened form.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> or
    /// <paramref name="expectedOutput"/> is null.</exception>
    /// <seealso cref="TrainStep"/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input), "Noise input tensor cannot be null.");
        }

        if (expectedOutput is null)
        {
            throw new ArgumentNullException(nameof(expectedOutput), "Real images tensor cannot be null.");
        }

        int batchSize = expectedOutput.Shape[0];

        // Use thread-safe random for cryptographically secure, thread-safe label generation
        var random = RandomHelper.ThreadSafeRandom;
        var realLabelIndices = new int[batchSize];
        var fakeLabelIndices = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            realLabelIndices[i] = random.Next(_numClasses);
            fakeLabelIndices[i] = random.Next(_numClasses);
        }

        var realLabels = CreateOneHotLabelsFromIndices(batchSize, realLabelIndices);
        var fakeLabels = CreateOneHotLabelsFromIndices(batchSize, fakeLabelIndices);

        TrainStep(expectedOutput, realLabels, input, fakeLabels);
    }

    /// <summary>
    /// Creates one-hot encoded label tensors from class indices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// One-hot encoding converts class indices (0, 1, 2, ...) into binary vectors
    /// where only one element is 1 and all others are 0.
    /// </para>
    /// <para><b>For Beginners:</b> One-hot encoding converts a class number into a format
    /// neural networks understand better.
    ///
    /// Example with 4 classes:
    /// <list type="bullet">
    /// <item><description>Class 0 becomes [1, 0, 0, 0]</description></item>
    /// <item><description>Class 1 becomes [0, 1, 0, 0]</description></item>
    /// <item><description>Class 2 becomes [0, 0, 1, 0]</description></item>
    /// <item><description>Class 3 becomes [0, 0, 0, 1]</description></item>
    /// </list>
    ///
    /// This helps the network learn to distinguish between classes more clearly.
    /// </para>
    /// </remarks>
    /// <param name="batchSize">The number of samples in the batch.</param>
    /// <param name="classIndices">Array of class indices, one per sample in the batch.</param>
    /// <returns>A tensor of shape [batchSize, numClasses] with one-hot encoded labels.</returns>
    private Tensor<T> CreateOneHotLabelsFromIndices(int batchSize, int[] classIndices)
    {
        var labels = new Tensor<T>(new int[] { batchSize, _numClasses });
        for (int i = 0; i < batchSize; i++)
        {
            int classIndex = classIndices[i];
            for (int c = 0; c < _numClasses; c++)
            {
                labels[i, c] = (classIndex >= 0 && classIndex < _numClasses && c == classIndex)
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return labels;
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

    /// <summary>
    /// Serializes AC-GAN-specific data including networks and optimizer states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method serializes all components needed to fully restore an AC-GAN's training state:
    /// <list type="bullet">
    /// <item><description>Number of classes</description></item>
    /// <item><description>Loss histories for monitoring training progress</description></item>
    /// <item><description>Generator and Discriminator networks with all learned weights</description></item>
    /// <item><description>Optimizer states (momentum, adaptive learning rates, timesteps)</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> When you save an AC-GAN during training, this method ensures
    /// that everything needed to resume training is saved:
    /// <list type="bullet">
    /// <item><description>The networks' learned knowledge (weights and biases)</description></item>
    /// <item><description>The optimizers' "memory" (like Adam's momentum vectors)</description></item>
    /// <item><description>Training history (loss values for monitoring)</description></item>
    /// </list>
    ///
    /// Without saving optimizer states, resuming training would be like starting with a new
    /// optimizer that has forgotten all the momentum and adaptive learning rates it built up,
    /// which can cause unstable training after loading.
    /// </para>
    /// </remarks>
    /// <param name="writer">The binary writer to serialize data to.</param>
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

        // Serialize Generator network
        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        // Serialize Discriminator network
        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);

        // Serialize optimizer states for training resumption
        // This preserves momentum vectors, adaptive learning rates, and timesteps
        var generatorOptimizerBytes = _generatorOptimizer.Serialize();
        writer.Write(generatorOptimizerBytes.Length);
        writer.Write(generatorOptimizerBytes);

        var discriminatorOptimizerBytes = _discriminatorOptimizer.Serialize();
        writer.Write(discriminatorOptimizerBytes.Length);
        writer.Write(discriminatorOptimizerBytes);
    }

    /// <summary>
    /// Deserializes AC-GAN-specific data including networks and optimizer states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method restores all components needed to continue AC-GAN training from a saved state:
    /// <list type="bullet">
    /// <item><description>Number of classes for classification</description></item>
    /// <item><description>Loss histories for training progress visualization</description></item>
    /// <item><description>Generator and Discriminator networks with all learned weights</description></item>
    /// <item><description>Optimizer states (momentum vectors, adaptive learning rates, timesteps)</description></item>
    /// </list>
    /// </para>
    /// <para><b>For Beginners:</b> When you load a saved AC-GAN, this method restores everything
    /// needed to continue training exactly where you left off:
    /// <list type="bullet">
    /// <item><description>The networks remember everything they learned</description></item>
    /// <item><description>The optimizers remember their momentum and learning rate adjustments</description></item>
    /// <item><description>Training can resume smoothly without any "warm-up" period</description></item>
    /// </list>
    ///
    /// This is especially important for Adam optimizer which maintains momentum vectors (m and v)
    /// and a timestep counter - losing these would cause training instability after loading.
    /// </para>
    /// </remarks>
    /// <param name="reader">The binary reader to deserialize data from.</param>
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

        // Deserialize Generator network
        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        // Deserialize Discriminator network
        int discriminatorDataLength = reader.ReadInt32();
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);

        // Deserialize optimizer states for training resumption
        // This restores momentum vectors, adaptive learning rates, and timesteps
        int generatorOptimizerDataLength = reader.ReadInt32();
        byte[] generatorOptimizerData = reader.ReadBytes(generatorOptimizerDataLength);
        _generatorOptimizer.Deserialize(generatorOptimizerData);

        int discriminatorOptimizerDataLength = reader.ReadInt32();
        byte[] discriminatorOptimizerData = reader.ReadBytes(discriminatorOptimizerDataLength);
        _discriminatorOptimizer.Deserialize(discriminatorOptimizerData);
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
