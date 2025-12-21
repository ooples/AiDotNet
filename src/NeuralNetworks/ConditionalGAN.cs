using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;

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
public class ConditionalGAN<T> : GenerativeAdversarialNetwork<T>
{
    private readonly List<T> _generatorLosses = new List<T>();

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
    /// Creates the combined ConditionalGAN architecture with correct dimension handling.
    /// </summary>
    /// <remarks>
    /// The generator architecture is returned unchanged. The condition vector is concatenated
    /// with the noise at runtime in GenerateConditional(), not by modifying the architecture.
    /// This preserves compatibility with the base CNN requirements for 3D input.
    /// </remarks>
    private static NeuralNetworkArchitecture<T> CreateConditionalGeneratorArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        int numConditionClasses)
    {
        // Return the generator architecture unchanged. The condition concatenation
        // is handled at runtime in GenerateConditional() and TrainStep(), not by
        // modifying the architecture dimensions. This avoids conflicts between
        // inputSize and dimension parameters for ThreeDimensional input types.
        _ = numConditionClasses; // Used at runtime for condition handling
        return generatorArchitecture;
    }

    /// <summary>
    /// Creates the discriminator architecture for conditional GAN.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateConditionalDiscriminatorArchitecture(
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numConditionClasses)
    {
        // Check if discriminator expects spatial input (3D/4D tensor with H, W, D > 0)
        bool isSpatialInput = discriminatorArchitecture.InputHeight > 0
            && discriminatorArchitecture.InputWidth > 0
            && discriminatorArchitecture.InputDepth > 0;

        int inputSize;
        int inputDepth;

        if (isSpatialInput)
        {
            // Spatial discriminator: conditions are added as extra channels
            // inputDepth increases by numConditionClasses
            inputDepth = discriminatorArchitecture.InputDepth + numConditionClasses;
            // inputSize may need to account for the extra channels across spatial dims
            inputSize = discriminatorArchitecture.InputSize > 0
                ? discriminatorArchitecture.InputSize + (numConditionClasses * discriminatorArchitecture.InputHeight * discriminatorArchitecture.InputWidth)
                : 0;
        }
        else
        {
            // Flat discriminator: conditions are concatenated to the input vector
            inputSize = discriminatorArchitecture.InputSize > 0
                ? discriminatorArchitecture.InputSize + numConditionClasses
                : 0;
            inputDepth = discriminatorArchitecture.InputDepth;
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: discriminatorArchitecture.InputType,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: discriminatorArchitecture.Complexity,
            inputSize: inputSize,
            inputHeight: discriminatorArchitecture.InputHeight,
            inputWidth: discriminatorArchitecture.InputWidth,
            inputDepth: inputDepth,
            outputSize: 1,
            layers: null);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalGAN{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">The neural network architecture for the generator.</param>
    /// <param name="discriminatorArchitecture">The neural network architecture for the discriminator.</param>
    /// <param name="numConditionClasses">The number of conditioning classes/categories.</param>
    /// <param name="inputType">The type of input the cGAN will process.</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="discriminatorOptimizer">Optional optimizer for the discriminator. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">Optional loss function.</param>
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
    /// - generatorOptimizer/discriminatorOptimizer: Custom learning algorithms (optional)
    /// </para>
    /// </remarks>
    public ConditionalGAN(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        int numConditionClasses,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(
            CreateConditionalGeneratorArchitecture(generatorArchitecture, numConditionClasses),
            CreateConditionalDiscriminatorArchitecture(discriminatorArchitecture, numConditionClasses),
            inputType,
            generatorOptimizer,
            discriminatorOptimizer,
            lossFunction)
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

        _numConditionClasses = numConditionClasses;
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
        T realLoss = TrainDiscriminatorOnBatch(realImagesWithConditions, realLabels);

        // Train discriminator on fake images with conditions
        Tensor<T> fakeImagesWithConditions = ConcatenateImageAndCondition(fakeImages, conditions);
        T fakeLoss = TrainDiscriminatorOnBatch(fakeImagesWithConditions, fakeLabels);

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
        T generatorLoss = TrainGeneratorOnBatch(newGeneratorInput, newFakeImagesWithConditions, allRealLabels);

        // Track losses
        _generatorLosses.Add(generatorLoss);
        if (_generatorLosses.Count > 100)
        {
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss);
    }

    /// <summary>
    /// Trains the discriminator on a batch of images.
    /// </summary>
    private T TrainDiscriminatorOnBatch(Tensor<T> images, Tensor<T> labels)
    {
        Discriminator.SetTrainingMode(true);

        // Forward pass
        var predictions = Discriminator.Predict(images);

        // Calculate loss
        var loss = CalculateBinaryLoss(predictions, labels);

        // Calculate gradients
        var outputGradients = CalculateBinaryGradients(predictions, labels);

        // Backpropagate
        Discriminator.Backpropagate(outputGradients);

        // Update parameters using base class method
        UpdateDiscriminatorWithOptimizer();

        return loss;
    }

    /// <summary>
    /// Trains the generator on a batch.
    /// </summary>
    private T TrainGeneratorOnBatch(Tensor<T> generatorInput, Tensor<T> fakeImagesWithConditions, Tensor<T> targetLabels)
    {
        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        // Get discriminator output
        var discriminatorOutput = Discriminator.Predict(fakeImagesWithConditions);

        // Calculate loss
        var loss = CalculateBinaryLoss(discriminatorOutput, targetLabels);

        // Calculate gradients
        var outputGradients = CalculateBinaryGradients(discriminatorOutput, targetLabels);

        // Backpropagate through discriminator to get input gradients
        var discriminatorInputGradients = Discriminator.Backpropagate(outputGradients);

        // Extract gradients for the image part (not the condition part)
        // Handle both spatial (4D) and flattened (2D) gradient formats
        int batchSize = generatorInput.Shape[0];
        Tensor<T> generatorGradients;

        if (discriminatorInputGradients.Shape.Length == 4)
        {
            // Spatial gradient format: [B, H, W, C+K] or [B, C+K, H, W]
            // Detect channel layout based on discriminator architecture.
            // Note: discArch.InputDepth already includes condition channels (C+K),
            // so we compare directly without adding _numConditionClasses again.
            var discArch = Discriminator.Architecture;
            bool isChannelsFirst = discArch.InputDepth > 0 && discriminatorInputGradients.Shape[1] == discArch.InputDepth;

            int height, width, totalChannels, imageChannels;
            if (isChannelsFirst)
            {
                // [B, C+K, H, W]
                totalChannels = discriminatorInputGradients.Shape[1];
                height = discriminatorInputGradients.Shape[2];
                width = discriminatorInputGradients.Shape[3];
                imageChannels = totalChannels - _numConditionClasses;

                // Extract image gradients (first imageChannels channels)
                generatorGradients = new Tensor<T>(new int[] { batchSize, imageChannels, height, width });
                for (int b = 0; b < batchSize; b++)
                {
                    for (int c = 0; c < imageChannels; c++)
                    {
                        for (int h = 0; h < height; h++)
                        {
                            for (int w = 0; w < width; w++)
                            {
                                generatorGradients[b, c, h, w] = discriminatorInputGradients[b, c, h, w];
                            }
                        }
                    }
                }
            }
            else
            {
                // [B, H, W, C+K]
                height = discriminatorInputGradients.Shape[1];
                width = discriminatorInputGradients.Shape[2];
                totalChannels = discriminatorInputGradients.Shape[3];
                imageChannels = totalChannels - _numConditionClasses;

                // Extract image gradients (first imageChannels channels)
                generatorGradients = new Tensor<T>(new int[] { batchSize, height, width, imageChannels });
                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            for (int c = 0; c < imageChannels; c++)
                            {
                                generatorGradients[b, h, w, c] = discriminatorInputGradients[b, h, w, c];
                            }
                        }
                    }
                }
            }
        }
        else
        {
            // Flattened gradient format: [B, image+K]
            int totalSize = discriminatorInputGradients.Length / batchSize;
            int imageSize = totalSize - _numConditionClasses;

            generatorGradients = new Tensor<T>(new int[] { batchSize, imageSize });
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < imageSize; i++)
                {
                    generatorGradients.SetFlatIndex(b * imageSize + i, discriminatorInputGradients.GetFlatIndexValue(b * totalSize + i));
                }
            }
        }

        // Backpropagate through generator
        Generator.Backpropagate(generatorGradients);

        // Update generator using base class method
        UpdateGeneratorWithOptimizer();

        return loss;
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

        var updatedParameters = GeneratorOptimizer.UpdateParameters(parameters, gradients);
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

        var updatedParameters = DiscriminatorOptimizer.UpdateParameters(parameters, gradients);
        Discriminator.UpdateParameters(updatedParameters);
    }

    /// <summary>
    /// Calculates binary cross-entropy loss with logits (numerically stable).
    /// </summary>
    /// <remarks>
    /// Uses the numerically stable formula: max(z,0) - z*t + log(1 + exp(-|z|))
    /// where z is the logit (pre-sigmoid prediction) and t is the target.
    /// This avoids numerical instability from computing log of values near 0 or 1.
    /// </remarks>
    private T CalculateBinaryLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        int batchSize = predictions.Shape[0];
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            T logit = predictions[i, 0];
            T target = targets[i, 0];

            // BCE with logits: max(z,0) - z*t + log(1 + exp(-|z|))
            T maxLogitZero = NumOps.GreaterThan(logit, NumOps.Zero) ? logit : NumOps.Zero;
            T absLogit = NumOps.GreaterThanOrEquals(logit, NumOps.Zero) ? logit : NumOps.Negate(logit);
            T expNegAbsLogit = NumOps.Exp(NumOps.Negate(absLogit));
            T logOnePlusExp = NumOps.Log(NumOps.Add(NumOps.One, expNegAbsLogit));

            T loss = NumOps.Add(
                NumOps.Subtract(maxLogitZero, NumOps.Multiply(logit, target)),
                logOnePlusExp);

            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Calculates gradients for binary cross-entropy loss with logits.
    /// </summary>
    /// <remarks>
    /// The gradient of BCE with logits with respect to the logit z is:
    /// dL/dz = sigmoid(z) - target = 1/(1+exp(-z)) - target
    /// This is consistent with the logits-based loss formula.
    /// </remarks>
    private Tensor<T> CalculateBinaryGradients(Tensor<T> predictions, Tensor<T> targets)
    {
        int batchSize = predictions.Shape[0];
        var gradients = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            T logit = predictions[i, 0];
            T target = targets[i, 0];

            // sigmoid(logit) = 1 / (1 + exp(-logit))
            T negLogit = NumOps.Negate(logit);
            T expNegLogit = NumOps.Exp(negLogit);
            T sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegLogit));

            // Gradient = sigmoid(logit) - target
            gradients[i, 0] = NumOps.Subtract(sigmoid, target);
        }

        return gradients;
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
    /// Concatenates noise and condition vectors.
    /// </summary>
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
    /// Handles both spatial (3D/4D) and flattened (2D) discriminator architectures.
    /// </summary>
    private Tensor<T> ConcatenateImageAndCondition(Tensor<T> images, Tensor<T> conditions)
    {
        int batchSize = images.Shape[0];
        int conditionSize = conditions.Shape[1];

        // Check if discriminator expects spatial input (3D/4D tensor)
        var discArch = Discriminator.Architecture;
        bool expectsSpatialInput = discArch.InputHeight > 0 && discArch.InputWidth > 0;

        if (expectsSpatialInput && images.Shape.Length >= 3)
        {
            // Spatial architecture: tile conditions across spatial dimensions
            // Input image shape: [B, H, W, C] or [B, C, H, W]
            // Output shape: [B, H, W, C + K] or [B, C + K, H, W]
            return ConcatenateSpatialImageAndCondition(images, conditions);
        }
        else
        {
            // Flattened architecture: concatenate flattened image with condition vector
            return ConcatenateFlattenedImageAndCondition(images, conditions);
        }
    }

    /// <summary>
    /// Concatenates flattened images with condition vectors for 1D discriminator input.
    /// </summary>
    private Tensor<T> ConcatenateFlattenedImageAndCondition(Tensor<T> images, Tensor<T> conditions)
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
    /// Concatenates spatial images with condition vectors by tiling conditions
    /// across spatial dimensions and appending as extra channels.
    /// </summary>
    /// <remarks>
    /// For a [B, H, W, C] image and [B, K] condition:
    /// - Expands condition to [B, H, W, K] by replicating across H and W
    /// - Concatenates along channel axis to produce [B, H, W, C + K]
    /// </remarks>
    private Tensor<T> ConcatenateSpatialImageAndCondition(Tensor<T> images, Tensor<T> conditions)
    {
        int batchSize = images.Shape[0];
        int conditionSize = conditions.Shape[1];

        // Determine image dimensions based on shape
        int height, width, channels;
        bool isChannelsFirst;

        if (images.Shape.Length == 4)
        {
            // 4D tensor: [B, C, H, W] (channels-first) or [B, H, W, C] (channels-last)
            // Detect layout using discriminator architecture, accounting for condition channels
            var discArch = Discriminator.Architecture;

            // The discriminator's InputDepth includes the appended condition channels (C+K).
            // Raw images only have C channels, so we subtract the condition size to get
            // the original image channel count for proper layout detection.
            int conditionChannelsAdded = conditionSize;
            int originalChannelCount = Math.Max(0, discArch.InputDepth - conditionChannelsAdded);

            // Check if shape[1] matches original image channels (channels-first: [B, C, H, W])
            // or shape[3] matches original image channels (channels-last: [B, H, W, C])
            if (originalChannelCount > 0 && images.Shape[1] == originalChannelCount)
            {
                // Channels-first format: [B, C, H, W]
                channels = images.Shape[1];
                height = images.Shape[2];
                width = images.Shape[3];
                isChannelsFirst = true;
            }
            else
            {
                // Channels-last format: [B, H, W, C] (default)
                height = images.Shape[1];
                width = images.Shape[2];
                channels = images.Shape[3];
                isChannelsFirst = false;
            }
        }
        else if (images.Shape.Length == 3)
        {
            // 3D tensor: [B, H*W, C] or similar - treat as spatial
            height = Discriminator.Architecture.InputHeight;
            width = Discriminator.Architecture.InputWidth;
            channels = images.Length / (batchSize * height * width);
            isChannelsFirst = false;
        }
        else
        {
            // Fall back to flattened for unexpected shapes
            return ConcatenateFlattenedImageAndCondition(images, conditions);
        }

        // Create output tensor with extra channels for conditions
        int[] outputShape = isChannelsFirst
            ? new int[] { batchSize, channels + conditionSize, height, width }
            : new int[] { batchSize, height, width, channels + conditionSize };
        var result = new Tensor<T>(outputShape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Copy original image channels
                    for (int c = 0; c < channels; c++)
                    {
                        T value;
                        if (images.Shape.Length == 4)
                        {
                            value = isChannelsFirst
                                ? images[b, c, h, w]
                                : images[b, h, w, c];
                        }
                        else
                        {
                            // For 3D, calculate linear index
                            int idx = h * width * channels + w * channels + c;
                            value = images.GetFlatIndexValue(b * (height * width * channels) + idx);
                        }

                        if (isChannelsFirst)
                        {
                            result[b, c, h, w] = value;
                        }
                        else
                        {
                            result[b, h, w, c] = value;
                        }
                    }

                    // Tile condition across spatial dimensions (replicate at each H, W position)
                    for (int k = 0; k < conditionSize; k++)
                    {
                        T condValue = conditions[b, k];
                        if (isChannelsFirst)
                        {
                            result[b, channels + k, h, w] = condValue;
                        }
                        else
                        {
                            result[b, h, w, channels + k] = condValue;
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Generates images conditioned on specific labels.
    /// </summary>
    public Tensor<T> GenerateConditional(Tensor<T> noise, Tensor<T> conditions)
    {
        Generator.SetTrainingMode(false);
        var input = ConcatenateTensors(noise, conditions);
        return Generator.Predict(input);
    }

    /// <summary>
    /// Generates random noise tensor using vectorized Gaussian noise generation.
    /// </summary>
    public new Tensor<T> GenerateRandomNoiseTensor(int batchSize, int noiseSize)
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
    public Tensor<T> CreateOneHotCondition(int batchSize, int classIndex)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                "Batch size must be positive.");
        }

        if (classIndex < 0 || classIndex >= _numConditionClasses)
        {
            throw new ArgumentOutOfRangeException(nameof(classIndex), classIndex,
                $"Class index must be between 0 and {_numConditionClasses - 1} (inclusive).");
        }

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

    /// <summary>
    /// Trains the conditional GAN on a batch of data.
    /// </summary>
    /// <param name="input">The input noise tensor for the generator.</param>
    /// <param name="expectedOutput">The tensor containing real images.</param>
    /// <remarks>
    /// <para>
    /// This method implements the standard Train interface by:
    /// 1. Generating random conditions for training
    /// 2. Using the input as noise for the generator
    /// 3. Using expectedOutput as the real images for the discriminator
    /// 4. Delegating to TrainStep for the actual training
    /// </para>
    /// <para><b>For Beginners:</b> This is the main training method that follows the base class contract.
    ///
    /// How it works:
    /// - The 'input' tensor is used as random noise for the generator
    /// - The 'expectedOutput' tensor contains real images to train the discriminator
    /// - Random class conditions are generated for conditional training
    /// - Both generator and discriminator are updated in each call
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        int batchSize = expectedOutput.Shape[0];

        // Generate random conditions for training (random class for each sample)
        var conditions = new Tensor<T>(new int[] { batchSize, _numConditionClasses });
        var random = RandomHelper.ThreadSafeRandom;

        for (int b = 0; b < batchSize; b++)
        {
            int randomClass = random.Next(_numConditionClasses);
            for (int c = 0; c < _numConditionClasses; c++)
            {
                conditions[b, c] = c == randomClass ? NumOps.One : NumOps.Zero;
            }
        }

        // Use input as noise and call TrainStep
        TrainStep(expectedOutput, conditions, input);
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        base.SerializeNetworkSpecificData(writer);
        writer.Write(_numConditionClasses);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        base.DeserializeNetworkSpecificData(reader);
        _numConditionClasses = reader.ReadInt32();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ConditionalGAN<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            _numConditionClasses,
            Architecture.InputType,
            null, // Use default optimizer
            null, // Use default optimizer
            null); // Use default loss function
    }
}
