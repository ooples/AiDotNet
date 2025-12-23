using System.IO;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Pix2Pix GAN for paired image-to-image translation tasks.
/// </summary>
/// <remarks>
/// <para>
/// Pix2Pix is a conditional GAN for paired image-to-image translation:
/// - Uses a U-Net generator with skip connections
/// - Uses a PatchGAN discriminator that classifies image patches
/// - Combines adversarial loss with L1 reconstruction loss
/// - Requires paired training data (input-output pairs)
/// - Works for various tasks: edges to photo, day to night, sketch to image, etc.
/// </para>
/// <para><b>For Beginners:</b> Pix2Pix transforms one type of image to another.
///
/// Key features:
/// - Learns from paired examples (input A becomes output B)
/// - Generator: U-Net architecture preserves spatial information
/// - Discriminator: PatchGAN focuses on local image patches
/// - Loss: Both "looks real" and "matches input"
///
/// Example use cases:
/// - Convert sketches to realistic photos
/// - Colorize black-and-white images
/// - Transform day scenes to night
/// - Semantic labels to photorealistic images
/// - Map to satellite image
///
/// Reference: Isola et al., "Image-to-Image Translation with Conditional
/// Adversarial Networks" (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class Pix2Pix<T> : NeuralNetworkBase<T>
{
    private readonly List<T> _discriminatorLosses = new List<T>();
    private readonly List<T> _generatorLosses = new List<T>();

    /// <summary>
    /// The optimizer for the generator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorOptimizer;

    /// <summary>
    /// The optimizer for the discriminator network.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorOptimizer;

    /// <summary>
    /// The coefficient for the L1 reconstruction loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls the trade-off between adversarial loss and L1 loss. Typical value is 100.
    /// Higher values encourage outputs to be closer to ground truth.
    /// </para>
    /// <para><b>For Beginners:</b> How important is matching the target exactly.
    ///
    /// - Higher (e.g., 100): output closely matches target
    /// - Lower (e.g., 10): more creative but less accurate
    /// - Paper uses 100 as default
    /// </para>
    /// </remarks>
    private double _l1Lambda;

    /// <summary>
    /// Gets the U-Net generator network.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> Generator { get; private set; }

    /// <summary>
    /// Gets the PatchGAN discriminator network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// PatchGAN classifies whether each N x N patch in an image is real or fake,
    /// rather than classifying the entire image. This encourages sharp high-frequency
    /// details and works well for image-to-image translation.
    /// </para>
    /// <para><b>For Beginners:</b> Discriminator checks local image quality.
    ///
    /// Instead of:
    /// - "Is the whole image real?" (standard discriminator)
    ///
    /// PatchGAN asks:
    /// - "Is this patch real? Is that patch real?" (many local checks)
    /// - This catches more detailed mistakes
    /// - Results in sharper, more realistic outputs
    /// </para>
    /// </remarks>
    public ConvolutionalNeuralNetwork<T> Discriminator { get; private set; }

    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Creates the combined Pix2Pix architecture with correct dimension handling.
    /// </summary>
    /// <param name="generatorArchitecture">The generator architecture.</param>
    /// <param name="inputType">The type of input.</param>
    /// <returns>The combined architecture for Pix2Pix.</returns>
    private static NeuralNetworkArchitecture<T> CreatePix2PixArchitecture(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        InputType inputType)
    {
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Deep,
                inputSize: 0,
                inputHeight: generatorArchitecture.InputHeight,
                inputWidth: generatorArchitecture.InputWidth,
                inputDepth: generatorArchitecture.InputDepth,
                outputSize: generatorArchitecture.OutputSize,
                layers: null);
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: generatorArchitecture.InputSize,
            outputSize: generatorArchitecture.OutputSize);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Pix2Pix{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">U-Net generator architecture.</param>
    /// <param name="discriminatorArchitecture">PatchGAN discriminator architecture.</param>
    /// <param name="inputType">Input type.</param>
    /// <param name="generatorOptimizer">Optional optimizer for the generator. If null, Adam optimizer is used.</param>
    /// <param name="discriminatorOptimizer">Optional optimizer for the discriminator. If null, Adam optimizer is used.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="l1Lambda">L1 loss coefficient. Default is 100.0.</param>
    /// <remarks>
    /// <para>
    /// The Pix2Pix constructor initializes both the generator and discriminator networks along with their
    /// respective optimizers. The L1 lambda coefficient controls how strongly the output should match
    /// the target image.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up Pix2Pix with sensible defaults.
    ///
    /// Key parameters:
    /// - Generator/discriminator architectures define the network structures
    /// - Optimizers control how the networks learn
    /// - L1 lambda (100.0) controls how closely output matches target
    /// </para>
    /// </remarks>
    public Pix2Pix(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorOptimizer = null,
        ILossFunction<T>? lossFunction = null,
        double l1Lambda = 100.0)
        : base(CreatePix2PixArchitecture(generatorArchitecture, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        if (generatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(generatorArchitecture), "Generator architecture cannot be null.");
        }
        if (discriminatorArchitecture is null)
        {
            throw new ArgumentNullException(nameof(discriminatorArchitecture), "Discriminator architecture cannot be null.");
        }
        if (l1Lambda < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(l1Lambda), l1Lambda, "L1 lambda must be non-negative.");
        }

        _l1Lambda = l1Lambda;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

        // Initialize optimizers (default to Adam if not provided)
        _generatorOptimizer = generatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Generator);
        _discriminatorOptimizer = discriminatorOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(Discriminator);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for Pix2Pix.
    /// </summary>
    /// <param name="inputImages">Input images (e.g., sketches, semantic maps).</param>
    /// <param name="targetImages">Target output images (e.g., photos).</param>
    /// <returns>Tuple of (discriminator loss, generator loss, L1 loss).</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Pix2Pix training algorithm:
    /// 1. Train discriminator on real and fake image pairs
    /// 2. Train generator with combined adversarial and L1 loss
    /// 3. The discriminator learns to distinguish real from fake
    /// 4. The generator learns to both fool the discriminator and match the target
    /// </para>
    /// <para><b>For Beginners:</b> One training round for Pix2Pix.
    ///
    /// The training process:
    /// - Discriminator learns to spot fake images
    /// - Generator learns to create realistic images that match target
    /// - L1 loss ensures output closely matches expected result
    /// - Returns loss values for monitoring progress
    /// </para>
    /// </remarks>
    public (T discriminatorLoss, T generatorLoss, T l1Loss) TrainStep(
        Tensor<T> inputImages,
        Tensor<T> targetImages)
    {
        if (inputImages is null)
        {
            throw new ArgumentNullException(nameof(inputImages), "Input images tensor cannot be null.");
        }

        if (targetImages is null)
        {
            throw new ArgumentNullException(nameof(targetImages), "Target images tensor cannot be null.");
        }

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(true);

        int batchSize = inputImages.Shape[0];

        // ----- Train Discriminator -----

        // Generate fake images (detached for discriminator training)
        var fakeImages = Generator.Predict(inputImages);

        // Concatenate input with real/fake images for discriminator
        var realPairs = ConcatenateImages(inputImages, targetImages);
        var fakePairs = ConcatenateImages(inputImages, fakeImages);

        // Real labels
        var realLabels = CreateLabelTensor(batchSize, NumOps.One);
        var fakeLabels = CreateLabelTensor(batchSize, NumOps.Zero);

        // Train on real pairs
        var realPredictions = Discriminator.Predict(realPairs);
        T realLoss = CalculateBinaryLoss(realPredictions, realLabels, batchSize);
        var realGradients = CalculateBinaryGradients(realPredictions, realLabels, batchSize);
        Discriminator.Backward(realGradients);

        // Train on fake pairs
        var fakePredictions = Discriminator.Predict(fakePairs);
        T fakeLossD = CalculateBinaryLoss(fakePredictions, fakeLabels, batchSize);
        var fakeGradients = CalculateBinaryGradients(fakePredictions, fakeLabels, batchSize);
        Discriminator.Backward(fakeGradients);
        UpdateDiscriminatorWithOptimizer();

        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLossD), NumOps.FromDouble(2.0));

        // ----- Train Generator -----

        Generator.SetTrainingMode(true);
        // Keep Discriminator in training mode - required for BackwardWithInputGradient
        // We just don't call UpdateDiscriminatorWithOptimizer() during generator training

        // Generate new fake images
        var newFakeImages = Generator.Predict(inputImages);

        // Adversarial loss: fool the discriminator
        var newFakePairs = ConcatenateImages(inputImages, newFakeImages);
        var genPredictions = Discriminator.Predict(newFakePairs);
        var allRealLabels = CreateLabelTensor(batchSize, NumOps.One);
        T advLoss = CalculateBinaryLoss(genPredictions, allRealLabels, batchSize);

        // L1 loss: match the target images
        T l1Loss = CalculateL1Loss(newFakeImages, targetImages);

        // Total generator loss
        T l1Coeff = NumOps.FromDouble(_l1Lambda);
        T generatorLoss = NumOps.Add(advLoss, NumOps.Multiply(l1Coeff, l1Loss));

        // Backpropagate adversarial gradients through discriminator to get input gradients
        var advGradients = CalculateBinaryGradients(genPredictions, allRealLabels, batchSize);
        var discInputGradients = Discriminator.BackwardWithInputGradient(advGradients);

        // Calculate L1 gradients
        var l1Gradients = CalculateL1Gradients(newFakeImages, targetImages);

        // Extract generator gradients from discInputGradients
        // discInputGradients contains gradients for [inputImages | newFakeImages]
        // We need only the second half (newFakeImages part)
        int inputTotalSize = inputImages.Length;
        int genOutputSize = newFakeImages.Length;
        int discInputTotalSize = discInputGradients.Length;

        var combinedGradients = new Tensor<T>(newFakeImages.Shape);

        // Combine adversarial and L1 gradients
        for (int b = 0; b < batchSize; b++)
        {
            int genSampleSize = genOutputSize / batchSize;
            int inputSampleSize = inputTotalSize / batchSize;
            int discSampleSize = discInputTotalSize / batchSize;

            for (int i = 0; i < genSampleSize; i++)
            {
                // The second half of each sample in discInputGradients corresponds to the generated image
                int discGenOffset = inputSampleSize + i;
                T advGrad = (discGenOffset < discSampleSize)
                    ? discInputGradients.GetFlat(b * discSampleSize + discGenOffset)
                    : NumOps.Zero;

                T l1Grad = l1Gradients.GetFlat(b * genSampleSize + i);

                // Combine: adversarial gradient + weighted L1 gradient
                combinedGradients.SetFlat(b * genSampleSize + i, NumOps.Add(advGrad, l1Grad));
            }
        }

        Generator.Backward(combinedGradients);
        UpdateGeneratorWithOptimizer();

        Discriminator.SetTrainingMode(true);

        // Track losses
        _discriminatorLosses.Add(discriminatorLoss);
        _generatorLosses.Add(generatorLoss);

        if (_discriminatorLosses.Count > 100)
        {
            _discriminatorLosses.RemoveAt(0);
            _generatorLosses.RemoveAt(0);
        }

        return (discriminatorLoss, generatorLoss, l1Loss);
    }

    /// <summary>
    /// Translates input images to output images.
    /// </summary>
    /// <param name="inputImages">The input images to translate.</param>
    /// <returns>The translated output images.</returns>
    public Tensor<T> Translate(Tensor<T> inputImages)
    {
        Generator.SetTrainingMode(false);
        return Generator.Predict(inputImages);
    }

    /// <summary>
    /// Concatenates two image tensors along the feature/channel dimension.
    /// Handles both 3D/4D spatial and 2D flattened inputs.
    /// </summary>
    private Tensor<T> ConcatenateImages(Tensor<T> images1, Tensor<T> images2)
    {
        if (images1.Shape.Length < 1 || images2.Shape.Length < 1)
        {
            throw new ArgumentException("Both image tensors must have at least one dimension.");
        }

        int batchSize1 = images1.Shape[0];
        int batchSize2 = images2.Shape[0];

        if (batchSize1 != batchSize2)
        {
            throw new ArgumentException(
                $"Batch size mismatch: images1 has {batchSize1} samples, images2 has {batchSize2} samples.");
        }

        int batchSize = batchSize1;

        // Check if we have spatial (3D/4D) tensors for PatchGAN discriminator
        // For 3D inputs, concatenate along channel dimension to preserve spatial structure
        if (Architecture.InputType == InputType.ThreeDimensional &&
            images1.Shape.Length >= 3 && images2.Shape.Length >= 3)
        {
            return ConcatenateSpatialImages(images1, images2);
        }

        // Fallback to flattened concatenation for non-spatial inputs
        return ConcatenateFlattenedImages(images1, images2);
    }

    /// <summary>
    /// Concatenates spatial image tensors along the channel dimension.
    /// Input: [B,H,W,C1] + [B,H,W,C2] => Output: [B,H,W,C1+C2]
    /// </summary>
    private Tensor<T> ConcatenateSpatialImages(Tensor<T> images1, Tensor<T> images2)
    {
        int batchSize = images1.Shape[0];

        // Determine spatial dimensions based on shape
        int height1, width1, channels1;
        int height2, width2, channels2;

        if (images1.Shape.Length == 4 && images2.Shape.Length == 4)
        {
            // 4D tensor: [B, H, W, C] (assuming channels-last)
            height1 = images1.Shape[1];
            width1 = images1.Shape[2];
            channels1 = images1.Shape[3];

            height2 = images2.Shape[1];
            width2 = images2.Shape[2];
            channels2 = images2.Shape[3];
        }
        else if (images1.Shape.Length == 3 && images2.Shape.Length == 3)
        {
            // 3D tensor: [B, H*W, C] - use architecture dimensions
            height1 = Architecture.InputHeight;
            width1 = Architecture.InputWidth;
            channels1 = images1.Shape[2];

            height2 = Architecture.InputHeight;
            width2 = Architecture.InputWidth;
            channels2 = images2.Shape[2];
        }
        else
        {
            // Fall back to flattened for mixed shapes
            return ConcatenateFlattenedImages(images1, images2);
        }

        // Validate spatial dimensions match
        if (height1 != height2 || width1 != width2)
        {
            throw new ArgumentException(
                $"Spatial dimensions must match: images1 is [{height1},{width1}], " +
                $"images2 is [{height2},{width2}].");
        }

        int height = height1;
        int width = width1;
        int totalChannels = channels1 + channels2;

        // Create output tensor with concatenated channels
        int[] outputShape = images1.Shape.Length == 4
            ? new int[] { batchSize, height, width, totalChannels }
            : new int[] { batchSize, height * width, totalChannels };
        var result = new Tensor<T>(outputShape);

        for (int b = 0; b < batchSize; b++)
        {
            if (images1.Shape.Length == 4)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        // Copy channels from images1
                        for (int c = 0; c < channels1; c++)
                        {
                            result[b, h, w, c] = images1[b, h, w, c];
                        }
                        // Copy channels from images2
                        for (int c = 0; c < channels2; c++)
                        {
                            result[b, h, w, channels1 + c] = images2[b, h, w, c];
                        }
                    }
                }
            }
            else
            {
                // 3D case: [B, H*W, C]
                int spatialSize = height * width;
                for (int s = 0; s < spatialSize; s++)
                {
                    // Copy channels from images1
                    for (int c = 0; c < channels1; c++)
                    {
                        result[b, s, c] = images1[b, s, c];
                    }
                    // Copy channels from images2
                    for (int c = 0; c < channels2; c++)
                    {
                        result[b, s, channels1 + c] = images2[b, s, c];
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Concatenates flattened image tensors along the feature dimension.
    /// </summary>
    private Tensor<T> ConcatenateFlattenedImages(Tensor<T> images1, Tensor<T> images2)
    {
        int batchSize = images1.Shape[0];
        int totalSize1 = images1.Shape.Aggregate(1, (a, b) => a * b);
        int totalSize2 = images2.Shape.Aggregate(1, (a, b) => a * b);
        int size1 = totalSize1 / batchSize;
        int size2 = totalSize2 / batchSize;

        var result = new Tensor<T>(new int[] { batchSize, size1 + size2 });

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < size1; i++)
            {
                result.SetFlat(b * (size1 + size2) + i, images1.GetFlat(b * size1 + i));
            }
            for (int i = 0; i < size2; i++)
            {
                result.SetFlat(b * (size1 + size2) + size1 + i, images2.GetFlat(b * size2 + i));
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the L1 loss between predictions and targets.
    /// </summary>
    private T CalculateL1Loss(Tensor<T> predictions, Tensor<T> targets)
    {
        T totalLoss = NumOps.Zero;
        int count = predictions.Shape.Aggregate(1, (a, b) => a * b);

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.GetFlat(i), targets.GetFlat(i));
            T absDiff = NumOps.GreaterThanOrEquals(diff, NumOps.Zero) ? diff : NumOps.Negate(diff);
            totalLoss = NumOps.Add(totalLoss, absDiff);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(count));
    }

    /// <summary>
    /// Calculates the gradients for L1 loss.
    /// </summary>
    private Tensor<T> CalculateL1Gradients(Tensor<T> predictions, Tensor<T> targets)
    {
        var gradients = new Tensor<T>(predictions.Shape);
        int count = predictions.Shape.Aggregate(1, (a, b) => a * b);
        T scale = NumOps.FromDouble(_l1Lambda / count);

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.GetFlat(i), targets.GetFlat(i));
            // Sign of difference
            T sign = NumOps.GreaterThanOrEquals(diff, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
            gradients.SetFlat(i, NumOps.Multiply(scale, sign));
        }

        return gradients;
    }

    /// <summary>
    /// Calculates the binary cross-entropy loss with logits (numerically stable).
    /// </summary>
    /// <remarks>
    /// Uses the numerically stable formula: max(z,0) - z*t + log(1 + exp(-|z|))
    /// where z is the logit (pre-sigmoid prediction) and t is the target.
    /// This avoids numerical instability from computing log of values near 0 or 1.
    /// </remarks>
    private T CalculateBinaryLoss(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
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
    /// Calculates the gradients for binary cross-entropy loss with logits.
    /// </summary>
    /// <remarks>
    /// The gradient of BCE with logits with respect to the logit z is:
    /// dL/dz = sigmoid(z) - target = 1/(1+exp(-z)) - target
    /// This is consistent with the logits-based loss formula.
    /// </remarks>
    private Tensor<T> CalculateBinaryGradients(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
        var gradients = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            T logit = predictions[i, 0];
            T target = targets[i, 0];

            // sigmoid(logit) = 1 / (1 + exp(-logit))
            T negLogit = NumOps.Negate(logit);
            T expNegLogit = NumOps.Exp(negLogit);
            T sigmoid = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegLogit));

            // Gradient = (sigmoid(logit) - target) / batchSize
            gradients[i, 0] = NumOps.Divide(
                NumOps.Subtract(sigmoid, target),
                NumOps.FromDouble(batchSize)
            );
        }

        return gradients;
    }

    /// <summary>
    /// Creates a tensor filled with a single label value.
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
    /// Resets both optimizer states for a fresh training run.
    /// </summary>
    public void ResetOptimizerState()
    {
        _generatorOptimizer.Reset();
        _discriminatorOptimizer.Reset();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Pix2Pix doesn't use layers directly
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Generator.Predict(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(input, expectedOutput);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.Pix2Pix,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorParameters", Generator.GetParameterCount() },
                { "DiscriminatorParameters", Discriminator.GetParameterCount() },
                { "L1Lambda", _l1Lambda }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_l1Lambda);

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

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _l1Lambda = reader.ReadDouble();

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

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Pix2Pix<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            Architecture.InputType,
            null, // Use default optimizer
            null, // Use default optimizer
            _lossFunction,
            _l1Lambda);
    }

    /// <summary>
    /// Updates the parameters of all networks in the Pix2Pix GAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int generatorCount = Generator.GetParameterCount();
        int discriminatorCount = Discriminator.GetParameterCount();

        if (parameters.Length != generatorCount + discriminatorCount)
        {
            throw new ArgumentException(
                $"Expected {generatorCount + discriminatorCount} parameters, " +
                $"but received {parameters.Length}.",
                nameof(parameters));
        }

        // Update Generator parameters
        var generatorParams = new Vector<T>(generatorCount);
        for (int i = 0; i < generatorCount; i++)
        {
            generatorParams[i] = parameters[i];
        }
        Generator.UpdateParameters(generatorParams);

        // Update Discriminator parameters
        var discriminatorParams = new Vector<T>(discriminatorCount);
        for (int i = 0; i < discriminatorCount; i++)
        {
            discriminatorParams[i] = parameters[generatorCount + i];
        }
        Discriminator.UpdateParameters(discriminatorParams);
    }
}
