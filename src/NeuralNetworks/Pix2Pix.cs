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
/// - Works for various tasks: edges→photo, day→night, sketch→image, etc.
/// </para>
/// <para><b>For Beginners:</b> Pix2Pix transforms one type of image to another.
///
/// Key features:
/// - Learns from paired examples (input A → output B)
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

    private double _initialLearningRate;
    private double _learningRateDecay;

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
    /// PatchGAN classifies whether each N×N patch in an image is real or fake,
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

    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Initializes a new instance of the <see cref="Pix2Pix{T}"/> class.
    /// </summary>
    /// <param name="generatorArchitecture">U-Net generator architecture.</param>
    /// <param name="discriminatorArchitecture">PatchGAN discriminator architecture.</param>
    /// <param name="inputType">Input type.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <param name="initialLearningRate">Initial learning rate. Default is 0.0002.</param>
    /// <param name="l1Lambda">L1 loss coefficient. Default is 100.0.</param>
    public Pix2Pix(
        NeuralNetworkArchitecture<T> generatorArchitecture,
        NeuralNetworkArchitecture<T> discriminatorArchitecture,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0002,
        double l1Lambda = 100.0)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Deep,
            generatorArchitecture.InputSize,
            generatorArchitecture.OutputSize,
            0, 0, 0,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        _l1Lambda = l1Lambda;
        _initialLearningRate = initialLearningRate;
        _learningRateDecay = 0.0001;

        Generator = new ConvolutionalNeuralNetwork<T>(generatorArchitecture);
        Discriminator = new ConvolutionalNeuralNetwork<T>(discriminatorArchitecture);

        // Initialize Generator optimizer state
        int genParamCount = Generator.GetParameterCount();
        _genMomentum = new Vector<T>(genParamCount);
        _genSecondMoment = new Vector<T>(genParamCount);
        _genBeta1Power = NumOps.One;
        _genBeta2Power = NumOps.One;
        _genCurrentLearningRate = initialLearningRate;

        // Initialize Discriminator optimizer state
        int discParamCount = Discriminator.GetParameterCount();
        _discMomentum = new Vector<T>(discParamCount);
        _discSecondMoment = new Vector<T>(discParamCount);
        _discBeta1Power = NumOps.One;
        _discBeta2Power = NumOps.One;
        _discCurrentLearningRate = initialLearningRate;

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for Pix2Pix.
    /// </summary>
    /// <param name="inputImages">Input images (e.g., sketches, semantic maps).</param>
    /// <param name="targetImages">Target output images (e.g., photos).</param>
    /// <returns>Tuple of (discriminator loss, generator loss, L1 loss).</returns>
    public (T discriminatorLoss, T generatorLoss, T l1Loss) TrainStep(
        Tensor<T> inputImages,
        Tensor<T> targetImages)
    {
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
        UpdateDiscriminatorParameters();

        T discriminatorLoss = NumOps.Divide(NumOps.Add(realLoss, fakeLossD), NumOps.FromDouble(2.0));

        // ----- Train Generator -----

        Generator.SetTrainingMode(true);
        Discriminator.SetTrainingMode(false);

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
        UpdateGeneratorParameters();

        Discriminator.SetTrainingMode(true);

        return (discriminatorLoss, generatorLoss, l1Loss);
    }

    /// <summary>
    /// Translates input images to output images.
    /// </summary>
    public Tensor<T> Translate(Tensor<T> inputImages)
    {
        Generator.SetTrainingMode(false);
        return Generator.Predict(inputImages);
    }

    private Tensor<T> ConcatenateImages(Tensor<T> images1, Tensor<T> images2)
    {
        // Simplified: concatenate along channel dimension
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

    private void UpdateGeneratorParameters()
    {
        var gradients = Generator.GetParameterGradients();
        var parameters = Generator.GetParameters();
        int paramCount = parameters.Length;

        // Adam hyperparameters - beta1=0.5 for Pix2Pix (paper recommendation)
        T beta1 = NumOps.FromDouble(0.5);
        T beta2 = NumOps.FromDouble(0.999);
        T epsilon = NumOps.FromDouble(1e-8);
        T lr = NumOps.FromDouble(_genCurrentLearningRate);

        // Update beta powers
        _genBeta1Power = NumOps.Multiply(_genBeta1Power, beta1);
        _genBeta2Power = NumOps.Multiply(_genBeta2Power, beta2);

        // Bias correction factors
        T beta1Correction = NumOps.Subtract(NumOps.One, _genBeta1Power);
        T beta2Correction = NumOps.Subtract(NumOps.One, _genBeta2Power);

        // Clip gradients
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradNormSq = NumOps.Zero;
        for (int i = 0; i < paramCount; i++)
            gradNormSq = NumOps.Add(gradNormSq, NumOps.Multiply(gradients[i], gradients[i]));
        T gradNorm = NumOps.Sqrt(gradNormSq);

        T scale = NumOps.One;
        if (NumOps.GreaterThan(gradNorm, maxGradNorm))
            scale = NumOps.Divide(maxGradNorm, NumOps.Add(gradNorm, epsilon));

        for (int i = 0; i < paramCount; i++)
        {
            T g = NumOps.Multiply(gradients[i], scale);

            // Update biased first moment estimate
            _genMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _genMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), g)
            );

            // Update biased second raw moment estimate
            _genSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _genSecondMoment[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta2), NumOps.Multiply(g, g))
            );

            // Compute bias-corrected estimates
            T mHat = NumOps.Divide(_genMomentum[i], beta1Correction);
            T vHat = NumOps.Divide(_genSecondMoment[i], beta2Correction);

            // Update parameters
            T update = NumOps.Divide(NumOps.Multiply(lr, mHat), NumOps.Add(NumOps.Sqrt(vHat), epsilon));
            parameters[i] = NumOps.Subtract(parameters[i], update);
        }

        Generator.UpdateParameters(parameters);

        // Learning rate decay
        _genCurrentLearningRate = _initialLearningRate / (1.0 + _learningRateDecay * Convert.ToDouble(_genBeta1Power));
    }

    private void UpdateDiscriminatorParameters()
    {
        var gradients = Discriminator.GetParameterGradients();
        var parameters = Discriminator.GetParameters();
        int paramCount = parameters.Length;

        // Adam hyperparameters - beta1=0.5 for Pix2Pix (paper recommendation)
        T beta1 = NumOps.FromDouble(0.5);
        T beta2 = NumOps.FromDouble(0.999);
        T epsilon = NumOps.FromDouble(1e-8);
        T lr = NumOps.FromDouble(_discCurrentLearningRate);

        // Update beta powers
        _discBeta1Power = NumOps.Multiply(_discBeta1Power, beta1);
        _discBeta2Power = NumOps.Multiply(_discBeta2Power, beta2);

        // Bias correction factors
        T beta1Correction = NumOps.Subtract(NumOps.One, _discBeta1Power);
        T beta2Correction = NumOps.Subtract(NumOps.One, _discBeta2Power);

        // Clip gradients
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradNormSq = NumOps.Zero;
        for (int i = 0; i < paramCount; i++)
            gradNormSq = NumOps.Add(gradNormSq, NumOps.Multiply(gradients[i], gradients[i]));
        T gradNorm = NumOps.Sqrt(gradNormSq);

        T scale = NumOps.One;
        if (NumOps.GreaterThan(gradNorm, maxGradNorm))
            scale = NumOps.Divide(maxGradNorm, NumOps.Add(gradNorm, epsilon));

        for (int i = 0; i < paramCount; i++)
        {
            T g = NumOps.Multiply(gradients[i], scale);

            _discMomentum[i] = NumOps.Add(
                NumOps.Multiply(beta1, _discMomentum[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), g)
            );

            _discSecondMoment[i] = NumOps.Add(
                NumOps.Multiply(beta2, _discSecondMoment[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta2), NumOps.Multiply(g, g))
            );

            T mHat = NumOps.Divide(_discMomentum[i], beta1Correction);
            T vHat = NumOps.Divide(_discSecondMoment[i], beta2Correction);

            T update = NumOps.Divide(NumOps.Multiply(lr, mHat), NumOps.Add(NumOps.Sqrt(vHat), epsilon));
            parameters[i] = NumOps.Subtract(parameters[i], update);
        }

        Discriminator.UpdateParameters(parameters);
        _discCurrentLearningRate = _initialLearningRate / (1.0 + _learningRateDecay * Convert.ToDouble(_discBeta1Power));
    }

    protected override void InitializeLayers() { }

    public override Tensor<T> Predict(Tensor<T> input) => Generator.Predict(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(input, expectedOutput);
    }

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

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_initialLearningRate);
        writer.Write(_learningRateDecay);
        writer.Write(_l1Lambda);

        // Write per-network learning rates
        writer.Write(_genCurrentLearningRate);
        writer.Write(_discCurrentLearningRate);

        var generatorBytes = Generator.Serialize();
        writer.Write(generatorBytes.Length);
        writer.Write(generatorBytes);

        var discriminatorBytes = Discriminator.Serialize();
        writer.Write(discriminatorBytes.Length);
        writer.Write(discriminatorBytes);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _initialLearningRate = reader.ReadDouble();
        _learningRateDecay = reader.ReadDouble();
        _l1Lambda = reader.ReadDouble();

        // Read per-network learning rates
        _genCurrentLearningRate = reader.ReadDouble();
        _discCurrentLearningRate = reader.ReadDouble();

        int generatorDataLength = reader.ReadInt32();
        byte[] generatorData = reader.ReadBytes(generatorDataLength);
        Generator.Deserialize(generatorData);

        int discriminatorDataLength = reader.ReadInt32();
        byte[] discriminatorData = reader.ReadBytes(discriminatorDataLength);
        Discriminator.Deserialize(discriminatorData);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new Pix2Pix<T>(
            Generator.Architecture,
            Discriminator.Architecture,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate,
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
