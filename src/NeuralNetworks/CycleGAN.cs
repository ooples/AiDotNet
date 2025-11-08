namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a CycleGAN for unpaired image-to-image translation.
/// </summary>
/// <remarks>
/// <para>
/// CycleGAN enables image-to-image translation without paired training data:
/// - Uses two generators (A→B and B→A) and two discriminators
/// - Enforces cycle consistency: A→B→A should equal A
/// - Works without paired examples (e.g., can learn horses→zebras from separate collections)
/// - Uses adversarial loss + cycle consistency loss + identity loss
/// </para>
/// <para><b>For Beginners:</b> CycleGAN translates images without matched pairs.
///
/// Key innovation:
/// - Doesn't need paired training data
/// - Learns from two separate collections of images
/// - Example: Photos of horses + Photos of zebras → can convert horses to zebras
///
/// How it works:
/// - Two generators: G (A→B) and F (B→A)
/// - Two discriminators: D_A and D_B
/// - Cycle consistency: G(F(B)) ≈ B and F(G(A)) ≈ A
/// - This prevents mode collapse and maintains content
///
/// Applications:
/// - Style transfer (Monet → Photo, Photo → Monet)
/// - Season transfer (Summer → Winter)
/// - Object transfiguration (Horse → Zebra)
/// - Domain adaptation
///
/// Reference: Zhu et al., "Unpaired Image-to-Image Translation using
/// Cycle-Consistent Adversarial Networks" (2017)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class CycleGAN<T> : NeuralNetworkBase<T>
{
    private Vector<T> _momentum;
    private Vector<T> _secondMoment;
    private T _beta1Power;
    private T _beta2Power;
    private double _currentLearningRate;
    private double _initialLearningRate;

    /// <summary>
    /// Coefficient for cycle consistency loss.
    /// </summary>
    /// <remarks>
    /// Controls the importance of cycle consistency. Typical value: 10.0.
    /// Higher values enforce stronger cycle consistency.
    /// </remarks>
    private double _cycleConsistencyLambda;

    /// <summary>
    /// Coefficient for identity loss.
    /// </summary>
    /// <remarks>
    /// Encourages G and F to preserve color composition. Typical value: 0.5 * cycleConsistencyLambda.
    /// </remarks>
    private double _identityLambda;

    /// <summary>
    /// Generator A→B.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> GeneratorAtoB { get; private set; }

    /// <summary>
    /// Generator B→A.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> GeneratorBtoA { get; private set; }

    /// <summary>
    /// Discriminator for domain A.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> DiscriminatorA { get; private set; }

    /// <summary>
    /// Discriminator for domain B.
    /// </summary>
    public ConvolutionalNeuralNetwork<T> DiscriminatorB { get; private set; }

    private ILossFunction<T> _lossFunction;

    public CycleGAN(
        NeuralNetworkArchitecture<T> generatorAtoB,
        NeuralNetworkArchitecture<T> generatorBtoA,
        NeuralNetworkArchitecture<T> discriminatorA,
        NeuralNetworkArchitecture<T> discriminatorB,
        InputType inputType,
        ILossFunction<T>? lossFunction = null,
        double initialLearningRate = 0.0002,
        double cycleConsistencyLambda = 10.0,
        double identityLambda = 5.0)
        : base(new NeuralNetworkArchitecture<T>(
            inputType,
            NeuralNetworkTaskType.ImageToImage,
            NetworkComplexity.High,
            generatorAtoB.InputSize,
            generatorAtoB.OutputSize,
            0, 0, 0,
            null), lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.ImageToImage))
    {
        _cycleConsistencyLambda = cycleConsistencyLambda;
        _identityLambda = identityLambda;
        _initialLearningRate = initialLearningRate;
        _currentLearningRate = initialLearningRate;

        _beta1Power = NumOps.One;
        _beta2Power = NumOps.One;

        GeneratorAtoB = new ConvolutionalNeuralNetwork<T>(generatorAtoB);
        GeneratorBtoA = new ConvolutionalNeuralNetwork<T>(generatorBtoA);
        DiscriminatorA = new ConvolutionalNeuralNetwork<T>(discriminatorA);
        DiscriminatorB = new ConvolutionalNeuralNetwork<T>(discriminatorB);

        _momentum = Vector<T>.Empty();
        _secondMoment = Vector<T>.Empty();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.ImageToImage);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for CycleGAN.
    /// </summary>
    public (T discLoss, T genLoss, T cycleLoss) TrainStep(
        Tensor<T> realA,
        Tensor<T> realB)
    {
        int batchSize = realA.Shape[0];

        // ----- Train Discriminators -----

        // Generate fake images
        var fakeB = GeneratorAtoB.Predict(realA);
        var fakeA = GeneratorBtoA.Predict(realB);

        // Train discriminator A
        var realAPred = DiscriminatorA.Predict(realA);
        var fakeAPred = DiscriminatorA.Predict(fakeA);
        T discALoss = CalculateDiscriminatorLoss(realAPred, fakeAPred, batchSize);

        // Train discriminator B
        var realBPred = DiscriminatorB.Predict(realB);
        var fakeBPred = DiscriminatorB.Predict(fakeB);
        T discBLoss = CalculateDiscriminatorLoss(realBPred, fakeBPred, batchSize);

        T discriminatorLoss = NumOps.Divide(NumOps.Add(discALoss, discBLoss), NumOps.FromDouble(2.0));

        // ----- Train Generators -----

        // Adversarial losses
        fakeB = GeneratorAtoB.Predict(realA);
        fakeA = GeneratorBtoA.Predict(realB);

        var fakeBPred2 = DiscriminatorB.Predict(fakeB);
        var fakeAPred2 = DiscriminatorA.Predict(fakeA);

        T advLossB = CalculateAdversarialLoss(fakeBPred2, batchSize);
        T advLossA = CalculateAdversarialLoss(fakeAPred2, batchSize);
        T advLoss = NumOps.Add(advLossB, advLossA);

        // Cycle consistency losses
        var reconstructedA = GeneratorBtoA.Predict(fakeB);
        var reconstructedB = GeneratorAtoB.Predict(fakeA);

        T cycleA = CalculateL1Loss(reconstructedA, realA);
        T cycleB = CalculateL1Loss(reconstructedB, realB);
        T cycleLoss = NumOps.Add(cycleA, cycleB);

        // Identity losses (optional)
        var identityA = GeneratorBtoA.Predict(realA);
        var identityB = GeneratorAtoB.Predict(realB);

        T idLossA = CalculateL1Loss(identityA, realA);
        T idLossB = CalculateL1Loss(identityB, realB);
        T identityLoss = NumOps.Add(idLossA, idLossB);

        // Total generator loss
        T cycleCoeff = NumOps.FromDouble(_cycleConsistencyLambda);
        T idCoeff = NumOps.FromDouble(_identityLambda);

        T generatorLoss = NumOps.Add(advLoss,
            NumOps.Add(
                NumOps.Multiply(cycleCoeff, cycleLoss),
                NumOps.Multiply(idCoeff, identityLoss)
            )
        );

        return (discriminatorLoss, generatorLoss, cycleLoss);
    }

    /// <summary>
    /// Translates image from domain A to domain B.
    /// </summary>
    public Tensor<T> TranslateAtoB(Tensor<T> imageA)
    {
        GeneratorAtoB.SetTrainingMode(false);
        return GeneratorAtoB.Predict(imageA);
    }

    /// <summary>
    /// Translates image from domain B to domain A.
    /// </summary>
    public Tensor<T> TranslateBtoA(Tensor<T> imageB)
    {
        GeneratorBtoA.SetTrainingMode(false);
        return GeneratorBtoA.Predict(imageB);
    }

    private T CalculateDiscriminatorLoss(Tensor<T> realPred, Tensor<T> fakePred, int batchSize)
    {
        var realLabels = CreateLabelTensor(batchSize, NumOps.One);
        var fakeLabels = CreateLabelTensor(batchSize, NumOps.Zero);

        T realLoss = CalculateBinaryLoss(realPred, realLabels, batchSize);
        T fakeLoss = CalculateBinaryLoss(fakePred, fakeLabels, batchSize);

        return NumOps.Divide(NumOps.Add(realLoss, fakeLoss), NumOps.FromDouble(2.0));
    }

    private T CalculateAdversarialLoss(Tensor<T> predictions, int batchSize)
    {
        var realLabels = CreateLabelTensor(batchSize, NumOps.One);
        return CalculateBinaryLoss(predictions, realLabels, batchSize);
    }

    private T CalculateL1Loss(Tensor<T> predictions, Tensor<T> targets)
    {
        T totalLoss = NumOps.Zero;
        int count = predictions.Data.Length;

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data[i], targets.Data[i]);
            T absDiff = NumOps.GreaterThanOrEquals(diff, NumOps.Zero) ? diff : NumOps.Negate(diff);
            totalLoss = NumOps.Add(totalLoss, absDiff);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(count));
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

    private Tensor<T> CreateLabelTensor(int batchSize, T value)
    {
        var tensor = new Tensor<T>(new int[] { batchSize, 1 });
        for (int i = 0; i < batchSize; i++)
        {
            tensor[i, 0] = value;
        }
        return tensor;
    }

    protected override void InitializeLayers() { }

    public override Tensor<T> Predict(Tensor<T> input) => GeneratorAtoB.Predict(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        TrainStep(input, expectedOutput);
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.CycleGAN,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "GeneratorAtoB_Parameters", GeneratorAtoB.GetParameterCount() },
                { "GeneratorBtoA_Parameters", GeneratorBtoA.GetParameterCount() },
                { "DiscriminatorA_Parameters", DiscriminatorA.GetParameterCount() },
                { "DiscriminatorB_Parameters", DiscriminatorB.GetParameterCount() },
                { "CycleConsistencyLambda", _cycleConsistencyLambda },
                { "IdentityLambda", _identityLambda }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_currentLearningRate);
        writer.Write(_cycleConsistencyLambda);
        writer.Write(_identityLambda);

        var genAtoB = GeneratorAtoB.Serialize();
        writer.Write(genAtoB.Length);
        writer.Write(genAtoB);

        var genBtoA = GeneratorBtoA.Serialize();
        writer.Write(genBtoA.Length);
        writer.Write(genBtoA);

        var discA = DiscriminatorA.Serialize();
        writer.Write(discA.Length);
        writer.Write(discA);

        var discB = DiscriminatorB.Serialize();
        writer.Write(discB.Length);
        writer.Write(discB);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _currentLearningRate = reader.ReadDouble();
        _cycleConsistencyLambda = reader.ReadDouble();
        _identityLambda = reader.ReadDouble();

        int genAtoB_Length = reader.ReadInt32();
        GeneratorAtoB.Deserialize(reader.ReadBytes(genAtoB_Length));

        int genBtoA_Length = reader.ReadInt32();
        GeneratorBtoA.Deserialize(reader.ReadBytes(genBtoA_Length));

        int discA_Length = reader.ReadInt32();
        DiscriminatorA.Deserialize(reader.ReadBytes(discA_Length));

        int discB_Length = reader.ReadInt32();
        DiscriminatorB.Deserialize(reader.ReadBytes(discB_Length));
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CycleGAN<T>(
            GeneratorAtoB.Architecture,
            GeneratorBtoA.Architecture,
            DiscriminatorA.Architecture,
            DiscriminatorB.Architecture,
            Architecture.InputType,
            _lossFunction,
            _initialLearningRate,
            _cycleConsistencyLambda,
            _identityLambda);
    }
}
