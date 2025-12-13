using System.IO;
using AiDotNet.Helpers;

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
    // GeneratorAtoB optimizer state
    private Vector<T> _genAtoBMomentum;
    private Vector<T> _genAtoBSecondMoment;
    private T _genAtoBBeta1Power;
    private T _genAtoBBeta2Power;
    private double _genAtoBCurrentLearningRate;

    // GeneratorBtoA optimizer state
    private Vector<T> _genBtoAMomentum;
    private Vector<T> _genBtoASecondMoment;
    private T _genBtoABeta1Power;
    private T _genBtoABeta2Power;
    private double _genBtoACurrentLearningRate;

    // DiscriminatorA optimizer state
    private Vector<T> _discAMomentum;
    private Vector<T> _discASecondMoment;
    private T _discABeta1Power;
    private T _discABeta2Power;
    private double _discACurrentLearningRate;

    // DiscriminatorB optimizer state
    private Vector<T> _discBMomentum;
    private Vector<T> _discBSecondMoment;
    private T _discBBeta1Power;
    private T _discBBeta2Power;
    private double _discBCurrentLearningRate;

    private double _initialLearningRate;
    private double _learningRateDecay;

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

    /// <summary>
    /// Creates the combined CycleGAN architecture with correct dimension handling.
    /// </summary>
    private static NeuralNetworkArchitecture<T> CreateCycleGANArchitecture(
        NeuralNetworkArchitecture<T> generatorAtoB,
        InputType inputType)
    {
        if (inputType == InputType.ThreeDimensional)
        {
            return new NeuralNetworkArchitecture<T>(
                inputType: inputType,
                taskType: NeuralNetworkTaskType.Generative,
                complexity: NetworkComplexity.Deep,
                inputSize: 0,
                inputHeight: generatorAtoB.InputHeight,
                inputWidth: generatorAtoB.InputWidth,
                inputDepth: generatorAtoB.InputDepth,
                outputSize: generatorAtoB.OutputSize,
                layers: null);
        }

        return new NeuralNetworkArchitecture<T>(
            inputType: inputType,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: generatorAtoB.InputSize,
            outputSize: generatorAtoB.OutputSize);
    }

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
        : base(CreateCycleGANArchitecture(generatorAtoB, inputType),
               lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative))
    {
        // Validate constructor inputs
        if (generatorAtoB is null)
        {
            throw new ArgumentNullException(nameof(generatorAtoB), "Generator A to B architecture cannot be null.");
        }
        if (generatorBtoA is null)
        {
            throw new ArgumentNullException(nameof(generatorBtoA), "Generator B to A architecture cannot be null.");
        }
        if (discriminatorA is null)
        {
            throw new ArgumentNullException(nameof(discriminatorA), "Discriminator A architecture cannot be null.");
        }
        if (discriminatorB is null)
        {
            throw new ArgumentNullException(nameof(discriminatorB), "Discriminator B architecture cannot be null.");
        }
        if (initialLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(initialLearningRate), initialLearningRate, "Initial learning rate must be positive.");
        }
        if (cycleConsistencyLambda < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(cycleConsistencyLambda), cycleConsistencyLambda, "Cycle consistency lambda must be non-negative.");
        }
        if (identityLambda < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(identityLambda), identityLambda, "Identity lambda must be non-negative.");
        }

        _cycleConsistencyLambda = cycleConsistencyLambda;
        _identityLambda = identityLambda;
        _initialLearningRate = initialLearningRate;
        _learningRateDecay = 0.0001;

        GeneratorAtoB = new ConvolutionalNeuralNetwork<T>(generatorAtoB);
        GeneratorBtoA = new ConvolutionalNeuralNetwork<T>(generatorBtoA);
        DiscriminatorA = new ConvolutionalNeuralNetwork<T>(discriminatorA);
        DiscriminatorB = new ConvolutionalNeuralNetwork<T>(discriminatorB);

        // Initialize GeneratorAtoB optimizer state
        // Beta powers start at beta^1 so first iteration's bias correction is non-zero
        int genAtoBParamCount = GeneratorAtoB.GetParameterCount();
        _genAtoBMomentum = new Vector<T>(genAtoBParamCount);
        _genAtoBSecondMoment = new Vector<T>(genAtoBParamCount);
        _genAtoBBeta1Power = NumOps.FromDouble(0.9);
        _genAtoBBeta2Power = NumOps.FromDouble(0.999);
        _genAtoBCurrentLearningRate = initialLearningRate;

        // Initialize GeneratorBtoA optimizer state
        int genBtoAParamCount = GeneratorBtoA.GetParameterCount();
        _genBtoAMomentum = new Vector<T>(genBtoAParamCount);
        _genBtoASecondMoment = new Vector<T>(genBtoAParamCount);
        _genBtoABeta1Power = NumOps.FromDouble(0.9);
        _genBtoABeta2Power = NumOps.FromDouble(0.999);
        _genBtoACurrentLearningRate = initialLearningRate;

        // Initialize DiscriminatorA optimizer state
        int discAParamCount = DiscriminatorA.GetParameterCount();
        _discAMomentum = new Vector<T>(discAParamCount);
        _discASecondMoment = new Vector<T>(discAParamCount);
        _discABeta1Power = NumOps.FromDouble(0.9);
        _discABeta2Power = NumOps.FromDouble(0.999);
        _discACurrentLearningRate = initialLearningRate;

        // Initialize DiscriminatorB optimizer state
        int discBParamCount = DiscriminatorB.GetParameterCount();
        _discBMomentum = new Vector<T>(discBParamCount);
        _discBSecondMoment = new Vector<T>(discBParamCount);
        _discBBeta1Power = NumOps.FromDouble(0.9);
        _discBBeta2Power = NumOps.FromDouble(0.999);
        _discBCurrentLearningRate = initialLearningRate;

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

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

        // Generate fake images (detached for discriminator training)
        var fakeB = GeneratorAtoB.Predict(realA);
        var fakeA = GeneratorBtoA.Predict(realB);

        // Train DiscriminatorA: real A vs fake A
        var realAPred = DiscriminatorA.Predict(realA);
        var fakeAPred = DiscriminatorA.Predict(fakeA);
        T discALoss = CalculateDiscriminatorLoss(realAPred, fakeAPred, batchSize);

        // Backprop for DiscriminatorA
        var realALabels = CreateLabelTensor(batchSize, NumOps.One);
        var fakeALabels = CreateLabelTensor(batchSize, NumOps.Zero);
        var realAGrad = CalculateBinaryGradients(realAPred, realALabels, batchSize);
        var fakeAGrad = CalculateBinaryGradients(fakeAPred, fakeALabels, batchSize);

        DiscriminatorA.Backward(realAGrad);
        DiscriminatorA.Backward(fakeAGrad);
        UpdateDiscriminatorAParameters();

        // Train DiscriminatorB: real B vs fake B
        var realBPred = DiscriminatorB.Predict(realB);
        var fakeBPred = DiscriminatorB.Predict(fakeB);
        T discBLoss = CalculateDiscriminatorLoss(realBPred, fakeBPred, batchSize);

        // Backprop for DiscriminatorB
        var realBLabels = CreateLabelTensor(batchSize, NumOps.One);
        var fakeBLabels = CreateLabelTensor(batchSize, NumOps.Zero);
        var realBGrad = CalculateBinaryGradients(realBPred, realBLabels, batchSize);
        var fakeBGrad = CalculateBinaryGradients(fakeBPred, fakeBLabels, batchSize);

        DiscriminatorB.Backward(realBGrad);
        DiscriminatorB.Backward(fakeBGrad);
        UpdateDiscriminatorBParameters();

        T discriminatorLoss = NumOps.Divide(NumOps.Add(discALoss, discBLoss), NumOps.FromDouble(2.0));

        // ----- Train Generators -----

        // Adversarial loss for GeneratorAtoB (fool DiscriminatorB)
        fakeB = GeneratorAtoB.Predict(realA);
        var fakeBPred2 = DiscriminatorB.Predict(fakeB);
        T advLossB = CalculateAdversarialLoss(fakeBPred2, batchSize);

        // Backprop adversarial loss through DiscriminatorB to GeneratorAtoB
        var genAtoBAdvGrad = CalculateBinaryGradients(fakeBPred2, CreateLabelTensor(batchSize, NumOps.One), batchSize);
        var discBInputGrad = DiscriminatorB.BackwardWithInputGradient(genAtoBAdvGrad);
        var genAtoBGrad = discBInputGrad.Clone();

        // Adversarial loss for GeneratorBtoA (fool DiscriminatorA)
        fakeA = GeneratorBtoA.Predict(realB);
        var fakeAPred2 = DiscriminatorA.Predict(fakeA);
        T advLossA = CalculateAdversarialLoss(fakeAPred2, batchSize);

        // Backprop adversarial loss through DiscriminatorA to GeneratorBtoA
        var genBtoAAdvGrad = CalculateBinaryGradients(fakeAPred2, CreateLabelTensor(batchSize, NumOps.One), batchSize);
        var discAInputGrad = DiscriminatorA.BackwardWithInputGradient(genBtoAAdvGrad);
        var genBtoAGrad = discAInputGrad.Clone();

        T advLoss = NumOps.Add(advLossB, advLossA);

        // Cycle consistency losses
        var reconstructedA = GeneratorBtoA.Predict(fakeB);
        var reconstructedB = GeneratorAtoB.Predict(fakeA);

        T cycleA = CalculateL1Loss(reconstructedA, realA);
        T cycleB = CalculateL1Loss(reconstructedB, realB);
        T cycleLoss = NumOps.Add(cycleA, cycleB);

        // Cycle consistency gradients: A -> B -> A (back to A)
        var cycleAGrad = CalculateL1Gradient(reconstructedA, realA, _cycleConsistencyLambda);
        var genBtoACycleGradInput = GeneratorBtoA.BackwardWithInputGradient(cycleAGrad);
        // Add cycle gradient to GeneratorAtoB (from fakeB that was used to reconstruct A)
        for (int i = 0; i < genAtoBGrad.Length; i++)
            genAtoBGrad.SetFlat(i, NumOps.Add(genAtoBGrad.GetFlat(i), genBtoACycleGradInput.GetFlat(i)));

        // Cycle consistency gradients: B -> A -> B (back to B)
        var cycleBGrad = CalculateL1Gradient(reconstructedB, realB, _cycleConsistencyLambda);
        var genAtoBCycleGradInput = GeneratorAtoB.BackwardWithInputGradient(cycleBGrad);
        // Add cycle gradient to GeneratorBtoA (from fakeA that was used to reconstruct B)
        for (int i = 0; i < genBtoAGrad.Length; i++)
            genBtoAGrad.SetFlat(i, NumOps.Add(genBtoAGrad.GetFlat(i), genAtoBCycleGradInput.GetFlat(i)));

        // Identity losses (optional, helps preserve color composition)
        var identityA = GeneratorBtoA.Predict(realA);
        var identityB = GeneratorAtoB.Predict(realB);

        T idLossA = CalculateL1Loss(identityA, realA);
        T idLossB = CalculateL1Loss(identityB, realB);
        T identityLoss = NumOps.Add(idLossA, idLossB);

        // Identity gradients
        var identityAGrad = CalculateL1Gradient(identityA, realA, _identityLambda);
        var identityBGrad = CalculateL1Gradient(identityB, realB, _identityLambda);

        // Apply combined gradients to generators
        GeneratorAtoB.Backward(genAtoBGrad);
        GeneratorAtoB.Backward(identityBGrad);
        UpdateGeneratorAtoBParameters();

        GeneratorBtoA.Backward(genBtoAGrad);
        GeneratorBtoA.Backward(identityAGrad);
        UpdateGeneratorBtoAParameters();

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
        int count = predictions.Length;

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.GetFlat(i), targets.GetFlat(i));
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

    private Tensor<T> CalculateBinaryGradients(Tensor<T> predictions, Tensor<T> targets, int batchSize)
    {
        var gradients = new Tensor<T>(predictions.Shape);
        T epsilon = NumOps.FromDouble(1e-10);
        T oneMinusEpsilon = NumOps.Subtract(NumOps.One, epsilon);

        for (int i = 0; i < batchSize; i++)
        {
            T p = predictions[i, 0];
            T t = targets[i, 0];

            // Clamp predictions to avoid numerical instability
            if (NumOps.LessThan(p, epsilon))
                p = epsilon;
            else if (NumOps.GreaterThan(p, oneMinusEpsilon))
                p = oneMinusEpsilon;

            // BCE gradient w.r.t. probability: dL/dp = (p - t) / (p * (1 - p))
            T oneMinusP = NumOps.Subtract(NumOps.One, p);
            T pTimesOneMinusP = NumOps.Multiply(p, oneMinusP);
            T gradient = NumOps.Divide(
                NumOps.Subtract(p, t),
                NumOps.Add(pTimesOneMinusP, epsilon)
            );
            gradients[i, 0] = NumOps.Divide(gradient, NumOps.FromDouble(batchSize));
        }

        return gradients;
    }

    private Tensor<T> CalculateL1Gradient(Tensor<T> predictions, Tensor<T> targets, double coefficient)
    {
        var gradients = new Tensor<T>(predictions.Shape);
        int count = predictions.Length;
        T coeff = NumOps.FromDouble(coefficient / count);

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.GetFlat(i), targets.GetFlat(i));
            // Sign of difference: 1 if positive, -1 if negative
            T sign = NumOps.GreaterThanOrEquals(diff, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
            gradients.SetFlat(i, NumOps.Multiply(coeff, sign));
        }

        return gradients;
    }

    /// <summary>
    /// Applies vectorized Adam update using Engine operations for SIMD/GPU acceleration.
    /// This follows the gold-standard pattern from the codebase's production-ready models.
    /// </summary>
    private Vector<T> ApplyVectorizedAdamUpdate(
        Vector<T> parameters,
        Vector<T> gradient,
        ref Vector<T> momentum,
        ref Vector<T> secondMoment,
        ref T beta1Power,
        ref T beta2Power,
        double learningRate)
    {
        T beta1 = NumOps.FromDouble(0.0); // GANs often use beta1=0
        T beta2 = NumOps.FromDouble(0.999);
        T oneMinusBeta1 = NumOps.FromDouble(1.0); // 1.0 - 0.0 = 1.0
        T oneMinusBeta2 = NumOps.FromDouble(0.001); // 1.0 - 0.999 = 0.001
        T epsilon = NumOps.FromDouble(1e-8);
        T lr = NumOps.FromDouble(learningRate);

        // Gradient clipping using vectorized L2 norm
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradient.L2Norm();

        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradient = (Vector<T>)Engine.Multiply(gradient, scaleFactor);
        }

        // Update beta powers
        beta1Power = NumOps.Multiply(beta1Power, beta1);
        beta2Power = NumOps.Multiply(beta2Power, beta2);

        // Compute bias correction factors
        T biasCorrection1 = NumOps.Subtract(NumOps.One, beta1Power);
        T biasCorrection2 = NumOps.Subtract(NumOps.One, beta2Power);

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        var mScaled = (Vector<T>)Engine.Multiply(momentum, beta1);
        var gradScaled = (Vector<T>)Engine.Multiply(gradient, oneMinusBeta1);
        momentum = (Vector<T>)Engine.Add(mScaled, gradScaled);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        var gradSquared = (Vector<T>)Engine.Multiply(gradient, gradient);
        var vScaled = (Vector<T>)Engine.Multiply(secondMoment, beta2);
        var gradSquaredScaled = (Vector<T>)Engine.Multiply(gradSquared, oneMinusBeta2);
        secondMoment = (Vector<T>)Engine.Add(vScaled, gradSquaredScaled);

        // Compute bias-corrected first moment: mHat = m / (1 - beta1^t)
        var mHat = (Vector<T>)Engine.Divide(momentum, biasCorrection1);

        // Compute bias-corrected second moment: vHat = v / (1 - beta2^t)
        var vHat = (Vector<T>)Engine.Divide(secondMoment, biasCorrection2);

        // Compute update: update = lr * mHat / (sqrt(vHat) + epsilon)
        var vHatSqrt = (Vector<T>)Engine.Sqrt(vHat);
        var epsilonVec = Vector<T>.CreateDefault(vHatSqrt.Length, epsilon);
        var denominator = (Vector<T>)Engine.Add(vHatSqrt, epsilonVec);
        var updateDiv = (Vector<T>)Engine.Divide(mHat, denominator);
        var update = (Vector<T>)Engine.Multiply(updateDiv, lr);

        // Apply update: parameters = parameters - update
        return (Vector<T>)Engine.Subtract(parameters, update);
    }

    private void UpdateGeneratorAtoBParameters()
    {
        var parameters = GeneratorAtoB.GetParameters();
        var gradients = GeneratorAtoB.GetParameterGradients();

        // Vectorized Adam update using Engine operations
        var updatedParams = ApplyVectorizedAdamUpdate(
            parameters, gradients,
            ref _genAtoBMomentum, ref _genAtoBSecondMoment,
            ref _genAtoBBeta1Power, ref _genAtoBBeta2Power,
            _genAtoBCurrentLearningRate);

        // Apply learning rate decay
        _genAtoBCurrentLearningRate = _initialLearningRate / (1.0 + _learningRateDecay * Convert.ToDouble(_genAtoBBeta1Power));

        GeneratorAtoB.UpdateParameters(updatedParams);
    }

    private void UpdateGeneratorBtoAParameters()
    {
        var parameters = GeneratorBtoA.GetParameters();
        var gradients = GeneratorBtoA.GetParameterGradients();

        // Vectorized Adam update using Engine operations
        var updatedParams = ApplyVectorizedAdamUpdate(
            parameters, gradients,
            ref _genBtoAMomentum, ref _genBtoASecondMoment,
            ref _genBtoABeta1Power, ref _genBtoABeta2Power,
            _genBtoACurrentLearningRate);

        // Apply learning rate decay
        _genBtoACurrentLearningRate = _initialLearningRate / (1.0 + _learningRateDecay * Convert.ToDouble(_genBtoABeta1Power));

        GeneratorBtoA.UpdateParameters(updatedParams);
    }

    private void UpdateDiscriminatorAParameters()
    {
        var parameters = DiscriminatorA.GetParameters();
        var gradients = DiscriminatorA.GetParameterGradients();

        // Vectorized Adam update using Engine operations
        var updatedParams = ApplyVectorizedAdamUpdate(
            parameters, gradients,
            ref _discAMomentum, ref _discASecondMoment,
            ref _discABeta1Power, ref _discABeta2Power,
            _discACurrentLearningRate);

        // Apply learning rate decay
        _discACurrentLearningRate = _initialLearningRate / (1.0 + _learningRateDecay * Convert.ToDouble(_discABeta1Power));

        DiscriminatorA.UpdateParameters(updatedParams);
    }

    private void UpdateDiscriminatorBParameters()
    {
        var parameters = DiscriminatorB.GetParameters();
        var gradients = DiscriminatorB.GetParameterGradients();

        // Vectorized Adam update using Engine operations
        var updatedParams = ApplyVectorizedAdamUpdate(
            parameters, gradients,
            ref _discBMomentum, ref _discBSecondMoment,
            ref _discBBeta1Power, ref _discBBeta2Power,
            _discBCurrentLearningRate);

        // Apply learning rate decay
        _discBCurrentLearningRate = _initialLearningRate / (1.0 + _learningRateDecay * Convert.ToDouble(_discBBeta1Power));

        DiscriminatorB.UpdateParameters(updatedParams);
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
        writer.Write(_initialLearningRate);
        writer.Write(_learningRateDecay);
        writer.Write(_cycleConsistencyLambda);
        writer.Write(_identityLambda);

        // Write per-network learning rates
        writer.Write(_genAtoBCurrentLearningRate);
        writer.Write(_genBtoACurrentLearningRate);
        writer.Write(_discACurrentLearningRate);
        writer.Write(_discBCurrentLearningRate);

        // Serialize GeneratorAtoB optimizer state
        SerializationHelper<T>.SerializeVector(writer, _genAtoBMomentum);
        SerializationHelper<T>.SerializeVector(writer, _genAtoBSecondMoment);
        SerializationHelper<T>.WriteValue(writer, _genAtoBBeta1Power);
        SerializationHelper<T>.WriteValue(writer, _genAtoBBeta2Power);

        // Serialize GeneratorBtoA optimizer state
        SerializationHelper<T>.SerializeVector(writer, _genBtoAMomentum);
        SerializationHelper<T>.SerializeVector(writer, _genBtoASecondMoment);
        SerializationHelper<T>.WriteValue(writer, _genBtoABeta1Power);
        SerializationHelper<T>.WriteValue(writer, _genBtoABeta2Power);

        // Serialize DiscriminatorA optimizer state
        SerializationHelper<T>.SerializeVector(writer, _discAMomentum);
        SerializationHelper<T>.SerializeVector(writer, _discASecondMoment);
        SerializationHelper<T>.WriteValue(writer, _discABeta1Power);
        SerializationHelper<T>.WriteValue(writer, _discABeta2Power);

        // Serialize DiscriminatorB optimizer state
        SerializationHelper<T>.SerializeVector(writer, _discBMomentum);
        SerializationHelper<T>.SerializeVector(writer, _discBSecondMoment);
        SerializationHelper<T>.WriteValue(writer, _discBBeta1Power);
        SerializationHelper<T>.WriteValue(writer, _discBBeta2Power);

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
        _initialLearningRate = reader.ReadDouble();
        _learningRateDecay = reader.ReadDouble();
        _cycleConsistencyLambda = reader.ReadDouble();
        _identityLambda = reader.ReadDouble();

        // Read per-network learning rates
        _genAtoBCurrentLearningRate = reader.ReadDouble();
        _genBtoACurrentLearningRate = reader.ReadDouble();
        _discACurrentLearningRate = reader.ReadDouble();
        _discBCurrentLearningRate = reader.ReadDouble();

        // Deserialize GeneratorAtoB optimizer state
        _genAtoBMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _genAtoBSecondMoment = SerializationHelper<T>.DeserializeVector(reader);
        _genAtoBBeta1Power = SerializationHelper<T>.ReadValue(reader);
        _genAtoBBeta2Power = SerializationHelper<T>.ReadValue(reader);

        // Deserialize GeneratorBtoA optimizer state
        _genBtoAMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _genBtoASecondMoment = SerializationHelper<T>.DeserializeVector(reader);
        _genBtoABeta1Power = SerializationHelper<T>.ReadValue(reader);
        _genBtoABeta2Power = SerializationHelper<T>.ReadValue(reader);

        // Deserialize DiscriminatorA optimizer state
        _discAMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _discASecondMoment = SerializationHelper<T>.DeserializeVector(reader);
        _discABeta1Power = SerializationHelper<T>.ReadValue(reader);
        _discABeta2Power = SerializationHelper<T>.ReadValue(reader);

        // Deserialize DiscriminatorB optimizer state
        _discBMomentum = SerializationHelper<T>.DeserializeVector(reader);
        _discBSecondMoment = SerializationHelper<T>.DeserializeVector(reader);
        _discBBeta1Power = SerializationHelper<T>.ReadValue(reader);
        _discBBeta2Power = SerializationHelper<T>.ReadValue(reader);

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

    /// <summary>
    /// Updates the parameters of all networks in the CycleGAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int genAtoBCount = GeneratorAtoB.GetParameterCount();
        int genBtoACount = GeneratorBtoA.GetParameterCount();
        int discACount = DiscriminatorA.GetParameterCount();
        int discBCount = DiscriminatorB.GetParameterCount();

        int offset = 0;

        // Update GeneratorAtoB parameters
        var genAtoBParams = new Vector<T>(genAtoBCount);
        for (int i = 0; i < genAtoBCount; i++)
            genAtoBParams[i] = parameters[offset + i];
        GeneratorAtoB.UpdateParameters(genAtoBParams);
        offset += genAtoBCount;

        // Update GeneratorBtoA parameters
        var genBtoAParams = new Vector<T>(genBtoACount);
        for (int i = 0; i < genBtoACount; i++)
            genBtoAParams[i] = parameters[offset + i];
        GeneratorBtoA.UpdateParameters(genBtoAParams);
        offset += genBtoACount;

        // Update DiscriminatorA parameters
        var discAParams = new Vector<T>(discACount);
        for (int i = 0; i < discACount; i++)
            discAParams[i] = parameters[offset + i];
        DiscriminatorA.UpdateParameters(discAParams);
        offset += discACount;

        // Update DiscriminatorB parameters
        var discBParams = new Vector<T>(discBCount);
        for (int i = 0; i < discBCount; i++)
            discBParams[i] = parameters[offset + i];
        DiscriminatorB.UpdateParameters(discBParams);
    }
}
