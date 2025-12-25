using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Optimizers;

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
    /// <summary>
    /// The optimizer used for training generator A→B.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the generator
    /// that transforms images from domain A to domain B. The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how the A→B generator
    /// learns from its mistakes and adjusts its parameters during training.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorAtoBOptimizer;

    /// <summary>
    /// The optimizer used for training generator B→A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the generator
    /// that transforms images from domain B to domain A. The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how the B→A generator
    /// learns from its mistakes and adjusts its parameters during training.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _generatorBtoAOptimizer;

    /// <summary>
    /// The optimizer used for training discriminator A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the discriminator
    /// that evaluates images in domain A (real vs. generated). The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how discriminator A
    /// learns to better distinguish real images from fake ones in domain A.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorAOptimizer;

    /// <summary>
    /// The optimizer used for training discriminator B.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This optimizer manages the gradient-based parameter updates for the discriminator
    /// that evaluates images in domain B (real vs. generated). The optimizer handles momentum,
    /// adaptive learning rates, and other algorithm-specific state.
    /// </para>
    /// <para><b>For Beginners:</b> This optimizer controls how discriminator B
    /// learns to better distinguish real images from fake ones in domain B.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _discriminatorBOptimizer;

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

    /// <summary>
    /// Initializes a new instance of the <see cref="CycleGAN{T}"/> class with the specified architecture and training parameters.
    /// </summary>
    /// <param name="generatorAtoB">The architecture for the generator that transforms images from domain A to domain B.</param>
    /// <param name="generatorBtoA">The architecture for the generator that transforms images from domain B to domain A.</param>
    /// <param name="discriminatorA">The architecture for the discriminator that evaluates images in domain A.</param>
    /// <param name="discriminatorB">The architecture for the discriminator that evaluates images in domain B.</param>
    /// <param name="inputType">The type of input data (e.g., ThreeDimensional for images).</param>
    /// <param name="generatorAtoBOptimizer">
    /// Optional optimizer for the A→B generator. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="generatorBtoAOptimizer">
    /// Optional optimizer for the B→A generator. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="discriminatorAOptimizer">
    /// Optional optimizer for discriminator A. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="discriminatorBOptimizer">
    /// Optional optimizer for discriminator B. If null, an Adam optimizer with default GAN settings is created.
    /// </param>
    /// <param name="lossFunction">Optional loss function. If null, the default loss function for generative tasks is used.</param>
    /// <param name="cycleConsistencyLambda">
    /// The coefficient for cycle consistency loss. Higher values enforce stronger cycle consistency. Default is 10.0.
    /// </param>
    /// <param name="identityLambda">
    /// The coefficient for identity loss. Helps preserve color composition. Default is 5.0.
    /// </param>
    /// <remarks>
    /// <para>
    /// This constructor creates a CycleGAN with four separate networks and optimizers:
    /// - Generator A→B: Transforms images from domain A to domain B
    /// - Generator B→A: Transforms images from domain B to domain A
    /// - Discriminator A: Evaluates whether images in domain A are real or generated
    /// - Discriminator B: Evaluates whether images in domain B are real or generated
    /// </para>
    /// <para><b>For Beginners:</b> CycleGAN needs four networks to work:
    /// - Two generators to translate images in both directions
    /// - Two discriminators to judge images in each domain
    ///
    /// The cycle consistency loss ensures that translating A→B→A gets back to the original,
    /// which helps maintain content while only changing style.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// Thrown when any of the architecture parameters is null.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when cycleConsistencyLambda or identityLambda is negative.
    /// </exception>
    public CycleGAN(
        NeuralNetworkArchitecture<T> generatorAtoB,
        NeuralNetworkArchitecture<T> generatorBtoA,
        NeuralNetworkArchitecture<T> discriminatorA,
        NeuralNetworkArchitecture<T> discriminatorB,
        InputType inputType,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorAtoBOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? generatorBtoAOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorAOptimizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? discriminatorBOptimizer = null,
        ILossFunction<T>? lossFunction = null,
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

        GeneratorAtoB = new ConvolutionalNeuralNetwork<T>(generatorAtoB);
        GeneratorBtoA = new ConvolutionalNeuralNetwork<T>(generatorBtoA);
        DiscriminatorA = new ConvolutionalNeuralNetwork<T>(discriminatorA);
        DiscriminatorB = new ConvolutionalNeuralNetwork<T>(discriminatorB);

        // Initialize optimizers - use provided optimizers or create default Adam optimizers
        _generatorAtoBOptimizer = generatorAtoBOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(GeneratorAtoB);
        _generatorBtoAOptimizer = generatorBtoAOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(GeneratorBtoA);
        _discriminatorAOptimizer = discriminatorAOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(DiscriminatorA);
        _discriminatorBOptimizer = discriminatorBOptimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(DiscriminatorB);

        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(NeuralNetworkTaskType.Generative);

        InitializeLayers();
    }

    /// <summary>
    /// Performs one training step for CycleGAN.
    /// </summary>
    /// <param name="realA">Real images from domain A.</param>
    /// <param name="realB">Real images from domain B.</param>
    /// <returns>A tuple containing discriminator loss, generator loss, and cycle consistency loss.</returns>
    /// <exception cref="ArgumentNullException">Thrown when realA or realB is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batch dimensions don't match or batch size is zero.</exception>
    public (T discLoss, T genLoss, T cycleLoss) TrainStep(
        Tensor<T> realA,
        Tensor<T> realB)
    {
        // Validate input tensors
        if (realA is null)
        {
            throw new ArgumentNullException(nameof(realA), "Real images from domain A cannot be null.");
        }

        if (realB is null)
        {
            throw new ArgumentNullException(nameof(realB), "Real images from domain B cannot be null.");
        }

        int batchSize = realA.Shape[0];

        if (batchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive.", nameof(realA));
        }

        if (realB.Shape[0] != batchSize)
        {
            throw new ArgumentException(
                $"Batch size mismatch: realA has batch size {batchSize}, but realB has batch size {realB.Shape[0]}. " +
                "Both tensors must have the same batch dimension.",
                nameof(realB));
        }

        // ----- Train Discriminators -----

        // Generate fake images (detached for discriminator training)
        var fakeB = GeneratorAtoB.Predict(realA);
        var fakeA = GeneratorBtoA.Predict(realB);

        // Train DiscriminatorA: real A vs fake A
        // Process real first: predict, compute grad, backward (preserves forward cache)
        var realAPred = DiscriminatorA.Predict(realA);
        var realALabels = CreateLabelTensor(batchSize, NumOps.One);
        var realAGrad = CalculateBinaryGradients(realAPred, realALabels, batchSize);
        DiscriminatorA.Backward(realAGrad);

        // Process fake second: predict, compute grad, backward (uses correct forward cache)
        var fakeAPred = DiscriminatorA.Predict(fakeA);
        var fakeALabels = CreateLabelTensor(batchSize, NumOps.Zero);
        var fakeAGrad = CalculateBinaryGradients(fakeAPred, fakeALabels, batchSize);
        DiscriminatorA.Backward(fakeAGrad);

        T discALoss = CalculateDiscriminatorLoss(realAPred, fakeAPred, batchSize);
        UpdateDiscriminatorAParameters();

        // Train DiscriminatorB: real B vs fake B
        // Process real first: predict, compute grad, backward (preserves forward cache)
        var realBPred = DiscriminatorB.Predict(realB);
        var realBLabels = CreateLabelTensor(batchSize, NumOps.One);
        var realBGrad = CalculateBinaryGradients(realBPred, realBLabels, batchSize);
        DiscriminatorB.Backward(realBGrad);

        // Process fake second: predict, compute grad, backward (uses correct forward cache)
        var fakeBPred = DiscriminatorB.Predict(fakeB);
        var fakeBLabels = CreateLabelTensor(batchSize, NumOps.Zero);
        var fakeBGrad = CalculateBinaryGradients(fakeBPred, fakeBLabels, batchSize);
        DiscriminatorB.Backward(fakeBGrad);

        T discBLoss = CalculateDiscriminatorLoss(realBPred, fakeBPred, batchSize);
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

        // Combine all generator gradients before backward to avoid gradient loss
        // GeneratorAtoB receives: adversarial + cycle + identity gradients
        var combinedGenAtoBGrad = new Tensor<T>(genAtoBGrad.Shape);
        for (int i = 0; i < genAtoBGrad.Length; i++)
            combinedGenAtoBGrad.SetFlat(i, NumOps.Add(genAtoBGrad.GetFlat(i), identityBGrad.GetFlat(i)));

        GeneratorAtoB.Backward(combinedGenAtoBGrad);
        UpdateGeneratorAtoBParameters();

        // GeneratorBtoA receives: adversarial + cycle + identity gradients
        var combinedGenBtoAGrad = new Tensor<T>(genBtoAGrad.Shape);
        for (int i = 0; i < genBtoAGrad.Length; i++)
            combinedGenBtoAGrad.SetFlat(i, NumOps.Add(genBtoAGrad.GetFlat(i), identityAGrad.GetFlat(i)));

        GeneratorBtoA.Backward(combinedGenBtoAGrad);
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
    /// <remarks>
    /// <para>
    /// This method temporarily sets the generator to evaluation mode for inference,
    /// then restores the original training mode after prediction. This ensures
    /// batch normalization and dropout behave correctly during both inference
    /// and subsequent training steps.
    /// </para>
    /// </remarks>
    public Tensor<T> TranslateAtoB(Tensor<T> imageA)
    {
        bool originalTrainingMode = GeneratorAtoB.IsTrainingMode;
        GeneratorAtoB.SetTrainingMode(false);
        var result = GeneratorAtoB.Predict(imageA);
        GeneratorAtoB.SetTrainingMode(originalTrainingMode);
        return result;
    }

    /// <summary>
    /// Translates image from domain B to domain A.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method temporarily sets the generator to evaluation mode for inference,
    /// then restores the original training mode after prediction. This ensures
    /// batch normalization and dropout behave correctly during both inference
    /// and subsequent training steps.
    /// </para>
    /// </remarks>
    public Tensor<T> TranslateBtoA(Tensor<T> imageB)
    {
        bool originalTrainingMode = GeneratorBtoA.IsTrainingMode;
        GeneratorBtoA.SetTrainingMode(false);
        var result = GeneratorBtoA.Predict(imageB);
        GeneratorBtoA.SetTrainingMode(originalTrainingMode);
        return result;
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
        // === Vectorized tensor fill using IEngine (Phase B: US-GPU-015) ===
        Engine.TensorFill(tensor, value);
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
    /// Updates the parameters of the generator A→B network using its optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method retrieves the current parameters and gradients from the generator,
    /// applies gradient clipping for training stability, and uses the configured optimizer
    /// to compute parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the A→B generator's weights
    /// based on how well it fooled the discriminator and maintained cycle consistency.
    /// </para>
    /// </remarks>
    private void UpdateGeneratorAtoBParameters()
    {
        var parameters = GeneratorAtoB.GetParameters();
        var gradients = GeneratorAtoB.GetParameterGradients();

        // Gradient clipping for training stability
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradients.L2Norm();
        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParams = _generatorAtoBOptimizer.UpdateParameters(parameters, gradients);
        GeneratorAtoB.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Updates the parameters of the generator B→A network using its optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method retrieves the current parameters and gradients from the generator,
    /// applies gradient clipping for training stability, and uses the configured optimizer
    /// to compute parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts the B→A generator's weights
    /// based on how well it fooled the discriminator and maintained cycle consistency.
    /// </para>
    /// </remarks>
    private void UpdateGeneratorBtoAParameters()
    {
        var parameters = GeneratorBtoA.GetParameters();
        var gradients = GeneratorBtoA.GetParameterGradients();

        // Gradient clipping for training stability
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradients.L2Norm();
        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParams = _generatorBtoAOptimizer.UpdateParameters(parameters, gradients);
        GeneratorBtoA.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Updates the parameters of discriminator A using its optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method retrieves the current parameters and gradients from the discriminator,
    /// applies gradient clipping for training stability, and uses the configured optimizer
    /// to compute parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts discriminator A's weights
    /// based on how well it distinguished real images from generated ones in domain A.
    /// </para>
    /// </remarks>
    private void UpdateDiscriminatorAParameters()
    {
        var parameters = DiscriminatorA.GetParameters();
        var gradients = DiscriminatorA.GetParameterGradients();

        // Gradient clipping for training stability
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradients.L2Norm();
        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParams = _discriminatorAOptimizer.UpdateParameters(parameters, gradients);
        DiscriminatorA.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Updates the parameters of discriminator B using its optimizer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method retrieves the current parameters and gradients from the discriminator,
    /// applies gradient clipping for training stability, and uses the configured optimizer
    /// to compute parameter updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts discriminator B's weights
    /// based on how well it distinguished real images from generated ones in domain B.
    /// </para>
    /// </remarks>
    private void UpdateDiscriminatorBParameters()
    {
        var parameters = DiscriminatorB.GetParameters();
        var gradients = DiscriminatorB.GetParameterGradients();

        // Gradient clipping for training stability
        T maxGradNorm = NumOps.FromDouble(5.0);
        T gradientNorm = gradients.L2Norm();
        if (NumOps.GreaterThan(gradientNorm, maxGradNorm))
        {
            T scaleFactor = NumOps.Divide(maxGradNorm, gradientNorm);
            gradients = (Vector<T>)Engine.Multiply(gradients, scaleFactor);
        }

        var updatedParams = _discriminatorBOptimizer.UpdateParameters(parameters, gradients);
        DiscriminatorB.UpdateParameters(updatedParams);
    }

    /// <summary>
    /// Resets the state of all optimizers to their initial values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets all four optimizers (both generators and both discriminators)
    /// to their initial state. This is useful when restarting training or when you want
    /// to clear accumulated momentum and adaptive learning rate information.
    /// </para>
    /// <para><b>For Beginners:</b> Call this method when you want to start fresh with
    /// training, as if the model had never been trained before. The network weights
    /// remain unchanged, but the optimizer's memory of past gradients is cleared.
    /// </para>
    /// </remarks>
    public void ResetOptimizerState()
    {
        _generatorAtoBOptimizer.Reset();
        _generatorBtoAOptimizer.Reset();
        _discriminatorAOptimizer.Reset();
        _discriminatorBOptimizer.Reset();
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

    /// <summary>
    /// Serializes CycleGAN-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the CycleGAN-specific configuration and all four networks.
    /// Optimizer state is managed by the optimizer implementations themselves.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the CycleGAN's settings and all
    /// four networks (two generators and two discriminators) to a file.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Serialize CycleGAN-specific hyperparameters
        writer.Write(_cycleConsistencyLambda);
        writer.Write(_identityLambda);

        // Serialize all four networks
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

    /// <summary>
    /// Deserializes CycleGAN-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method deserializes the CycleGAN-specific configuration and all four networks.
    /// After deserialization, the optimizers are reset to their initial state.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the CycleGAN's settings and all
    /// four networks (two generators and two discriminators) from a file.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Deserialize CycleGAN-specific hyperparameters
        _cycleConsistencyLambda = reader.ReadDouble();
        _identityLambda = reader.ReadDouble();

        // Deserialize all four networks
        int genAtoB_Length = reader.ReadInt32();
        GeneratorAtoB.Deserialize(reader.ReadBytes(genAtoB_Length));

        int genBtoA_Length = reader.ReadInt32();
        GeneratorBtoA.Deserialize(reader.ReadBytes(genBtoA_Length));

        int discA_Length = reader.ReadInt32();
        DiscriminatorA.Deserialize(reader.ReadBytes(discA_Length));

        int discB_Length = reader.ReadInt32();
        DiscriminatorB.Deserialize(reader.ReadBytes(discB_Length));

        // Reset optimizer state after loading network weights
        ResetOptimizerState();
    }

    /// <summary>
    /// Creates a new instance of the CycleGAN with the same configuration.
    /// </summary>
    /// <returns>A new CycleGAN instance with the same architecture and hyperparameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a fresh CycleGAN instance with the same network architectures
    /// and hyperparameters. The new instance has freshly initialized optimizers.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a copy of the CycleGAN structure
    /// but with new, untrained networks and fresh optimizers.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CycleGAN<T>(
            GeneratorAtoB.Architecture,
            GeneratorBtoA.Architecture,
            DiscriminatorA.Architecture,
            DiscriminatorB.Architecture,
            Architecture.InputType,
            generatorAtoBOptimizer: null,
            generatorBtoAOptimizer: null,
            discriminatorAOptimizer: null,
            discriminatorBOptimizer: null,
            _lossFunction,
            _cycleConsistencyLambda,
            _identityLambda);
    }

    /// <summary>
    /// Updates the parameters of all networks in the CycleGAN.
    /// </summary>
    /// <param name="parameters">The new parameters vector containing parameters for all networks.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters), "Parameters vector cannot be null.");
        }

        int genAtoBCount = GeneratorAtoB.GetParameterCount();
        int genBtoACount = GeneratorBtoA.GetParameterCount();
        int discACount = DiscriminatorA.GetParameterCount();
        int discBCount = DiscriminatorB.GetParameterCount();

        int totalCount = genAtoBCount + genBtoACount + discACount + discBCount;

        if (parameters.Length != totalCount)
        {
            throw new ArgumentException(
                $"Parameters vector length mismatch: expected {totalCount} " +
                $"(GeneratorAtoB: {genAtoBCount}, GeneratorBtoA: {genBtoACount}, " +
                $"DiscriminatorA: {discACount}, DiscriminatorB: {discBCount}), " +
                $"but received {parameters.Length}.",
                nameof(parameters));
        }

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
