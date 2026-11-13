using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Implements Vision Transformer (ViT)-specific adversarial attacks that exploit the
/// self-attention mechanism and patch-based architecture.
/// </summary>
/// <remarks>
/// <para>
/// Unlike CNN-targeted attacks (FGSM, PGD) that rely on local gradient information,
/// ViT attacks exploit the global self-attention mechanism. This attack perturbs patches
/// that the model attends to most, generating adversarial examples that are more effective
/// against ViT architectures while using smaller perturbation budgets.
/// </para>
/// <para>
/// <b>For Beginners:</b> Vision Transformers look at images in small patches and use
/// "attention" to decide which patches matter most. This attack finds the most important
/// patches and makes tiny changes to them, which is more effective than changing random
/// pixels. It's like knowing which cards matter in a card game and only swapping those.
/// </para>
/// <para>
/// <b>References:</b>
/// - On the adversarial robustness of Vision Transformers (Bhojanapalli et al., 2021)
/// - ViT adversarial robustness analysis and improved training (CVPR 2024)
/// - Towards Robust Vision Transformer (Mo et al., 2022)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("On the Adversarial Robustness of Vision Transformers", "https://arxiv.org/abs/2103.15670", Year = 2021, Authors = "Srinadh Bhojanapalli, Ayan Chakrabarti, Daniel Glasner, Daliang Li, Thomas Unterthiner, Andreas Veit")]
public class ViTAdversarialAttack<T, TInput, TOutput> : AdversarialAttackBase<T, TInput, TOutput>
{
    private readonly int _patchSize;
    private readonly int _numAttackPatches;
    private readonly int _numSteps;

    /// <summary>
    /// Initializes a new ViT-specific adversarial attack.
    /// </summary>
    /// <param name="options">Base attack configuration (epsilon, norm type).</param>
    /// <param name="patchSize">
    /// Size of image patches in pixels. Should match the ViT model's patch size. Default: 16.
    /// </param>
    /// <param name="numAttackPatches">
    /// Number of most-attended patches to perturb. Default: 4.
    /// Fewer patches = more targeted attack with smaller perturbation footprint.
    /// </param>
    /// <param name="numSteps">Number of PGD-like steps for iterative perturbation. Default: 10.</param>
    public ViTAdversarialAttack(
        AdversarialAttackOptions<T> options,
        int patchSize = 16,
        int numAttackPatches = 4,
        int numSteps = 10) : base(options)
    {
        if (patchSize <= 0) throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be positive.");
        if (numAttackPatches <= 0) throw new ArgumentOutOfRangeException(nameof(numAttackPatches), "Number of attack patches must be positive.");
        if (numSteps <= 0) throw new ArgumentOutOfRangeException(nameof(numSteps), "Number of steps must be positive.");

        _patchSize = patchSize;
        _numAttackPatches = numAttackPatches;
        _numSteps = numSteps;
    }

    /// <inheritdoc />
    public override TInput GenerateAdversarialExample(
        TInput input, TOutput trueLabel, IFullModel<T, TInput, TOutput> targetModel)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (targetModel == null) throw new ArgumentNullException(nameof(targetModel));

        // For vector inputs, apply patch-aware perturbation
        if (input is Vector<T> vectorInput)
        {
            var adversarial = GeneratePatchAdversarial(vectorInput, trueLabel, targetModel);
            return (TInput)(object)adversarial;
        }

        // Non-vector input types are not supported for ViT attacks
        throw new NotSupportedException(
            $"ViT adversarial attack requires Vector<{typeof(T).Name}> input, but received {typeof(TInput).Name}.");
    }

    /// <inheritdoc />
    public override TInput CalculatePerturbation(TInput original, TInput adversarial)
    {
        if (original is Vector<T> origVec && adversarial is Vector<T> advVec)
        {
            var engine = AiDotNetEngine.Current;
            return (TInput)(object)engine.Subtract<T>(advVec, origVec);
        }

        throw new NotSupportedException(
            $"ViT adversarial attack requires Vector<{typeof(T).Name}> input for perturbation calculation.");
    }

    /// <summary>
    /// Patch-restricted PGD: an L-infinity PGD attack whose perturbation lives only on the most salient patches.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Patches are ranked by the L1 norm of the loss gradient inside them (gradient saliency, as in the
    /// patch attacks studied for ViTs: Bhojanapalli et al. 2021; Fu et al. 2022, Patch-Fool), and the
    /// top <c>numAttackPatches</c> are attacked. Each step is x + stepSize * sign(g) on those patches,
    /// projected to the epsilon ball and clipped to [0, 1] (Madry et al. 2018), with stepSize = epsilon / steps.
    /// The loss is the same MSE against the label vector that FGSM and PGD use. The attack stops as soon as the
    /// predicted class (argmax) differs from the true class.
    /// </para>
    /// </remarks>
    private Vector<T> GeneratePatchAdversarial(
        Vector<T> input, TOutput trueLabel, IFullModel<T, TInput, TOutput> targetModel)
    {
        var engine = AiDotNetEngine.Current;
        int length = input.Length;
        var vectorLabel = ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel);
        int trueClass = ArgMaxIndex(vectorLabel);

        int patchElements = _patchSize * _patchSize;
        int numPatches = Math.Max((length + patchElements - 1) / patchElements, 1);
        int attackPatchCount = Math.Min(_numAttackPatches, numPatches);

        // Patch saliency from the gradient at the clean input; the mask covers the top-k patches.
        var mask = SalientPatchMask(LossGradient(input, vectorLabel, targetModel), patchElements, numPatches, attackPatchCount);

        var epsilon = NumOps.FromDouble(Options.Epsilon);
        var stepSize = NumOps.FromDouble(Options.Epsilon / Math.Max(1, _numSteps));
        var adversarial = input.Clone();

        for (int step = 0; step < _numSteps; step++)
        {
            var gradient = LossGradient(adversarial, vectorLabel, targetModel);
            var update = engine.Multiply<T>(engine.Multiply<T>(SignVector(gradient), mask), stepSize);
            adversarial = engine.Add<T>(adversarial, Options.IsTargeted ? engine.Negate<T>(update) : update);
            adversarial = engine.Add<T>(input, ProjectLInfinity(engine.Subtract<T>(adversarial, input), epsilon));
            adversarial = engine.Clamp<T>(adversarial, NumOps.Zero, NumOps.One);

            if (PredictedClass(adversarial, targetModel) != trueClass)
            {
                break;
            }
        }

        return adversarial;
    }

    private int PredictedClass(Vector<T> input, IFullModel<T, TInput, TOutput> targetModel)
    {
        var output = targetModel.Predict((TInput)(object)input);
        return ArgMaxIndex(ConversionsHelper.ConvertToVector<T, TOutput>(output));
    }

    /// <summary>
    /// d MSE(model(x), label) / dx: the recorded backward pass for neural networks, central differences otherwise.
    /// </summary>
    private Vector<T> LossGradient(Vector<T> input, Vector<T> vectorLabel, IFullModel<T, TInput, TOutput> targetModel)
    {
        if (targetModel is NeuralNetworks.NeuralNetworkBase<T> nnModel)
        {
            var engine = AiDotNetEngine.Current;
            var inputTensor = Tensor<T>.FromVector(input);
            var targetTensor = Tensor<T>.FromVector(vectorLabel);
            using var tape = new GradientTape<T>();
            var outputTensor = nnModel.ForwardForTraining(inputTensor);
            var diff = engine.TensorSubtract(outputTensor, engine.Reshape(targetTensor, outputTensor.Shape.ToArray()));
            var squared = engine.TensorMultiply(diff, diff);
            var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
            var loss = engine.ReduceMean(squared, allAxes, keepDims: false);
            var grads = tape.ComputeGradients(loss, [inputTensor]);
            if (!grads.TryGetValue(inputTensor, out var g))
            {
                throw new InvalidOperationException("ViT attack: gradient computation failed - no gradient for input tensor.");
            }

            return g.ToVector();
        }

        return EstimateInputGradient(
            input,
            (TInput)(object)input,
            targetModel,
            output => ComputeMeanSquaredError(output, vectorLabel),
            NumOps.FromDouble(1e-4));
    }

    /// <summary>
    /// A 0/1 mask over the input that is 1 on the <paramref name="count"/> patches with the largest
    /// sum of |gradient| (lowest patch index first on ties).
    /// </summary>
    private Vector<T> SalientPatchMask(Vector<T> gradient, int patchElements, int numPatches, int count)
    {
        var engine = AiDotNetEngine.Current;
        int length = gradient.Length;
        int padded = numPatches * patchElements;

        var absGradient = engine.Abs<T>(gradient).AsTensor();
        if (padded > length)
        {
            absGradient = engine.TensorConcatenate(new[] { absGradient, new Tensor<T>(new[] { padded - length }) }, 0);
        }

        var saliency = engine.ReduceSum(engine.Reshape(absGradient, new[] { numPatches, patchElements }), new[] { 1 }, keepDims: false);
        var lowest = engine.TensorAddScalar(new Tensor<T>(new[] { numPatches }), NumOps.MinValue);
        var patchMask = new Tensor<T>(new[] { numPatches });
        for (int k = 0; k < count; k++)
        {
            int best = engine.TensorArgMax(saliency, 0).Data.Span[0];
            var chosen = Vector<T>.CreateStandardBasis(numPatches, best).AsTensor();
            patchMask = engine.TensorAdd(patchMask, chosen);
            saliency = engine.TensorWhere(chosen, lowest, saliency);
        }

        var elementMask = engine.TensorRepeatInterleave(patchMask, patchElements, 0);
        return engine.TensorSlice(elementMask, new[] { 0 }, new[] { length }).ToVector();
    }
}
