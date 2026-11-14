using System.Text;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Base class for adversarial attack implementations.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract partial class AdversarialAttackBase<T, TInput, TOutput> : IAdversarialAttack<T, TInput, TOutput>, IModelShape
{
    // --- declared state (ModelStateRegistry) ---
    // Identical in every model base because these bases are siblings over the same interfaces rather
    // than one hierarchy; the logic itself lives once in ModelStateRegistry/ModelStateEnvelope.

    /// <summary>State that is not a parameter vector, declared once and persisted by this base.</summary>
    private readonly AiDotNet.Models.ModelStateRegistry<T> _declaredState = new();
    private bool _declaredStateRegistered;

    /// <summary>
    /// Declare state here that the parameter vector does not carry -- a retained training set,
    /// fitted knots, kernel centres, an ensemble's children. Both halves of the payload are driven
    /// by the declaration, so they cannot drift.
    /// </summary>
    /// <param name="state">The registry to declare into.</param>
    protected virtual void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
    }
    /// <summary>Generated state declarations for fields declared across this model's hierarchy.</summary>
    /// <param name="state">The registry to declare into.</param>
    /// <remarks>
    /// Emitted by ModelStateGenerator into the partial model, so a model author declares nothing. The
    /// hand-written <c>RegisterState</c> beside it exists only for state the classifier genuinely
    /// cannot place; anything it CAN place belongs here, where it cannot be forgotten.
    /// </remarks>
    protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        RegisterGeneratedStateCore(state);
    }

    /// <summary>The declared state, registered once and lazily so it runs after the constructor.</summary>
    protected AiDotNet.Models.ModelStateRegistry<T> DeclaredState
    {
        get
        {
            if (!_declaredStateRegistered)
            {
                _declaredStateRegistered = true;
                RegisterGeneratedState(_declaredState);
                RegisterState(_declaredState);
            }
            return _declaredState;
        }
    }
    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The machine epsilon of T: the gap between 1 and the next representable value (2^-10 for Half,
    /// 2^-23 for float, 2^-52 for double).
    /// </summary>
    private protected static readonly double RelativeResolution = ResolutionFrom(1);

    /// <summary>
    /// Configuration options for the attack.
    /// </summary>
    protected AdversarialAttackOptions<T> Options { get; private set; }

    /// <summary>
    /// Random number generator for stochastic operations.
    /// </summary>
    protected Random Random;

    /// <summary>
    /// Initializes a new instance of the adversarial attack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    protected AdversarialAttackBase(AdversarialAttackOptions<T> options)
    {
        Guard.NotNull(options);
        Options = options;
        Random = RandomHelper.CreateSeededRandom(Options.RandomSeed);
    }

    /// <inheritdoc/>
    public abstract TInput GenerateAdversarialExample(TInput input, TOutput trueLabel, IFullModel<T, TInput, TOutput> targetModel);

    /// <inheritdoc/>
    public virtual TInput[] GenerateAdversarialBatch(TInput[] inputs, TOutput[] trueLabels, IFullModel<T, TInput, TOutput> targetModel)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (trueLabels == null)
        {
            throw new ArgumentNullException(nameof(trueLabels));
        }

        if (inputs.Length != trueLabels.Length)
        {
            throw new ArgumentException("Number of labels must match number of inputs.", nameof(trueLabels));
        }

        var adversarialExamples = new TInput[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            adversarialExamples[i] = GenerateAdversarialExample(inputs[i], trueLabels[i], targetModel);
        }

        return adversarialExamples;
    }

    /// <inheritdoc/>
    public abstract TInput CalculatePerturbation(TInput original, TInput adversarial);

    /// <inheritdoc/>
    public virtual AdversarialAttackOptions<T> GetOptions()
    {
        return Options;
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        // Reset any state if needed
    }

    /// <inheritdoc/>
    public virtual byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var json = JsonConvert.SerializeObject(Options, Formatting.None);
        return AiDotNet.Models.ModelStateEnvelope.Append(DeclaredState, Encoding.UTF8.GetBytes(json));
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        // Strips and applies any declared-state trailer, so the body below reads the payload
        // exactly as it did before this existed.
        data = AiDotNet.Models.ModelStateEnvelope.Extract(DeclaredState, data);
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // Read the payload back as the runtime type of the current options: Serialize writes every property of
        // a derived options class (AutoAttackOptions, for example), and reading it as the base class dropped them.
        var json = Encoding.UTF8.GetString(data);
        Options = JsonConvert.DeserializeObject(json, Options.GetType()) as AdversarialAttackOptions<T>
            ?? new AdversarialAttackOptions<T>();

        // Re-initialize Random with the deserialized seed to ensure consistent behavior
        Random = RandomHelper.CreateSeededRandom(Options.RandomSeed);
    }

    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        // Validate path doesn't contain directory traversal attempts BEFORE normalization
        // Path.GetFullPath normalizes and resolves ".." sequences, so we must check the original input
        if (filePath.Contains(".."))
        {
            throw new ArgumentException("File path cannot contain directory traversal sequences.", nameof(filePath));
        }

        Helpers.ModelPersistenceGuard.EnforceBeforeSave();

        var fullPath = Path.GetFullPath(filePath);

        // Ensure parent directory exists
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            var data = Serialize();
            byte[] envelopedData = ModelFileHeader.WrapWithHeader(
                data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Json,
                dynamicShapeInfo: GetDynamicShapeInfo());
            File.WriteAllBytes(fullPath, envelopedData);
        }
    }

    /// <summary>
    /// Returns the input shape for this attack configuration.
    /// Attacks are config/strategy objects rather than inference models, so shape is typically empty.
    /// Subclasses that wrap a target model should override to delegate to the target model's shape.
    /// </summary>
    public virtual int[] GetInputShape()
    {
        return Array.Empty<int>();
    }

    /// <summary>
    /// Returns the output shape for this attack configuration.
    /// Attacks are config/strategy objects rather than inference models, so shape is typically empty.
    /// Subclasses that wrap a target model should override to delegate to the target model's shape.
    /// </summary>
    public virtual int[] GetOutputShape()
    {
        return Array.Empty<int>();
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        // Validate path doesn't contain directory traversal attempts BEFORE normalization
        // Path.GetFullPath normalizes and resolves ".." sequences, so we must check the original input
        if (filePath.Contains(".."))
        {
            throw new ArgumentException("File path cannot contain directory traversal sequences.", nameof(filePath));
        }

        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();

        var fullPath = Path.GetFullPath(filePath);

        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        var data = File.ReadAllBytes(fullPath);

        // Extract payload from AIMF envelope
        data = ModelFileHeader.ExtractPayload(data);

        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            Deserialize(data);
        }
    }

    /// <summary>
    /// Returns the sign of each element in a vector (-1, 0, or 1) using vectorized operations.
    /// </summary>
    protected Vector<T> SignVector(Vector<T> vector)
    {
        return Engine.Sign<T>(vector);
    }

    /// <summary>
    /// Computes the L-infinity norm of a vector (maximum absolute value).
    /// </summary>
    protected T ComputeLInfinityNorm(Vector<T> vector)
    {
        if (vector.Length == 0)
        {
            return NumOps.Zero;
        }

        // max(|v|) with the running maximum seeded at zero, as a single engine reduction.
        var absVector = Engine.Abs<T>(vector).AsTensor();
        return Engine.TensorMaxValue(Engine.TensorMax(absVector, NumOps.Zero));
    }

    /// <summary>
    /// Estimates d loss(model(x)) / dx with central finite differences for black-box models.
    /// </summary>
    /// <remarks>
    /// Each probe x +/- delta*e_i is clamped to the valid input domain [0, 1], and the difference is
    /// divided by the actual clamped step width; a coordinate whose width collapses gets a zero gradient.
    /// The probes need one model call each, so the loop runs over model calls, never over arithmetic.
    /// </remarks>
    /// <param name="vectorInput">The point at which to estimate the gradient.</param>
    /// <param name="referenceInput">An input used to rebuild the model's input type from a vector.</param>
    /// <param name="targetModel">The black-box model.</param>
    /// <param name="lossOfOutput">The loss as a function of the model output.</param>
    /// <param name="delta">The finite-difference step.</param>
    protected Vector<T> EstimateInputGradient(
        Vector<T> vectorInput,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel,
        Func<Vector<T>, T> lossOfOutput,
        T delta)
    {
        int n = vectorInput.Length;

        // A step below the type's resolution rounds x + delta back to x (1e-4 is below half an ulp of
        // Half anywhere in [0.0625, 1]), so every probe would collapse and the gradient would be zero.
        // Eight ulps at 1 keeps the probe representable across the whole [0, 1] domain.
        var minimumDelta = NumOps.FromDouble(8 * RelativeResolution);
        if (NumOps.LessThan(delta, minimumDelta))
        {
            delta = minimumDelta;
        }

        var plusAll = Engine.Clamp<T>(Engine.Add<T>(vectorInput, Engine.Fill<T>(n, delta)), NumOps.Zero, NumOps.One);
        var minusAll = Engine.Clamp<T>(Engine.Subtract<T>(vectorInput, Engine.Fill<T>(n, delta)), NumOps.Zero, NumOps.One);

        var plusLosses = new List<T>(n);
        var minusLosses = new List<T>(n);
        for (int i = 0; i < n; i++)
        {
            plusLosses.Add(LossAtProbe(vectorInput, plusAll, i, referenceInput, targetModel, lossOfOutput));
            minusLosses.Add(LossAtProbe(vectorInput, minusAll, i, referenceInput, targetModel, lossOfOutput));
        }

        var width = Engine.Subtract<T>(plusAll, minusAll).AsTensor();
        var lossDiff = Engine.Subtract<T>(new Vector<T>(plusLosses), new Vector<T>(minusLosses)).AsTensor();
        var positive = Engine.TensorGreaterThan(width, NumOps.Zero);
        var zeros = new Tensor<T>(width.Shape.ToArray());
        var safeWidth = Engine.TensorWhere(positive, width, Engine.TensorAddScalar(zeros, NumOps.One));
        return Engine.TensorWhere(positive, Engine.TensorDivide(lossDiff, safeWidth), zeros).ToVector();
    }

    /// <summary>
    /// Evaluates the loss with coordinate <paramref name="index"/> of the input replaced by the probe value.
    /// </summary>
    private static T LossAtProbe(
        Vector<T> vectorInput,
        Vector<T> probeValues,
        int index,
        TInput referenceInput,
        IFullModel<T, TInput, TOutput> targetModel,
        Func<Vector<T>, T> lossOfOutput)
    {
        var probe = vectorInput.Clone();
        probeValues.AsSpan().Slice(index, 1).CopyTo(probe.AsWritableSpan().Slice(index, 1));
        var modelInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(probe, referenceInput);
        var output = ConversionsHelper.ConvertToVector<T, TOutput>(targetModel.Predict(modelInput));
        return lossOfOutput(output);
    }

    /// <summary>
    /// The small constant added to a norm before dividing by it (torchattacks uses 1e-10).
    /// </summary>
    /// <remarks>
    /// 1e-10 underflows to zero in narrow types such as Half; there the guard falls back to the
    /// library's default stability epsilon, which still is a positive value of T.
    /// </remarks>
    protected static T DivisionGuard()
    {
        return PositiveConstant(1e-10);
    }

    /// <summary>
    /// <paramref name="value"/> in T, or the library's default stability epsilon when it underflows to zero.
    /// </summary>
    /// <remarks>
    /// Stability constants such as 1e-10 or Adam's 1e-8 are zero in Half, which turns 0 / (0 + eps) into NaN.
    /// </remarks>
    private protected static T PositiveConstant(double value)
    {
        var constant = NumOps.FromDouble(value);
        return NumOps.GreaterThan(constant, NumOps.Zero)
            ? constant
            : NumOps.FromDouble(NumericalStabilityHelper.DefaultEpsilon);
    }

    /// <summary>
    /// The smallest power of two 2^-k (k &gt;= <paramref name="exponent"/>) with 1 + 2^-k still distinct from 1 in T.
    /// </summary>
    private static double ResolutionFrom(int exponent)
    {
        double next = Math.Pow(2.0, -(exponent + 1));
        return exponent >= 60 || NumOps.ToDouble(NumOps.FromDouble(1.0 + next)) <= 1.0
            ? Math.Pow(2.0, -exponent)
            : ResolutionFrom(exponent + 1);
    }

    /// <summary>
    /// Mean squared error between a model output and a target vector, as engine reductions.
    /// </summary>
    /// <remarks>
    /// Computed as (||diff||_2 / sqrt(n))^2 so the sum of squares is never formed in T: in Half it overflows
    /// once it passes 65504 (for example 1000 logits that are each 10 away from the target).
    /// </remarks>
    protected T ComputeMeanSquaredError(Vector<T> output, Vector<T> target)
    {
        var diff = Engine.Subtract<T>(output, target);
        var rootMeanSquare = NumOps.Divide(ComputeL2Norm(diff), NumOps.Sqrt(NumOps.FromDouble(diff.Length)));
        return NumOps.Multiply(rootMeanSquare, rootMeanSquare);
    }

    /// <summary>
    /// Index of the largest entry (first one on ties), or 0 for an empty vector.
    /// </summary>
    protected int ArgMaxIndex(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return 0;
        }

        // TensorArgMax returns the first maximal index, matching a strict ">" scan from index 0.
        var index = Engine.TensorArgMax(values.AsTensor(), 0);
        return index.Data.Span[0];
    }

    /// <summary>
    /// Computes the L2 norm of a vector using vectorized operations.
    /// </summary>
    /// <remarks>
    /// The engine's tensor norm accumulates the squares in double, so a Half vector whose norm exceeds 256
    /// (or a Gaussian direction with more than 65504 entries) still gets a finite norm instead of infinity.
    /// </remarks>
    protected T ComputeL2Norm(Vector<T> vector)
    {
        if (vector.Length == 0)
        {
            return NumOps.Zero;
        }

        return Engine.TensorNorm(vector.AsTensor()).Data.Span[0];
    }

    /// <summary>
    /// Projects perturbation to satisfy L-infinity constraint using vectorized operations.
    /// </summary>
    protected Vector<T> ProjectLInfinity(Vector<T> perturbation, T epsilon)
    {
        return Engine.Clamp<T>(perturbation, NumOps.Negate(epsilon), epsilon);
    }

    /// <summary>
    /// Projects perturbation to satisfy L2 constraint using vectorized operations.
    /// </summary>
    protected Vector<T> ProjectL2(Vector<T> perturbation, T epsilon)
    {
        var norm = ComputeL2Norm(perturbation);
        if (NumOps.LessThanOrEquals(norm, epsilon))
        {
            return perturbation;
        }

        var scale = NumOps.Divide(epsilon, norm);
        return Engine.Multiply<T>(perturbation, scale);
    }
}
