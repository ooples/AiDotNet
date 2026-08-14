using System.Text;
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
public abstract class AdversarialAttackBase<T, TInput, TOutput> : IAdversarialAttack<T, TInput, TOutput>, IModelShape
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

        var json = Encoding.UTF8.GetString(data);
        Options = JsonConvert.DeserializeObject<AdversarialAttackOptions<T>>(json) ?? new AdversarialAttackOptions<T>();

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
        var absVector = Engine.Abs<T>(vector);
        // Find max of absolute values
        T maxValue = NumOps.Zero;
        for (int i = 0; i < absVector.Length; i++)
        {
            if (NumOps.GreaterThan(absVector[i], maxValue))
            {
                maxValue = absVector[i];
            }
        }
        return maxValue;
    }

    /// <summary>
    /// Computes the L2 norm of a vector using vectorized operations.
    /// </summary>
    protected T ComputeL2Norm(Vector<T> vector)
    {
        return Engine.Norm<T>(vector);
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
