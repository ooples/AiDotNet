using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using System.Text;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Base class for adversarial attack implementations.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public abstract class AdversarialAttackBase<T, TInput, TOutput> : IAdversarialAttack<T, TInput, TOutput>
{
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
        Options = options ?? throw new ArgumentNullException(nameof(options));
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
        var json = JsonConvert.SerializeObject(Options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
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

        var fullPath = Path.GetFullPath(filePath);

        // Ensure parent directory exists
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var data = Serialize();
        File.WriteAllBytes(fullPath, data);
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

        var fullPath = Path.GetFullPath(filePath);

        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        var data = File.ReadAllBytes(fullPath);
        Deserialize(data);
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
