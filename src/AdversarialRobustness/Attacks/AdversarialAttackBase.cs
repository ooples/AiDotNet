using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using System.Text;

namespace AiDotNet.AdversarialRobustness.Attacks;

/// <summary>
/// Base class for adversarial attack implementations.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public abstract class AdversarialAttackBase<T> : IAdversarialAttack<T>
{
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
    public abstract Vector<T> GenerateAdversarialExample(Vector<T> input, int trueLabel, IPredictiveModel<T, Vector<T>, Vector<T>> targetModel);

    /// <inheritdoc/>
    public virtual Matrix<T> GenerateAdversarialBatch(Matrix<T> inputs, Vector<int> trueLabels, IPredictiveModel<T, Vector<T>, Vector<T>> targetModel)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs));
        }

        if (trueLabels == null)
        {
            throw new ArgumentNullException(nameof(trueLabels));
        }

        if (inputs.Rows != trueLabels.Length)
        {
            throw new ArgumentException("Number of labels must match number of input rows.", nameof(trueLabels));
        }

        var adversarialExamples = new Matrix<T>(inputs.Rows, inputs.Columns);
        for (int i = 0; i < inputs.Rows; i++)
        {
            var adversarial = GenerateAdversarialExample(inputs.GetRow(i), trueLabels[i], targetModel);
            adversarialExamples.SetRow(i, adversarial);
        }

        return adversarialExamples;
    }

    /// <inheritdoc/>
    public virtual Vector<T> CalculatePerturbation(Vector<T> original, Vector<T> adversarial)
    {
        if (original == null)
        {
            throw new ArgumentNullException(nameof(original));
        }

        if (adversarial == null)
        {
            throw new ArgumentNullException(nameof(adversarial));
        }

        if (original.Length != adversarial.Length)
        {
            throw new ArgumentException("Original and adversarial examples must have the same length.");
        }

        var perturbation = new Vector<T>(original.Length);
        for (int i = 0; i < original.Length; i++)
        {
            perturbation[i] = NumOps.Subtract(adversarial[i], original[i]);
        }

        return perturbation;
    }

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

        // Validate path doesn't contain directory traversal attempts
        var fullPath = Path.GetFullPath(filePath);
        if (fullPath.Contains(".."))
        {
            throw new ArgumentException("File path cannot contain directory traversal sequences.", nameof(filePath));
        }

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

        // Validate path doesn't contain directory traversal attempts
        var fullPath = Path.GetFullPath(filePath);
        if (fullPath.Contains(".."))
        {
            throw new ArgumentException("File path cannot contain directory traversal sequences.", nameof(filePath));
        }

        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        var data = File.ReadAllBytes(fullPath);
        Deserialize(data);
    }

    /// <summary>
    /// Returns the sign of a value (-1, 0, or 1).
    /// </summary>
    protected static T Sign(T value)
    {
        if (NumOps.GreaterThan(value, NumOps.Zero)) return NumOps.One;
        if (NumOps.LessThan(value, NumOps.Zero)) return NumOps.Negate(NumOps.One);
        return NumOps.Zero;
    }

    /// <summary>
    /// Computes the L-infinity norm of a vector.
    /// </summary>
    protected static T ComputeLInfinityNorm(Vector<T> vector)
    {
        T maxValue = NumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            var absValue = NumOps.Abs(vector[i]);
            if (NumOps.GreaterThan(absValue, maxValue))
            {
                maxValue = absValue;
            }
        }
        return maxValue;
    }

    /// <summary>
    /// Computes the L2 norm of a vector.
    /// </summary>
    protected static T ComputeL2Norm(Vector<T> vector)
    {
        double sumSquares = 0.0;
        for (int i = 0; i < vector.Length; i++)
        {
            var d = NumOps.ToDouble(vector[i]);
            sumSquares += d * d;
        }
        return NumOps.FromDouble(Math.Sqrt(sumSquares));
    }

    /// <summary>
    /// Projects perturbation to satisfy L-infinity constraint.
    /// </summary>
    protected Vector<T> ProjectLInfinity(Vector<T> perturbation, T epsilon)
    {
        var projected = new Vector<T>(perturbation.Length);
        for (int i = 0; i < perturbation.Length; i++)
        {
            projected[i] = MathHelper.Clamp(perturbation[i], NumOps.Negate(epsilon), epsilon);
        }
        return projected;
    }

    /// <summary>
    /// Projects perturbation to satisfy L2 constraint.
    /// </summary>
    protected Vector<T> ProjectL2(Vector<T> perturbation, T epsilon)
    {
        var norm = ComputeL2Norm(perturbation);
        if (NumOps.LessThanOrEquals(norm, epsilon))
        {
            return perturbation;
        }

        var projected = new Vector<T>(perturbation.Length);
        var scale = NumOps.Divide(epsilon, norm);
        for (int i = 0; i < perturbation.Length; i++)
        {
            projected[i] = NumOps.Multiply(perturbation[i], scale);
        }
        return projected;
    }
}
