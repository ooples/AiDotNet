using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

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
    protected readonly AdversarialAttackOptions<T> Options;

    /// <summary>
    /// Random number generator for stochastic operations.
    /// </summary>
    protected readonly Random Random;

    /// <summary>
    /// Initializes a new instance of the adversarial attack.
    /// </summary>
    /// <param name="options">The configuration options for the attack.</param>
    protected AdversarialAttackBase(AdversarialAttackOptions<T> options)
    {
        Options = options;
        Random = new Random(options.RandomSeed);
    }

    /// <inheritdoc/>
    public abstract T[] GenerateAdversarialExample(T[] input, int trueLabel, Func<T[], T[]> targetModel);

    /// <inheritdoc/>
    public virtual T[][] GenerateAdversarialBatch(T[][] inputs, int[] trueLabels, Func<T[], T[]> targetModel)
    {
        var adversarialExamples = new T[inputs.Length][];
        for (int i = 0; i < inputs.Length; i++)
        {
            adversarialExamples[i] = GenerateAdversarialExample(inputs[i], trueLabels[i], targetModel);
        }
        return adversarialExamples;
    }

    /// <inheritdoc/>
    public virtual T[] CalculatePerturbation(T[] original, T[] adversarial)
    {
        if (original.Length != adversarial.Length)
        {
            throw new ArgumentException("Original and adversarial examples must have the same length.");
        }

        var perturbation = new T[original.Length];
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
        var json = System.Text.Json.JsonSerializer.Serialize(Options);
        return System.Text.Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        // Deserialization logic for restoring state
    }

    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        var data = Serialize();
        File.WriteAllBytes(filePath, data);
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        var data = File.ReadAllBytes(filePath);
        Deserialize(data);
    }

    /// <summary>
    /// Clips a value to be within the specified range.
    /// </summary>
    protected static T Clip(T value, T min, T max)
    {
        if (NumOps.LessThan(value, min)) return min;
        if (NumOps.GreaterThan(value, max)) return max;
        return value;
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
    protected static T ComputeLInfinityNorm(T[] vector)
    {
        T maxValue = NumOps.Zero;
        foreach (var value in vector)
        {
            var absValue = NumOps.Abs(value);
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
    protected static T ComputeL2Norm(T[] vector)
    {
        double sumSquares = 0.0;
        foreach (var value in vector)
        {
            var d = NumOps.ToDouble(value);
            sumSquares += d * d;
        }
        return NumOps.FromDouble(Math.Sqrt(sumSquares));
    }

    /// <summary>
    /// Projects perturbation to satisfy L-infinity constraint.
    /// </summary>
    protected T[] ProjectLInfinity(T[] perturbation, T epsilon)
    {
        var projected = new T[perturbation.Length];
        for (int i = 0; i < perturbation.Length; i++)
        {
            projected[i] = Clip(perturbation[i], NumOps.Negate(epsilon), epsilon);
        }
        return projected;
    }

    /// <summary>
    /// Projects perturbation to satisfy L2 constraint.
    /// </summary>
    protected T[] ProjectL2(T[] perturbation, T epsilon)
    {
        var norm = ComputeL2Norm(perturbation);
        if (NumOps.LessThanOrEquals(norm, epsilon))
        {
            return perturbation;
        }

        var projected = new T[perturbation.Length];
        var scale = NumOps.Divide(epsilon, norm);
        for (int i = 0; i < perturbation.Length; i++)
        {
            projected[i] = NumOps.Multiply(perturbation[i], scale);
        }
        return projected;
    }
}
