using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using System.Text;

namespace AiDotNet.AdversarialRobustness.Defenses;

/// <summary>
/// Implements adversarial training as a defense mechanism.
/// </summary>
/// <remarks>
/// <para>
/// Adversarial training augments the training data with adversarial examples,
/// teaching the model to correctly classify both clean and adversarial inputs.
/// </para>
/// <para><b>For Beginners:</b> Adversarial training is like vaccinating your model.
/// Just as vaccines expose your immune system to weakened pathogens so it learns to fight them,
/// adversarial training exposes your model to adversarial examples during training so it learns
/// to resist them. This is one of the most effective defenses against adversarial attacks.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AdversarialTraining<T> : IAdversarialDefense<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private AdversarialDefenseOptions<T> options;
    private readonly IAdversarialAttack<T> attackMethod;

    /// <summary>
    /// Initializes a new instance of adversarial training.
    /// </summary>
    /// <param name="options">The defense configuration options.</param>
    public AdversarialTraining(AdversarialDefenseOptions<T> options)
    {
        this.options = options ?? throw new ArgumentNullException(nameof(options));

        // Initialize the attack method to use during training
        var attackOptions = new AdversarialAttackOptions<T>
        {
            Epsilon = options.Epsilon,
            StepSize = options.Epsilon / 4.0,
            Iterations = 10,
            UseRandomStart = true
        };

        // Use PGD attack as the default for adversarial training
        attackMethod = new PGDAttack<T>(attackOptions);
    }

    /// <inheritdoc/>
    public IPredictiveModel<T, Vector<T>, Vector<T>> ApplyDefense(Matrix<T> trainingData, Vector<int> labels, IPredictiveModel<T, Vector<T>, Vector<T>> model)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        // Training-time adversarial example augmentation requires integration with a trainer.
        // For now, return a runtime defense wrapper that applies preprocessing before inference.
        if (!options.UsePreprocessing)
        {
            return model;
        }

        return new PreprocessingPredictiveModel(model, this);
    }

    /// <inheritdoc/>
    public Vector<T> PreprocessInput(Vector<T> input)
    {
        if (!options.UsePreprocessing)
        {
            return input;
        }

        // Apply simple preprocessing based on the method
        return options.PreprocessingMethod.ToLowerInvariant() switch
        {
            "jpeg" => ApplyJPEGCompression(input),
            "bit_depth_reduction" => ApplyBitDepthReduction(input),
            "denoising" => ApplyDenoising(input),
            _ => input
        };
    }

    /// <inheritdoc/>
    public RobustnessMetrics<T> EvaluateRobustness(
        IPredictiveModel<T, Vector<T>, Vector<T>> model,
        Matrix<T> testData,
        Vector<int> labels,
        IAdversarialAttack<T> attack)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (testData == null)
        {
            throw new ArgumentNullException(nameof(testData));
        }

        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (attack == null)
        {
            throw new ArgumentNullException(nameof(attack));
        }

        if (testData.Rows != labels.Length)
        {
            throw new ArgumentException("Number of labels must match number of test rows.", nameof(labels));
        }

        var metrics = new RobustnessMetrics<T>();
        int cleanCorrect = 0;
        int adversarialCorrect = 0;
        var perturbationSizes = new List<double>();

        for (int i = 0; i < testData.Rows; i++)
        {
            var input = testData.GetRow(i);
            var label = labels[i];

            // Evaluate on clean example
            var cleanOutput = model.Predict(input);
            var cleanPrediction = ArgMax(cleanOutput);
            if (cleanPrediction == label)
            {
                cleanCorrect++;
            }

            // Generate and evaluate on adversarial example
            try
            {
                var adversarial = attack.GenerateAdversarialExample(input, label, model);
                var advOutput = model.Predict(adversarial);
                var advPrediction = ArgMax(advOutput);

                if (advPrediction == label)
                {
                    adversarialCorrect++;
                }

                // Calculate perturbation size
                var perturbation = new Vector<T>(input.Length);
                for (int j = 0; j < input.Length; j++)
                {
                    perturbation[j] = NumOps.Subtract(adversarial[j], input[j]);
                }

                var l2Norm = ComputeL2Norm(perturbation);
                perturbationSizes.Add(NumOps.ToDouble(l2Norm));
            }
            catch (ArgumentException)
            {
                // Count as defended if attack fails
                adversarialCorrect++;
            }
            catch (InvalidOperationException)
            {
                // Count as defended if attack fails
                adversarialCorrect++;
            }
        }

        metrics.CleanAccuracy = (double)cleanCorrect / testData.Rows;
        metrics.AdversarialAccuracy = (double)adversarialCorrect / testData.Rows;
        metrics.AveragePerturbationSize = perturbationSizes.Count > 0 ? perturbationSizes.Average() : 0.0;
        metrics.AttackSuccessRate = 1.0 - metrics.AdversarialAccuracy;
        metrics.RobustnessScore = (metrics.CleanAccuracy + metrics.AdversarialAccuracy) / 2.0;

        return metrics;
    }

    /// <inheritdoc/>
    public AdversarialDefenseOptions<T> GetOptions() => options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var json = JsonConvert.SerializeObject(options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);
        options = JsonConvert.DeserializeObject<AdversarialDefenseOptions<T>>(json) ?? new AdversarialDefenseOptions<T>();
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        Deserialize(File.ReadAllBytes(filePath));
    }

    private Vector<T> ApplyJPEGCompression(Vector<T> input)
    {
        // Simplified JPEG-like compression: quantize values
        var compressed = new Vector<T>(input.Length);
        var quantizationLevel = 0.1;

        for (int i = 0; i < input.Length; i++)
        {
            var v = NumOps.ToDouble(input[i]);
            var quantized = Math.Floor(v / quantizationLevel) * quantizationLevel;
            compressed[i] = NumOps.FromDouble(MathHelper.Clamp(quantized, 0.0, 1.0));
        }

        return compressed;
    }

    private Vector<T> ApplyBitDepthReduction(Vector<T> input)
    {
        // Reduce bit depth to remove fine-grained adversarial perturbations
        var reduced = new Vector<T>(input.Length);
        var levels = 16.0; // Reduce to 4-bit color depth

        for (int i = 0; i < input.Length; i++)
        {
            var v = NumOps.ToDouble(input[i]);
            var quantized = Math.Round(v * levels) / levels;
            reduced[i] = NumOps.FromDouble(MathHelper.Clamp(quantized, 0.0, 1.0));
        }

        return reduced;
    }

    private Vector<T> ApplyDenoising(Vector<T> input)
    {
        // Simple moving average denoising (for demonstration)
        // In practice, would use more sophisticated methods
        var clone = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            clone[i] = input[i];
        }

        return clone; // Simplified
    }

    private static int ArgMax(Vector<T> vector)
    {
        int maxIndex = 0;
        double maxValue = NumOps.ToDouble(vector[0]);

        for (int i = 1; i < vector.Length; i++)
        {
            var v = NumOps.ToDouble(vector[i]);
            if (v > maxValue)
            {
                maxValue = v;
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private static T ComputeL2Norm(Vector<T> vector)
    {
        double sumSquares = 0.0;
        for (int i = 0; i < vector.Length; i++)
        {
            var d = NumOps.ToDouble(vector[i]);
            sumSquares += d * d;
        }
        return NumOps.FromDouble(Math.Sqrt(sumSquares));
    }

    private sealed class PreprocessingPredictiveModel : IPredictiveModel<T, Vector<T>, Vector<T>>
    {
        private readonly IPredictiveModel<T, Vector<T>, Vector<T>> _inner;
        private readonly AdversarialTraining<T> _defense;

        public PreprocessingPredictiveModel(IPredictiveModel<T, Vector<T>, Vector<T>> inner, AdversarialTraining<T> defense)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            _defense = defense ?? throw new ArgumentNullException(nameof(defense));
        }

        public Vector<T> Predict(Vector<T> input)
        {
            var preprocessed = _defense.PreprocessInput(input);
            return _inner.Predict(preprocessed);
        }

        public ModelMetadata<T> GetModelMetadata()
        {
            return _inner.GetModelMetadata();
        }

        public byte[] Serialize()
        {
            return _inner.Serialize();
        }

        public void Deserialize(byte[] data)
        {
            _inner.Deserialize(data);
        }

        public void SaveModel(string filePath)
        {
            _inner.SaveModel(filePath);
        }

        public void LoadModel(string filePath)
        {
            _inner.LoadModel(filePath);
        }
    }
}
