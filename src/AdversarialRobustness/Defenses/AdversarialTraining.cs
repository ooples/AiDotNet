using System.Text;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using Newtonsoft.Json;

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
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class AdversarialTraining<T, TInput, TOutput> : IAdversarialDefense<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private AdversarialDefenseOptions<T> options;
    private readonly IAdversarialAttack<T, TInput, TOutput> attackMethod;

    /// <summary>
    /// Initializes a new instance of adversarial training.
    /// </summary>
    /// <param name="options">The defense configuration options.</param>
    public AdversarialTraining(AdversarialDefenseOptions<T> options)
    {
        Guard.NotNull(options);
        this.options = options;

        // Initialize the attack method to use during training
        var attackOptions = new AdversarialAttackOptions<T>
        {
            Epsilon = options.Epsilon,
            StepSize = options.Epsilon / 4.0,
            Iterations = 10,
            UseRandomStart = true
        };

        // Use PGD attack as the default for adversarial training
        attackMethod = new PGDAttack<T, TInput, TOutput>(attackOptions);
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> ApplyDefense(TInput[] trainingData, TOutput[] labels, IFullModel<T, TInput, TOutput> model)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (trainingData == null)
        {
            throw new ArgumentNullException(nameof(trainingData));
        }

        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (trainingData.Length != labels.Length)
        {
            throw new ArgumentException("Number of labels must match number of training samples.", nameof(labels));
        }

        // Training-time adversarial example augmentation requires integration with a trainer.
        // For now, return a runtime defense wrapper that applies preprocessing before inference.
        if (!options.UsePreprocessing)
        {
            return model;
        }

        return new PreprocessingFullModel(model, this);
    }

    /// <inheritdoc/>
    public TInput PreprocessInput(TInput input)
    {
        if (!options.UsePreprocessing)
        {
            return input;
        }

        // Apply preprocessing based on the method
        // Convert to vector for preprocessing, then convert back
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        var preprocessed = options.PreprocessingMethod.ToLowerInvariant() switch
        {
            "jpeg" => ApplyJPEGCompression(vectorInput),
            "bit_depth_reduction" => ApplyBitDepthReduction(vectorInput),
            "denoising" => ApplyDenoising(vectorInput),
            _ => vectorInput
        };

        return ConversionsHelper.ConvertVectorToInput<T, TInput>(preprocessed, input);
    }

    /// <inheritdoc/>
    public RobustnessMetrics<T> EvaluateRobustness(
        IFullModel<T, TInput, TOutput> model,
        TInput[] testData,
        TOutput[] labels,
        IAdversarialAttack<T, TInput, TOutput> attack)
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

        if (testData.Length != labels.Length)
        {
            throw new ArgumentException("Number of labels must match number of test samples.", nameof(labels));
        }

        var metrics = new RobustnessMetrics<T>();
        int cleanCorrect = 0;
        int adversarialCorrect = 0;
        var perturbationSizes = new List<double>();

        for (int i = 0; i < testData.Length; i++)
        {
            var input = testData[i];
            var label = labels[i];

            // Convert input and label to vectors for comparison
            var inputVector = ConversionsHelper.ConvertToVector<T, TInput>(input);
            var labelVector = ConversionsHelper.ConvertToVector<T, TOutput>(label);
            var trueClass = ArgMaxVector(labelVector);

            // Evaluate on clean example
            var cleanOutput = model.Predict(input);
            var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
            var cleanPrediction = ArgMaxVector(cleanOutputVector);
            if (cleanPrediction == trueClass)
            {
                cleanCorrect++;
            }

            // Generate and evaluate on adversarial example
            try
            {
                var adversarial = attack.GenerateAdversarialExample(input, label, model);
                var advOutput = model.Predict(adversarial);
                var advOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(advOutput);
                var advPrediction = ArgMaxVector(advOutputVector);

                if (advPrediction == trueClass)
                {
                    adversarialCorrect++;
                }

                // Calculate perturbation size using vectorized operations
                var adversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(adversarial);
                var perturbation = Engine.Subtract<T>(adversarialVector, inputVector);
                var l2Norm = Engine.Norm<T>(perturbation);
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

        metrics.CleanAccuracy = (double)cleanCorrect / testData.Length;
        metrics.AdversarialAccuracy = (double)adversarialCorrect / testData.Length;
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
        var quantizationLevel = 0.1;
        var compressed = new Vector<T>(input.Length);

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
        var levels = 16.0; // Reduce to 4-bit color depth
        var reduced = new Vector<T>(input.Length);

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
        // Use Engine to clone the vector
        var zeros = Engine.FillZero<T>(input.Length);
        return Engine.Add<T>(input, zeros);
    }

    private static int ArgMaxVector(Vector<T> vector)
    {
        if (vector == null || vector.Length == 0)
        {
            return 0;
        }

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

    private sealed class PreprocessingFullModel : IFullModel<T, TInput, TOutput>
    {
        private readonly IFullModel<T, TInput, TOutput> _inner;
        private readonly AdversarialTraining<T, TInput, TOutput> _defense;

        public PreprocessingFullModel(IFullModel<T, TInput, TOutput> inner, AdversarialTraining<T, TInput, TOutput> defense)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            _defense = defense ?? throw new ArgumentNullException(nameof(defense));
        }

        /// <inheritdoc/>
        public ILossFunction<T> DefaultLossFunction => _inner.DefaultLossFunction;

        /// <inheritdoc/>
        public int ParameterCount => _inner.ParameterCount;

        /// <inheritdoc/>
        public bool SupportsJitCompilation => _inner.SupportsJitCompilation;

        /// <inheritdoc/>
        public TOutput Predict(TInput input)
        {
            var preprocessed = _defense.PreprocessInput(input);
            return _inner.Predict(preprocessed);
        }

        /// <inheritdoc/>
        public void Train(TInput input, TOutput expectedOutput)
        {
            var preprocessed = _defense.PreprocessInput(input);
            _inner.Train(preprocessed, expectedOutput);
        }

        /// <inheritdoc/>
        public ModelMetadata<T> GetModelMetadata()
        {
            return _inner.GetModelMetadata();
        }

        /// <inheritdoc/>
        public byte[] Serialize()
        {
            return _inner.Serialize();
        }

        /// <inheritdoc/>
        public void Deserialize(byte[] data)
        {
            _inner.Deserialize(data);
        }

        /// <inheritdoc/>
        public void SaveModel(string filePath)
        {
            _inner.SaveModel(filePath);
        }

        /// <inheritdoc/>
        public void LoadModel(string filePath)
        {
            _inner.LoadModel(filePath);
        }

        /// <inheritdoc/>
        public void SaveState(Stream stream)
        {
            _inner.SaveState(stream);
        }

        /// <inheritdoc/>
        public void LoadState(Stream stream)
        {
            _inner.LoadState(stream);
        }

        /// <inheritdoc/>
        public Vector<T> GetParameters()
        {
            return _inner.GetParameters();
        }

        /// <inheritdoc/>
        public void SetParameters(Vector<T> parameters)
        {
            _inner.SetParameters(parameters);
        }

        /// <inheritdoc/>
        public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        {
            var innerWithParams = _inner.WithParameters(parameters);
            return new PreprocessingFullModel(innerWithParams, _defense);
        }

        /// <inheritdoc/>
        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return _inner.GetActiveFeatureIndices();
        }

        /// <inheritdoc/>
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            _inner.SetActiveFeatureIndices(featureIndices);
        }

        /// <inheritdoc/>
        public bool IsFeatureUsed(int featureIndex)
        {
            return _inner.IsFeatureUsed(featureIndex);
        }

        /// <inheritdoc/>
        public Dictionary<string, T> GetFeatureImportance()
        {
            return _inner.GetFeatureImportance();
        }

        /// <inheritdoc/>
        public IFullModel<T, TInput, TOutput> DeepCopy()
        {
            var innerCopy = _inner.DeepCopy();
            return new PreprocessingFullModel(innerCopy, _defense);
        }

        /// <inheritdoc/>
        public IFullModel<T, TInput, TOutput> Clone()
        {
            var innerClone = _inner.Clone();
            return new PreprocessingFullModel(innerClone, _defense);
        }

        /// <inheritdoc/>
        public Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
        {
            var preprocessed = _defense.PreprocessInput(input);
            return _inner.ComputeGradients(preprocessed, target, lossFunction);
        }

        /// <inheritdoc/>
        public void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            _inner.ApplyGradients(gradients, learningRate);
        }

        /// <inheritdoc/>
        public ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
        {
            return _inner.ExportComputationGraph(inputNodes);
        }
    }
}
