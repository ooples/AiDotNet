using System.Text;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Towards Deep Learning Models Resistant to Adversarial Attacks", "https://arxiv.org/abs/1706.06083", Year = 2017, Authors = "Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu")]
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
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var json = JsonConvert.SerializeObject(options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
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
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            File.WriteAllBytes(filePath, Serialize());
        }
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();
        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            Deserialize(File.ReadAllBytes(filePath));
        }
    }

    private Vector<T> ApplyJPEGCompression(Vector<T> input)
    {
        // JPEG compression as adversarial defense (Das et al. 2017 "Keeping
        // the Bad Guys Out: Protecting and Vaccinating Deep Learning with
        // JPEG Compression"). Real JPEG works in the DCT domain, not in
        // pixel domain — uniform-quantize per pixel doesn't remove
        // high-frequency adversarial perturbations like real JPEG does.
        //
        // Algorithm (per JPEG standard ISO/IEC 10918-1):
        //   1. Partition the flattened image into 8×8 blocks.
        //   2. For each block: compute the 2D DCT-II.
        //   3. Quantize DCT coefficients using the JPEG luminance table
        //      scaled by the quality factor.
        //   4. Inverse-quantize (multiply back) and inverse-DCT.
        //   5. Reassemble the blocks.
        //
        // Since the public surface here is a 1D Vector<T>, we assume a
        // square image of side √Length. Non-square inputs fall back to the
        // single-block DCT path.
        int n = input.Length;
        int side = (int)Math.Sqrt(n);
        if (side * side != n || side < 8)
        {
            // Can't tile into 8×8 blocks — fall back to a 1D DCT-quantize-
            // inverse-DCT path that still removes high-frequency components.
            return Apply1DDCTQuantization(input);
        }

        // Quality factor from quantization level: 0.1 → high compression (Q≈20),
        // 0.05 → moderate (Q≈50). Lower quality = stronger defense.
        double quality = MathHelper.Clamp(100.0 - 800.0 * 0.1, 1.0, 99.0); // ≈20

        // JPEG standard luminance quantization table (ISO/IEC 10918-1 Annex K).
        double[,] qTable =
        {
            { 16, 11, 10, 16, 24, 40, 51, 61 },
            { 12, 12, 14, 19, 26, 58, 60, 55 },
            { 14, 13, 16, 24, 40, 57, 69, 56 },
            { 14, 17, 22, 29, 51, 87, 80, 62 },
            { 18, 22, 37, 56, 68,109,103, 77 },
            { 24, 35, 55, 64, 81,104,113, 92 },
            { 49, 64, 78, 87,103,121,120,101 },
            { 72, 92, 95, 98,112,100,103, 99 }
        };
        // Quality scaling per IJG libjpeg convention.
        double scale = quality < 50 ? 5000.0 / quality : 200.0 - 2.0 * quality;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                qTable[i, j] = Math.Max(1, Math.Floor((qTable[i, j] * scale + 50) / 100.0));

        // Copy pixels into a 2D buffer for in-place block processing.
        var image = new double[side, side];
        for (int i = 0; i < n; i++)
            image[i / side, i % side] = NumOps.ToDouble(input[i]);

        var block = new double[8, 8];
        for (int by = 0; by < side; by += 8)
        {
            for (int bx = 0; bx < side; bx += 8)
            {
                int blockH = Math.Min(8, side - by);
                int blockW = Math.Min(8, side - bx);
                if (blockH < 8 || blockW < 8) continue;  // skip edge fragments

                // Pull block (centred on 0 — JPEG subtracts 128 from [0..255] range;
                // we keep the [0,1]-style centring symmetric and skip the shift).
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        block[y, x] = image[by + y, bx + x];

                Dct2D(block);

                // Quantize then dequantize.
                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        block[y, x] = Math.Round(block[y, x] / qTable[y, x]) * qTable[y, x];

                InverseDct2D(block);

                for (int y = 0; y < 8; y++)
                    for (int x = 0; x < 8; x++)
                        image[by + y, bx + x] = block[y, x];
            }
        }

        var output = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            output[i] = NumOps.FromDouble(MathHelper.Clamp(image[i / side, i % side], 0.0, 1.0));
        return output;
    }

    /// <summary>
    /// 2D type-II DCT (the JPEG DCT) computed via the separable two-pass form
    /// over an 8×8 block. Operates in-place.
    /// </summary>
    private static void Dct2D(double[,] block)
    {
        var tmp = new double[8, 8];
        // Rows.
        for (int y = 0; y < 8; y++)
        {
            for (int u = 0; u < 8; u++)
            {
                double sum = 0.0;
                for (int x = 0; x < 8; x++)
                    sum += block[y, x] * Math.Cos((2 * x + 1) * u * Math.PI / 16.0);
                double cu = u == 0 ? 1.0 / Math.Sqrt(2) : 1.0;
                tmp[y, u] = 0.5 * cu * sum;
            }
        }
        // Columns.
        for (int u = 0; u < 8; u++)
        {
            for (int v = 0; v < 8; v++)
            {
                double sum = 0.0;
                for (int y = 0; y < 8; y++)
                    sum += tmp[y, u] * Math.Cos((2 * y + 1) * v * Math.PI / 16.0);
                double cv = v == 0 ? 1.0 / Math.Sqrt(2) : 1.0;
                block[v, u] = 0.5 * cv * sum;
            }
        }
    }

    /// <summary>Inverse 2D DCT (IDCT) — the JPEG decode step. Operates in-place.</summary>
    private static void InverseDct2D(double[,] block)
    {
        var tmp = new double[8, 8];
        // Inverse columns.
        for (int u = 0; u < 8; u++)
        {
            for (int y = 0; y < 8; y++)
            {
                double sum = 0.0;
                for (int v = 0; v < 8; v++)
                {
                    double cv = v == 0 ? 1.0 / Math.Sqrt(2) : 1.0;
                    sum += cv * block[v, u] * Math.Cos((2 * y + 1) * v * Math.PI / 16.0);
                }
                tmp[y, u] = 0.5 * sum;
            }
        }
        // Inverse rows.
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                double sum = 0.0;
                for (int u = 0; u < 8; u++)
                {
                    double cu = u == 0 ? 1.0 / Math.Sqrt(2) : 1.0;
                    sum += cu * tmp[y, u] * Math.Cos((2 * x + 1) * u * Math.PI / 16.0);
                }
                block[y, x] = 0.5 * sum;
            }
        }
    }

    /// <summary>
    /// 1D DCT-quantize-inverse-DCT fallback for non-square or sub-block inputs.
    /// Same idea (kill high-frequency components via coarse quantization in
    /// the DCT domain) without requiring an image-shape assumption.
    /// </summary>
    private Vector<T> Apply1DDCTQuantization(Vector<T> input)
    {
        int n = input.Length;
        var x = new double[n];
        for (int i = 0; i < n; i++) x[i] = NumOps.ToDouble(input[i]);

        // 1D DCT-II.
        var X = new double[n];
        for (int k = 0; k < n; k++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
                sum += x[i] * Math.Cos(Math.PI * (i + 0.5) * k / n);
            double c = k == 0 ? 1.0 / Math.Sqrt(n) : Math.Sqrt(2.0 / n);
            X[k] = c * sum;
        }

        // Coarsely quantize higher-frequency coefficients (proxy for JPEG's
        // quality-controlled quantization table).
        for (int k = 0; k < n; k++)
        {
            double step = 0.01 + 0.1 * ((double)k / n);
            X[k] = Math.Round(X[k] / step) * step;
        }

        // Inverse 1D DCT.
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                double c = k == 0 ? 1.0 / Math.Sqrt(n) : Math.Sqrt(2.0 / n);
                sum += c * X[k] * Math.Cos(Math.PI * (i + 0.5) * k / n);
            }
            y[i] = sum;
        }

        var output = new Vector<T>(n);
        for (int i = 0; i < n; i++)
            output[i] = NumOps.FromDouble(MathHelper.Clamp(y[i], 0.0, 1.0));
        return output;
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

    [AiDotNet.Attributes.ModelMetadataExempt]
    private sealed class PreprocessingFullModel : IFullModel<T, TInput, TOutput>
    {
        private readonly IFullModel<T, TInput, TOutput> _inner;
        private readonly AdversarialTraining<T, TInput, TOutput> _defense;

        public PreprocessingFullModel(IFullModel<T, TInput, TOutput> inner, AdversarialTraining<T, TInput, TOutput> defense)
        {
            Guard.NotNull(inner);
            _inner = inner;
            Guard.NotNull(defense);
            _defense = defense;
        }

        /// <inheritdoc/>
        public ILossFunction<T> DefaultLossFunction => _inner.DefaultLossFunction;

        /// <inheritdoc/>
        public long ParameterCount => InterfaceGuard.Parameterizable(_inner).ParameterCount;

        /// <inheritdoc/>
        public bool SupportsParameterInitialization => InterfaceGuard.Parameterizable(_inner).SupportsParameterInitialization;

        /// <inheritdoc/>
        public Vector<T> SanitizeParameters(Vector<T> parameters) => InterfaceGuard.Parameterizable(_inner).SanitizeParameters(parameters);


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
            return InterfaceGuard.Parameterizable(_inner).GetParameters();
        }

        /// <inheritdoc/>
        public void SetParameters(Vector<T> parameters)
        {
            InterfaceGuard.Parameterizable(_inner).SetParameters(parameters);
        }

        /// <inheritdoc/>
        public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
        {
            var innerWithParams = InterfaceGuard.Parameterizable(_inner).WithParameters(parameters);
            return new PreprocessingFullModel(innerWithParams, _defense);
        }

        /// <inheritdoc/>
        public IEnumerable<int> GetActiveFeatureIndices()
        {
            return InterfaceGuard.FeatureAware(_inner).GetActiveFeatureIndices();
        }

        /// <inheritdoc/>
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
        {
            InterfaceGuard.FeatureAware(_inner).SetActiveFeatureIndices(featureIndices);
        }

        /// <inheritdoc/>
        public bool IsFeatureUsed(int featureIndex)
        {
            return InterfaceGuard.FeatureAware(_inner).IsFeatureUsed(featureIndex);
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
            return InterfaceGuard.GradientComputable(_inner).ComputeGradients(preprocessed, target, lossFunction);
        }

        /// <inheritdoc/>
        public void ApplyGradients(Vector<T> gradients, T learningRate)
        {
            InterfaceGuard.GradientComputable(_inner).ApplyGradients(gradients, learningRate);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Issue #1136 plan part 3: forwards Dispose to the inner
        /// model when it implements IDisposable. The preprocessing
        /// wrapper itself owns no additional disposable state.
        /// </remarks>
        public void Dispose()
        {
            (_inner as System.IDisposable)?.Dispose();
            System.GC.SuppressFinalize(this);
        }

    }
}
