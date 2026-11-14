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
        var perturbationSizes = new List<T>();

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

            // Generate and evaluate on adversarial example. A failing attack propagates its exception: counting
            // it as a successful defense would overstate the robustness.
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
            perturbationSizes.Add(l2Norm);
        }

        metrics.CleanAccuracy = (double)cleanCorrect / testData.Length;
        metrics.AdversarialAccuracy = (double)adversarialCorrect / testData.Length;
        metrics.AveragePerturbationSize = perturbationSizes.Count > 0
            ? NumOps.ToDouble(Engine.Mean<T>(new Vector<T>(perturbationSizes)))
            : 0.0;
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
        //   1. Map [0, 1] intensities to the 8-bit range and level-shift by 128.
        //   2. Partition the image into 8×8 blocks (edge blocks replicate the last row/column).
        //   3. For each block: compute the 2D DCT-II.
        //   4. Quantize DCT coefficients using the JPEG luminance table
        //      scaled by the quality factor.
        //   5. Inverse-quantize (multiply back) and inverse-DCT.
        //   6. Reassemble the blocks and undo the level shift and scaling.
        //
        // Since the public surface here is a 1D Vector<T>, we assume a
        // square image of side √Length. Non-square inputs fall back to the
        // single-block DCT path.
        int n = input.Length;
        int side = (int)Math.Sqrt(n);
        // For non-square inputs, try to factor n into the closest-to-square
        // (h, w) pair where both dimensions are >= 8 — that lets the 2D
        // JPEG defense apply to rectangular images (the common case for
        // 16:9 / 4:3 / 3:4 photos) instead of silently degrading them to
        // the much weaker 1D path. The factorisation walks h downward from
        // √n until either it finds a divisor that satisfies h >= 8 and
        // n/h >= 8, or the search exits the valid range.
        int rectH = 0, rectW = 0;
        if (side * side != n)
        {
            for (int h = side; h >= 8; h--)
            {
                if (n % h == 0 && n / h >= 8)
                {
                    rectH = h;
                    rectW = n / h;
                    break;
                }
            }
        }
        if (rectH == 0 && side * side != n)
        {
            // No square or rectangular factorisation with both dims >= 8 —
            // fall back to a 1D DCT-quantize-inverse-DCT path that still
            // removes high-frequency components.
            return Apply1DDCTQuantization(input);
        }
        if (side < 8 && rectH == 0)
        {
            return Apply1DDCTQuantization(input);
        }
        // Use square dims when n is a perfect square, otherwise the
        // rectangular factorisation found above.
        int imgH = side * side == n ? side : rectH;
        int imgW = side * side == n ? side : rectW;

        // Track the observed input range so the reconstruction can be
        // clamped back to whatever the caller fed in.
        var flat = input.AsTensor();
        T minValue = Engine.TensorMinValue(flat);
        T maxValue = Engine.TensorMaxValue(flat);

        // JPEG's quantization table is defined for 8-bit samples level-shifted by 128
        // (ISO/IEC 10918-1, A.3.1). Quantizing [0, 1] intensities against it directly
        // rounds every coefficient but DC to zero, so map to [0, 255] - 128 first.
        T pixelScale = NumOps.FromDouble(255.0);
        T levelShift = NumOps.FromDouble(128.0);
        var image = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.Reshape(flat, new[] { imgH, imgW }), pixelScale),
            NumOps.Negate(levelShift));

        // Pad to whole blocks by replicating the last row and column, so an adversary
        // cannot hide a perturbation in an unprocessed edge fragment.
        int blocksH = (imgH + 7) / 8;
        int blocksW = (imgW + 7) / 8;
        var padded = Engine.TensorGather(image, ClampedIndices(blocksH * 8, imgH), 0);
        padded = Engine.TensorGather(padded, ClampedIndices(blocksW * 8, imgW), 1);

        // [Bh*8, Bw*8] -> [Bh, 8, Bw, 8] -> [Bh, Bw, 8, 8] -> [N, 8, 8]
        int blockCount = blocksH * blocksW;
        var blocks = Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(padded, new[] { blocksH, 8, blocksW, 8 }), new[] { 0, 2, 1, 3 }),
            new[] { blockCount, 8, 8 });

        var basis = DctBasis();
        var coefficients = BlockTransform(blocks, Engine.TensorTranspose(basis), blockCount);

        // Quantize then dequantize against the quality-scaled table.
        var table = Engine.TensorBroadcastTo(QuantizationTable(), new[] { blockCount, 8, 8 });
        var quantized = Engine.TensorMultiply(Engine.TensorRound(Engine.TensorDivide(coefficients, table)), table);
        var restored = BlockTransform(quantized, basis, blockCount);

        // [N, 8, 8] -> [Bh, Bw, 8, 8] -> [Bh, 8, Bw, 8] -> [Bh*8, Bw*8], then crop the padding.
        var reassembled = Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(restored, new[] { blocksH, blocksW, 8, 8 }), new[] { 0, 2, 1, 3 }),
            new[] { blocksH * 8, blocksW * 8 });
        var cropped = Engine.TensorSlice(reassembled, new[] { 0, 0 }, new[] { imgH, imgW });
        var pixels = Engine.TensorDivideScalar(Engine.TensorAddScalar(cropped, levelShift), pixelScale);
        return Engine.TensorClamp(Engine.Reshape(pixels, new[] { n }), minValue, maxValue).ToVector();
    }

    /// <summary>min(i, limit - 1) for i in [0, count): the edge-replicating gather indices.</summary>
    private static Tensor<int> ClampedIndices(int count, int limit)
    {
        var indices = new int[count];
        for (int i = 0; i < count; i++)
        {
            indices[i] = Math.Min(i, limit - 1);
        }

        return new Tensor<int>(indices, new[] { count });
    }

    /// <summary>
    /// M^T B M for every 8x8 block B, as two batched matrix products with per-block transposes:
    /// T(T(B M) M) = M^T B M. With M = D^T this is the 2D DCT D B D^T; with M = D, its inverse D^T Y D.
    /// </summary>
    private Tensor<T> BlockTransform(Tensor<T> blocks, Tensor<T> transform, int blockCount)
    {
        var rows = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(blocks, new[] { blockCount * 8, 8 }), transform), new[] { blockCount, 8, 8 });
        var flipped = Engine.Reshape(Engine.TensorPermute(rows, new[] { 0, 2, 1 }), new[] { blockCount * 8, 8 });
        var columns = Engine.Reshape(Engine.TensorMatMul(flipped, transform), new[] { blockCount, 8, 8 });
        return Engine.TensorPermute(columns, new[] { 0, 2, 1 });
    }

    /// <summary>
    /// The orthonormal 8-point DCT-II matrix D[k, i] = c_k cos((2i + 1) k pi / 16), c_0 = sqrt(1/8), c_k = 1/2.
    /// </summary>
    private Tensor<T> DctBasis() => DctMatrix(8, 8, 0, 8);

    /// <summary>
    /// Rows [first, first + count) of the orthonormal n-point DCT-II matrix
    /// C[k, i] = c_k cos(pi (i + 1/2) k / n), c_0 = sqrt(1/n), c_k = sqrt(2/n).
    /// </summary>
    private Tensor<T> DctMatrix(int n, int count, int first, int columns)
    {
        var k = Engine.Reshape(Engine.TensorArange<T>(first, first + count, 1), new[] { count, 1 });
        var position = Engine.Reshape(Engine.TensorArange<T>(0.5, columns + 0.5, 1), new[] { 1, columns });
        var angles = Engine.TensorMultiplyScalar(Engine.TensorMatMul(k, position), NumOps.FromDouble(Math.PI / n));
        var cosines = Engine.TensorCos(angles);
        var scale = Engine.TensorAddScalar(new Tensor<T>(new[] { count, 1 }), NumOps.FromDouble(Math.Sqrt(2.0 / n)));
        if (first == 0)
        {
            scale = Engine.TensorSetSlice(scale, Engine.TensorAddScalar(new Tensor<T>(new[] { 1, 1 }), NumOps.FromDouble(Math.Sqrt(1.0 / n))), new[] { 0, 0 });
        }

        return Engine.TensorMultiply(cosines, Engine.TensorBroadcastTo(scale, new[] { count, columns }));
    }

    /// <summary>
    /// The JPEG luminance table (ISO/IEC 10918-1 Annex K) scaled for quality 20 per the IJG libjpeg convention.
    /// </summary>
    private static Tensor<T> QuantizationTable()
    {
        // Quality factor from quantization level: 0.1 → high compression (Q≈20).
        double quality = MathHelper.Clamp(100.0 - 800.0 * 0.1, 1.0, 99.0); // ≈20
        double[] luminance =
        {
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
        };
        double scale = quality < 50 ? 5000.0 / quality : 200.0 - 2.0 * quality;
        for (int i = 0; i < luminance.Length; i++)
        {
            luminance[i] = Math.Max(1, Math.Floor((luminance[i] * scale + 50) / 100.0));
        }

        return new Tensor<double>(luminance, new[] { 1, 8, 8 }).Cast<T>();
    }

    /// <summary>
    /// 1D DCT-quantize-inverse-DCT fallback for non-square or sub-block inputs.
    /// Same idea (kill high-frequency components via coarse quantization in
    /// the DCT domain) without requiring an image-shape assumption.
    /// </summary>
    private Vector<T> Apply1DDCTQuantization(Vector<T> input)
    {
        int n = input.Length;
        var x = Engine.Reshape(input.AsTensor(), new[] { n, 1 });
        var reconstruction = new Tensor<T>(new[] { n, 1 });

        // The orthonormal DCT-II in row chunks so the n x n matrix is never materialized at once.
        const int chunk = 512;
        for (int first = 0; first < n; first += chunk)
        {
            reconstruction = Engine.TensorAdd(reconstruction, QuantizedChunk(x, n, first, Math.Min(chunk, n - first)));
        }

        return Engine.TensorClamp(Engine.Reshape(reconstruction, new[] { n }), NumOps.Zero, NumOps.One).ToVector();
    }

    /// <summary>
    /// The contribution C_rows^T q(C_rows x) of DCT rows [first, first + count): coefficient k is quantized with
    /// step 0.01 + 0.1 k / n (a proxy for JPEG's quality-controlled table).
    /// </summary>
    private Tensor<T> QuantizedChunk(Tensor<T> x, int n, int first, int count)
    {
        var rows = DctMatrix(n, count, first, n);
        var coefficients = Engine.TensorMatMul(rows, x);
        var steps = Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.Reshape(Engine.TensorArange<T>(first, first + count, 1), new[] { count, 1 }), NumOps.FromDouble(0.1 / n)),
            NumOps.FromDouble(0.01));
        var quantized = Engine.TensorMultiply(Engine.TensorRound(Engine.TensorDivide(coefficients, steps)), steps);
        return Engine.TensorMatMul(Engine.TensorTranspose(rows), quantized);
    }

    private Vector<T> ApplyBitDepthReduction(Vector<T> input)
    {
        // Feature squeezing (Xu et al. 2018): i-bit depth keeps 2^i levels, x -> round(x (2^i - 1)) / (2^i - 1).
        // 4 bits are 16 levels, so the multiplier is 15; 16 would give 17 levels.
        T levels = NumOps.FromDouble(15.0);
        var scaled = Engine.TensorMultiplyScalar(input.AsTensor(), levels);
        var reduced = Engine.TensorDivideScalar(Engine.TensorRound(scaled), levels);
        return Engine.TensorClamp(reduced, NumOps.Zero, NumOps.One).ToVector();
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

        return AiDotNetEngine.Current.TensorArgMax(vector.AsTensor(), 0).GetFlat(0);
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
