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
    private readonly IAdversarialAttack<T, TInput, TOutput>? customAttack;

    /// <summary>
    /// Initializes a new instance of adversarial training that generates its training examples with
    /// <see cref="AdversarialDefenseOptions{T}.AttackMethod"/>.
    /// </summary>
    /// <param name="options">The defense configuration options.</param>
    public AdversarialTraining(AdversarialDefenseOptions<T> options)
    {
        Guard.NotNull(options);
        this.options = options;
    }

    /// <summary>
    /// Initializes a new instance of adversarial training that generates its training examples with a custom attack.
    /// </summary>
    /// <param name="options">The defense configuration options.</param>
    /// <param name="trainingAttack">The attack that replaces <see cref="AdversarialDefenseOptions{T}.AttackMethod"/>.</param>
    public AdversarialTraining(AdversarialDefenseOptions<T> options, IAdversarialAttack<T, TInput, TOutput> trainingAttack)
        : this(options)
    {
        Guard.NotNull(trainingAttack);
        customAttack = trainingAttack;
    }

    /// <summary>
    /// Adversarially trains <paramref name="model"/> (or an ensemble of copies of it) and wraps the result with the
    /// configured preprocessing defense.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For <see cref="AdversarialDefenseOptions{T}.TrainingEpochs"/> epochs the data is shuffled and split into
    /// mini-batches of <see cref="AdversarialDefenseOptions{T}.BatchSize"/>. In each batch the first
    /// round(<see cref="AdversarialDefenseOptions{T}.AdversarialRatio"/> * size) samples are replaced by adversarial
    /// examples generated against the current model (Madry et al. 2018; mixed clean and adversarial batches per
    /// Goodfellow et al. 2015). A model that computes gradients takes one step of
    /// <see cref="AdversarialDefenseOptions{T}.LearningRate"/> on the batch-mean gradient; any other model is
    /// trained once per epoch on the stacked mixed epoch (matrix rows or tensor slices), or sample by sample through
    /// its own <c>Train</c> for other input types.
    /// </para>
    /// <para>
    /// With <see cref="AdversarialDefenseOptions{T}.UseEnsemble"/> the procedure runs on
    /// <see cref="AdversarialDefenseOptions{T}.EnsembleSize"/> deep copies, each with its own shuffling and attack
    /// random starts, and the returned model predicts the mean of their outputs. The returned model applies
    /// <see cref="PreprocessInput"/> before every prediction.
    /// </para>
    /// </remarks>
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

        ValidateOptions();

        int memberCount = options.UseEnsemble ? options.EnsembleSize : 1;
        var members = new IFullModel<T, TInput, TOutput>[memberCount];
        int baseSeed = options.Seed ?? RandomHelper.CreateSecureRandom().Next();
        for (int m = 0; m < memberCount; m++)
        {
            // Ensemble members are independent copies; a single model is trained in place, as before.
            members[m] = options.UseEnsemble ? model.DeepCopy() : model;
            Train(members[m], trainingData, labels, unchecked(baseSeed + 7919 * m));
        }

        if (!options.UseEnsemble && !options.UsePreprocessing)
        {
            return model;
        }

        return new PreprocessingFullModel(members, this);
    }

    /// <summary>
    /// Runs the mixed clean/adversarial training loop on one model.
    /// </summary>
    private void Train(IFullModel<T, TInput, TOutput> model, TInput[] trainingData, TOutput[] labels, int seed)
    {
        int n = trainingData.Length;
        if (options.TrainingEpochs == 0 || n == 0)
        {
            return;
        }

        var random = RandomHelper.CreateSeededRandom(seed);
        var attack = customAttack ?? CreateTrainingAttack(seed);
        var gradientModel = model as IGradientComputable<T, TInput, TOutput>;
        T learningRate = NumOps.FromDouble(options.LearningRate);
        T fullBatch = NumOps.FromDouble(Math.Min(options.BatchSize, n));
        T lastBatch = NumOps.FromDouble(n % options.BatchSize == 0 ? Math.Min(options.BatchSize, n) : n % options.BatchSize);
        var order = Enumerable.Range(0, n).ToArray();

        for (int epoch = 0; epoch < options.TrainingEpochs; epoch++)
        {
            Shuffle(order, random);
            var epochInputs = new List<TInput>(gradientModel is null ? n : 0);
            var epochLabels = new List<TOutput>(gradientModel is null ? n : 0);
            for (int start = 0; start < n; start += options.BatchSize)
            {
                int size = Math.Min(options.BatchSize, n - start);
                int adversarialCount = (int)Math.Round(options.AdversarialRatio * size, MidpointRounding.AwayFromZero);
                Vector<T>? gradientSum = null;
                for (int b = 0; b < size; b++)
                {
                    int index = order[start + b];
                    var sample = b < adversarialCount
                        ? attack.GenerateAdversarialExample(trainingData[index], labels[index], model)
                        : trainingData[index];
                    if (gradientModel is null)
                    {
                        epochInputs.Add(sample);
                        epochLabels.Add(labels[index]);
                        continue;
                    }

                    var gradient = gradientModel.ComputeGradients(sample, labels[index]);
                    gradientSum = gradientSum is null ? gradient : Engine.Add<T>(gradientSum, gradient);
                }

                if (gradientModel is not null && gradientSum is not null)
                {
                    gradientModel.ApplyGradients(Engine.Divide<T>(gradientSum, size == options.BatchSize ? fullBatch : lastBatch), learningRate);
                }
            }

            if (gradientModel is null)
            {
                TrainOnEpoch(model, epochInputs, epochLabels);
            }
        }
    }

    /// <summary>
    /// Trains a model without gradients on one epoch of mixed samples: one <c>Train</c> call on the stacked epoch when
    /// the samples are matrix rows or tensor slices (so a model that refits from scratch, such as a closed-form
    /// regression, fits the whole adversarially augmented epoch), otherwise one call per sample.
    /// </summary>
    private void TrainOnEpoch(IFullModel<T, TInput, TOutput> model, List<TInput> inputs, List<TOutput> labels)
    {
        if (inputs.Count == 0)
        {
            return;
        }

        var matrices = inputs.OfType<Matrix<T>>().ToArray();
        var vectors = labels.OfType<Vector<T>>().ToArray();
        if (matrices.Length == inputs.Count && vectors.Length == labels.Count)
        {
            int columns = matrices[0].Columns;
            var stacked = new Matrix<T>(matrices.Sum(m => m.Rows), columns);
            int row = 0;
            foreach (var m in matrices)
            {
                for (int i = 0; i < m.Rows; i++)
                {
                    stacked.SetRow(row++, m.GetRow(i));
                }
            }

            var targets = Engine.TensorConcatenate(vectors.Select(v => v.AsTensor()).ToArray(), 0).ToVector();
            model.Train((TInput)(object)stacked, (TOutput)(object)targets);
            return;
        }

        var tensors = inputs.OfType<Tensor<T>>().ToArray();
        var tensorLabels = labels.OfType<Tensor<T>>().ToArray();
        if (tensors.Length == inputs.Count && tensorLabels.Length == labels.Count)
        {
            model.Train((TInput)(object)Engine.TensorConcatenate(tensors, 0), (TOutput)(object)Engine.TensorConcatenate(tensorLabels, 0));
            return;
        }

        for (int i = 0; i < inputs.Count; i++)
        {
            model.Train(inputs[i], labels[i]);
        }
    }

    /// <summary>
    /// The training attack named by <see cref="AdversarialDefenseOptions{T}.AttackMethod"/>: epsilon from the options,
    /// step epsilon / 4, <see cref="AdversarialDefenseOptions{T}.AttackIterations"/> iterations, a random start.
    /// </summary>
    private IAdversarialAttack<T, TInput, TOutput> CreateTrainingAttack(int seed)
    {
        var attackOptions = new AdversarialAttackOptions<T>
        {
            Epsilon = options.Epsilon,
            StepSize = options.Epsilon / 4.0,
            Iterations = options.AttackIterations,
            UseRandomStart = true,
            RandomSeed = seed
        };

        return options.AttackMethod switch
        {
            AdversarialAttackMethod.Fgsm => new FGSMAttack<T, TInput, TOutput>(attackOptions),
            AdversarialAttackMethod.Pgd => new PGDAttack<T, TInput, TOutput>(attackOptions),
            AdversarialAttackMethod.CarliniWagner => new CWAttack<T, TInput, TOutput>(attackOptions),
            AdversarialAttackMethod.AutoAttack => new AutoAttack<T, TInput, TOutput>(attackOptions),
            _ => throw new InvalidOperationException(
                $"AdversarialDefenseOptions.AttackMethod has the undefined value {(int)options.AttackMethod}.")
        };
    }

    private static void Shuffle(int[] order, Random random)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    private void ValidateOptions()
    {
        if (!(options.AdversarialRatio >= 0.0 && options.AdversarialRatio <= 1.0))
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.AdversarialRatio must lie in [0, 1] (was {options.AdversarialRatio}).");
        }

        if (options.TrainingEpochs < 0)
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.TrainingEpochs must not be negative (was {options.TrainingEpochs}).");
        }

        if (options.BatchSize < 1)
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.BatchSize must be at least 1 (was {options.BatchSize}).");
        }

        if (options.AttackIterations < 1)
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.AttackIterations must be at least 1 (was {options.AttackIterations}).");
        }

        if (!(options.LearningRate > 0.0 && !double.IsInfinity(options.LearningRate)))
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.LearningRate must be a positive finite number (was {options.LearningRate}).");
        }

        if (options.UseEnsemble && options.EnsembleSize < 1)
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.EnsembleSize must be at least 1 (was {options.EnsembleSize}).");
        }
    }

    /// <inheritdoc/>
    public TInput PreprocessInput(TInput input)
    {
        if (!options.UsePreprocessing)
        {
            return input;
        }

        // Convert to vector for preprocessing, then convert back
        var vectorInput = ConversionsHelper.ConvertToVector<T, TInput>(input);

        var preprocessed = options.CustomPreprocessor is { } custom
            ? custom.Preprocess(vectorInput)
            : options.PreprocessingMethod switch
            {
                AdversarialPreprocessingMethod.JpegCompression => ApplyJPEGCompression(vectorInput),
                AdversarialPreprocessingMethod.BitDepthReduction => ApplyBitDepthReduction(vectorInput),
                AdversarialPreprocessingMethod.MedianSmoothing => ApplyMedianSmoothing(vectorInput),
                _ => throw new InvalidOperationException(
                    $"AdversarialDefenseOptions.PreprocessingMethod has the undefined value {(int)options.PreprocessingMethod}.")
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
    /// <remarks>
    /// The basis is built in double and converted to T once: in Half the angles k (i + 1/2) pi / n reach
    /// tens of radians, where the spacing of Half is 1/64 to 1/16, so cosines computed in T were off by
    /// several thousandths.
    /// </remarks>
    internal Tensor<T> DctMatrix(int n, int count, int first, int columns)
    {
        var k = Engine.Reshape(Engine.TensorArange<double>(first, first + count, 1), new[] { count, 1 });
        var position = Engine.Reshape(Engine.TensorArange<double>(0.5, columns + 0.5, 1), new[] { 1, columns });
        var angles = Engine.TensorMultiplyScalar(Engine.TensorMatMul(k, position), Math.PI / n);
        var cosines = Engine.TensorCos(angles);
        var scale = Engine.TensorAddScalar(new Tensor<double>(new[] { count, 1 }), Math.Sqrt(2.0 / n));
        if (first == 0)
        {
            scale = Engine.TensorSetSlice(scale, Engine.TensorAddScalar(new Tensor<double>(new[] { 1, 1 }), Math.Sqrt(1.0 / n)), new[] { 0, 0 });
        }

        return Engine.TensorMultiply(cosines, Engine.TensorBroadcastTo(scale, new[] { count, columns })).Contiguous().Cast<T>();
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
    /// <remarks>
    /// The reconstruction is clamped to the observed [min, max] of the input, as the 2D JPEG path does, so the
    /// output range does not depend on whether the input length factorises into 8 x 8 blocks.
    /// </remarks>
    private Vector<T> Apply1DDCTQuantization(Vector<T> input)
    {
        int n = input.Length;
        if (n == 0)
        {
            return input;
        }

        T minValue = Engine.TensorMinValue(input.AsTensor());
        T maxValue = Engine.TensorMaxValue(input.AsTensor());
        var x = Engine.Reshape(input.AsTensor(), new[] { n, 1 });
        var reconstruction = new Tensor<T>(new[] { n, 1 });

        // The orthonormal DCT-II in row chunks so the n x n matrix is never materialized at once.
        const int chunk = 512;
        for (int first = 0; first < n; first += chunk)
        {
            reconstruction = Engine.TensorAdd(reconstruction, QuantizedChunk(x, n, first, Math.Min(chunk, n - first)));
        }

        return Engine.TensorClamp(Engine.Reshape(reconstruction, new[] { n }), minValue, maxValue).ToVector();
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
        // Feature squeezing (Xu et al. 2018): b-bit depth keeps 2^b levels, x -> round(x (2^b - 1)) / (2^b - 1).
        if (options.BitDepth < 1 || options.BitDepth > 30)
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.BitDepth must lie in [1, 30] (was {options.BitDepth}).");
        }

        T levels = NumOps.FromDouble((1 << options.BitDepth) - 1);
        var scaled = Engine.TensorMultiplyScalar(Engine.TensorClamp(input.AsTensor(), NumOps.Zero, NumOps.One), levels);
        var reduced = Engine.TensorDivideScalar(Engine.TensorRound(scaled), levels);
        return Engine.TensorClamp(reduced, NumOps.Zero, NumOps.One).ToVector();
    }

    /// <summary>
    /// Local spatial smoothing (Xu et al. 2018, Section 3.2): every pixel becomes the median of its k x k window.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The window and border follow scipy.ndimage.median_filter, the implementation Xu et al. use: the window covers
    /// offsets [-(k / 2), k - 1 - k / 2] on each axis, borders are reflected (d c b a | a b c d), and the median of an
    /// even-sized window is the upper middle value (rank k*k / 2). A square input of side s is filtered as an s x s
    /// image; any other length as a 1 x n row with a 1 x k window.
    /// </para>
    /// </remarks>
    private Vector<T> ApplyMedianSmoothing(Vector<T> input)
    {
        int k = options.MedianFilterSize;
        if (k < 1)
        {
            throw new InvalidOperationException(
                $"AdversarialDefenseOptions.MedianFilterSize must be at least 1 (was {k}).");
        }

        int n = input.Length;
        if (n == 0 || k == 1)
        {
            return input;
        }

        int side = (int)Math.Sqrt(n);
        bool square = side * side == n && side > 1;
        int height = square ? side : 1;
        int width = square ? side : n;
        int rowsWindow = square ? k : 1;

        var image = Engine.Reshape(input.AsTensor(), new[] { height, width });
        var shifted = new List<Tensor<T>>(rowsWindow * k);
        for (int dy = -(rowsWindow / 2); dy <= rowsWindow - 1 - rowsWindow / 2; dy++)
        {
            var rows = Engine.TensorGather(image, ReflectedIndices(height, dy), 0);
            for (int dx = -(k / 2); dx <= k - 1 - k / 2; dx++)
            {
                shifted.Add(Engine.TensorGather(rows, ReflectedIndices(width, dx), 1));
            }
        }

        var (sorted, _) = Engine.TensorSort(Engine.TensorStack(shifted.ToArray(), 0), 0, false);
        int rank = shifted.Count / 2;
        var median = Engine.TensorSlice(sorted, new[] { rank, 0, 0 }, new[] { 1, height, width });
        return Engine.Reshape(median, new[] { n }).ToVector();
    }

    /// <summary>
    /// The indices i + offset for i in [0, count), reflected at the borders as scipy's "reflect" mode does.
    /// </summary>
    private static Tensor<int> ReflectedIndices(int count, int offset)
    {
        var indices = new int[count];
        int period = 2 * count;
        for (int i = 0; i < count; i++)
        {
            int j = ((i + offset) % period + period) % period;
            indices[i] = j < count ? j : period - 1 - j;
        }

        return new Tensor<int>(indices, new[] { count });
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
        private readonly IFullModel<T, TInput, TOutput>[] _members;
        private readonly AdversarialTraining<T, TInput, TOutput> _defense;

        public PreprocessingFullModel(IFullModel<T, TInput, TOutput> inner, AdversarialTraining<T, TInput, TOutput> defense)
            : this(new[] { inner }, defense)
        {
        }

        /// <summary>
        /// Wraps one model or an ensemble. Predictions average the members' outputs; every other member of the
        /// model contract (parameters, persistence, gradients) is the first member's.
        /// </summary>
        public PreprocessingFullModel(IFullModel<T, TInput, TOutput>[] members, AdversarialTraining<T, TInput, TOutput> defense)
        {
            Guard.NotNull(members);
            if (members.Length == 0)
            {
                throw new ArgumentException("At least one model is required.", nameof(members));
            }

            _members = members;
            _inner = members[0];
            Guard.NotNull(defense);
            _defense = defense;
        }

        /// <summary>The ensemble members (one entry when no ensemble is used).</summary>
        internal IReadOnlyList<IFullModel<T, TInput, TOutput>> Members => _members;

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
            var first = _inner.Predict(preprocessed);
            if (_members.Length == 1)
            {
                return first;
            }

            var sum = ConversionsHelper.ConvertToVector<T, TOutput>(first);
            for (int m = 1; m < _members.Length; m++)
            {
                sum = AiDotNetEngine.Current.Add<T>(sum, ConversionsHelper.ConvertToVector<T, TOutput>(_members[m].Predict(preprocessed)));
            }

            var mean = AiDotNetEngine.Current.Divide<T>(sum, NumOps.FromDouble(_members.Length));
            return ConversionsHelper.ConvertVectorToInput<T, TOutput>(mean, first);
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
            return new PreprocessingFullModel(_members.Select(m => m.DeepCopy()).ToArray(), _defense);
        }

        /// <inheritdoc/>
        public IFullModel<T, TInput, TOutput> Clone()
        {
            return new PreprocessingFullModel(_members.Select(m => m.Clone()).ToArray(), _defense);
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
