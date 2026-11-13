using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// FedGEN — Data-free federated distillation using a lightweight generator on the server.
/// </summary>
/// <remarks>
/// <para>
/// FedGEN (Zhu et al., 2021) learns a small conditional generator G_w(y, ε) on the server that maps a label and
/// Gaussian noise to the latent feature space of the clients' predictors. The generator is trained so that the
/// label-count-weighted ensemble of client predictor heads classifies its samples as the requested label:
/// min_w E_{y ~ p̂(y), ε ~ N(0, I)} CE(Σ_k α_k^y h_k(G_w(y, ε)), y), with α_k^y = n_k^y / Σ_j n_j^y.
/// Clients then distill this ensemble knowledge into their own head with the data-free term
/// E_{ŷ ~ p̂(y), z ~ G_w(·|ŷ)} CE(h(z; θ_p), ŷ).
/// </para>
/// <para>
/// <b>Model layout:</b> the first C·L entries of a client's parameter vector are its predictor weights W [C, L]
/// (row-major) and the next C entries its bias b, where C is the class count and L the latent dimension; the
/// remaining entries (the feature extractor) are not touched by distillation. The generator is a two-layer
/// MLP [onehot(y); ε] → ReLU(hidden) → z, sized by the constructor options.
/// </para>
/// <para>
/// <b>Generator objective:</b> following the paper and the authors' released code (serverpFedGen.train_generator),
/// each server step minimizes
/// <c>TeacherLossWeight · L_teacher − StudentLossWeight · KL + DiversityLossWeight · L_div</c> (ensemble_alpha = 1,
/// ensemble_beta = 0 and ensemble_eta = 1 in the official configs). L_teacher is selected by
/// <see cref="TeacherLoss"/>; KL = KL(softmax(Σ_k α_k^y h_k(z)) ‖ softmax(h_s(z))) averaged over every entry, with
/// h_s the sample-count-weighted average head (the server model); and
/// L_div = exp(−mean_{i,j} ‖ε_i − ε_j‖²/d_ε · ‖z_i − z_j‖₁/L).
/// </para>
/// <para>
/// <b>Generator optimizer:</b> as in the official code, the generator is trained with Adam (lr 3e-4,
/// betas (0.9, 0.999), eps 1e-8, L2 weight decay 1e-2 added to the gradient) and the learning rate decays by
/// 0.98 after every round (ExponentialLR). <see cref="GeneratorOptimizer"/> and the related properties change
/// this.
/// </para>
/// <para>
/// <b>Knowledge exchange:</b> <see cref="ExtractKnowledge(Vector{T}, Matrix{T})"/> returns the head as a
/// [C, L + 5] matrix (W | b | per-class sample count as four base-256 digits, least significant first). The
/// digits are exact in every numeric type (BFloat16 holds integers exactly only up to 256 and Half up to 2048), so
/// counts of any size reach the server unchanged. <see cref="AggregateKnowledge"/> trains the generator
/// against the ensemble of these heads and returns a single-row matrix holding the generator parameters followed
/// by the global label distribution p̂(y), which <see cref="ApplyKnowledge"/> uses on the client.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of needing a shared dataset, the server learns a tiny "imagination" network
/// that produces feature vectors every client's classifier agrees on. Clients then practise on these imagined
/// features, which transfers what the other clients know without sharing any real data.
/// </para>
/// <para>
/// Reference: Zhu et al. (2021), "Data-Free Knowledge Distillation for Heterogeneous Federated Learning".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FedGENDistillation<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedDistillationStrategy<T>
{
    private readonly int _numClasses;
    private readonly int _featureDim;
    private readonly int _noiseDim;
    private readonly int _hiddenDim;
    private readonly int _batchSize;
    private readonly int _generatorEpochs;
    private readonly double _generatorLearningRate;
    private readonly int _distillationEpochs;
    private readonly double _distillationLearningRate;
    private readonly Random _rng;

    // Per-class sample counts travel as CountDigits base-CountBase digits so they stay exact in narrow T.
    private const int CountDigits = 4;
    private const int CountBase = 256;

    private double _teacherLossWeight = 1.0;
    private double _studentLossWeight;
    private double _diversityLossWeight = 1.0;
    private double _adamBeta1 = 0.9;
    private double _adamBeta2 = 0.999;
    private double _adamEpsilon = 1e-8;
    private double _generatorWeightDecay = 1e-2;
    private double _generatorLearningRateDecay = 0.98;

    // Generator optimizer state, kept in double (Adam's second moment of a small gradient underflows Half). Slots
    // follow the update order W2, b2, W1, b1.
    private readonly double[][] _firstMoments;
    private readonly double[][] _secondMoments;
    private int _generatorSteps;
    private int _completedRounds;

    // Server-side generator G_w(y, ε) = W2ᵀ ReLU(W1ᵀ [onehot(y); ε] + b1) + b2.
    private Tensor<T> _w1;
    private Tensor<T> _b1;
    private Tensor<T> _w2;
    private Tensor<T> _b2;

    /// <summary>
    /// Creates a new FedGEN distillation strategy.
    /// </summary>
    /// <param name="numClasses">Number of output classes C.</param>
    /// <param name="featureDim">Latent feature dimension L of the client predictors (generator output size).</param>
    /// <param name="generatorEpochs">Server generator training steps per round. Default: 5.</param>
    /// <param name="generatorLearningRate">Initial generator learning rate. Default: 3e-4 (ensemble_lr of the official
    /// configs).</param>
    /// <param name="seed">Random seed for generator initialization and sampling. Default: 42.</param>
    /// <param name="noiseDim">Dimension of the generator noise ε. Default: 16.</param>
    /// <param name="generatorHiddenDim">Hidden width of the generator MLP. Default: 32.</param>
    /// <param name="batchSize">Number of (label, noise) samples per generator or distillation step. Default: 32.</param>
    /// <param name="distillationEpochs">Client distillation steps per round. Default: 5.</param>
    /// <param name="distillationLearningRate">Client distillation learning rate. Default: 0.01.</param>
    public FedGENDistillation(int numClasses = 10, int featureDim = 64,
        int generatorEpochs = 5, double generatorLearningRate = 3e-4, int seed = 42,
        int noiseDim = 16, int generatorHiddenDim = 32, int batchSize = 32,
        int distillationEpochs = 5, double distillationLearningRate = 0.01)
    {
        if (numClasses < 2) throw new ArgumentOutOfRangeException(nameof(numClasses), "Need at least two classes.");
        if (featureDim < 1) throw new ArgumentOutOfRangeException(nameof(featureDim), "Feature dimension must be positive.");
        if (noiseDim < 1) throw new ArgumentOutOfRangeException(nameof(noiseDim), "Noise dimension must be positive.");
        if (generatorHiddenDim < 1) throw new ArgumentOutOfRangeException(nameof(generatorHiddenDim), "Hidden dimension must be positive.");
        if (batchSize < 1) throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be positive.");
        if (generatorEpochs < 0) throw new ArgumentOutOfRangeException(nameof(generatorEpochs));
        if (distillationEpochs < 0) throw new ArgumentOutOfRangeException(nameof(distillationEpochs));

        _numClasses = numClasses;
        _featureDim = featureDim;
        _noiseDim = noiseDim;
        _hiddenDim = generatorHiddenDim;
        _batchSize = batchSize;
        _generatorEpochs = generatorEpochs;
        _generatorLearningRate = generatorLearningRate;
        _distillationEpochs = distillationEpochs;
        _distillationLearningRate = distillationLearningRate;
        _rng = RandomHelper.CreateSeededRandom(seed);

        int inputDim = _numClasses + _noiseDim;
        _w1 = XavierUniform(inputDim, _hiddenDim);
        _b1 = new Tensor<T>(new[] { _hiddenDim });
        _w2 = XavierUniform(_hiddenDim, _featureDim);
        _b2 = new Tensor<T>(new[] { _featureDim });
        _firstMoments = new[] { new double[_w2.Length], new double[_b2.Length], new double[_w1.Length], new double[_b1.Length] };
        _secondMoments = new[] { new double[_w2.Length], new double[_b2.Length], new double[_w1.Length], new double[_b1.Length] };
    }

    /// <summary>
    /// Weight of the teacher term in the generator objective (ensemble_alpha in the official code). Default: 1.
    /// </summary>
    public double TeacherLossWeight
    {
        get => _teacherLossWeight;
        set => _teacherLossWeight = NonNegative(value, nameof(TeacherLossWeight));
    }

    /// <summary>
    /// Weight of the adversarial student term: the generator maximizes the KL divergence between the teacher
    /// ensemble and the server model on its samples (ensemble_beta in the official code). Default: 0 (off).
    /// </summary>
    public double StudentLossWeight
    {
        get => _studentLossWeight;
        set => _studentLossWeight = NonNegative(value, nameof(StudentLossWeight));
    }

    /// <summary>
    /// Weight of the diversity term that keeps different noise draws from collapsing onto one feature
    /// (ensemble_eta in the official code). Default: 1.
    /// </summary>
    public double DiversityLossWeight
    {
        get => _diversityLossWeight;
        set => _diversityLossWeight = NonNegative(value, nameof(DiversityLossWeight));
    }

    /// <summary>
    /// Form of the teacher term. Default: <see cref="FedGenTeacherLoss.EnsembleCrossEntropy"/>.
    /// </summary>
    public FedGenTeacherLoss TeacherLoss { get; set; } = FedGenTeacherLoss.EnsembleCrossEntropy;

    /// <summary>
    /// Update rule of the server generator. Default: <see cref="FedGenGeneratorOptimizer.Adam"/>.
    /// </summary>
    public FedGenGeneratorOptimizer GeneratorOptimizer { get; set; } = FedGenGeneratorOptimizer.Adam;

    /// <summary>Adam first-moment decay β1. Default: 0.9.</summary>
    public double AdamBeta1
    {
        get => _adamBeta1;
        set => _adamBeta1 = UnitInterval(value, nameof(AdamBeta1));
    }

    /// <summary>Adam second-moment decay β2. Default: 0.999.</summary>
    public double AdamBeta2
    {
        get => _adamBeta2;
        set => _adamBeta2 = UnitInterval(value, nameof(AdamBeta2));
    }

    /// <summary>Adam denominator term ε. Default: 1e-8.</summary>
    public double AdamEpsilon
    {
        get => _adamEpsilon;
        set => _adamEpsilon = NonNegative(value, nameof(AdamEpsilon));
    }

    /// <summary>L2 weight decay added to the generator gradient (weight_decay in the official configs). Default: 1e-2.</summary>
    public double GeneratorWeightDecay
    {
        get => _generatorWeightDecay;
        set => _generatorWeightDecay = NonNegative(value, nameof(GeneratorWeightDecay));
    }

    /// <summary>
    /// Factor applied to the generator learning rate after every round (ExponentialLR gamma in the official code).
    /// Default: 0.98; 1 keeps it constant.
    /// </summary>
    public double GeneratorLearningRateDecay
    {
        get => _generatorLearningRateDecay;
        set => _generatorLearningRateDecay = NonNegative(value, nameof(GeneratorLearningRateDecay));
    }

    /// <summary>The generator learning rate of the current round: the initial rate times decay^rounds.</summary>
    internal double CurrentGeneratorLearningRate =>
        _generatorLearningRate * Math.Pow(_generatorLearningRateDecay, _completedRounds);

    /// <summary>The current generator parameters (W1, b1, W2, b2).</summary>
    internal (Tensor<T> W1, Tensor<T> B1, Tensor<T> W2, Tensor<T> B2) GeneratorState => (_w1, _b1, _w2, _b2);

    /// <summary>Number of client parameters that form the predictor head (C·L weights plus C biases).</summary>
    public int HeadParameterCount => _numClasses * _featureDim + _numClasses;

    /// <summary>Number of generator parameters.</summary>
    public int GeneratorParameterCount =>
        (_numClasses + _noiseDim) * _hiddenDim + _hiddenDim + _hiddenDim * _featureDim + _featureDim;

    /// <inheritdoc/>
    /// <remarks>Every class counts as one local sample; use <see cref="ExtractHeadKnowledge"/> with label counts for the paper's α weights.</remarks>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        var uniform = new int[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            uniform[c] = 1;
        }

        return ExtractHeadKnowledge(localModelParameters, uniform);
    }

    /// <summary>
    /// Extracts the client's predictor head together with its per-class sample counts.
    /// </summary>
    /// <param name="localModelParameters">Client parameters; the first <see cref="HeadParameterCount"/> are the head.</param>
    /// <param name="labelCounts">Number of local training samples of each class (length C, non-negative).</param>
    /// <returns>A [C, L + 5] matrix: predictor weights, bias, sample count as four base-256 digits (least
    /// significant first).</returns>
    public Matrix<T> ExtractHeadKnowledge(Vector<T> localModelParameters, int[] labelCounts)
    {
        Guard.NotNull(localModelParameters);
        Guard.NotNull(labelCounts);
        if (labelCounts.Length != _numClasses || labelCounts.Any(c => c < 0))
        {
            throw new ArgumentException($"Label counts must hold {_numClasses} non-negative entries.", nameof(labelCounts));
        }

        var (weights, bias) = ReadHead(localModelParameters);
        // Counts stay int until they are split into digits below CountBase, which every T represents exactly.
        var digits = new int[_numClasses * CountDigits];
        for (int c = 0; c < _numClasses; c++)
        {
            long remaining = labelCounts[c];
            for (int d = 0; d < CountDigits; d++)
            {
                digits[c * CountDigits + d] = (int)(remaining % CountBase);
                remaining /= CountBase;
            }
        }

        var counts = new Tensor<int>(digits, new[] { _numClasses, CountDigits }).Cast<T>();
        return Engine.TensorConcatenate(new[] { weights, bias.Reshape(new[] { _numClasses, 1 }), counts }, 1).ToMatrix();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Trains the server generator against the α-weighted ensemble of the client heads (α_k^y ∝ w_k n_k^y) and
    /// returns [generator parameters | p̂(y)] as a single row.
    /// </remarks>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientKnowledge);
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        int k = clientKnowledge.Count;
        var heads = new (Tensor<T> Weights, Tensor<T> Bias)[k];
        var weightedCounts = new double[k, _numClasses];
        var classTotals = new double[_numClasses];
        int index = 0;
        foreach (var (clientId, knowledge) in clientKnowledge)
        {
            if (knowledge.Rows != _numClasses || knowledge.Columns != _featureDim + 1 + CountDigits)
            {
                throw new ArgumentException(
                    $"Client {clientId} knowledge is {knowledge.Rows}x{knowledge.Columns}; expected {_numClasses}x{_featureDim + 1 + CountDigits}.",
                    nameof(clientKnowledge));
            }

            var tensor = Tensor<T>.FromMatrix(knowledge);
            heads[index] = (
                Engine.TensorSlice(tensor, new[] { 0, 0 }, new[] { _numClasses, _featureDim }),
                Engine.TensorSlice(tensor, new[] { 0, _featureDim }, new[] { _numClasses, 1 }).Reshape(new[] { _numClasses }));
            var digits = Engine.TensorSlice(tensor, new[] { 0, _featureDim + 1 }, new[] { _numClasses, CountDigits })
                .Cast<double>().ToArray();
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            for (int c = 0; c < _numClasses; c++)
            {
                long count = 0;
                for (int d = CountDigits - 1; d >= 0; d--)
                {
                    count = count * CountBase + (long)Math.Round(digits[c * CountDigits + d]);
                }

                weightedCounts[index, c] = Math.Max(0, w * count);
                classTotals[c] += weightedCounts[index, c];
            }

            index++;
        }

        double grandTotal = classTotals.Sum();
        if (grandTotal <= 0)
            throw new ArgumentException("Clients reported no labelled samples.", nameof(clientKnowledge));

        // p̂(y) and the per-class ensemble weights α_k^y (zero for classes nobody has).
        var labelDistribution = new double[_numClasses];
        var alpha = new double[k][];
        for (int c = 0; c < _numClasses; c++)
        {
            labelDistribution[c] = classTotals[c] / grandTotal;
        }

        for (int i = 0; i < k; i++)
        {
            alpha[i] = new double[_numClasses];
            for (int c = 0; c < _numClasses; c++)
            {
                alpha[i][c] = classTotals[c] > 0 ? weightedCounts[i, c] / classTotals[c] : 0;
            }
        }

        // The student is the server model: the client heads averaged by their (weighted) sample counts.
        var headWeights = new Tensor<T>[k];
        var headBiases = new Tensor<T>[k];
        var sampleTotals = new double[k];
        for (int i = 0; i < k; i++)
        {
            headWeights[i] = heads[i].Weights;
            headBiases[i] = heads[i].Bias;
            for (int c = 0; c < _numClasses; c++)
            {
                sampleTotals[i] += weightedCounts[i, c];
            }
        }

        var student = (
            Infrastructure.FederatedTensorMath<T>.WeightedSum(headWeights, sampleTotals, grandTotal),
            Infrastructure.FederatedTensorMath<T>.WeightedSum(headBiases, sampleTotals, grandTotal));

        for (int epoch = 0; epoch < _generatorEpochs; epoch++)
        {
            var labels = SampleLabels(labelDistribution);
            var noise = SampleNoise(_batchSize);
            var (input, hidden, latent) = Generate(labels, NoiseTensor(noise, _batchSize));
            var objective = GeneratorObjective(heads, alpha, student, labels, latent, noise);
            BackpropGenerator(input, hidden, objective.GradLatent);
        }

        _completedRounds++;

        var packed = Engine.TensorConcatenate(new[]
        {
            _w1.Reshape(new[] { _w1.Length }), _b1, _w2.Reshape(new[] { _w2.Length }), _b2,
            new Tensor<double>(labelDistribution, new[] { _numClasses }).Cast<T>(),
        }, 0);
        return packed.Reshape(new[] { 1, packed.Length }).ToMatrix();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Distills the generator into the client head: gradient descent on CE(h(G(ŷ, ε)), ŷ) with ŷ ~ p̂(y). Only the
    /// first <see cref="HeadParameterCount"/> parameters change; <paramref name="temperature"/> is not used because
    /// the paper's term is a plain cross-entropy.
    /// </remarks>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        Guard.NotNull(localModelParameters);
        Guard.NotNull(globalKnowledge);
        int generatorCount = GeneratorParameterCount;
        if (globalKnowledge.Rows != 1 || globalKnowledge.Columns != generatorCount + _numClasses)
        {
            throw new ArgumentException(
                $"Global knowledge must be 1x{generatorCount + _numClasses} (generator parameters and label distribution).",
                nameof(globalKnowledge));
        }

        var packed = Tensor<T>.FromMatrix(globalKnowledge).Reshape(new[] { generatorCount + _numClasses });
        var generator = UnpackGenerator(packed);
        var labelDistribution = Engine.TensorSlice(packed, new[] { generatorCount }, new[] { _numClasses }).Cast<double>().ToArray();

        var (weights, bias) = ReadHead(localModelParameters);
        var learningRate = NumOps.FromDouble(_distillationLearningRate);
        for (int epoch = 0; epoch < _distillationEpochs; epoch++)
        {
            var labels = SampleLabels(labelDistribution);
            var latent = Forward(generator, labels, NoiseTensor(SampleNoise(_batchSize), _batchSize)).Latent;
            var gradLogits = Engine.TensorMultiplyScalar(
                Engine.TensorSubtract(Engine.Softmax(HeadLogits(latent, (weights, bias)), 1), OneHot(labels)),
                NumOps.FromDouble(1.0 / _batchSize));

            weights = Engine.TensorSubtract(weights,
                Engine.TensorMultiplyScalar(Engine.TensorMatMul(Engine.TensorTranspose(gradLogits), latent), learningRate));
            bias = Engine.TensorSubtract(bias,
                Engine.TensorMultiplyScalar(Engine.ReduceSum(gradLogits, new[] { 0 }, false), learningRate));
        }

        int headCount = HeadParameterCount;
        var head = Engine.TensorConcatenate(new[] { weights.Reshape(new[] { _numClasses * _featureDim }), bias }, 0);
        var rest = localModelParameters.Length > headCount
            ? Engine.TensorSlice(localModelParameters.AsTensor(), new[] { headCount }, new[] { localModelParameters.Length - headCount })
            : new Tensor<T>(new[] { 0 });
        return Engine.TensorConcatenate(new[] { head, rest }, 0).ToVector();
    }

    /// <summary>
    /// Draws latent features from the current server generator.
    /// </summary>
    /// <param name="label">Class to generate.</param>
    /// <param name="count">Number of samples.</param>
    /// <returns>A [count, L] matrix of generated latent features.</returns>
    public Matrix<T> GenerateFeatures(int label, int count)
    {
        if (label < 0 || label >= _numClasses) throw new ArgumentOutOfRangeException(nameof(label));
        if (count < 1) throw new ArgumentOutOfRangeException(nameof(count));

        var labels = new int[count];
        for (int i = 0; i < count; i++)
        {
            labels[i] = label;
        }

        var generator = (_w1, _b1, _w2, _b2);
        return Forward(generator, labels, NoiseTensor(SampleNoise(count), count)).Latent.ToMatrix();
    }

    /// <summary>
    /// One server step of the FedGen generator objective and its gradient with respect to the generated features.
    /// </summary>
    /// <remarks>
    /// <para>Returns the unweighted terms (teacher loss, KL of the student from the teacher ensemble, diversity
    /// loss) and dJ/dZ of J = TeacherLossWeight · teacher − StudentLossWeight · KL + DiversityLossWeight ·
    /// diversity, following serverpFedGen.train_generator and DiversityLoss(metric = 'l1') of the official code.
    /// The losses are formed in double from the logits.</para>
    /// </remarks>
    /// <param name="heads">Client predictor heads (W [C, L], b [C]).</param>
    /// <param name="alpha">Per-client, per-class ensemble weights α_k^y.</param>
    /// <param name="student">The server model head.</param>
    /// <param name="labels">Sampled labels y (length B).</param>
    /// <param name="latent">Generated features Z [B, L].</param>
    /// <param name="noise">The noise ε that produced <paramref name="latent"/>, row-major [B, noiseDim].</param>
    internal (double Teacher, double Student, double Diversity, Tensor<T> GradLatent) GeneratorObjective(
        (Tensor<T> Weights, Tensor<T> Bias)[] heads,
        double[][] alpha,
        (Tensor<T> Weights, Tensor<T> Bias) student,
        int[] labels,
        Tensor<T> latent,
        double[] noise)
    {
        int batch = labels.Length;
        int k = heads.Length;
        int classes = _numClasses;
        int cells = batch * classes;

        var logitList = new Tensor<T>[k];
        var alphaList = new Tensor<T>[k];
        var weightList = new Tensor<T>[k];
        for (int i = 0; i < k; i++)
        {
            logitList[i] = HeadLogits(latent, heads[i]);
            alphaList[i] = Column(labels, alpha[i]);
            weightList[i] = heads[i].Weights;
        }

        var logits = Engine.TensorStack(logitList, 0); // [K, B, C]
        var alphas = Engine.TensorStack(alphaList, 0); // [K, B, C], row i holds α_k^{y_i}
        var ensemble = Engine.ReduceSum(Engine.TensorMultiply(alphas, logits), new[] { 0 }, false); // [B, C]

        // Teacher ensemble p = softmax(Σ_k α_k^y l_k) and its log, in double.
        var logP = LogSoftmaxRows(ensemble.Cast<double>().ToArray(), batch, classes);

        double teacher = 0;
        var ensembleGrad = new double[cells];         // dJ/d(ensemble logits)
        var clientGrad = new double[k * cells];       // dJ/d(client logits) from the per-client teacher form
        if (TeacherLoss == FedGenTeacherLoss.EnsembleCrossEntropy)
        {
            // CE(Σ_k α_k^y l_k, y) averaged over the batch; dCE/ds = (softmax(s) - onehot(y)) / B.
            for (int i = 0; i < batch; i++)
            {
                teacher -= logP[i * classes + labels[i]] / batch;
                for (int c = 0; c < classes; c++)
                {
                    double target = c == labels[i] ? 1.0 : 0.0;
                    ensembleGrad[i * classes + c] = _teacherLossWeight * (Math.Exp(logP[i * classes + c]) - target) / batch;
                }
            }
        }
        else
        {
            // Σ_k mean_i α_k^{y_i} CE(l_k,i, y_i); dCE_k/dl_k = α_k^{y_i} (softmax(l_k) - onehot(y)) / B.
            var clientLogP = LogSoftmaxRows(logits.Cast<double>().ToArray(), k * batch, classes);
            for (int j = 0; j < k; j++)
            {
                for (int i = 0; i < batch; i++)
                {
                    double weight = alpha[j][labels[i]];
                    int row = (j * batch + i) * classes;
                    teacher -= weight * clientLogP[row + labels[i]] / batch;
                    for (int c = 0; c < classes; c++)
                    {
                        double target = c == labels[i] ? 1.0 : 0.0;
                        clientGrad[row + c] = _teacherLossWeight * weight * (Math.Exp(clientLogP[row + c]) - target) / batch;
                    }
                }
            }
        }

        // Student KL(p || q) with q = softmax(server head), averaged over all B * C entries (F.kl_div 'mean').
        // The generator maximizes it, so its gradient enters with -StudentLossWeight.
        var logQ = LogSoftmaxRows(HeadLogits(latent, student).Cast<double>().ToArray(), batch, classes);
        double kl = 0;
        var studentGrad = new double[cells];
        for (int i = 0; i < batch; i++)
        {
            double rowKl = 0;
            for (int c = 0; c < classes; c++)
            {
                int idx = i * classes + c;
                rowKl += Math.Exp(logP[idx]) * (logP[idx] - logQ[idx]);
            }

            kl += rowKl / cells;
            for (int c = 0; c < classes; c++)
            {
                int idx = i * classes + c;
                double p = Math.Exp(logP[idx]);
                double q = Math.Exp(logQ[idx]);
                studentGrad[idx] = -_studentLossWeight * (q - p) / cells;
                ensembleGrad[idx] -= _studentLossWeight * (p * (logP[idx] - logQ[idx]) - p * rowKl) / cells;
            }
        }

        // dJ/dZ through the clients: G_k = α_k ⊙ dJ/ds + dJ/dl_k, then Σ_k G_k W_k as one batched product.
        var ensembleGradTensor = new Tensor<double>(ensembleGrad, new[] { 1, batch, classes }).Cast<T>();
        var perClient = Engine.TensorAdd(
            Engine.TensorMultiply(alphas, Engine.TensorBroadcastTo(ensembleGradTensor, new[] { k, batch, classes })),
            new Tensor<double>(clientGrad, new[] { k, batch, classes }).Cast<T>());
        var gradLatent = Engine.ReduceSum(
            Engine.TensorMatMul(perClient, Engine.TensorStack(weightList, 0)), new[] { 0 }, false); // [B, L]
        gradLatent = Engine.TensorAdd(gradLatent,
            Engine.TensorMatMul(new Tensor<double>(studentGrad, new[] { batch, classes }).Cast<T>(), student.Weights));

        // Diversity exp(-mean_{i,j} d_ε(i, j) d_z(i, j)), d_ε = mean squared noise difference and d_z = mean
        // absolute feature difference. d/dz_i = D * (-2 / (B^2 L)) Σ_j d_ε(i, j) sign(z_i - z_j).
        var noiseDistance = new double[batch * batch];
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < batch; j++)
            {
                double sum = 0;
                for (int d = 0; d < _noiseDim; d++)
                {
                    double delta = noise[i * _noiseDim + d] - noise[j * _noiseDim + d];
                    sum += delta * delta;
                }

                noiseDistance[i * batch + j] = sum / _noiseDim;
            }
        }

        var pairShape = new[] { batch, batch, _featureDim };
        var differences = Engine.TensorSubtract(
            Engine.TensorBroadcastTo(latent.Reshape(new[] { batch, 1, _featureDim }), pairShape),
            Engine.TensorBroadcastTo(latent.Reshape(new[] { 1, batch, _featureDim }), pairShape)); // z_i - z_j
        var featureDistance = Engine.ReduceMean(Engine.TensorAbs(differences), new[] { 2 }, false).Cast<double>().ToArray();
        double meanProduct = 0;
        for (int p = 0; p < noiseDistance.Length; p++)
        {
            meanProduct += noiseDistance[p] * featureDistance[p];
        }

        double diversity = Math.Exp(-meanProduct / noiseDistance.Length);
        var noiseWeights = Engine.TensorBroadcastTo(
            new Tensor<double>(noiseDistance, new[] { batch, batch, 1 }).Cast<T>(), pairShape);
        var diversityGrad = Engine.ReduceSum(
            Engine.TensorMultiply(Engine.TensorSign(differences), noiseWeights), new[] { 1 }, false); // [B, L]
        double diversityScale = _diversityLossWeight * diversity * -2.0 / ((double)batch * batch * _featureDim);
        gradLatent = Engine.TensorAdd(gradLatent, Engine.TensorMultiplyScalar(diversityGrad, NumOps.FromDouble(diversityScale)));

        return (teacher, kl, diversity, gradLatent);
    }

    /// <summary>Row-wise log-softmax of a row-major [rows, columns] array, in double.</summary>
    private static double[] LogSoftmaxRows(double[] values, int rows, int columns)
    {
        var result = new double[values.Length];
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity;
            for (int c = 0; c < columns; c++)
            {
                max = Math.Max(max, values[r * columns + c]);
            }

            double sum = 0;
            for (int c = 0; c < columns; c++)
            {
                sum += Math.Exp(values[r * columns + c] - max);
            }

            double logSum = max + Math.Log(sum);
            for (int c = 0; c < columns; c++)
            {
                result[r * columns + c] = values[r * columns + c] - logSum;
            }
        }

        return result;
    }

    private (Tensor<T> Input, Tensor<T> Hidden, Tensor<T> Latent) Generate(int[] labels, Tensor<T> noise)
    {
        return Forward((_w1, _b1, _w2, _b2), labels, noise);
    }

    private (Tensor<T> Input, Tensor<T> Hidden, Tensor<T> Latent) Forward(
        (Tensor<T> W1, Tensor<T> B1, Tensor<T> W2, Tensor<T> B2) generator, int[] labels, Tensor<T> noise)
    {
        int batch = labels.Length;
        var input = Engine.TensorConcatenate(new[] { OneHot(labels), noise }, 1);
        var hidden = Engine.TensorReLU(Engine.TensorAdd(
            Engine.TensorMatMul(input, generator.W1),
            Engine.TensorBroadcastTo(generator.B1.Reshape(new[] { 1, _hiddenDim }), new[] { batch, _hiddenDim })));
        var latent = Engine.TensorAdd(
            Engine.TensorMatMul(hidden, generator.W2),
            Engine.TensorBroadcastTo(generator.B2.Reshape(new[] { 1, _featureDim }), new[] { batch, _featureDim }));
        return (input, hidden, latent);
    }

    /// <summary>
    /// Back-propagates dJ/dZ through the generator and applies one optimizer step at the current learning rate.
    /// </summary>
    internal void BackpropGenerator(Tensor<T> input, Tensor<T> hidden, Tensor<T> gradLatent)
    {
        var gradW2 = Engine.TensorMatMul(Engine.TensorTranspose(hidden), gradLatent);
        var gradB2 = Engine.ReduceSum(gradLatent, new[] { 0 }, false);
        var gradHidden = Engine.TensorMultiply(
            Engine.TensorMatMul(gradLatent, Engine.TensorTranspose(_w2)),
            Engine.TensorGreaterThan(hidden, NumOps.Zero));
        var gradW1 = Engine.TensorMatMul(Engine.TensorTranspose(input), gradHidden);
        var gradB1 = Engine.ReduceSum(gradHidden, new[] { 0 }, false);

        _generatorSteps++;
        double learningRate = CurrentGeneratorLearningRate;
        _w2 = OptimizerStep(0, _w2, gradW2, learningRate);
        _b2 = OptimizerStep(1, _b2, gradB2, learningRate);
        _w1 = OptimizerStep(2, _w1, gradW1, learningRate);
        _b1 = OptimizerStep(3, _b1, gradB1, learningRate);
    }

    /// <summary>
    /// torch.optim.Adam (or SGD) update with L2 weight decay: g' = g + λθ; Adam then keeps m = β1 m + (1 − β1) g',
    /// v = β2 v + (1 − β2) g'² and moves θ by lr · (m / (1 − β1^t)) / (sqrt(v / (1 − β2^t)) + ε).
    /// </summary>
    private Tensor<T> OptimizerStep(int slot, Tensor<T> parameter, Tensor<T> gradient, double learningRate)
    {
        var theta = parameter.Cast<double>().ToArray();
        var grad = gradient.Cast<double>().ToArray();
        var m = _firstMoments[slot];
        var v = _secondMoments[slot];
        bool adam = GeneratorOptimizer == FedGenGeneratorOptimizer.Adam;
        double correction1 = 1 - Math.Pow(_adamBeta1, _generatorSteps);
        double correction2 = 1 - Math.Pow(_adamBeta2, _generatorSteps);
        for (int i = 0; i < theta.Length; i++)
        {
            double g = grad[i] + _generatorWeightDecay * theta[i];
            if (adam)
            {
                m[i] = _adamBeta1 * m[i] + (1 - _adamBeta1) * g;
                v[i] = _adamBeta2 * v[i] + (1 - _adamBeta2) * g * g;
                theta[i] -= learningRate * (m[i] / correction1) / (Math.Sqrt(v[i] / correction2) + _adamEpsilon);
            }
            else
            {
                theta[i] -= learningRate * g;
            }
        }

        return new Tensor<double>(theta, parameter.Shape.ToArray()).Cast<T>();
    }

    private (Tensor<T> W1, Tensor<T> B1, Tensor<T> W2, Tensor<T> B2) UnpackGenerator(Tensor<T> packed)
    {
        int inputDim = _numClasses + _noiseDim;
        int offset = 0;
        Tensor<T> Take(int length, int[] shape)
        {
            var part = Engine.TensorSlice(packed, new[] { offset }, new[] { length }).Reshape(shape);
            offset += length;
            return part;
        }

        var w1 = Take(inputDim * _hiddenDim, new[] { inputDim, _hiddenDim });
        var b1 = Take(_hiddenDim, new[] { _hiddenDim });
        var w2 = Take(_hiddenDim * _featureDim, new[] { _hiddenDim, _featureDim });
        var b2 = Take(_featureDim, new[] { _featureDim });
        return (w1, b1, w2, b2);
    }

    private (Tensor<T> Weights, Tensor<T> Bias) ReadHead(Vector<T> parameters)
    {
        int headCount = HeadParameterCount;
        if (parameters.Length < headCount)
        {
            throw new ArgumentException(
                $"Model has {parameters.Length} parameters; the predictor head needs {headCount} ({_numClasses}x{_featureDim} weights and {_numClasses} biases).",
                nameof(parameters));
        }

        var flat = parameters.AsTensor();
        var weights = Engine.TensorSlice(flat, new[] { 0 }, new[] { _numClasses * _featureDim }).Reshape(new[] { _numClasses, _featureDim });
        var bias = Engine.TensorSlice(flat, new[] { _numClasses * _featureDim }, new[] { _numClasses });
        return (weights, bias);
    }

    private Tensor<T> HeadLogits(Tensor<T> latent, (Tensor<T> Weights, Tensor<T> Bias) head)
    {
        int batch = latent.Shape[0];
        return Engine.TensorAdd(
            Engine.TensorMatMul(latent, Engine.TensorTranspose(head.Weights)),
            Engine.TensorBroadcastTo(head.Bias.Reshape(new[] { 1, _numClasses }), new[] { batch, _numClasses }));
    }

    private Tensor<T> OneHot(int[] labels)
    {
        var values = new int[labels.Length * _numClasses];
        for (int i = 0; i < labels.Length; i++)
        {
            values[i * _numClasses + labels[i]] = 1;
        }

        return new Tensor<int>(values, new[] { labels.Length, _numClasses }).Cast<T>();
    }

    /// <summary>[B, C] matrix whose row i repeats the per-class weight of sample i's label.</summary>
    private Tensor<T> Column(int[] labels, double[] perClass)
    {
        var values = new double[labels.Length * _numClasses];
        for (int i = 0; i < labels.Length; i++)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                values[i * _numClasses + c] = perClass[labels[i]];
            }
        }

        return new Tensor<double>(values, new[] { labels.Length, _numClasses }).Cast<T>();
    }

    private int[] SampleLabels(double[] distribution)
    {
        var labels = new int[_batchSize];
        for (int i = 0; i < _batchSize; i++)
        {
            double u = _rng.NextDouble();
            double cumulative = 0;
            int label = _numClasses - 1;
            for (int c = 0; c < _numClasses; c++)
            {
                cumulative += distribution[c];
                if (u < cumulative)
                {
                    label = c;
                    break;
                }
            }

            labels[i] = label;
        }

        return labels;
    }

    private double[] SampleNoise(int count)
    {
        var noise = new double[count * _noiseDim];
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = Gaussian();
        }

        return noise;
    }

    private Tensor<T> NoiseTensor(double[] noise, int count)
    {
        return new Tensor<double>(noise, new[] { count, _noiseDim }).Cast<T>();
    }

    private static double UnitInterval(double value, string name)
    {
        if (double.IsNaN(value) || value < 0 || value >= 1)
        {
            throw new ArgumentOutOfRangeException(name, "Moment decay must be in [0, 1).");
        }

        return value;
    }

    private static double NonNegative(double value, string name)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value < 0)
        {
            throw new ArgumentOutOfRangeException(name, "Loss weights must be finite and non-negative.");
        }

        return value;
    }

    private double Gaussian()
    {
        double u1 = 1.0 - _rng.NextDouble();
        double u2 = _rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private Tensor<T> XavierUniform(int fanIn, int fanOut)
    {
        double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
        var values = new double[fanIn * fanOut];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (_rng.NextDouble() * 2 - 1) * limit;
        }

        return new Tensor<double>(values, new[] { fanIn, fanOut }).Cast<T>();
    }
}
