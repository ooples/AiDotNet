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
/// <b>Knowledge exchange:</b> <see cref="ExtractKnowledge(Vector{T}, Matrix{T})"/> returns the head as a
/// [C, L + 2] matrix (W | b | per-class sample count). <see cref="AggregateKnowledge"/> trains the generator
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
    /// <param name="generatorLearningRate">Generator learning rate. Default: 0.01.</param>
    /// <param name="seed">Random seed for generator initialization and sampling. Default: 42.</param>
    /// <param name="noiseDim">Dimension of the generator noise ε. Default: 16.</param>
    /// <param name="generatorHiddenDim">Hidden width of the generator MLP. Default: 32.</param>
    /// <param name="batchSize">Number of (label, noise) samples per generator or distillation step. Default: 32.</param>
    /// <param name="distillationEpochs">Client distillation steps per round. Default: 5.</param>
    /// <param name="distillationLearningRate">Client distillation learning rate. Default: 0.01.</param>
    public FedGENDistillation(int numClasses = 10, int featureDim = 64,
        int generatorEpochs = 5, double generatorLearningRate = 0.01, int seed = 42,
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
    }

    /// <summary>Number of client parameters that form the predictor head (C·L weights plus C biases).</summary>
    public int HeadParameterCount => _numClasses * _featureDim + _numClasses;

    /// <summary>Number of generator parameters.</summary>
    public int GeneratorParameterCount =>
        (_numClasses + _noiseDim) * _hiddenDim + _hiddenDim + _hiddenDim * _featureDim + _featureDim;

    /// <inheritdoc/>
    /// <remarks>Every class counts as one local sample; use the overload with label counts for the paper's α weights.</remarks>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        var uniform = new int[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            uniform[c] = 1;
        }

        return ExtractKnowledge(localModelParameters, uniform);
    }

    /// <summary>
    /// Extracts the client's predictor head together with its per-class sample counts.
    /// </summary>
    /// <param name="localModelParameters">Client parameters; the first <see cref="HeadParameterCount"/> are the head.</param>
    /// <param name="labelCounts">Number of local training samples of each class (length C, non-negative).</param>
    /// <returns>A [C, L + 2] matrix: predictor weights, bias, sample count.</returns>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, int[] labelCounts)
    {
        Guard.NotNull(localModelParameters);
        Guard.NotNull(labelCounts);
        if (labelCounts.Length != _numClasses || labelCounts.Any(c => c < 0))
        {
            throw new ArgumentException($"Label counts must hold {_numClasses} non-negative entries.", nameof(labelCounts));
        }

        var (weights, bias) = ReadHead(localModelParameters);
        var counts = new Tensor<int>(labelCounts.ToArray(), new[] { _numClasses, 1 }).Cast<T>();
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
            if (knowledge.Rows != _numClasses || knowledge.Columns != _featureDim + 2)
            {
                throw new ArgumentException(
                    $"Client {clientId} knowledge is {knowledge.Rows}x{knowledge.Columns}; expected {_numClasses}x{_featureDim + 2}.",
                    nameof(clientKnowledge));
            }

            var tensor = Tensor<T>.FromMatrix(knowledge);
            heads[index] = (
                Engine.TensorSlice(tensor, new[] { 0, 0 }, new[] { _numClasses, _featureDim }),
                Engine.TensorSlice(tensor, new[] { 0, _featureDim }, new[] { _numClasses, 1 }).Reshape(new[] { _numClasses }));
            var counts = Engine.TensorSlice(tensor, new[] { 0, _featureDim + 1 }, new[] { _numClasses, 1 }).Cast<double>().ToArray();
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            for (int c = 0; c < _numClasses; c++)
            {
                weightedCounts[index, c] = Math.Max(0, w * counts[c]);
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

        var learningRate = NumOps.FromDouble(_generatorLearningRate);
        for (int epoch = 0; epoch < _generatorEpochs; epoch++)
        {
            var labels = SampleLabels(labelDistribution);
            var (input, hidden, latent) = Generate(labels, SampleNoise());
            var targets = OneHot(labels);

            // Ensemble logits s = Σ_k α_k^y (Z W_kᵀ + b_k); CE gradient dS = (softmax(s) - Y) / B.
            var alphaColumns = new Tensor<T>[k];
            var ensemble = new Tensor<T>(new[] { _batchSize, _numClasses });
            for (int i = 0; i < k; i++)
            {
                alphaColumns[i] = Column(labels, alpha[i]);
                ensemble = Engine.TensorAdd(ensemble, Engine.TensorMultiply(HeadLogits(latent, heads[i]), alphaColumns[i]));
            }

            var gradLogits = Engine.TensorMultiplyScalar(
                Engine.TensorSubtract(Engine.Softmax(ensemble, 1), targets), NumOps.FromDouble(1.0 / _batchSize));

            // dZ = Σ_k (α_k ⊙ dS) W_k.
            var gradLatent = new Tensor<T>(new[] { _batchSize, _featureDim });
            for (int i = 0; i < k; i++)
            {
                gradLatent = Engine.TensorAdd(gradLatent,
                    Engine.TensorMatMul(Engine.TensorMultiply(gradLogits, alphaColumns[i]), heads[i].Weights));
            }

            BackpropGenerator(input, hidden, gradLatent, learningRate);
        }

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
            var latent = Forward(generator, labels, SampleNoise()).Latent;
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

        var noise = new double[count * _noiseDim];
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = Gaussian();
        }

        var generator = (_w1, _b1, _w2, _b2);
        return Forward(generator, labels, new Tensor<double>(noise, new[] { count, _noiseDim }).Cast<T>()).Latent.ToMatrix();
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

    private void BackpropGenerator(Tensor<T> input, Tensor<T> hidden, Tensor<T> gradLatent, T learningRate)
    {
        var gradW2 = Engine.TensorMatMul(Engine.TensorTranspose(hidden), gradLatent);
        var gradB2 = Engine.ReduceSum(gradLatent, new[] { 0 }, false);
        var gradHidden = Engine.TensorMultiply(
            Engine.TensorMatMul(gradLatent, Engine.TensorTranspose(_w2)),
            Engine.TensorGreaterThan(hidden, NumOps.Zero));
        var gradW1 = Engine.TensorMatMul(Engine.TensorTranspose(input), gradHidden);
        var gradB1 = Engine.ReduceSum(gradHidden, new[] { 0 }, false);

        _w2 = Engine.TensorSubtract(_w2, Engine.TensorMultiplyScalar(gradW2, learningRate));
        _b2 = Engine.TensorSubtract(_b2, Engine.TensorMultiplyScalar(gradB2, learningRate));
        _w1 = Engine.TensorSubtract(_w1, Engine.TensorMultiplyScalar(gradW1, learningRate));
        _b1 = Engine.TensorSubtract(_b1, Engine.TensorMultiplyScalar(gradB1, learningRate));
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

    private Tensor<T> SampleNoise()
    {
        var noise = new double[_batchSize * _noiseDim];
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = Gaussian();
        }

        return new Tensor<double>(noise, new[] { _batchSize, _noiseDim }).Cast<T>();
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
