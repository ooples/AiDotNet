namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// The output of an SDF network query: the signed distance, the geometry feature vector, and (when requested) the
/// SDF's spatial gradient.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class NeuSSdfOutput<T>
{
    /// <summary>Creates an output.</summary>
    public NeuSSdfOutput(Tensor<T> sdf, Tensor<T> features, Tensor<T>? gradient)
    {
        Sdf = sdf;
        Features = features;
        Gradient = gradient;
    }

    /// <summary>The signed distances, <c>[N, 1]</c>.</summary>
    public Tensor<T> Sdf { get; }

    /// <summary>The geometry features, <c>[N, F]</c>.</summary>
    public Tensor<T> Features { get; }

    /// <summary>∇ₓ SDF, <c>[N, 3]</c>, differentiable with respect to the network's parameters; null when not requested.</summary>
    public Tensor<T>? Gradient { get; }
}

/// <summary>
/// The SDF multilayer perceptron of NeuS (Wang et al. 2021, reference <c>models/fields.py</c> <c>SDFNetwork</c>) and of
/// SparseNeuS (Long et al. 2022, <c>LatentSDFLayer</c>): a frequency-embedded point, optionally with a latent feature,
/// through softplus(β = 100) layers with an input skip connection, giving the signed distance and a feature vector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>Spatial gradient.</b> The reference takes ∇ₓf by <c>torch.autograd.grad(..., create_graph=True)</c> so the
/// Eikonal loss and the normals in the color network differentiate into the weights. The engine's tape has no
/// second-order mode, so this class writes the gradient's reverse pass out in engine operations
/// (<see cref="Forward"/> with <c>withGradient</c>): with <c>zₗ</c> the pre-activations, the row of the last layer's
/// weight for the SDF output is propagated back through <c>Wₗᵀ</c> and <c>softplus'(z) = sigmoid(βz)</c>, split at the
/// skip and latent concatenations, and through the embedding's Jacobian (<see cref="NeuSFrequencyEmbedder{T}.Backward"/>).
/// Every step is a tape-recorded op, so the tape's ordinary backward then gives exactly the second-order terms
/// autograd's double backward gives. A latent input that depends on the point (SparseNeuS's trilinearly sampled
/// geometry volume) contributes through its spatial Jacobian, which the caller supplies.
/// </para>
/// <para>
/// <b>Initialization.</b> The geometric initialization of Atzmon &amp; Lipman (SAL), as both references apply it: the
/// network starts as an approximate SDF of a sphere of radius <c>bias</c> (0.5), with the embedding's frequency
/// channels and the latent channels zeroed so the initial shape is smooth.
/// </para>
/// </remarks>
internal sealed class NeuSSdfMlp<T>
{
    private static readonly double InvSqrt2 = 1.0 / Math.Sqrt(2.0);

    private readonly NeuSFrequencyEmbedder<T> _embedder;
    private readonly NeuSLinear<T>[] _layers;
    private readonly bool[] _skip;
    private readonly bool[] _latent;
    private readonly string _prefix;

    private NeuSSdfMlp(NeuSFrequencyEmbedder<T> embedder, NeuSLinear<T>[] layers, bool[] skip, bool[] latent,
        int latentDims, double scale, double beta, string prefix)
    {
        _embedder = embedder;
        _layers = layers;
        _skip = skip;
        _latent = latent;
        LatentDims = latentDims;
        Scale = scale;
        Beta = beta;
        _prefix = prefix;
    }

    /// <summary>The latent feature width (0 for plain NeuS).</summary>
    public int LatentDims { get; }

    /// <summary>The input scale (NeuS's <c>scale</c>; the SDF is divided by it).</summary>
    public double Scale { get; }

    /// <summary>The softplus sharpness β.</summary>
    public double Beta { get; }

    /// <summary>The output width: the SDF plus <see cref="FeatureDims"/> features.</summary>
    public int OutputDims => _layers[_layers.Length - 1].OutputDims;

    /// <summary>The geometry feature width.</summary>
    public int FeatureDims => OutputDims - 1;

    /// <summary>The embedded input width.</summary>
    public int EmbeddingDims => _embedder.OutputDims;

    /// <summary>The linear layers, first to last.</summary>
    public IReadOnlyList<NeuSLinear<T>> Layers => _layers;

    /// <summary>The trainable tensors.</summary>
    public IEnumerable<Tensor<T>> Parameters => _layers.SelectMany(l => l.Parameters);

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => _layers.Sum(l => l.ParameterCount);

    /// <summary>The parameters under the reference's state-dict names.</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters()
    {
        for (int l = 0; l < _layers.Length; l++)
            foreach (var p in _layers[l].NamedParameters(_prefix + "lin" + l)) yield return p;
    }

    /// <summary>Loads one named parameter; returns whether the name belongs to this network.</summary>
    public bool TryLoad(string name, Tensor<T> value)
    {
        for (int l = 0; l < _layers.Length; l++)
            if (_layers[l].TryLoad(_prefix + "lin" + l, name, value)) return true;
        return false;
    }

    /// <summary>
    /// NeuS's <c>SDFNetwork</c>: <c>dims = [γ(3)] + [hidden]·layers + [1 + features]</c>, with layer <c>l + 1 ∈ skip</c>
    /// narrowed by the embedding width so the skip concatenation restores <c>hidden</c>.
    /// </summary>
    /// <param name="hidden">The hidden width (256).</param>
    /// <param name="hiddenLayers">The hidden layer count (8).</param>
    /// <param name="outputDims">The output width (257: the SDF and 256 features).</param>
    /// <param name="skipIn">The layers that take the embedded input again (4).</param>
    /// <param name="multires">The embedding frequencies (6).</param>
    /// <param name="bias">The geometric-initialization sphere radius (0.5).</param>
    /// <param name="scale">The input scale (1).</param>
    /// <param name="geometricInit">Whether to use the geometric initialization.</param>
    /// <param name="weightNorm">Whether to weight-normalize.</param>
    /// <param name="insideOutside">Whether the camera is inside the surface (flips the initial sphere).</param>
    /// <param name="beta">The softplus β (100).</param>
    /// <param name="random">The initialization generator.</param>
    public static NeuSSdfMlp<T> CreateNeuS(int hidden, int hiddenLayers, int outputDims, int[] skipIn, int multires,
        double bias, double scale, bool geometricInit, bool weightNorm, bool insideOutside, double beta, Random random)
    {
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden), "The hidden width must be positive.");
        if (hiddenLayers <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenLayers), "At least one hidden layer is needed.");
        if (outputDims < 1) throw new ArgumentOutOfRangeException(nameof(outputDims), "The output needs the SDF channel.");
        var embedder = new NeuSFrequencyEmbedder<T>(3, multires);
        var dims = new int[hiddenLayers + 2];
        dims[0] = embedder.OutputDims;
        for (int i = 1; i <= hiddenLayers; i++) dims[i] = hidden;
        dims[hiddenLayers + 1] = outputDims;
        int numLinear = dims.Length - 1;
        var skip = new bool[numLinear];
        foreach (int s in skipIn ?? Array.Empty<int>())
            if (s > 0 && s < numLinear) skip[s] = true;
        foreach (int s in skipIn ?? Array.Empty<int>())
            if (s > 0 && s < numLinear && dims[s] <= dims[0])
                throw new ArgumentException($"Skip layer {s} needs a hidden width larger than the embedding ({dims[0]}).", nameof(skipIn));

        var layers = new NeuSLinear<T>[numLinear];
        for (int l = 0; l < numLinear; l++)
        {
            int inDim = dims[l];
            int outDim = l + 1 < numLinear && skip[l + 1] ? dims[l + 1] - dims[0] : dims[l + 1];
            double[] weight, layerBias;
            if (!geometricInit)
            {
                (weight, layerBias) = NeuSLinear<T>.DefaultInit(inDim, outDim, random);
            }
            else
            {
                weight = new double[outDim * inDim];
                layerBias = new double[outDim];
                if (l == numLinear - 1)
                {
                    double mean = (insideOutside ? -1.0 : 1.0) * Math.Sqrt(Math.PI) / Math.Sqrt(inDim);
                    for (int i = 0; i < weight.Length; i++) weight[i] = mean + 1e-4 * NeuSHost<T>.NextGaussian(random);
                    for (int o = 0; o < outDim; o++) layerBias[o] = insideOutside ? bias : -bias;
                }
                else
                {
                    double std = Math.Sqrt(2.0) / Math.Sqrt(outDim);
                    for (int o = 0; o < outDim; o++)
                        for (int i = 0; i < inDim; i++)
                        {
                            bool zero = multires > 0 && ((l == 0 && i >= 3) || (l != 0 && skip[l] && i >= inDim - (dims[0] - 3)));
                            weight[o * inDim + i] = zero ? 0.0 : std * NeuSHost<T>.NextGaussian(random);
                        }
                }
            }

            layers[l] = new NeuSLinear<T>(weight, layerBias, inDim, outDim, weightNorm);
        }

        return new NeuSSdfMlp<T>(embedder, layers, skip, new bool[numLinear], 0, scale, beta, string.Empty);
    }

    /// <summary>
    /// SparseNeuS's <c>LatentSDFLayer</c>: <c>layers − 1</c> linear layers; every layer but the first takes the latent
    /// feature concatenated to its input (<c>dims_in = [γ(3)] + [hidden + latent]·(layers − 2)</c>), and every layer
    /// outputs <c>hidden</c> channels (the reference's <c>dims_out[layers − 2]</c> is <c>hidden</c>, so the SDF network
    /// yields the SDF and <c>hidden − 1</c> features).
    /// </summary>
    /// <param name="hidden">The hidden width (128).</param>
    /// <param name="numLayers">The reference's <c>n_layers</c> (4: three linear layers).</param>
    /// <param name="latentDims">The latent width (16).</param>
    /// <param name="skipIn">The layers that take the embedded input again (the reference passes 4, which three layers never reach).</param>
    /// <param name="multires">The embedding frequencies (6).</param>
    /// <param name="bias">The geometric-initialization sphere radius (0.5).</param>
    /// <param name="geometricInit">Whether to use the geometric initialization.</param>
    /// <param name="weightNorm">Whether to weight-normalize.</param>
    /// <param name="beta">The softplus β (100).</param>
    /// <param name="random">The initialization generator.</param>
    /// <param name="prefix">The state-dict prefix (<c>"sdf_layer."</c>).</param>
    public static NeuSSdfMlp<T> CreateLatent(int hidden, int numLayers, int latentDims, int[] skipIn, int multires,
        double bias, bool geometricInit, bool weightNorm, double beta, Random random, string prefix)
    {
        if (hidden <= 1) throw new ArgumentOutOfRangeException(nameof(hidden), "The hidden width must exceed 1.");
        if (numLayers < 3) throw new ArgumentOutOfRangeException(nameof(numLayers), "The latent SDF layer needs n_layers >= 3.");
        if (latentDims <= 0) throw new ArgumentOutOfRangeException(nameof(latentDims), "The latent width must be positive.");
        var embedder = new NeuSFrequencyEmbedder<T>(3, multires);
        int numLinear = numLayers - 1;
        int embedding = embedder.OutputDims;
        var skip = new bool[numLinear];
        foreach (int s in skipIn ?? Array.Empty<int>())
            if (s >= 0 && s < numLinear) skip[s] = true;
        var latent = new bool[numLinear];
        for (int l = 1; l < numLinear; l++) latent[l] = true;

        var layers = new NeuSLinear<T>[numLinear];
        for (int l = 0; l < numLinear; l++)
        {
            int dimsIn = l == 0 ? embedding : hidden + latentDims;
            int inDim = skip[l] ? dimsIn + embedding : dimsIn;
            int outDim = hidden;
            double[] weight, layerBias;
            if (!geometricInit)
            {
                (weight, layerBias) = NeuSLinear<T>.DefaultInit(inDim, outDim, random);
            }
            else
            {
                weight = new double[outDim * inDim];
                layerBias = new double[outDim];
                if (l == numLinear - 1)
                {
                    double mean = Math.Sqrt(Math.PI) / Math.Sqrt(inDim);
                    for (int o = 0; o < outDim; o++)
                    {
                        // The reference zeroes the latent columns and the LAST latentDims bias entries.
                        layerBias[o] = o >= outDim - latentDims ? 0.0 : -bias;
                        for (int i = 0; i < inDim; i++)
                            weight[o * inDim + i] = i >= inDim - latentDims ? 0.0 : mean + 1e-4 * NeuSHost<T>.NextGaussian(random);
                    }
                }
                else
                {
                    double std = Math.Sqrt(2.0) / Math.Sqrt(outDim);
                    int zeroTail = multires > 0 && l == 0 ? inDim - 3
                        : multires > 0 && skip[l] ? embedding - 3 + latentDims
                        : latentDims;
                    for (int o = 0; o < outDim; o++)
                        for (int i = 0; i < inDim; i++)
                        {
                            bool zero = multires > 0 && l == 0 ? i >= 3 : i >= inDim - zeroTail;
                            weight[o * inDim + i] = zero ? 0.0 : std * NeuSHost<T>.NextGaussian(random);
                        }
                }
            }

            layers[l] = new NeuSLinear<T>(weight, layerBias, inDim, outDim, weightNorm);
        }

        return new NeuSSdfMlp<T>(embedder, layers, skip, latent, latentDims, 1.0, beta, prefix);
    }

    /// <summary>
    /// Queries the network at <c>[N, 3]</c> points.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="points">The points.</param>
    /// <param name="latent">The latent features, <c>[N, LatentDims]</c>, or null when <see cref="LatentDims"/> is 0.</param>
    /// <param name="latentJacobian">∂latent/∂xᵢ for i = x, y, z (three <c>[N, LatentDims]</c> tensors), or null when the
    /// latent does not depend on the point.</param>
    /// <param name="withGradient">Whether to compute ∇ₓ SDF.</param>
    public NeuSSdfOutput<T> Forward(IEngine engine, Tensor<T> points, Tensor<T>? latent, Tensor<T>[]? latentJacobian,
        bool withGradient)
    {
        if (points is null) throw new ArgumentNullException(nameof(points));
        if (points.Shape.Length != 2 || points.Shape[1] != 3)
            throw new ArgumentException("Points must be [N, 3].", nameof(points));
        if (LatentDims > 0 && latent is null) throw new ArgumentNullException(nameof(latent), "This network needs a latent input.");
        int n = points.Shape[0];
        var u = Scale == 1.0 ? points : engine.TensorMultiplyScalar(points, NeuSHost<T>.Scalar(Scale));
        var inputs = _embedder.Embed(engine, u);
        int embedding = _embedder.OutputDims;

        var weights = new Tensor<T>[_layers.Length];
        var preActivations = new Tensor<T>[_layers.Length];
        var x = inputs;
        for (int l = 0; l < _layers.Length; l++)
        {
            if (_skip[l]) x = engine.TensorMultiplyScalar(engine.TensorConcatenate(new[] { x, inputs }, 1), NeuSHost<T>.Scalar(InvSqrt2));
            if (_latent[l] && latent is not null) x = engine.TensorConcatenate(new[] { x, latent }, 1);
            weights[l] = _layers[l].EffectiveWeight(engine);
            var z = _layers[l].Forward(engine, x, weights[l]);
            if (l < _layers.Length - 1)
            {
                preActivations[l] = z;
                x = Softplus(engine, z);
            }
            else
            {
                x = z;
            }
        }

        var sdf = engine.TensorSlice(x, new[] { 0, 0 }, new[] { n, 1 });
        if (Scale != 1.0) sdf = engine.TensorMultiplyScalar(sdf, NeuSHost<T>.Scalar(1.0 / Scale));
        var features = engine.TensorSlice(x, new[] { 0, 1 }, new[] { n, OutputDims - 1 });
        if (!withGradient) return new NeuSSdfOutput<T>(sdf, features, null);

        // Reverse pass of ∂(output₀)/∂(input), written in engine ops so it stays differentiable.
        Tensor<T>? gradInputs = null;
        Tensor<T>? gradLatent = null;
        Tensor<T>? gradZ = null;
        for (int l = _layers.Length - 1; l >= 0; l--)
        {
            int inDim = _layers[l].InputDims;
            Tensor<T> gradIn = l == _layers.Length - 1
                ? engine.TensorBroadcastTo(engine.TensorSlice(weights[l], new[] { 0, 0 }, new[] { 1, inDim }), new[] { n, inDim })
                : engine.TensorMatMul(gradZ ?? throw new InvalidOperationException("Missing upstream gradient."), weights[l]);
            int width = inDim;
            if (_latent[l] && LatentDims > 0)
            {
                width -= LatentDims;
                gradLatent = Accumulate(engine, gradLatent, engine.TensorSlice(gradIn, new[] { 0, width }, new[] { n, LatentDims }));
                gradIn = engine.TensorSlice(gradIn, new[] { 0, 0 }, new[] { n, width });
            }

            if (_skip[l])
            {
                width -= embedding;
                var half = NeuSHost<T>.Scalar(InvSqrt2);
                gradInputs = Accumulate(engine, gradInputs,
                    engine.TensorMultiplyScalar(engine.TensorSlice(gradIn, new[] { 0, width }, new[] { n, embedding }), half));
                gradIn = engine.TensorMultiplyScalar(engine.TensorSlice(gradIn, new[] { 0, 0 }, new[] { n, width }), half);
            }

            if (l > 0)
            {
                var slope = engine.TensorSigmoid(engine.TensorMultiplyScalar(preActivations[l - 1], NeuSHost<T>.Scalar(Beta)));
                gradZ = engine.TensorMultiply(gradIn, slope);
            }
            else
            {
                gradInputs = Accumulate(engine, gradInputs, gradIn);
            }
        }

        // u = scale·x and sdf = out₀ / scale, so ∂sdf/∂x = ∂out₀/∂u through the embedding.
        var gradient = _embedder.Backward(engine, u,
            gradInputs ?? throw new InvalidOperationException("The reverse pass produced no input gradient."));
        if (gradLatent is not null && latentJacobian is not null)
        {
            if (latentJacobian.Length != 3) throw new ArgumentException("The latent Jacobian needs three components.", nameof(latentJacobian));
            var columns = new Tensor<T>[3];
            for (int d = 0; d < 3; d++)
                columns[d] = engine.ReduceSum(engine.TensorMultiply(gradLatent, latentJacobian[d]), new[] { 1 }, true);
            var viaLatent = engine.TensorConcatenate(columns, 1);
            if (Scale != 1.0) viaLatent = engine.TensorMultiplyScalar(viaLatent, NeuSHost<T>.Scalar(1.0 / Scale));
            gradient = engine.TensorAdd(gradient, viaLatent);
        }

        return new NeuSSdfOutput<T>(sdf, features, gradient);
    }

    /// <summary>
    /// <c>softplus_β(z) = log(1 + exp(βz)) / β</c>, in the overflow-free form <c>relu(z) + log1p(exp(−β|z|)) / β</c>.
    /// </summary>
    /// <param name="engine">The engine.</param>
    /// <param name="z">The input.</param>
    public Tensor<T> Softplus(IEngine engine, Tensor<T> z)
    {
        var tail = engine.TensorLog1p(engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorAbs(z), NeuSHost<T>.Scalar(-Beta))));
        return engine.TensorAdd(engine.TensorReLU(z), engine.TensorMultiplyScalar(tail, NeuSHost<T>.Scalar(1.0 / Beta)));
    }

    private static Tensor<T> Accumulate(IEngine engine, Tensor<T>? total, Tensor<T> term)
        => total is null ? term : engine.TensorAdd(total, term);
}
