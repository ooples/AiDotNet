namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// NeuS's color network (reference <c>models/fields.py</c> <c>RenderingNetwork</c>, the IDR rendering MLP): the color
/// at a point from the point, the embedded view direction, the SDF normal and the SDF network's feature vector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <c>dims = [d_in + d_feature] + [hidden]·layers + [3]</c>, ReLU between layers, a sigmoid on the output
/// (<c>squeeze_out</c>), weight normalization, PyTorch's default linear initialization. The paper's configuration is
/// mode <see cref="NeuSRenderingMode.Idr"/>, 4 hidden layers of 256 and 4 view-direction frequencies.
/// <para>
/// Mode <see cref="NeuSRenderingMode.VolumeRadiance"/> is instant-nsr-pl's <c>VolumeRadiance</c> on a
/// <c>VanillaMLP</c>: the input is <c>[feature, SH(view)?, normal]</c>, where the caller passes the SDF network's whole
/// output as the feature and the unit normal as the normal; <c>kaiming_uniform_(nonlinearity='relu')</c> weights, zero
/// biases and no weight normalization.
/// </para>
/// </remarks>
internal sealed class NeuSRenderingNetwork<T>
{
    private readonly NeuSFrequencyEmbedder<T> _viewEmbedder;
    private readonly NeuSLinear<T>[] _layers;
    private readonly int _shDegree;

    /// <summary>Creates the network.</summary>
    /// <param name="featureDims">The SDF feature width (256).</param>
    /// <param name="mode">Which inputs the color depends on.</param>
    /// <param name="hidden">The hidden width (256).</param>
    /// <param name="hiddenLayers">The hidden layer count (4).</param>
    /// <param name="viewFrequencies">The view-direction embedding frequencies (4).</param>
    /// <param name="weightNorm">Whether to weight-normalize.</param>
    /// <param name="squeezeOut">Whether to apply a sigmoid to the output.</param>
    /// <param name="random">The initialization generator.</param>
    /// <param name="shDegree">With <see cref="NeuSRenderingMode.VolumeRadiance"/>, the view direction's
    /// spherical-harmonics degree, or 0 to leave the view direction out (instant-nsr-pl's <c>with_viewdir = False</c>).</param>
    public NeuSRenderingNetwork(int featureDims, NeuSRenderingMode mode, int hidden, int hiddenLayers, int viewFrequencies,
        bool weightNorm, bool squeezeOut, Random random, int shDegree = 0)
    {
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden), "The hidden width must be positive.");
        if (hiddenLayers < 0) throw new ArgumentOutOfRangeException(nameof(hiddenLayers), "The hidden layer count cannot be negative.");
        if (shDegree < 0 || shDegree > NeuSSphericalHarmonics.MaxDegree) throw new ArgumentOutOfRangeException(nameof(shDegree));
        if (random is null) throw new ArgumentNullException(nameof(random));
        Mode = mode;
        SqueezeOut = squeezeOut;
        _shDegree = mode == NeuSRenderingMode.VolumeRadiance ? shDegree : 0;
        _viewEmbedder = new NeuSFrequencyEmbedder<T>(3, viewFrequencies);
        int input = mode switch
        {
            NeuSRenderingMode.Idr => 3 + _viewEmbedder.OutputDims + 3 + featureDims,
            NeuSRenderingMode.NoViewDirection => 3 + 3 + featureDims,
            NeuSRenderingMode.NoNormal => 3 + _viewEmbedder.OutputDims + featureDims,
            NeuSRenderingMode.VolumeRadiance => featureDims + _shDegree * _shDegree + 3,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };
        var dims = new int[hiddenLayers + 2];
        dims[0] = input;
        for (int i = 1; i <= hiddenLayers; i++) dims[i] = hidden;
        dims[hiddenLayers + 1] = 3;
        _layers = new NeuSLinear<T>[dims.Length - 1];
        for (int l = 0; l < _layers.Length; l++)
        {
            if (mode == NeuSRenderingMode.VolumeRadiance)
            {
                // VanillaMLP.make_linear without sphere_init: kaiming_uniform_(relu), bound sqrt(6 / fan_in); zero bias.
                double bound = Math.Sqrt(6.0 / dims[l]);
                var kaiming = new double[dims[l + 1] * dims[l]];
                for (int i = 0; i < kaiming.Length; i++) kaiming[i] = (random.NextDouble() * 2.0 - 1.0) * bound;
                _layers[l] = new NeuSLinear<T>(kaiming, new double[dims[l + 1]], dims[l], dims[l + 1], weightNorm: false);
                continue;
            }

            var (weight, bias) = NeuSLinear<T>.DefaultInit(dims[l], dims[l + 1], random);
            _layers[l] = new NeuSLinear<T>(weight, bias, dims[l], dims[l + 1], weightNorm);
        }
    }

    /// <summary>Which inputs the color depends on.</summary>
    public NeuSRenderingMode Mode { get; }

    /// <summary>Whether the output passes through a sigmoid.</summary>
    public bool SqueezeOut { get; }

    /// <summary>The linear layers.</summary>
    public IReadOnlyList<NeuSLinear<T>> Layers => _layers;

    /// <summary>The trainable tensors.</summary>
    public IEnumerable<Tensor<T>> Parameters => _layers.SelectMany(l => l.Parameters);

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => _layers.Sum(l => l.ParameterCount);

    /// <summary>The parameters under the reference's state-dict names (<c>lin{l}.*</c>).</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters()
    {
        for (int l = 0; l < _layers.Length; l++)
            foreach (var p in _layers[l].NamedParameters("lin" + l)) yield return p;
    }

    /// <summary>Loads one named parameter; returns whether the name belongs to this network.</summary>
    public bool TryLoad(string name, Tensor<T> value)
    {
        for (int l = 0; l < _layers.Length; l++)
            if (_layers[l].TryLoad("lin" + l, name, value)) return true;
        return false;
    }

    /// <summary>The color at each point, <c>[N, 3]</c>.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="points">The points, <c>[N, 3]</c>.</param>
    /// <param name="normals">The SDF gradients, <c>[N, 3]</c>.</param>
    /// <param name="viewDirections">The ray directions, <c>[N, 3]</c>.</param>
    /// <param name="features">The SDF features, <c>[N, F]</c>.</param>
    public Tensor<T> Forward(IEngine engine, Tensor<T> points, Tensor<T> normals, Tensor<T> viewDirections, Tensor<T> features)
    {
        if (Mode == NeuSRenderingMode.VolumeRadiance)
        {
            // torch.cat([features, dirs_embd?, normal]); the directions are constants, so their encoding is host-side.
            var inputs = new List<Tensor<T>> { features };
            if (_shDegree > 0)
            {
                int n = viewDirections.Shape[0];
                var basis = NeuSSphericalHarmonics.Evaluate(NeuSHost<T>.ToDoubles(viewDirections), _shDegree);
                inputs.Add(NeuSHost<T>.FromDoubles(basis, n, _shDegree * _shDegree));
            }

            inputs.Add(normals);
            return Mlp(engine, engine.TensorConcatenate(inputs.ToArray(), 1));
        }

        var views = _viewEmbedder.Embed(engine, viewDirections);
        var parts = Mode switch
        {
            NeuSRenderingMode.Idr => new[] { points, views, normals, features },
            NeuSRenderingMode.NoViewDirection => new[] { points, normals, features },
            _ => new[] { points, views, features },
        };
        return Mlp(engine, engine.TensorConcatenate(parts, 1));
    }

    private Tensor<T> Mlp(IEngine engine, Tensor<T> x)
    {
        for (int l = 0; l < _layers.Length; l++)
        {
            x = _layers[l].Forward(engine, x);
            if (l < _layers.Length - 1) x = engine.TensorReLU(x);
        }

        return SqueezeOut ? engine.TensorSigmoid(x) : x;
    }
}
