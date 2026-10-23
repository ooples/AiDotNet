using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A 2-D convolution with optional weight or spectral normalization, stride, padding and depthwise grouping —
/// PyTorch's <c>Conv2d(...)</c>, <c>weight_norm(Conv2d(...))</c> and <c>spectral_norm(Conv2d(...))</c> as GAN encoders and
/// discriminators use them.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The kernel V is initialized as PyTorch's default (U(−1/√fan_in, 1/√fan_in) with fan_in the input channels per group
/// times the kernel area), as is the bias. With weight normalization (Salimans and Kingma 2016) the kernel is
/// <c>W = g · V / ‖V‖</c> per output channel, g starting at ‖V‖. With spectral normalization (Miyato et al. 2018) the kernel is
/// <c>W = V / σ</c>, <c>σ = uᵀ V v</c> over V reshaped to <c>[out, in/groups · kh · kw]</c>; u and v (persistent buffers,
/// as in <c>torch.nn.utils.spectral_norm</c>) start from 15 power iterations, as PyTorch's current spectral-norm
/// parametrization starts, take one more on every training forward and stay fixed in evaluation. Groups are 1 or the channel count (a depthwise convolution). Input and output are
/// <c>[batch, channels, height, width]</c>.
/// </para>
/// <para><b>For Beginners:</b> An image-style convolution whose weights are kept at a controlled scale, which makes
/// adversarial training more stable.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4, 8, 8",
    TestConstructorArgs = "4, 4, 3, 3, 1, 1, 1, 1, 1, AiDotNet.NeuralNetworks.Layers.ConvolutionNormalization.Spectral")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class NormedConv2DLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _in;
    private readonly int _out;
    private readonly int _kernelHeight;
    private readonly int _kernelWidth;
    private readonly int _strideHeight;
    private readonly int _strideWidth;
    private readonly int _paddingHeight;
    private readonly int _paddingWidth;
    private readonly int _groups;
    private readonly ConvolutionNormalization _normalization;
    private readonly bool _useBias;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _kernel;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _gain;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T> _bias;
    [Buffer]
    private Tensor<T> _u;
    [Buffer]
    private Tensor<T> _v;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates the layer.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernelHeight">Kernel height.</param>
    /// <param name="kernelWidth">Kernel width.</param>
    /// <param name="strideHeight">Vertical stride.</param>
    /// <param name="strideWidth">Horizontal stride.</param>
    /// <param name="paddingHeight">Zero padding above and below.</param>
    /// <param name="paddingWidth">Zero padding left and right.</param>
    /// <param name="groups">1, or the channel count for a depthwise convolution.</param>
    /// <param name="normalization">No, weight or spectral normalization of the kernel.</param>
    /// <param name="useBias">Whether the layer has a bias (true).</param>
    public NormedConv2DLayer([LayerState] int inChannels, [LayerState] int outChannels, [LayerState] int kernelHeight,
        [LayerState] int kernelWidth, [LayerState] int strideHeight, [LayerState] int strideWidth, [LayerState] int paddingHeight,
        [LayerState] int paddingWidth, [LayerState] int groups, [LayerState] ConvolutionNormalization normalization, [LayerState] bool useBias = true)
        : base(new[] { inChannels }, new[] { outChannels })
    {
        if (inChannels <= 0 || outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inChannels));
        if (kernelHeight <= 0 || kernelWidth <= 0 || strideHeight <= 0 || strideWidth <= 0 || paddingHeight < 0 || paddingWidth < 0)
            throw new ArgumentOutOfRangeException(nameof(kernelHeight));
        if (groups != 1 && (groups != inChannels || groups != outChannels))
            throw new ArgumentException("Groups are 1 or, for a depthwise convolution, the channel count.", nameof(groups));
        _in = inChannels;
        _out = outChannels;
        _kernelHeight = kernelHeight;
        _kernelWidth = kernelWidth;
        _strideHeight = strideHeight;
        _strideWidth = strideWidth;
        _paddingHeight = paddingHeight;
        _paddingWidth = paddingWidth;
        _groups = groups;
        _normalization = normalization;
        _useBias = useBias;

        var random = RandomSeed.HasValue
            ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(RandomSeed.Value)
            : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        int perGroup = inChannels / groups, fanIn = perGroup * kernelHeight * kernelWidth;
        double bound = 1.0 / Math.Sqrt(fanIn);
        _kernel = new Tensor<T>(new[] { outChannels, perGroup, kernelHeight, kernelWidth });
        for (int i = 0; i < _kernel.Length; i++) _kernel[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        _bias = new Tensor<T>(new[] { outChannels });
        if (useBias)
            for (int i = 0; i < outChannels; i++) _bias[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
        _u = NormalizedGaussian(outChannels, random);
        _v = NormalizedGaussian(fanIn, random);
        // torch.nn.utils.parametrizations.spectral_norm runs 15 power iterations when it is registered, so a fresh
        // layer's σ estimate is already close to the largest singular value; the random u and v alone can make
        // uᵀ V v nearly zero and an untrained network's output enormous.
        if (normalization == ConvolutionNormalization.Spectral) PowerIterations(15);
        _gain = new Tensor<T>(new[] { outChannels });
        for (int r = 0; r < outChannels; r++) _gain[r] = NumOps.FromDouble(RowNorm(r));

        RegisterTrainableParameter(_kernel, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gain, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    private Tensor<T> NormalizedGaussian(int length, Random random)
    {
        var result = new Tensor<T>(new[] { length });
        var values = new double[length];
        double norm = 0;
        for (int i = 0; i < length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            values[i] = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            norm += values[i] * values[i];
        }
        for (int i = 0; i < length; i++) result[i] = NumOps.FromDouble(values[i] / Math.Max(Math.Sqrt(norm), 1e-12));
        return result;
    }

    private double RowNorm(int row)
    {
        int per = _kernel.Length / _out;
        double sum = 0;
        for (int i = 0; i < per; i++)
        {
            double v = NumOps.ToDouble(_kernel[row * per + i]);
            sum += v * v;
        }
        return Math.Sqrt(sum);
    }

    // Power iterations on the current kernel, updating u and v in place.
    private void PowerIterations(int count)
    {
        int rows = _out, per = _kernel.Length / rows;
        double[] u = new double[rows], v = new double[per], w = new double[_kernel.Length];
        for (int i = 0; i < rows; i++) u[i] = NumOps.ToDouble(_u[i]);
        for (int i = 0; i < w.Length; i++) w[i] = NumOps.ToDouble(_kernel[i]);
        void Normalize(double[] a)
        {
            double n = Math.Sqrt(a.Sum(x => x * x));
            for (int i = 0; i < a.Length; i++) a[i] /= Math.Max(n, 1e-12);
        }
        for (int step = 0; step < count; step++)
        {
            for (int c = 0; c < per; c++)
            {
                double sum = 0;
                for (int r = 0; r < rows; r++) sum += w[r * per + c] * u[r];
                v[c] = sum;
            }
            Normalize(v);
            for (int r = 0; r < rows; r++)
            {
                double sum = 0;
                for (int c = 0; c < per; c++) sum += w[r * per + c] * v[c];
                u[r] = sum;
            }
            Normalize(u);
        }
        for (int i = 0; i < rows; i++) _u[i] = NumOps.FromDouble(u[i]);
        for (int i = 0; i < per; i++) _v[i] = NumOps.FromDouble(v[i]);
    }

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The kernel's shape in PyTorch's layout, <c>[out, in / groups, kh, kw]</c>.</summary>
    internal int[] TorchWeightShape => _kernel.Shape.ToArray();

    /// <summary>
    /// Loads a PyTorch convolution: the weight in <see cref="TorchWeightShape"/> order (<c>weight_orig</c> for a spectrally
    /// normalized one, <c>weight_v</c> / <c>parametrizations.weight.original1</c> for a weight-normalized one), the bias;
    /// for weight normalization the gain <c>weight_g</c> / <c>original0</c> (one per output channel; without it the gain
    /// is ‖V‖, so W = V), and for spectral normalization the stored singular-vector estimates <c>weight_u</c> <c>[out]</c>
    /// and <c>weight_v</c> <c>[in/groups · kh · kw]</c>.
    /// </summary>
    internal void LoadTorchWeights(double[] weight, double[]? bias, double[]? u = null, double[]? v = null, double[]? gain = null)
    {
        if (weight.Length != _kernel.Length)
            throw new ArgumentException($"Expected {_kernel.Length} kernel values, got {weight.Length}.", nameof(weight));
        if (_normalization == ConvolutionNormalization.Spectral && (u is null || v is null))
            throw new ArgumentException("A spectrally normalized convolution loads its weight_u and weight_v too.");
        if (gain is not null && _normalization != ConvolutionNormalization.Weight)
            throw new InvalidOperationException("A weight-norm gain loads only into a weight-normalized convolution.");
        if (gain is not null && gain.Length != _out)
            throw new ArgumentException($"Expected {_out} gain values, got {gain.Length}.", nameof(gain));
        for (int i = 0; i < weight.Length; i++) _kernel[i] = NumOps.FromDouble(weight[i]);
        if (bias is not null)
        {
            if (!_useBias) throw new InvalidOperationException("The checkpoint has a bias but this convolution has none.");
            if (bias.Length != _bias.Length) throw new ArgumentException($"Expected {_bias.Length} bias values.", nameof(bias));
            for (int i = 0; i < bias.Length; i++) _bias[i] = NumOps.FromDouble(bias[i]);
        }
        if (u is not null && v is not null)
        {
            if (u.Length != _u.Length || v.Length != _v.Length) throw new ArgumentException("The singular-vector estimates do not match the kernel.");
            for (int i = 0; i < u.Length; i++) _u[i] = NumOps.FromDouble(u[i]);
            for (int i = 0; i < v.Length; i++) _v[i] = NumOps.FromDouble(v[i]);
            Engine.InvalidatePersistentTensor(_u);
            Engine.InvalidatePersistentTensor(_v);
        }
        for (int r = 0; r < _out; r++) _gain[r] = NumOps.FromDouble(gain is not null ? gain[r] : RowNorm(r));
        Engine.InvalidatePersistentTensor(_kernel);
        Engine.InvalidatePersistentTensor(_gain);
        Engine.InvalidatePersistentTensor(_bias);
    }

    /// <summary>The effective kernel W.</summary>
    internal Tensor<T> Kernel()
    {
        if (_normalization == ConvolutionNormalization.None) return _kernel;
        int rows = _out, per = _kernel.Length / rows;
        var w = Engine.Reshape(_kernel, new[] { rows, per });
        if (_normalization == ConvolutionNormalization.Weight)
        {
            var norms = Engine.TensorPow(Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiply(w, w), new[] { 1 }, keepDims: true),
                NumOps.FromDouble(1e-24)), NumOps.FromDouble(-0.5));                                          // [rows, 1]
            var scale = Engine.TensorTile(Engine.TensorMultiply(Engine.Reshape(_gain, new[] { rows, 1 }), norms), new[] { 1, per });
            return Engine.Reshape(Engine.TensorMultiply(w, scale), _kernel._shape);
        }
        double[] uVec = new double[rows], vVec = new double[per];
        for (int r = 0; r < rows; r++) uVec[r] = NumOps.ToDouble(_u[r]);
        for (int c = 0; c < per; c++) vVec[c] = NumOps.ToDouble(_v[c]);
        if (IsTrainingMode)
        {
            // One power iteration (torch.nn.utils.spectral_norm): v ← Wᵀu / ‖Wᵀu‖, u ← W v / ‖W v‖, both kept.
            var host = _kernel.ToVector();
            void Normalize(double[] a)
            {
                double n = Math.Sqrt(a.Sum(x => x * x));
                for (int i = 0; i < a.Length; i++) a[i] /= Math.Max(n, 1e-12);
            }
            for (int c = 0; c < per; c++)
            {
                double s = 0;
                for (int r = 0; r < rows; r++) s += NumOps.ToDouble(host[r * per + c]) * uVec[r];
                vVec[c] = s;
            }
            Normalize(vVec);
            for (int r = 0; r < rows; r++)
            {
                double s = 0;
                for (int c = 0; c < per; c++) s += NumOps.ToDouble(host[r * per + c]) * vVec[c];
                uVec[r] = s;
            }
            Normalize(uVec);
            for (int r = 0; r < rows; r++) _u[r] = NumOps.FromDouble(uVec[r]);
            for (int c = 0; c < per; c++) _v[c] = NumOps.FromDouble(vVec[c]);
            Engine.InvalidatePersistentTensor(_u);
            Engine.InvalidatePersistentTensor(_v);
        }
        var uT = new Tensor<T>(new[] { 1, rows });
        var vT = new Tensor<T>(new[] { per, 1 });
        for (int r = 0; r < rows; r++) uT[0, r] = NumOps.FromDouble(uVec[r]);
        for (int c = 0; c < per; c++) vT[c, 0] = NumOps.FromDouble(vVec[c]);
        var sigma = Engine.TensorMatMul(Engine.TensorMatMul(uT, w), vT);                                     // [1, 1]
        var inverse = Engine.TensorTile(Engine.TensorPow(sigma, NumOps.FromDouble(-1)), new[] { rows, per });
        return Engine.Reshape(Engine.TensorMultiply(w, inverse), _kernel._shape);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 4 || input.Shape[1] != _in)
            throw new ArgumentException($"Expected [batch, {_in}, height, width], got [{string.Join(", ", input.Shape)}].", nameof(input));
        var kernel = Kernel();
        var stride = new[] { _strideHeight, _strideWidth };
        var padding = new[] { _paddingHeight, _paddingWidth };
        var y = _groups == 1
            ? Engine.Conv2D(input, kernel, stride, padding, new[] { 1, 1 })
            : Engine.DepthwiseConv2D(input, kernel, stride, padding);
        if (!_useBias) return y;
        int batch = y.Shape[0], height = y.Shape[2], width = y.Shape[3];
        return Engine.TensorAdd(y, Engine.TensorTile(Engine.Reshape(_bias, new[] { 1, _out, 1, 1 }), new[] { batch, 1, height, width }));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <summary>Persists the constructor arguments.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _in.ToString(inv);
        metadata["OutChannels"] = _out.ToString(inv);
        metadata["KernelHeight"] = _kernelHeight.ToString(inv);
        metadata["KernelWidth"] = _kernelWidth.ToString(inv);
        metadata["StrideHeight"] = _strideHeight.ToString(inv);
        metadata["StrideWidth"] = _strideWidth.ToString(inv);
        metadata["PaddingHeight"] = _paddingHeight.ToString(inv);
        metadata["PaddingWidth"] = _paddingWidth.ToString(inv);
        metadata["Groups"] = _groups.ToString(inv);
        metadata["Normalization"] = _normalization.ToString();
        metadata["UseBias"] = _useBias.ToString(inv);
        return metadata;
    }
}
