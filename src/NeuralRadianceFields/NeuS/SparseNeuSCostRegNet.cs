namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// SparseNeuS's cost-volume regularization network (reference <c>tsparse/modules.py</c> <c>SparseCostRegNet</c>): a
/// 3-D U-Net of convolution + batch norm + ReLU blocks with three stride-2 downsamplings, three transposed-convolution
/// upsamplings and additive skips, evaluated densely over the active voxels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The reference runs torchsparse convolutions on the voxels visible in at least two views. A sparse stride-1
/// convolution reads only active inputs and writes only active outputs; a stride-2 one writes a coarse voxel when any
/// voxel of its 2×2×2 block is active; a transposed one writes back to the finer level's active voxels. This class
/// runs the same kernels as dense convolutions with the inactive voxels held at zero after every block (and excluded
/// from the batch statistics), which gives those outputs on the active voxels. Volume dimensions must be divisible
/// by 8.
/// </para>
/// </remarks>
internal sealed class SparseNeuSCostRegNet<T>
{
    private readonly SparseNeuSConvBlock<T>[] _blocks;
    private static readonly string[] Names = { "conv0", "conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv9", "conv11" };

    /// <summary>Creates the network.</summary>
    /// <param name="inChannels">The cost volume's channels (32: variance and mean of 16 compressed features).</param>
    /// <param name="outChannels">The geometry encoding's channels (<c>regnet_d_out</c> = 16).</param>
    /// <param name="random">The initialization generator.</param>
    public SparseNeuSCostRegNet(int inChannels, int outChannels, Random random)
    {
        SparseNeuSConvBlock<T> Conv(int i, int o, int stride)
            => new(3, i, o, 3, stride, 1, false, true, SparseNeuSActivation.ReLU, random, "net.0", "net.1");
        SparseNeuSConvBlock<T> Deconv(int i, int o)
            => new(3, i, o, 3, 2, 1, true, true, SparseNeuSActivation.ReLU, random, "net.0", "net.1");
        InChannels = inChannels;
        OutChannels = outChannels;
        _blocks = new[]
        {
            Conv(inChannels, outChannels, 1), Conv(outChannels, 16, 2), Conv(16, 16, 1), Conv(16, 32, 2), Conv(32, 32, 1),
            Conv(32, 64, 2), Conv(64, 64, 1), Deconv(64, 32), Deconv(32, 16), Deconv(16, outChannels),
        };
    }

    /// <summary>The input channels.</summary>
    public int InChannels { get; }

    /// <summary>The output channels.</summary>
    public int OutChannels { get; }

    /// <summary>The trainable tensors.</summary>
    public IEnumerable<Tensor<T>> Parameters => _blocks.SelectMany(b => b.Parameters);

    /// <summary>The trainable parameter count.</summary>
    public int ParameterCount => _blocks.Sum(b => b.ParameterCount);

    /// <summary>The parameters under <paramref name="prefix"/>.</summary>
    public IEnumerable<KeyValuePair<string, Tensor<T>>> NamedParameters(string prefix)
    {
        for (int i = 0; i < _blocks.Length; i++)
            foreach (var p in _blocks[i].NamedParameters(prefix + Names[i])) yield return p;
    }

    /// <summary>Loads one named parameter; returns whether the name belongs to this network.</summary>
    public bool TryLoad(string prefix, string name, Tensor<T> value)
    {
        for (int i = 0; i < _blocks.Length; i++)
            if (_blocks[i].TryLoad(prefix + Names[i], name, value)) return true;
        return false;
    }

    /// <summary>Regularizes a <c>[1, C, X, Y, Z]</c> cost volume.</summary>
    /// <param name="engine">The engine.</param>
    /// <param name="volume">The cost volume (zero at inactive voxels).</param>
    /// <param name="active">The 0/1 active-voxel mask, <c>X·Y·Z</c> values x-slowest.</param>
    /// <param name="dims">X, Y, Z (each divisible by 8).</param>
    /// <param name="training">Whether batch normalization uses batch statistics.</param>
    public Tensor<T> Forward(IEngine engine, Tensor<T> volume, double[] active, int[] dims, bool training)
    {
        for (int d = 0; d < 3; d++)
            if (dims[d] % 8 != 0) throw new ArgumentException("Volume dimensions must be divisible by 8.", nameof(dims));
        var mask0 = active;
        var mask1 = Downsample(mask0, dims);
        var dims1 = new[] { dims[0] / 2, dims[1] / 2, dims[2] / 2 };
        var mask2 = Downsample(mask1, dims1);
        var dims2 = new[] { dims1[0] / 2, dims1[1] / 2, dims1[2] / 2 };
        var mask3 = Downsample(mask2, dims2);

        var b = _blocks;
        var conv0 = b[0].Forward(engine, volume, training, mask0);
        var conv2 = b[2].Forward(engine, b[1].Forward(engine, conv0, training, mask1), training, mask1);
        var conv4 = b[4].Forward(engine, b[3].Forward(engine, conv2, training, mask2), training, mask2);
        var x = b[6].Forward(engine, b[5].Forward(engine, conv4, training, mask3), training, mask3);
        x = engine.TensorAdd(conv4, b[7].Forward(engine, x, training, mask2));
        x = engine.TensorAdd(conv2, b[8].Forward(engine, x, training, mask1));
        return engine.TensorAdd(conv0, b[9].Forward(engine, x, training, mask0));
    }

    /// <summary>A coarse voxel is active when any voxel of its 2×2×2 block is (torchsparse's stride-2 output set).</summary>
    private static double[] Downsample(double[] mask, int[] dims)
    {
        int x2 = dims[0] / 2, y2 = dims[1] / 2, z2 = dims[2] / 2;
        var coarse = new double[x2 * y2 * z2];
        for (int x = 0; x < dims[0]; x++)
            for (int y = 0; y < dims[1]; y++)
                for (int z = 0; z < dims[2]; z++)
                    if (mask[(x * dims[1] + y) * dims[2] + z] > 0)
                        coarse[((x / 2) * y2 + y / 2) * z2 + z / 2] = 1.0;
        return coarse;
    }
}
