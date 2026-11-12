using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// IFNet_sdi_recur.IFBlock: the input (and the flows) bilinearly reduced by the scale, two stride-2 convolution-PReLU
/// pairs, eight residual convolution-PReLU pairs and a 4x4 stride-2 transposed convolution to four flow channels and a
/// mask logit, upsampled back to the input's size.
/// </summary>
internal sealed class IacBlock<T> : LazyWeightModule<T>
{
    private readonly AmtConvPRelu<T> _down1, _down2;
    private readonly AmtConvPRelu<T>[] _body;
    private readonly AmtDeconv<T> _last;

    public IacBlock(int inChannels, int c)
    {
        _down1 = new AmtConvPRelu<T>(inChannels, c / 2, 3, 2);
        _down2 = new AmtConvPRelu<T>(c / 2, c, 3, 2);
        _body = Enumerable.Range(0, 8).Select(_ => new AmtConvPRelu<T>(c, c)).ToArray();
        _last = new AmtDeconv<T>(c, 5);
    }

    public (Tensor<T> Flow, Tensor<T> Mask) Forward(Tensor<T> x, Tensor<T>? flow, int scale)
    {
        int h = x.Shape[2], w = x.Shape[3];
        if (scale != 1) x = AmtOps.Resize(Engine, x, h / scale, w / scale);
        if (flow is not null)
            x = Engine.TensorConcatenate(new[] { x, Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, h / scale, w / scale), NumOps.FromDouble(1.0 / scale)) }, 1);
        x = _down2.Forward(_down1.Forward(x));
        var y = x;
        foreach (var layer in _body) y = layer.Forward(y);
        var tmp = AmtOps.Resize(Engine, _last.Forward(Engine.TensorAdd(y, x)), h, w);
        return (Engine.TensorMultiplyScalar(Engine.TensorNarrow(tmp, 1, 0, 4), NumOps.FromDouble(2.0 * scale)), Engine.TensorNarrow(tmp, 1, 4, 1));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => new LazyWeightModule<T>[] { _down1, _down2 }.Concat(_body).Concat(new LazyWeightModule<T>[] { _last });
}

/// <summary>refine.Contextnet: four Conv2 levels, each level's feature warped along the halved flow.</summary>
internal sealed class IacContext<T> : LazyWeightModule<T>
{
    private readonly M2mConv2<T>[] _levels;

    public IacContext(int c)
        => _levels = new[] { new M2mConv2<T>(3, c), new M2mConv2<T>(c, 2 * c), new M2mConv2<T>(2 * c, 4 * c), new M2mConv2<T>(4 * c, 8 * c) };

    public Tensor<T>[] Forward(Tensor<T> x, Tensor<T> flow)
    {
        var features = new Tensor<T>[4];
        for (int i = 0; i < 4; i++)
        {
            x = _levels[i].Forward(x);
            flow = Engine.TensorMultiplyScalar(AmtOps.Resize(Engine, flow, flow.Shape[2] / 2, flow.Shape[3] / 2), NumOps.FromDouble(0.5));
            features[i] = AmtOps.Warp(Engine, x, flow);
        }
        return features;
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children() => _levels;
}

/// <summary>refine.Unet: the frames, warped frames, mask logit and flows encoded with both frames' warped context, decoded
/// to a sigmoid residual.</summary>
internal sealed class IacUnet<T> : LazyWeightModule<T>
{
    private readonly M2mConv2<T>[] _down;
    private readonly M2mDeconv<T>[] _up;
    private readonly AmtConv2d<T> _out;

    public IacUnet(int c)
    {
        _down = new[] { new M2mConv2<T>(17, 2 * c), new M2mConv2<T>(4 * c, 4 * c), new M2mConv2<T>(8 * c, 8 * c), new M2mConv2<T>(16 * c, 16 * c) };
        _up = new[] { new M2mDeconv<T>(32 * c, 8 * c), new M2mDeconv<T>(16 * c, 4 * c), new M2mDeconv<T>(8 * c, 2 * c), new M2mDeconv<T>(4 * c, c) };
        _out = new AmtConv2d<T>(c, 3, 3);
    }

    private Tensor<T> Cat(params Tensor<T>[] parts) => Engine.TensorConcatenate(parts, 1);

    public Tensor<T> Forward(Tensor<T> img0, Tensor<T> img1, Tensor<T> warped0, Tensor<T> warped1, Tensor<T> mask, Tensor<T> flow,
        Tensor<T>[] c0, Tensor<T>[] c1)
    {
        var s0 = _down[0].Forward(Cat(img0, img1, warped0, warped1, mask, flow));
        var s1 = _down[1].Forward(Cat(s0, c0[0], c1[0]));
        var s2 = _down[2].Forward(Cat(s1, c0[1], c1[1]));
        var s3 = _down[3].Forward(Cat(s2, c0[2], c1[2]));
        var x = _up[0].Forward(Cat(s3, c0[3], c1[3]));
        x = _up[1].Forward(Cat(x, s2));
        x = _up[2].Forward(Cat(x, s1));
        x = _up[3].Forward(Cat(x, s0));
        return Engine.Sigmoid(_out.Forward(x));
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => _down.Cast<LazyWeightModule<T>>().Concat(_up).Concat(new LazyWeightModule<T>[] { _out });
}

/// <summary>One IFNet_sdi pass: the students' flows and blends, the refined frame and, with a ground truth, the teacher.</summary>
internal sealed class IacOutput<T>
{
    public IacOutput(Tensor<T>[] flows, Tensor<T>[] merged, Tensor<T> refined, Tensor<T> mask, Tensor<T>? teacher, Tensor<T>? distill)
    {
        Flows = flows;
        Merged = merged;
        Refined = refined;
        Mask = mask;
        Teacher = teacher;
        Distill = distill;
    }

    /// <summary>Each student's accumulated flows [N, 4, H, W] (to frame 0, then to frame 1).</summary>
    public Tensor<T>[] Flows { get; }

    /// <summary>Each student's mask blend of the two warped frames.</summary>
    public Tensor<T>[] Merged { get; }

    /// <summary>The last blend plus the U-Net's residual, clamped to [0, 1].</summary>
    public Tensor<T> Refined { get; }

    /// <summary>The last student's sigmoid blending mask.</summary>
    public Tensor<T> Mask { get; }

    /// <summary>The privileged teacher's blend, when a ground truth was given and the network distills.</summary>
    public Tensor<T>? Teacher { get; }

    /// <summary>The distillation loss of the students' flows toward the teacher's, where the teacher is better.</summary>
    public Tensor<T>? Distill { get; }
}

/// <summary>
/// IFNet_sdi (IFNet_sdi_recur.py): RIFE's three coarse-to-fine IFBlocks at scales 4, 2 and 1 conditioned on the
/// five-channel index map (target distance, reference distance and reference frame), the privileged teacher block that
/// also reads the ground truth, and the Contextnet-U-Net refinement.
/// </summary>
internal sealed class IacNetwork<T> : LazyWeightModule<T>
{
    private static readonly int[] Scales = { 4, 2, 1 };
    private readonly IacBlock<T>[] _students;
    private readonly IacBlock<T>? _teacher;
    private readonly IacContext<T> _context;
    private readonly IacUnet<T> _unet;

    public IacNetwork(IReadOnlyList<int> widths, int teacherWidth, int contextChannels, bool distill)
    {
        _students = new[] { new IacBlock<T>(6 + 5, widths[0]), new IacBlock<T>(13 + 4 + 5, widths[1]), new IacBlock<T>(13 + 4 + 5, widths[2]) };
        if (distill) _teacher = new IacBlock<T>(16 + 4 + 5, teacherWidth);
        _context = new IacContext<T>(contextChannels);
        _unet = new IacUnet<T>(contextChannels);
    }

    private Tensor<T> Cat(params Tensor<T>[] parts) => Engine.TensorConcatenate(parts, 1);

    private Tensor<T> Blend(Tensor<T> warped0, Tensor<T> warped1, Tensor<T> mask)
    {
        var full = Engine.TensorBroadcastTo(mask, warped0.Shape.ToArray());
        return Engine.TensorAdd(Engine.TensorMultiply(warped0, full), Engine.TensorMultiply(warped1, Engine.ScalarMinusTensor(NumOps.One, full)));
    }

    /// <summary>The pass on frames [N, 3, H, W] in [0, 1] with sides divisible by 32 and the index map [N, 5, H, W].</summary>
    public IacOutput<T> Forward(Tensor<T> img0, Tensor<T> img1, Tensor<T> indexMap, Tensor<T>? gt)
    {
        var flows = new Tensor<T>[3];
        var masks = new Tensor<T>[3];
        var merged = new Tensor<T>[3];
        Tensor<T>? flow = null, mask = null;
        Tensor<T> warped0 = img0, warped1 = img1;
        for (int i = 0; i < 3; i++)
        {
            if (flow is not null && mask is not null)
            {
                var (flowD, maskD) = _students[i].Forward(Cat(img0, img1, indexMap, warped0, warped1, mask), flow, Scales[i]);
                flow = Engine.TensorAdd(flow, flowD);
                mask = Engine.TensorAdd(mask, maskD);
            }
            else
            {
                (flow, mask) = _students[i].Forward(Cat(img0, img1, indexMap), null, Scales[i]);
            }
            masks[i] = Engine.Sigmoid(mask);
            flows[i] = flow;
            warped0 = AmtOps.Warp(Engine, img0, Engine.TensorNarrow(flow, 1, 0, 2));
            warped1 = AmtOps.Warp(Engine, img1, Engine.TensorNarrow(flow, 1, 2, 2));
            merged[i] = Blend(warped0, warped1, masks[i]);
        }
        if (flow is null || mask is null) throw new InvalidOperationException("IFNet has no student block.");
        Tensor<T>? teacher = null, distill = null;
        if (gt is not null && _teacher is not null)
        {
            var (flowD, maskD) = _teacher.Forward(Cat(img0, img1, indexMap, warped0, warped1, mask, gt), flow, 1);
            var flowTeacher = Engine.TensorAdd(flow, flowD);
            teacher = Blend(AmtOps.Warp(Engine, img0, Engine.TensorNarrow(flowTeacher, 1, 0, 2)),
                AmtOps.Warp(Engine, img1, Engine.TensorNarrow(flowTeacher, 1, 2, 2)), Engine.Sigmoid(Engine.TensorAdd(mask, maskD)));
            var teacherError = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(teacher, gt)), new[] { 1 }, true).ToArray();
            var fixedTeacher = Engine.StopGradient(flowTeacher);
            for (int i = 0; i < 3; i++)
            {
                // Where the student's blend is worse than the teacher's by more than 0.01, pull its flows to the teacher's.
                var studentError = Engine.ReduceMean(Engine.TensorAbs(Engine.TensorSubtract(merged[i], gt)), new[] { 1 }, true).ToArray();
                var where = new Tensor<T>(new[] { img0.Shape[0], 1, img0.Shape[2], img0.Shape[3] });
                for (int p = 0; p < where.Length; p++)
                    where[p] = NumOps.ToDouble(studentError[p]) > NumOps.ToDouble(teacherError[p]) + 0.01 ? NumOps.One : NumOps.Zero;
                var distance = Engine.TensorSqrt(Engine.ReduceMean(Engine.TensorSquare(Engine.TensorSubtract(fixedTeacher, flows[i])), new[] { 1 }, true));
                var term = Engine.ReduceMean(Engine.TensorMultiply(distance, where), new[] { 0, 1, 2, 3 }, false);
                distill = distill is null ? term : Engine.TensorAdd(distill, term);
            }
        }
        var c0 = _context.Forward(img0, Engine.TensorNarrow(flow, 1, 0, 2));
        var c1 = _context.Forward(img1, Engine.TensorNarrow(flow, 1, 2, 2));
        var residual = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(_unet.Forward(img0, img1, warped0, warped1, mask, flow, c0, c1), NumOps.FromDouble(2.0)), NumOps.FromDouble(-1.0));
        var refined = Engine.TensorClamp(Engine.TensorAdd(merged[2], residual), NumOps.Zero, NumOps.One);
        return new IacOutput<T>(flows, merged, refined, masks[2], teacher, distill);
    }

    protected override IEnumerable<LazyWeightModule<T>?> Children()
        => _students.Cast<LazyWeightModule<T>?>().Concat(new LazyWeightModule<T>?[] { _teacher, _context, _unet });
}
