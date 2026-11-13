using AiDotNet.Helpers;

namespace AiDotNet.VisionLanguage;

/// <summary>
/// An exponential-moving-average teacher of a set of student layers (Tarvainen and Valpola 2017; BYOL; DINO; iBOT;
/// the EMA teacher of SigLIP 2's self-distillation and masked prediction): the teacher's parameters follow the
/// student's as <c>θ_t ← m·θ_t + (1 − m)·θ_s</c> and are never trained.
/// </summary>
/// <remarks>
/// The teacher shares the student's layers and holds only its own parameter vectors: <see cref="Run"/> swaps the
/// teacher's parameters in, runs the forward pass without recording gradients, and restores the student's. A layer
/// whose parameters materialize after the teacher was taken (layers resolve their input width on first use) is
/// copied from the student when first seen.
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal sealed class EmaTeacher<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IEngine _engine;
    private readonly IReadOnlyList<ILayer<T>> _layers;
    private Vector<T>[]? _teacher;

    public EmaTeacher(IEngine engine, IReadOnlyList<ILayer<T>> layers, double momentum)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _layers = layers ?? throw new ArgumentNullException(nameof(layers));
        if (momentum < 0 || momentum > 1) throw new ArgumentOutOfRangeException(nameof(momentum), "The momentum must lie in [0, 1].");
        Momentum = momentum;
    }

    /// <summary>The EMA momentum <c>m</c>.</summary>
    public double Momentum { get; }

    /// <summary>The teacher's parameters, one vector per layer (empty before the teacher is taken).</summary>
    internal IReadOnlyList<Vector<T>> Parameters => _teacher ?? Array.Empty<Vector<T>>();

    /// <summary>Moves the teacher toward the student: <c>θ_t ← m·θ_t + (1 − m)·θ_s</c>; the first call copies the student.</summary>
    public void Update()
    {
        var teacher = Synchronized();
        var rate = NumOps.FromDouble(1 - Momentum);
        for (int i = 0; i < _layers.Count; i++)
            teacher[i] = _engine.Lerp(teacher[i], _layers[i].GetParameters(), rate);
    }

    /// <summary>Runs <paramref name="forward"/> with the teacher's parameters and no gradient recording.</summary>
    public Tensor<T> Run(Func<Tensor<T>> forward)
    {
        if (forward is null) throw new ArgumentNullException(nameof(forward));
        var teacher = Synchronized();
        var student = _layers.Select(layer => Copy(layer.GetParameters())).ToArray();
        try
        {
            for (int i = 0; i < _layers.Count; i++) _layers[i].SetParameters(teacher[i]);
            using (new Tensors.Engines.Autodiff.NoGradScope<T>())
                return forward();
        }
        finally
        {
            for (int i = 0; i < _layers.Count; i++) _layers[i].SetParameters(student[i]);
        }
    }

    // The teacher's vectors, taken from the student for any layer the teacher has not yet seen at its current size.
    private Vector<T>[] Synchronized()
    {
        var previous = _teacher;
        var teacher = new Vector<T>[_layers.Count];
        for (int i = 0; i < _layers.Count; i++)
        {
            var student = _layers[i].GetParameters();
            teacher[i] = previous is not null && previous[i].Length == student.Length ? previous[i] : Copy(student);
        }
        _teacher = teacher;
        return teacher;
    }

    private static Vector<T> Copy(Vector<T> parameters) => Vector<T>.FromArray(parameters.ToArray());
}
