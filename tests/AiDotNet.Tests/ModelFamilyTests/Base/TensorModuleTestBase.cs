using System;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for Tensor-to-Tensor modules whose only required constructor argument is a
/// WIDTH — <c>CenteringMechanism(int dimension)</c>, <c>RelationModule(int hiddenDimension)</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>The base owns the width and passes it to the factory.</b> The generated fixture implements
/// <see cref="CreateModel(int)"/> as <c>new X&lt;double&gt;(width)</c>, and every invariant feeds an
/// input built from the same <see cref="Width"/>. A module constructed for one width and fed another
/// is the defect class that kept recurring in this generator — an architecture pinning 64x32 while the
/// fixture fed 80 mel bins, a pin of <c>outputSize: 4</c> against a declared horizon of 96 — and it
/// cannot occur here, because there is only one number.
/// </para>
/// <para>
/// <b>No training invariant.</b> These are components trained by the method that owns them:
/// <c>RelationModule.Train</c> is empty, and <c>CenteringMechanism.Train</c> updates a running
/// centre rather than learnable weights. Asserting that training changes parameters would fail for a
/// reason that is not a defect — the mistake already made once with the <c>PolicyBase</c> models.
/// </para>
/// </remarks>
public abstract class TensorModuleTestBase<T>
{
    /// <summary>Shared numeric operations, matching the sibling family bases.</summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a numeric value to double for assertion.</summary>
    protected static double ToD(T value) => Convert.ToDouble(value);

    /// <summary>The single width both the module and its inputs are built from.</summary>
    protected virtual int Width => 8;

    /// <summary>Rows in each input batch.</summary>
    protected virtual int BatchSize => 2;

    /// <summary>Subclasses construct their module at exactly the width they are handed.</summary>
    protected abstract IFullModel<T, Tensor<T>, Tensor<T>> CreateModel(int width);

    private IFullModel<T, Tensor<T>, Tensor<T>> CreateModel() => CreateModel(Width);

    /// <summary>Deterministic [BatchSize, Width] input, identical on every call.</summary>
    private Tensor<T> CreateInput()
    {
        var tensor = new Tensor<T>([BatchSize, Width]);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.FromDouble(((i % 13) / 13.0) - 0.4);
        }

        return tensor;
    }

    [Fact(Timeout = 120000)]
    public async Task Construction_ProducesAModule()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();

        Assert.NotNull(CreateModel());
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_ReturnsFiniteValues()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();

        var output = model.Predict(CreateInput());

        Assert.NotNull(output);
        Assert.True(output.Length > 0, "A module returned an empty output for a non-empty input.");
        for (int i = 0; i < output.Length; i++)
        {
            double value = ToD(output[i]);
            Assert.False(double.IsNaN(value), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(value), $"Output[{i}] is infinite.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_IsDeterministic()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();
        var input = CreateInput();

        var first = model.Predict(input);
        var second = model.Predict(input);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(ToD(first[i]), ToD(second[i]), 6);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task WithParameters_RoundTripsTheParameterVector()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();
        model.Predict(CreateInput());

        var original = model.GetParameters();

        // A module may legitimately carry state but no learnable weights; then there is nothing to
        // round-trip, and asserting a non-empty vector would test a capability it does not claim.
        if (original.Length == 0)
        {
            return;
        }

        var restored = model.WithParameters(original).GetParameters();

        Assert.Equal(original.Length, restored.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(ToD(original[i]), ToD(restored[i]), 6);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DeepCopy_PredictsIdentically()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();
        var input = CreateInput();
        model.Predict(input);

        var clone = model.DeepCopy();
        var fromOriginal = model.Predict(input);
        var fromClone = clone.Predict(input);

        Assert.Equal(fromOriginal.Length, fromClone.Length);
        for (int i = 0; i < fromOriginal.Length; i++)
        {
            Assert.Equal(ToD(fromOriginal[i]), ToD(fromClone[i]), 6);
        }
    }
}

/// <summary>Double-precision convenience form, matching the other family bases.</summary>
public abstract class TensorModuleTestBase : TensorModuleTestBase<double>
{
}
