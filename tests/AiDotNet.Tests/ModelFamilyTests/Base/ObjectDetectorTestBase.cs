using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for the vision detectors — <c>ObjectDetectorBase</c>,
/// <c>TextDetectorBase</c> and <c>OCRBase</c>, all of which are
/// <c>ModelBase&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>There is deliberately no training invariant here, and that is the point of this remark.</b>
/// <c>ObjectDetectorBase.Train</c> is an empty body — "Training object detectors requires
/// specialized loss. Override in subclasses." Asserting that training changes parameters would
/// therefore fail on every detector for a reason that is not a defect.
/// </para>
/// <para>
/// That mistake was made once already in this area: the six <c>PolicyBase</c> models were routed to
/// the RL family, whose fixture asserts exactly that, and 28 of 33 emitted tests failed with no
/// model defect behind any of them. A generated fixture that cannot pass is worse than no fixture,
/// because it reports a working model as broken. So the invariants below are restricted to what
/// this family actually guarantees: construction, a finite deterministic forward, detections that
/// are well-formed, an honest parameter surface, and an independent clone.
/// </para>
/// </remarks>
public abstract class ObjectDetectorTestBase<T>
{
    /// <summary>Shared numeric operations, matching the convention of the sibling family bases.</summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a numeric value to double for assertion, as the other bases do.</summary>
    protected static double ToD(T value) => Convert.ToDouble(value);

    /// <summary>Subclasses return their concrete detector.</summary>
    protected abstract IFullModel<T, Tensor<T>, Tensor<T>> CreateModel();

    /// <summary>Per-sample input shape fed to the detector, as [C, H, W].</summary>
    protected virtual int[] InputShape => [3, 64, 64];

    /// <summary>Declared output shape. Detectors size their own head, so this is advisory.</summary>
    protected virtual int[] OutputShape => [1, 0];

    /// <summary>
    /// Runs one forward pass so lazy layers resolve their shapes before the parameter surface is
    /// read.
    /// </summary>
    /// <remarks>
    /// These detectors build their backbone lazily: <c>GetParameters()</c> on a freshly constructed
    /// model throws <c>ParameterLayoutNotReadyException</c> ("an unresolved layout is not an empty
    /// parameter vector"), which is the framework correctly refusing to report a half-built model
    /// rather than a defect. A warm-up is what any real caller does before touching weights, so the
    /// invariants below do it too.
    /// </remarks>
    private void WarmUp(IFullModel<T, Tensor<T>, Tensor<T>> model) => model.Predict(CreateImage());

    /// <summary>Deterministic image tensor — same content for every call, so repeat runs compare.</summary>
    private Tensor<T> CreateImage()
    {
        int[] shape = [1, .. InputShape];
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            // A smooth ramp rather than noise: a detector fed pure noise can legitimately return
            // nothing, which would make the well-formedness checks below vacuous.
            tensor[i] = NumOps.FromDouble(((i % 251) / 251.0) - 0.5);
        }

        return tensor;
    }

    [Fact(Timeout = 120000)]
    public async Task Construction_ProducesAUsableDetector()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();

        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_ReturnsFiniteValues()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();

        var output = model.Predict(CreateImage());

        Assert.NotNull(output);
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
        var image = CreateImage();

        var first = model.Predict(image);
        var second = model.Predict(image);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(
                ToD(first[i]),
                ToD(second[i]),
                6);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ProducesWellFormedDetections()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();

        // Only the object detectors expose Detect; text detectors and OCR models answer through
        // Predict alone, so this invariant is skipped rather than failed for them.
        if (model is not ObjectDetectorBase<T> detector)
        {
            return;
        }

        var result = detector.Detect(CreateImage());

        Assert.NotNull(result);
        Assert.NotNull(result.Detections);
        foreach (var detection in result.Detections)
        {
            double confidence = ToD(detection.Confidence);
            Assert.False(double.IsNaN(confidence), "Detection confidence is NaN.");
            Assert.InRange(confidence, 0.0, 1.0);
            Assert.True(detection.ClassId >= 0, $"Negative class id {detection.ClassId}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ParameterCount_IsPositive()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();

        WarmUp(model);
        var parameters = model.GetParameters();

        Assert.NotNull(parameters);
        Assert.True(
            parameters.Length > 0,
            "A detector with a backbone and a head reports an empty parameter vector, so nothing "
                + "downstream — optimisation, serialization, clone fidelity — can see its weights.");
    }

    [Fact(Timeout = 120000)]
    public async Task WithParameters_RoundTripsTheParameterVector()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();

        WarmUp(model);
        var original = model.GetParameters();
        var restored = model.WithParameters(original).GetParameters();

        Assert.Equal(original.Length, restored.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(
                ToD(original[i]),
                ToD(restored[i]),
                6);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DeepCopy_PredictsIdenticallyAndIsIndependent()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        var model = CreateModel();
        var image = CreateImage();

        WarmUp(model);
        var clone = model.DeepCopy();
        var fromOriginal = model.Predict(image);
        var fromClone = clone.Predict(image);

        Assert.Equal(fromOriginal.Length, fromClone.Length);
        for (int i = 0; i < fromOriginal.Length; i++)
        {
            Assert.Equal(
                ToD(fromOriginal[i]),
                ToD(fromClone[i]),
                6);
        }

        // Independence: writing through the clone's parameter surface must not reach the original.
        var originalParameters = model.GetParameters();
        if (originalParameters.Length == 0)
        {
            return;
        }

        var mutated = new Vector<T>(originalParameters.Length);
        for (int i = 0; i < originalParameters.Length; i++)
        {
            mutated[i] = NumOps.Add(originalParameters[i], NumOps.FromDouble(1.0));
        }

        clone.WithParameters(mutated);
        var afterMutation = model.GetParameters();

        for (int i = 0; i < originalParameters.Length; i++)
        {
            Assert.Equal(
                ToD(originalParameters[i]),
                ToD(afterMutation[i]),
                6);
        }
    }
}

/// <summary>Double-precision convenience form, matching the other family bases.</summary>
public abstract class ObjectDetectorTestBase : ObjectDetectorTestBase<double>
{
}
