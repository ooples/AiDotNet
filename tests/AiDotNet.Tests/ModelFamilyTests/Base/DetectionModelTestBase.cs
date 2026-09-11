using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Shared base for the computer-vision detection and OCR families: object detection
/// (<see cref="ObjectDetectionTestBase{T}"/>), text detection
/// (<see cref="TextDetectionTestBase{T}"/>) and text recognition (<see cref="OCRTestBase{T}"/>).
/// </summary>
/// <remarks>
/// <para>
/// These models are <c>IFullModel&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c> built on
/// <c>ModelBase</c>, NOT <c>INeuralNetworkModel</c>: they hold their layers as discrete private
/// fields rather than an enumerable layer collection, and expose no architecture object. So they
/// cannot use <c>NeuralNetworkModelTestBase</c>, whose invariants are written against
/// <c>Layers</c>, <c>GetArchitecture()</c> and <c>GetNamedLayerActivations()</c>. This base
/// carries the part of the contract they DO share - the <c>IFullModel</c> surface - and each
/// domain base adds the invariants specific to its output format.
/// </para>
/// <para>
/// Two of the invariants here exist because the shared contract was silently broken:
/// <see cref="Train_ShouldChangeParameters"/> pins that training does something at all, and
/// <see cref="Clone_ShouldNotShareParameterStorage"/> pins that a clone owns its own weights.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type the model is expressed in.</typeparam>
public abstract class DetectionModelTestBase<T>
{
    /// <summary>Numeric operations for <typeparamref name="T"/>.</summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Converts a double to <typeparamref name="T"/>.</summary>
    protected static T ToT(double value) => NumOps.FromDouble(value);

    /// <summary>Converts a <typeparamref name="T"/> to double.</summary>
    protected static double ToD(T value) => NumOps.ToDouble(value);

    /// <summary>
    /// Builds the model under test. The generated fixtures override this.
    /// </summary>
    protected abstract IFullModel<T, Tensor<T>, Tensor<T>> CreateModel();

    /// <summary>
    /// Shape of the image tensor fed to the model, as NCHW. Detection and OCR backbones are
    /// convolutional and downsample by up to 32x, so the spatial dimensions must stay a multiple
    /// of 32 for the feature-pyramid levels to line up.
    /// </summary>
    protected virtual int[] InputShape => [1, 3, 64, 64];

    /// <summary>
    /// Number of training steps the training invariants take. Detection losses are slow per step,
    /// so this stays small; the invariants assert that something changed, not that the model
    /// converged.
    /// </summary>
    protected virtual int TrainingIterations => 2;

    /// <summary>
    /// Creates a deterministic pseudo-random image in [0, 1], the range the detector
    /// preprocessing expects.
    /// </summary>
    protected Tensor<T> CreateRandomImage(Random rng)
    {
        var tensor = new Tensor<T>(InputShape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = ToT(rng.NextDouble());
        }

        return tensor;
    }

    /// <summary>
    /// Creates a target tensor shaped like the model output, for the training invariants.
    /// </summary>
    protected Tensor<T> CreateTargetLike(Tensor<T> output, Random rng)
    {
        var target = new Tensor<T>(output.Shape);
        for (int i = 0; i < target.Length; i++)
        {
            target[i] = ToT(rng.NextDouble());
        }

        return target;
    }

    private static Vector<T> ParametersOf(IFullModel<T, Tensor<T>, Tensor<T>> model)
        => ((IParameterizable<T, Tensor<T>, Tensor<T>>)model).GetParameters();

    [Fact(Timeout = 120000)]
    public async Task ForwardPass_ShouldProduceFiniteOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var output = model.Predict(CreateRandomImage(rng));

        Assert.True(output.Length > 0, "Model produced an empty output tensor.");
        for (int i = 0; i < output.Length; i++)
        {
            double value = ToD(output[i]);
            Assert.False(double.IsNaN(value), $"Output[{i}] is NaN.");
            Assert.False(double.IsInfinity(value), $"Output[{i}] is Infinity.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var image = CreateRandomImage(rng);

        var first = model.Predict(image);
        var second = model.Predict(image);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(ToD(first[i]), ToD(second[i]), 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var first = model.Predict(CreateRandomImage(rng));
        var second = model.Predict(CreateRandomImage(rng));

        // A model whose output ignores its input is not reading the image at all - the failure
        // mode a constant-returning stub would show.
        Assert.Equal(first.Length, second.Length);
        bool anyDifference = false;
        for (int i = 0; i < first.Length && !anyDifference; i++)
        {
            if (Math.Abs(ToD(first[i]) - ToD(second[i])) > 1e-12)
            {
                anyDifference = true;
            }
        }

        Assert.True(anyDifference, "Two different images produced byte-identical output.");
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();

        Assert.True(ParametersOf(model).Length > 0, "Model reports no trainable parameters.");
    }

    [Fact(Timeout = 120000)]
    public async Task Metadata_ShouldExist()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();

        Assert.NotNull(model.GetModelMetadata());
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_ShouldProduceIdenticalOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var image = CreateRandomImage(rng);

        var clone = model.Clone();

        var original = model.Predict(image);
        var copied = clone.Predict(image);

        Assert.Equal(original.Length, copied.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(ToD(original[i]), ToD(copied[i]), 10);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Clone_ShouldNotShareParameterStorage()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var model = CreateModel();

        var before = ParametersOf(model);
        Assert.True(before.Length > 0, "Model reports no trainable parameters.");

        var clone = model.Clone();

        // Perturb the CLONE. A clone that shares weight storage with its original - the classic
        // MemberwiseClone shallow copy - will drag the original along with it, and every
        // downstream user who cloned a model to fine-tune it would silently corrupt the source.
        var mutated = new Vector<T>(before.Length);
        for (int i = 0; i < before.Length; i++)
        {
            mutated[i] = NumOps.Add(before[i], ToT(1.0));
        }

        ((IParameterizable<T, Tensor<T>, Tensor<T>>)clone).SetParameters(mutated);

        var after = ParametersOf(model);
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.Equal(ToD(before[i]), ToD(after[i]), 10);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Train_ShouldChangeParameters()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var image = CreateRandomImage(rng);
        var target = CreateTargetLike(model.Predict(image), rng);

        var before = ParametersOf(model);
        Assert.True(before.Length > 0, "Model reports no trainable parameters.");

        for (int step = 0; step < TrainingIterations; step++)
        {
            model.Train(image, target);
        }

        var after = ParametersOf(model);
        Assert.Equal(before.Length, after.Length);

        bool anyChange = false;
        for (int i = 0; i < before.Length && !anyChange; i++)
        {
            if (Math.Abs(ToD(before[i]) - ToD(after[i])) > 1e-12)
            {
                anyChange = true;
            }
        }

        Assert.True(
            anyChange,
            "Train() left every parameter untouched. The model cannot learn: either the training "
            + "step is a no-op or no gradient reaches the parameters.");
    }

    [Fact(Timeout = 300000)]
    public async Task Train_ShouldProduceFinitePredictions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var image = CreateRandomImage(rng);
        var target = CreateTargetLike(model.Predict(image), rng);

        for (int step = 0; step < TrainingIterations; step++)
        {
            model.Train(image, target);
        }

        var output = model.Predict(image);
        for (int i = 0; i < output.Length; i++)
        {
            double value = ToD(output[i]);
            Assert.False(double.IsNaN(value), $"Output[{i}] is NaN after training.");
            Assert.False(double.IsInfinity(value), $"Output[{i}] is Infinity after training.");
        }
    }
}
