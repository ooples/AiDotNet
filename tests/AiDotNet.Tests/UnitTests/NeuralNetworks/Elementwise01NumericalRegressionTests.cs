#if NET5_0_OR_GREATER
using System.Reflection;
using System.Runtime.Serialization;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public partial class Elementwise01NumericalRegressionTests
{
    private static object? Invoke(Type type, object? instance, string name, params object[] args)
        => (type.GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance)
            ?? throw new InvalidOperationException($"Missing {name}")).Invoke(instance, args);

    // Exercise value-only numerical helpers without constructing unrelated model architectures.
    private static TModel NumericModel<T, TModel>() where TModel : NeuralNetworkBase<T>
    {
        var model = (TModel)FormatterServices.GetUninitializedObject(typeof(TModel));
        (typeof(NeuralNetworkBase<T>).GetField("NumOps", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("Missing numeric operations")).SetValue(model, MathHelper.GetNumericOperations<T>());
        return model;
    }

    [Fact]
    public void CreditColumnNormalization_HandlesFullScaleHalfAndZeroDecimal()
    {
        var half = new Matrix<Half>(new[,] { { (Half)60000 }, { (Half)60000 } });
        Invoke(typeof(CreditRuleBase<Half>), null, "NormalizeColumns", half, MathHelper.GetNumericOperations<Half>());
        Assert.InRange((double)half[0, 0], 0.706, 0.708);
        Assert.Equal(half[0, 0], half[1, 0]);

        const decimal tiny = 0.0000000000001234567890123456m;
        var decimals = new Matrix<decimal>(new[,] { { 0m, 3m, tiny }, { 0m, 4m, 0m } });
        Invoke(typeof(CreditRuleBase<decimal>), null, "NormalizeColumns", decimals, MathHelper.GetNumericOperations<decimal>());
        Assert.Equal(0m, decimals[0, 0]);
        Assert.Equal(0m, decimals[1, 0]);
        Assert.InRange(decimals[0, 1], 0.599999m, 0.600001m);
        Assert.InRange(decimals[1, 1], 0.799999m, 0.800001m);
        Assert.Equal(tiny, decimals[0, 2]);
    }

    [Fact]
    public void CreditRowRescaling_MatchesReferenceNormAndRetainsZeroRows()
    {
        var signal = new Matrix<Half>(new[,] { { (Half)60000, (Half)60000 }, { (Half)0, (Half)0 } });
        var reference = new Matrix<Half>(new[,] { { (Half)3, (Half)4 }, { (Half)1, (Half)2 } });
        Invoke(typeof(CreditRuleBase<Half>), null, "RescaleRowsToMatch", signal, reference, MathHelper.GetNumericOperations<Half>());
        Assert.InRange((double)signal[0, 0], 3.53, 3.54);
        Assert.Equal(signal[0, 0], signal[0, 1]);
        Assert.Equal((Half)0, signal[1, 0]);

        var zero = new Matrix<decimal>(new[,] { { 0m, 0m } });
        Invoke(typeof(CreditRuleBase<decimal>), null, "RescaleRowsToMatch", zero,
            new Matrix<decimal>(new[,] { { 3m, 4m } }), MathHelper.GetNumericOperations<decimal>());
        Assert.Equal(0m, zero[0, 0]);
    }

    [Fact]
    public void BernoulliNormalization_PreservesHalfExtremesAndMidpoint()
    {
        var model = NumericModel<Half, DeepBeliefNetwork<Half>>();
        var input = new Tensor<Half>(new[] { (Half)(-65504), (Half)0, (Half)65504 }, new[] { 1, 3 });
        var output = Assert.IsType<Tensor<Half>>(Invoke(typeof(DeepBeliefNetwork<Half>), model, "NormalizeBernoulliVisibleInput", input));
        Assert.Equal(new[] { (Half)0, (Half)0.5, (Half)1 }, output.ToArray());
    }

    [Fact]
    public void LargeHalfZeroChecks_DetectASingleNonzeroElement()
    {
        var state = new Vector<Half>(70000);
        Assert.Equal(true, Invoke(typeof(HopeNetwork<Half>), null, "IsMetaStateZero", state));
        state[69999] = (Half)1;
        Assert.Equal(false, Invoke(typeof(HopeNetwork<Half>), null, "IsMetaStateZero", state));

        var model = NumericModel<Half, LSTMNeuralNetwork<Half>>();
        var values = new Tensor<Half>(new[] { 70000 });
        Assert.Equal(true, Invoke(typeof(LSTMNeuralNetwork<Half>), model, "IsZeroTensor", values));
        values.SetFlat(69999, (Half)1);
        Assert.Equal(false, Invoke(typeof(LSTMNeuralNetwork<Half>), model, "IsZeroTensor", values));
    }

    [Fact]
    public void HalfHtmAnomalyRate_UsesExactCountsBeyondHalfRange()
    {
        var model = NumericModel<Half, HTMNetwork<Half>>();
        var memory = new TemporalMemoryLayer<Half>(70000, 1);
        memory.PreviousState = new Vector<Half>(70000);
        memory.PreviousState.AsWritableSpan().Slice(0, 35000).Fill((Half)1);
        var baseType = typeof(NeuralNetworkBase<Half>);
        (baseType.GetField("_layers", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("Missing layer storage"))
            .SetValue(model, new List<ILayer<Half>> { memory, memory });
        (baseType.GetField("_architectureLayerOwnershipEstablished", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("Missing ownership state")).SetValue(model, 1);

        Assert.Equal((Half)0.5, Invoke(typeof(HTMNetwork<Half>), model, "CalculateAnomalyScore"));
    }

    [Fact]
    public void HalfEmbeddingAndCrf_RejectRoundedOutOfRangeBoundary()
    {
        var labels = new Tensor<Half>(new[] { (Half)2052 }, new[] { 1, 1 });
        var embedding = new EmbeddingLayer<Half>(2052, 1);
        Assert.Throws<InputContractViolationException>(() => embedding.Forward(labels));
        var embeddingError = Assert.Throws<TargetInvocationException>(() =>
            Invoke(typeof(EmbeddingLayer<Half>), embedding, "ValidateIndicesOrThrow", labels, 2052));
        Assert.IsType<ArgumentException>(embeddingError.InnerException);
        var parameters = embedding.GetParameters();
        parameters.AsWritableSpan().Fill((Half)0);
        parameters[2050] = (Half)0.5;
        embedding.SetParameters(parameters);
        var valid = new Tensor<Half>(new[] { (Half)2050 }, new[] { 1, 1 });
        Assert.Equal((Half)0.5, embedding.Forward(valid).GetFlat(0));

        var crf = new ConditionalRandomFieldLayer<Half>(numClasses: 1, scalarActivation: null);
        var crfError = Assert.Throws<TargetInvocationException>(() =>
            Invoke(typeof(ConditionalRandomFieldLayer<Half>), crf, "BuildLabelOneHotForBatch", labels, 0, 1, 2052));
        Assert.IsType<ArgumentOutOfRangeException>(crfError.InnerException);
    }

    [Fact]
    public void HalfFusedReadout_AcceptsLastRepresentableValidClass()
    {
        var layer = new DenseLayer<Half>(2049, (IActivationFunction<Half>)new IdentityActivation<Half>());
        var input = new Tensor<Half>(new[] { (Half)1 }, new[] { 1, 1 });
        _ = layer.Forward(input);
        var loss = layer.TryComputeFusedLoss(input,
            new Tensor<Half>(new[] { (Half)2048 }, new[] { 1 }), new CrossEntropyWithLogitsLoss<Half>());
        Assert.NotNull(loss);
        Assert.InRange((double)loss.GetFlat(0), 7.0, 9.0);
    }
}
#endif
