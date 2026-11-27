using System;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// #2447: ModelBase.Serialize sizes its payload up front so that the buffer is never doubled. The size left
/// out the declared state, so the first state byte of any model that has state overflowed a buffer that was
/// already the size of the parameters and doubled it.
/// </summary>
public sealed class ModelSerializePayloadSizeTests
{
    private const int ParameterCount = 1 << 20;   // 8 MB as doubles
    private const int StateBytes = 1 << 20;

    [Fact]
    public void Serialize_OfAModelWithDeclaredState_RoundTripsParametersAndState()
    {
        var model = new StatefulModel(ParameterCount, StateBytes);
        for (int i = 0; i < ParameterCount; i++) model.Weight[i] = (i % 997) * 0.001 - 0.5;
        for (int i = 0; i < StateBytes; i++) model.Blob[i] = (byte)(i * 31);

        byte[] payload = model.Serialize();
        var restored = new StatefulModel(ParameterCount, StateBytes);
        restored.Deserialize(payload);

        // Everything the parameters and the state hold is in the payload, and nothing more than a header.
        Assert.InRange(payload.LongLength, 8L * ParameterCount + StateBytes, 8L * ParameterCount + StateBytes + 4096);
        for (int i = 0; i < ParameterCount; i++)
        {
            if (restored.Weight[i] != model.Weight[i]) Assert.Equal(model.Weight[i], restored.Weight[i]);
        }
        Assert.Equal(model.Blob.ToArray(), restored.Blob.ToArray());
    }

    [Fact]
    public void SerializedPayloadBytes_IsTheExactLengthOfThePayload()
    {
        var model = new StatefulModel(parameterCount: 37, stateBytes: 300);
        string typeName = typeof(StatefulModel).FullName ?? nameof(StatefulModel);

        byte[] payload = model.Serialize();

        // Read the state's length off the payload itself: everything after the header and the parameters.
        long withoutState = ModelBase<double, Tensor<double>, Tensor<double>>.SerializedPayloadBytes(typeName, 37, 0);
        long state = payload.LongLength - withoutState;
        Assert.True(state >= 300, $"The payload holds {state} bytes of state for a 300-byte declaration.");
        Assert.Equal(
            payload.LongLength,
            ModelBase<double, Tensor<double>, Tensor<double>>.SerializedPayloadBytes(typeName, 37, checked((int)state)));

        // The header arithmetic, checked against bytes this test lays out itself: magic, the 7-bit length
        // prefix (two bytes from 128 UTF-8 bytes on), the name, the count, the doubles.
        Assert.Equal(4 + 1 + 3 + 4 + 8 * 2 + 5,
            ModelBase<double, Tensor<double>, Tensor<double>>.SerializedPayloadBytes("abc", 2, 5));
        Assert.Equal(4 + 2 + 200 + 4,
            ModelBase<double, Tensor<double>, Tensor<double>>.SerializedPayloadBytes(new string('n', 200), 0, 0));
        Assert.Equal(4 + 1 + 2 + 4,   // one two-byte character
            ModelBase<double, Tensor<double>, Tensor<double>>.SerializedPayloadBytes("é", 0, 0));
    }

#if NET8_0_OR_GREATER
    [Fact]
    public void Serialize_OfAModelWithDeclaredState_DoesNotDoubleTheParameterBuffer()
    {
        var model = new StatefulModel(ParameterCount, StateBytes);
        model.Serialize();   // first call: one-time registration and JIT

        long before = GC.GetAllocatedBytesForCurrentThread();
        model.GetParameters();
        long flatRead = GC.GetAllocatedBytesForCurrentThread() - before;

        before = GC.GetAllocatedBytesForCurrentThread();
        byte[] payload = model.Serialize();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // Serialize reads the flat vector, converts it in bounded blocks (a slice and its double copy each) and
        // writes the payload. Sized exactly, the buffers are one payload plus the state's own small buffer:
        // under two payloads. Sized without the state they were the parameter-sized buffer, its doubling when
        // the first state byte overflowed it, and the final copy: about four.
        long conversionBlocks = 2L * 8 * ParameterCount;
        long bound = flatRead + conversionBlocks + 2L * payload.LongLength;
        Assert.True(allocated <= bound,
            $"Serialize allocated {allocated:N0} bytes for a {payload.LongLength:N0}-byte payload " +
            $"({flatRead:N0} of it the flat read, {conversionBlocks:N0} the conversion blocks); at most " +
            $"{bound:N0} is expected when the payload buffer is sized once.");
    }
#endif

    private sealed class StatefulModel : ModelBase<double, Tensor<double>, Tensor<double>>
    {
        public StatefulModel(int parameterCount, int stateBytes)
        {
            Weight = new Tensor<double>(new[] { parameterCount });
            Blob = new Vector<byte>(stateBytes);
        }

        public Tensor<double> Weight { get; }
        public Vector<byte> Blob { get; private set; }
        public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

        protected override void RegisterComponents() =>
            RegisterParameterComponent("weight", new TensorListParameterSource<double>(() => new[] { Weight }));

        protected override void RegisterState(ModelStateRegistry<double> state) =>
            state.DeclareByteVector("blob", () => Blob, value => Blob = value ?? new Vector<byte>(0));

        public override Tensor<double> Predict(Tensor<double> input) => input;
        public override void Train(Tensor<double> input, Tensor<double> expectedOutput) =>
            throw new NotSupportedException("A serialization fixture.");
        public override IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters) =>
            throw new NotSupportedException("A serialization fixture.");
    }
}
