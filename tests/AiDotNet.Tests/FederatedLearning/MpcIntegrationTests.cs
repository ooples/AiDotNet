using AiDotNet.FederatedLearning.MPC;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Comprehensive integration tests for MPC beyond secure aggregation (#540).
/// </summary>
public class MpcIntegrationTests
{
    private static Tensor<double> CreateTensor(params double[] values)
    {
        var tensor = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++)
        {
            tensor[i] = values[i];
        }

        return tensor;
    }

    // ========== ArithmeticSecretSharing Tests ==========

    [Fact]
    public void ArithmeticSS_ShareAndReconstruct_RecoversOriginal()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 3, seed: 42);
        var secret = CreateTensor(1.5, 2.5, 3.5);

        var shares = ss.Share(secret, numberOfParties: 3);

        Assert.Equal(3, shares.Length);

        var reconstructed = ss.Reconstruct(shares);

        Assert.NotNull(reconstructed);
        Assert.Equal(secret.Shape[0], reconstructed.Shape[0]);
        for (int i = 0; i < secret.Shape[0]; i++)
        {
            Assert.Equal(secret[i], reconstructed[i], 6);
        }
    }

    [Fact]
    public void ArithmeticSS_ShareSize_MatchesOriginal()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 3, seed: 42);
        var secret = CreateTensor(1.0, 2.0, 3.0, 4.0, 5.0);

        var shares = ss.Share(secret, numberOfParties: 3);

        for (int i = 0; i < shares.Length; i++)
        {
            Assert.Equal(secret.Shape[0], shares[i].Shape[0]);
        }
    }

    [Fact]
    public void ArithmeticSS_SecureAdd_ProducesCorrectResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var a = CreateTensor(1.0, 2.0, 3.0);
        var b = CreateTensor(4.0, 5.0, 6.0);

        var sharesA = ss.Share(a, 2);
        var sharesB = ss.Share(b, 2);

        var resultShares = ss.SecureAdd(sharesA, sharesB);
        var result = ss.Reconstruct(resultShares);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(a[i] + b[i], result[i], 6);
        }
    }

    [Fact]
    public void ArithmeticSS_SecureMultiply_ProducesCorrectResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var a = CreateTensor(2.0, 3.0);
        var b = CreateTensor(4.0, 5.0);

        var sharesA = ss.Share(a, 2);
        var sharesB = ss.Share(b, 2);

        var resultShares = ss.SecureMultiply(sharesA, sharesB);
        var result = ss.Reconstruct(resultShares);

        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(a[i] * b[i], result[i], 4);
        }
    }

    [Fact]
    public void ArithmeticSS_ScalarMultiply_ProducesCorrectResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var a = CreateTensor(1.0, 2.0, 3.0);
        double scalar = 2.5;

        var shares = ss.Share(a, 2);
        var resultShares = ss.ScalarMultiply(shares, scalar);
        var result = ss.Reconstruct(resultShares);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(a[i] * scalar, result[i], 6);
        }
    }

    [Fact]
    public void ArithmeticSS_TooFewParties_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ArithmeticSecretSharing<double>(numberOfParties: 1));
    }

    [Fact]
    public void ArithmeticSS_PreGenerateBeaverTriples_Succeeds()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);

        // Should not throw - needs shape and count
        ss.PreGenerateBeaverTriples(shape: new[] { 3 }, count: 10);
    }

    [Fact]
    public void ArithmeticSS_ShareNull_Throws()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);

        Assert.Throws<ArgumentNullException>(() => ss.Share(null, 2));
    }

    [Fact]
    public void ArithmeticSS_ReconstructNull_Throws()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);

        Assert.Throws<ArgumentException>(() => ss.Reconstruct(null));
    }

    // ========== BooleanSecretSharing Tests ==========

    [Fact]
    public void BooleanSS_ShareAndReconstruct_RecoversOriginal()
    {
        var ss = new BooleanSecretSharing(numberOfParties: 2);
        var secret = new byte[] { 0xFF, 0x00, 0xAB, 0xCD };

        var shares = ss.Share(secret);

        Assert.Equal(2, shares.Length);

        var reconstructed = ss.Reconstruct(shares);

        Assert.Equal(secret, reconstructed);
    }

    [Fact]
    public void BooleanSS_SecureXor_ProducesCorrectResult()
    {
        var ss = new BooleanSecretSharing(numberOfParties: 2);
        var a = new byte[] { 0xFF, 0x00 };
        var b = new byte[] { 0x0F, 0xF0 };

        var sharesA = ss.Share(a);
        var sharesB = ss.Share(b);

        var resultShares = ss.SecureXor(sharesA, sharesB);
        var result = ss.Reconstruct(resultShares);

        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal((byte)(a[i] ^ b[i]), result[i]);
        }
    }

    [Fact]
    public void BooleanSS_SecureAnd_ProducesCorrectResult()
    {
        var ss = new BooleanSecretSharing(numberOfParties: 2);
        var a = new byte[] { 0xFF, 0xAB };
        var b = new byte[] { 0x0F, 0xCD };

        var sharesA = ss.Share(a);
        var sharesB = ss.Share(b);

        // SecureAnd requires a BooleanTriple
        var andTriple = ss.GenerateAndTriple(byteLength: a.Length);
        var resultShares = ss.SecureAnd(sharesA, sharesB, andTriple);
        var result = ss.Reconstruct(resultShares);

        for (int i = 0; i < a.Length; i++)
        {
            Assert.Equal((byte)(a[i] & b[i]), result[i]);
        }
    }

    [Fact]
    public void BooleanSS_GenerateAndTriple_ProducesValidTriple()
    {
        var ss = new BooleanSecretSharing(numberOfParties: 2);

        var triple = ss.GenerateAndTriple(byteLength: 4);

        Assert.NotNull(triple);
        Assert.NotNull(triple.SharesU);
        Assert.NotNull(triple.SharesV);
        Assert.NotNull(triple.SharesW);
        Assert.Equal(2, triple.SharesU.Length); // one per party
        Assert.Equal(2, triple.SharesV.Length);
        Assert.Equal(2, triple.SharesW.Length);
    }

    [Fact]
    public void BooleanSS_TooFewParties_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new BooleanSecretSharing(numberOfParties: 1));
    }

    [Fact]
    public void BooleanSS_ShareEmptySecret_Throws()
    {
        var ss = new BooleanSecretSharing(numberOfParties: 2);

        Assert.Throws<ArgumentException>(() => ss.Share(Array.Empty<byte>()));
    }

    [Fact]
    public void BooleanSS_ShareNullSecret_Throws()
    {
        var ss = new BooleanSecretSharing(numberOfParties: 2);

        Assert.Throws<ArgumentException>(() => ss.Share(null));
    }

    // ========== BaseObliviousTransfer Tests ==========

    [Fact]
    public void BaseOT_Transfer_ReturnsSelectedMessage()
    {
        var ot = new BaseObliviousTransfer();
        var message0 = new byte[] { 1, 2, 3, 4 };
        var message1 = new byte[] { 5, 6, 7, 8 };

        // Choose message 0
        var result0 = ot.Transfer(message0, message1, choiceBit: 0);
        Assert.Equal(message0, result0);

        // Choose message 1
        var result1 = ot.Transfer(message0, message1, choiceBit: 1);
        Assert.Equal(message1, result1);
    }

    [Fact]
    public void BaseOT_BatchTransfer_ReturnsCorrectMessages()
    {
        var ot = new BaseObliviousTransfer();
        var messages0 = new byte[][] { new byte[] { 1 }, new byte[] { 3 }, new byte[] { 5 } };
        var messages1 = new byte[][] { new byte[] { 2 }, new byte[] { 4 }, new byte[] { 6 } };
        var choices = new int[] { 0, 1, 0 };

        var results = ot.BatchTransfer(messages0, messages1, choices);

        Assert.Equal(3, results.Length);
        Assert.Equal(messages0[0], results[0]); // choice 0
        Assert.Equal(messages1[1], results[1]); // choice 1
        Assert.Equal(messages0[2], results[2]); // choice 0
    }

    // ========== ExtendedObliviousTransfer Tests ==========

    [Fact]
    public void ExtendedOT_Initialize_Succeeds()
    {
        var extOt = new ExtendedObliviousTransfer();

        // Should not throw
        extOt.Initialize();
    }

    [Fact]
    public void ExtendedOT_Transfer_Works()
    {
        var extOt = new ExtendedObliviousTransfer();
        extOt.Initialize();

        var message0 = new byte[] { 10, 20 };
        var message1 = new byte[] { 30, 40 };

        var result = extOt.Transfer(message0, message1, choiceBit: 1);
        Assert.Equal(message1, result);
    }

    [Fact]
    public void ExtendedOT_BatchTransfer_Works()
    {
        var extOt = new ExtendedObliviousTransfer();
        extOt.Initialize();

        var m0 = new byte[][] { new byte[] { 1 }, new byte[] { 3 } };
        var m1 = new byte[][] { new byte[] { 2 }, new byte[] { 4 } };
        var choices = new int[] { 1, 0 };

        var results = extOt.BatchTransfer(m0, m1, choices);

        Assert.Equal(2, results.Length);
        Assert.Equal(m1[0], results[0]);
        Assert.Equal(m0[1], results[1]);
    }

    // ========== GarbledCircuit Tests ==========

    [Fact]
    public void GarbledCircuit_GarbleAndGate_ProducesValidData()
    {
        var generator = new GarbledCircuitGenerator();
        var gates = new List<CircuitGate>
        {
            new CircuitGate { Type = GateType.And, InputWire0 = 0, InputWire1 = 1, OutputWire = 2 }
        };

        var garbled = generator.Garble(gates, inputWireCount: 2, outputWireCount: 1);

        Assert.NotNull(garbled);
        Assert.Equal(2, garbled.InputWireCount);
        Assert.Equal(1, garbled.OutputWireCount);
        Assert.NotNull(garbled.GarbledTables);
        Assert.NotNull(garbled.InputWireLabels);
        Assert.NotNull(garbled.DecodingTable);
    }

    [Fact]
    public void GarbledCircuit_EvaluateAndGate_ProducesCorrectOutput()
    {
        var generator = new GarbledCircuitGenerator();
        var evaluator = new GarbledCircuitEvaluator();
        var gates = new List<CircuitGate>
        {
            new CircuitGate { Type = GateType.And, InputWire0 = 0, InputWire1 = 1, OutputWire = 2 }
        };

        var garbled = generator.Garble(gates, inputWireCount: 2, outputWireCount: 1);

        // Test AND(1, 1) = 1
        var inputLabels = new byte[][] { garbled.InputWireLabels[0][1], garbled.InputWireLabels[1][1] };
        var outputLabels = evaluator.Evaluate(garbled, inputLabels);

        Assert.NotNull(outputLabels);
        Assert.Single(outputLabels);

        var decoded = evaluator.Decode(outputLabels, garbled.DecodingTable);
        Assert.Single(decoded);
        Assert.Equal(1, decoded[0]); // AND(1, 1) = 1
    }

    [Fact]
    public void GarbledCircuit_EvaluateAndGate_ZeroInputs_ProducesZero()
    {
        var generator = new GarbledCircuitGenerator();
        var evaluator = new GarbledCircuitEvaluator();
        var gates = new List<CircuitGate>
        {
            new CircuitGate { Type = GateType.And, InputWire0 = 0, InputWire1 = 1, OutputWire = 2 }
        };

        var garbled = generator.Garble(gates, inputWireCount: 2, outputWireCount: 1);

        // Test AND(0, 1) = 0
        var inputLabels = new byte[][] { garbled.InputWireLabels[0][0], garbled.InputWireLabels[1][1] };
        var outputLabels = evaluator.Evaluate(garbled, inputLabels);
        var decoded = evaluator.Decode(outputLabels, garbled.DecodingTable);

        Assert.Equal(0, decoded[0]); // AND(0, 1) = 0
    }

    [Fact]
    public void GarbledCircuit_XorGate_ProducesCorrectOutput()
    {
        var generator = new GarbledCircuitGenerator(enableFreeXor: true);
        var evaluator = new GarbledCircuitEvaluator(enableFreeXor: true);
        var gates = new List<CircuitGate>
        {
            new CircuitGate { Type = GateType.Xor, InputWire0 = 0, InputWire1 = 1, OutputWire = 2 }
        };

        var garbled = generator.Garble(gates, inputWireCount: 2, outputWireCount: 1);

        // Test XOR(1, 0) = 1
        var inputLabels = new byte[][] { garbled.InputWireLabels[0][1], garbled.InputWireLabels[1][0] };
        var outputLabels = evaluator.Evaluate(garbled, inputLabels);
        var decoded = evaluator.Decode(outputLabels, garbled.DecodingTable);

        Assert.Equal(1, decoded[0]); // XOR(1, 0) = 1
    }

    [Fact]
    public void GarbledCircuit_MultiGateCircuit_Works()
    {
        var generator = new GarbledCircuitGenerator();
        var evaluator = new GarbledCircuitEvaluator();

        // Build: output = (input0 AND input1) XOR input2
        var gates = new List<CircuitGate>
        {
            new CircuitGate { Type = GateType.And, InputWire0 = 0, InputWire1 = 1, OutputWire = 3 },
            new CircuitGate { Type = GateType.Xor, InputWire0 = 3, InputWire1 = 2, OutputWire = 4 }
        };

        var garbled = generator.Garble(gates, inputWireCount: 3, outputWireCount: 1);

        // Test (1 AND 1) XOR 0 = 1
        var inputLabels = new byte[][]
        {
            garbled.InputWireLabels[0][1],
            garbled.InputWireLabels[1][1],
            garbled.InputWireLabels[2][0]
        };
        var outputLabels = evaluator.Evaluate(garbled, inputLabels);
        var decoded = evaluator.Decode(outputLabels, garbled.DecodingTable);

        Assert.Equal(1, decoded[0]); // (1 AND 1) XOR 0 = 1
    }

    // ========== SecureComparisonProtocol Tests ==========

    [Fact]
    public void SecureComparison_Compare_ReturnsResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var comparison = new SecureComparisonProtocol<double>(ss);
        var a = CreateTensor(5.0);
        var b = CreateTensor(3.0);

        var sharesA = ss.Share(a, 2);
        var sharesB = ss.Share(b, 2);

        var result = comparison.Compare(sharesA, sharesB);

        Assert.NotNull(result);
    }

    [Fact]
    public void SecureComparison_SecureMax_ReturnsResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var comparison = new SecureComparisonProtocol<double>(ss);
        var a = CreateTensor(5.0, 2.0);
        var b = CreateTensor(3.0, 8.0);

        var sharesA = ss.Share(a, 2);
        var sharesB = ss.Share(b, 2);

        var maxShares = comparison.SecureMax(sharesA, sharesB);

        Assert.NotNull(maxShares);
        Assert.Equal(2, maxShares.Length);
    }

    [Fact]
    public void SecureComparison_SecureMin_ReturnsResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var comparison = new SecureComparisonProtocol<double>(ss);
        var a = CreateTensor(5.0, 2.0);
        var b = CreateTensor(3.0, 8.0);

        var sharesA = ss.Share(a, 2);
        var sharesB = ss.Share(b, 2);

        var minShares = comparison.SecureMin(sharesA, sharesB);

        Assert.NotNull(minShares);
    }

    [Fact]
    public void SecureComparison_SecureNormSquared_ReturnsResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var comparison = new SecureComparisonProtocol<double>(ss);
        var a = CreateTensor(3.0, 4.0);

        var shares = ss.Share(a, 2);

        var normSquaredShares = comparison.SecureNormSquared(shares);

        Assert.NotNull(normSquaredShares);
    }

    [Fact]
    public void SecureComparison_NullProtocol_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SecureComparisonProtocol<double>(null));
    }

    // ========== SecureClippingProtocol Tests ==========

    [Fact]
    public void SecureClipping_ClipByNorm_ReturnsResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var clipping = new SecureClippingProtocol<double>(ss, clipNorm: 1.0);
        var largeGradient = CreateTensor(100.0, 200.0, 300.0);

        var gradientShares = ss.Share(largeGradient, 2);
        var clippedShares = clipping.ClipByNorm(gradientShares);

        Assert.NotNull(clippedShares);
        Assert.Equal(2, clippedShares.Length); // 2 shares

        var clipped = ss.Reconstruct(clippedShares);
        Assert.NotNull(clipped);
        Assert.Equal(3, clipped.Shape[0]);
    }

    [Fact]
    public void SecureClipping_ClipByValue_ReturnsResult()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var clipping = new SecureClippingProtocol<double>(ss, clipNorm: 1.0);
        var gradient = CreateTensor(0.5, -5.0, 3.0, -0.2);

        var gradientShares = ss.Share(gradient, 2);
        var clippedShares = clipping.ClipByValue(gradientShares);

        Assert.NotNull(clippedShares);

        var clipped = ss.Reconstruct(clippedShares);
        Assert.NotNull(clipped);
        Assert.Equal(4, clipped.Shape[0]);
    }

    [Fact]
    public void SecureClipping_ClipMultipleClients_Works()
    {
        var ss = new ArithmeticSecretSharing<double>(numberOfParties: 2, seed: 42);
        var clipping = new SecureClippingProtocol<double>(ss, clipNorm: 1.0);

        // Create gradient shares for multiple clients
        var clientGradientShares = new List<Tensor<double>[]>
        {
            ss.Share(CreateTensor(10.0, 20.0), 2),
            ss.Share(CreateTensor(0.1, 0.2), 2),
            ss.Share(CreateTensor(50.0, 100.0), 2)
        };

        var clipped = clipping.ClipMultipleClients(clientGradientShares);

        Assert.NotNull(clipped);
        Assert.Equal(3, clipped.Count);
    }

    [Fact]
    public void SecureClipping_NullProtocol_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SecureClippingProtocol<double>(null));
    }

    // ========== HybridMpcProtocol Tests ==========

    [Fact]
    public void HybridMpc_DefaultConstructor_Succeeds()
    {
        var hybrid = new HybridMpcProtocol<double>();

        Assert.NotNull(hybrid);
        Assert.NotNull(hybrid.ArithmeticScheme);
        Assert.NotNull(hybrid.BooleanScheme);
        Assert.NotNull(hybrid.CircuitGenerator);
        Assert.NotNull(hybrid.CircuitEvaluator);
        Assert.NotNull(hybrid.Comparison);
        Assert.NotNull(hybrid.Clipping);
    }

    [Fact]
    public void HybridMpc_WithOptions_Succeeds()
    {
        var options = new MpcOptions
        {
            Protocol = MpcProtocol.AdditiveSecretSharing,
            SecurityModel = MpcSecurityModel.SemiHonest,
            Threshold = 3,
            ClippingNormThreshold = 2.0
        };
        var hybrid = new HybridMpcProtocol<double>(options);

        Assert.NotNull(hybrid);
    }

    [Fact]
    public void HybridMpc_SecureClipGradient_Works()
    {
        var hybrid = new HybridMpcProtocol<double>(new MpcOptions { ClippingNormThreshold = 1.0 });
        var gradient = CreateTensor(100.0, 200.0);

        // Share first, then clip
        var gradientShares = hybrid.Share(gradient, 2);
        var clippedShares = hybrid.SecureClipGradient(gradientShares);

        Assert.NotNull(clippedShares);

        var clipped = hybrid.Reconstruct(clippedShares);
        Assert.NotNull(clipped);
    }

    [Fact]
    public void HybridMpc_SecureWeightedSum_ProducesResult()
    {
        var hybrid = new HybridMpcProtocol<double>();
        var clientGradientShares = new List<Tensor<double>[]>
        {
            hybrid.Share(CreateTensor(1.0, 2.0), 2),
            hybrid.Share(CreateTensor(3.0, 4.0), 2)
        };
        var weights = new List<double> { 0.5, 0.5 };

        var resultShares = hybrid.SecureWeightedSum(clientGradientShares, weights);

        Assert.NotNull(resultShares);

        var result = hybrid.Reconstruct(resultShares);
        Assert.NotNull(result);
    }

    [Fact]
    public void HybridMpc_SecureClippedAggregation_EndToEnd()
    {
        var hybrid = new HybridMpcProtocol<double>(new MpcOptions { ClippingNormThreshold = 5.0 });
        var clientGradientShares = new List<Tensor<double>[]>
        {
            hybrid.Share(CreateTensor(100.0, 200.0), 2),
            hybrid.Share(CreateTensor(0.5, 0.3), 2),
            hybrid.Share(CreateTensor(50.0, 80.0), 2)
        };
        var weights = new List<double> { 0.33, 0.34, 0.33 };

        var resultShares = hybrid.SecureClippedAggregation(clientGradientShares, weights);

        Assert.NotNull(resultShares);

        var result = hybrid.Reconstruct(resultShares);
        Assert.NotNull(result);
    }

    // ========== MpcOptions Defaults ==========

    [Fact]
    public void MpcOptions_DefaultValues()
    {
        var options = new MpcOptions();

        Assert.Equal(MpcProtocol.AdditiveSecretSharing, options.Protocol);
        Assert.Equal(MpcSecurityModel.SemiHonest, options.SecurityModel);
        Assert.Equal(3, options.Threshold);
        Assert.Equal(128, options.BaseObliviousTransferCount);
        Assert.Equal(1024, options.ObliviousTransferBatchSize);
        Assert.Equal(128, options.SecurityParameterBits);
        Assert.Equal(64, options.FieldBitLength);
        Assert.Equal(0.5, options.CovertDeterrenceFactor);
        Assert.Equal(1.0, options.ClippingNormThreshold);
        Assert.True(options.EnableFreeXor);
        Assert.True(options.EnableHalfGates);
        Assert.Null(options.RandomSeed);
    }

    [Fact]
    public void MpcProtocol_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(MpcProtocol), MpcProtocol.AdditiveSecretSharing));
    }

    [Fact]
    public void MpcSecurityModel_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(MpcSecurityModel), MpcSecurityModel.SemiHonest));
    }

    // ========== GateType enum ==========

    [Fact]
    public void GateType_HasAllExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(GateType), GateType.And));
        Assert.True(Enum.IsDefined(typeof(GateType), GateType.Xor));
        Assert.True(Enum.IsDefined(typeof(GateType), GateType.Not));
        Assert.True(Enum.IsDefined(typeof(GateType), GateType.Or));
    }

    [Fact]
    public void CircuitGate_DefaultValues()
    {
        var gate = new CircuitGate();

        Assert.Equal(GateType.And, gate.Type);
        Assert.Equal(0, gate.InputWire0);
        Assert.Equal(-1, gate.InputWire1); // -1 for unary gates
        Assert.Equal(0, gate.OutputWire);
    }
}
