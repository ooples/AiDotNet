#if NET5_0_OR_GREATER
using System;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Every model-family network's training forward is the network it predicts with: in eval mode, ForwardForTraining
/// (what Train differentiates) returns what Predict returns. A model that overrides PredictCore with its own forward
/// and leaves ForwardForTraining on the default Layers walk trains a different network than it runs (measured:
/// VGG16BNBackbone's Layers list holds its convolutions and batch norms, not its pools and ReLUs).
/// </summary>
[Collection(GpuEngineSwapCollection.Name)]
public class TrainingForwardMatchesPredictSweepTests
{
    private readonly ITestOutputHelper _output;

    public TrainingForwardMatchesPredictSweepTests(ITestOutputHelper output) => _output = output;

    public static System.Collections.Generic.IEnumerable<object[]> ModelFamilies() => GpuPredictScopeSweepTests.ModelFamilies();

    [Theory(Timeout = 300000)]
    [MemberData(nameof(ModelFamilies))]
    public async Task TrainingForward_MatchesPredict(string fixtureType)
    {
        await Task.Yield();
        var fixture = (IAsyncLifetime)Activator.CreateInstance(typeof(TrainingForwardMatchesPredictSweepTests).Assembly.GetType(fixtureType)!)!;
        await fixture.InitializeAsync();
        object? network = null;
        try
        {
            var type = fixture.GetType();
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            network = type.GetMethod("CreateNetwork", flags, null, Type.EmptyTypes, null)!.Invoke(fixture, null)!;
            var shape = (int[])type.GetProperty("EffectiveInputShape", flags)!.GetValue(fixture)!;
            var makeConstant = type.GetMethod("CreateConstantTensor", flags, null, new[] { typeof(int[]), typeof(double) }, null)!;
            var element = CompiledPredictBatchSweepTests.FindNetworkElementType(network.GetType());
            var report = (string)typeof(TrainingForwardMatchesPredictSweepTests)
                .GetMethod(nameof(Compare), BindingFlags.Static | BindingFlags.NonPublic)!.MakeGenericMethod(element)
                .Invoke(null, new[] { network, makeConstant.Invoke(fixture, new object[] { shape, 0.37 })! })!;
            _output.WriteLine($"{type.Name}: {report}");
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
        }
        finally
        {
            (network as IDisposable)?.Dispose();
            await fixture.DisposeAsync();
        }
    }

    private static string Compare<T>(NeuralNetworkBase<T> network, Tensor<T> input)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // Train differentiates ForwardForTraining unless the model trains its own way: an overridden Train, declared
        // objectives (each with its own forward), or a learning rule (a GAN's generator/discriminator steps).
        var own = OwnTrainingPath(network.GetType(), typeof(NeuralNetworkBase<T>), typeof(Tensor<T>));
        if (own is not null) return $"trains through its own {own}: ForwardForTraining is not its training forward";
        var designed = DesignedTrainingForward(network.GetType());
        if (designed is not null) return $"by design: {designed}";
        network.SetTrainingMode(false);
        var predicted = network.Predict(input).ToArray();
        if (!predicted.SequenceEqual(network.Predict(input).ToArray())) return "Predict is not deterministic: nothing to compare";

        // The input Train hands ForwardForTraining (ForwardPreparedForTraining's preparation; that funnel itself switches
        // to training mode, where dropout would make the comparison meaningless).
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        var baseType = typeof(NeuralNetworkBase<T>);
        object Call(string name, params object?[] args)
        {
            try { return baseType.GetMethod(name, flags)!.Invoke(network, args)!; }
            catch (TargetInvocationException ex) when (ex.InnerException is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                throw;
            }
        }
        var pairInput = ((ValueTuple<Tensor<T>, Tensor<T>?>)Call("PrepareTrainingPair", input, null, 0L)).Item1;
        var trainingInput = (Tensor<T>)Call("PrepareInputForTraining", pairInput);
        Tensor<T> trainedOutput;
        network.SetTrainingMode(false);   // a Predict may leave the mode changed
        using (new NoGradScope<T>())
            trainedOutput = network.ForwardForTraining(trainingInput);
        var trained = trainedOutput.ToArray();
        network.SetTrainingMode(false);   // a training forward may switch the mode on

        string? rawMismatch = Mismatch(ops, predicted, trained);
        if (rawMismatch is null) return $"{predicted.Length} values match";

        // A classifier trains its logits (the loss applies the stable log-softmax or log-sigmoid) and predicts the
        // probabilities: over the last axis, or the class (channel) axis of a segmentation map.
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        for (int axis = trainedOutput.Rank - 1; axis >= 0; axis--)
            if (Mismatch(ops, predicted, engine.Softmax(trainedOutput, axis: axis).ToArray()) is null)
                return $"{predicted.Length} values match: Predict is the softmax of the trained logits over axis {axis}";
        if (Mismatch(ops, predicted, engine.Sigmoid(trainedOutput).ToArray()) is null)
            return $"{predicted.Length} values match: Predict is the sigmoid of the trained logits";
        // A domain base's PostprocessOutput (softmax over logits, a decode) is inference-only: training differentiates
        // the raw output, and Predict returns it post-processed.
        var postprocess = network.GetType().GetMethod("PostprocessOutput",
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic, null, new[] { typeof(Tensor<T>) }, null);
        if (postprocess is not null)
        {
            // One that rejects the raw output (a genre-keyed decode expecting the class count) is not this model's
            // relation between the two.
            Tensor<T>? processed = null;
            try
            {
                using (new NoGradScope<T>())
                    processed = (Tensor<T>)postprocess.Invoke(network, new object[] { trainedOutput })!;
            }
            catch (TargetInvocationException ex) when (ex.InnerException is ArgumentException or InvalidOperationException or IndexOutOfRangeException)
            {
            }
            if (processed is not null && Mismatch(ops, predicted, processed.ToArray()) is null)
                return $"{predicted.Length} values match after PostprocessOutput";
        }
        // Deep supervision: the training output stacks several heads (ABINet's vision, language and fusion) for one
        // loss each, and Predict returns one of them.
        if (predicted.Length > 0 && trained.Length > predicted.Length && trained.Length % predicted.Length == 0)
        {
            for (int head = 0; head < trained.Length / predicted.Length; head++)
                if (Mismatch(ops, predicted, trained.Skip(head * predicted.Length).Take(predicted.Length).ToArray()) is null)
                    return $"{predicted.Length} values match head {head} of {trained.Length / predicted.Length} trained heads";
        }
        Assert.Fail($"{network.GetType().Name}: {rawMismatch}");
        return "";
    }

    private static string? Mismatch<T>(AiDotNet.Tensors.Interfaces.INumericOperations<T> ops, T[] predicted, T[] trained)
    {
        if (predicted.Length != trained.Length) return $"Predict returned {predicted.Length} values, ForwardForTraining {trained.Length}";
        for (int i = 0; i < predicted.Length; i++)
        {
            double p = ops.ToDouble(predicted[i]), t = ops.ToDouble(trained[i]);
            if (!(Math.Abs(p - t) <= 1e-4 + 1e-3 * Math.Abs(p) || double.IsNaN(p) && double.IsNaN(t)))
                return $"[{i}] Predict {p}, ForwardForTraining {t}";
        }
        return null;
    }

    // Models whose training forward is a different function of the input by design, each for its paper's reason.
    private static readonly System.Collections.Generic.Dictionary<string, string> DesignedTrainingForwards = new()
    {
        // Greydanus et al. 2019 Eq. 3: training fits the symplectic field (dH/dp, -dH/dq); Predict returns H.
        ["HamiltonianNeuralNetwork"] = "trains the Hamiltonian's symplectic gradient field, predicts the Hamiltonian",
        // Document-level labels: the per-node logits are mean-pooled for the loss; inference keeps them per node.
        ["LayoutGraph"] = "trains document-level pooled logits, predicts per-node logits",
        // Memory-bank video segmentation: inference attends to the bank of earlier frames, training to a provisional
        // entry built under the tape from the frame itself.
        ["Cutie"] = "trains against a provisional memory entry, predicts against the frame memory bank",
        // MIA-VSR Sec. 3.3 (the model's docs): the training forward samples Gumbel keep-masks for the L_mask term.
        ["MIAVSR"] = "samples Gumbel keep-masks in training",
        // Pennington et al. 2014 Eq. 8: training fits w_i . w~_j + b_i + b~_j to log X_ij over co-occurring pairs;
        // inference returns the word vectors W + W~ (Section 4.3).
        ["GloVe"] = "trains the co-occurrence pair objective, predicts the word vectors",
    };

    private static string? DesignedTrainingForward(Type model)
    {
        for (var t = model; t is not null; t = t.BaseType)
        {
            string name = t.IsGenericType ? t.Name.Substring(0, t.Name.IndexOf('`')) : t.Name;
            if (DesignedTrainingForwards.TryGetValue(name, out var reason)) return reason;
            // Video inpainting trains on a fresh random hole mask each step (the PyTorch video-inpainting recipe);
            // inference fills the deterministic default mask.
            if (name == "VideoInpaintingBase") return "trains on random hole masks, predicts with the default mask";
        }
        return null;
    }

    private static string? OwnTrainingPath(Type model, Type networkBase, Type tensor)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        foreach (var (name, parameters) in new[]
                 {
                     ("Train", new[] { tensor, tensor }),
                     ("DefineTrainingObjectives", new[] { tensor, tensor }),
                     ("TryApplyLearningRule", new[] { tensor, tensor }),
                 })
        {
            var method = model.GetMethod(name, flags, null, parameters, null);
            if (method is not null && method.DeclaringType != networkBase) return $"{name} ({method.DeclaringType!.Name})";
        }
        return null;
    }
}
#endif
