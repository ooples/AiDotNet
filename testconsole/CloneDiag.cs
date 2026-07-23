using System;
using System.Reflection;
using AiDotNet.Classification.ImbalancedEnsemble;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

internal static class CloneDiag
{
    public static void Run()
    {
        // Reproduce ClassificationModelTestBase.Clone_ShouldProduceIdenticalPredictions
        // for BalancedRandomForestClassifier and dump enough state to localise the
        // round-trip bug.
        var rng = new Random(42);
        const int trainSamples = 100, testSamples = 30, features = 5, numClasses = 3;
        var trainX = new Matrix<double>(trainSamples, features);
        var trainY = new Vector<double>(trainSamples);
        var testX = new Matrix<double>(testSamples, features);
        for (int i = 0; i < trainSamples; i++)
        {
            for (int j = 0; j < features; j++) trainX[i, j] = rng.NextDouble();
            trainY[i] = rng.Next(numClasses);
        }
        for (int i = 0; i < testSamples; i++)
            for (int j = 0; j < features; j++) testX[i, j] = rng.NextDouble();

        var model = new BalancedRandomForestClassifier<double>();
        model.Train(trainX, trainY);
        var cloned = model.Clone();

        var pred1 = model.Predict(testX);
        var pred2 = cloned.Predict(testX);

        int diffs = 0;
        for (int i = 0; i < pred1.Length; i++)
            if (pred1[i] != pred2[i]) diffs++;
        Console.WriteLine($"Predictions diff: {diffs}/{pred1.Length}");
        Console.WriteLine($"NumClasses    orig={model.NumClasses}  clone={(cloned as BalancedRandomForestClassifier<double>)?.NumClasses}");
        Console.WriteLine($"NumFeatures   orig={model.NumFeatures} clone={(cloned as BalancedRandomForestClassifier<double>)?.NumFeatures}");
        Console.WriteLine($"ClassLabels   orig={Vec(model.ClassLabels)}  clone={Vec((cloned as BalancedRandomForestClassifier<double>)?.ClassLabels)}");

        // Reflect into _trees on both
        var treesField = typeof(BalancedRandomForestClassifier<double>)
            .GetField("_trees", BindingFlags.NonPublic | BindingFlags.Instance);
        var origTrees = treesField?.GetValue(model) as System.Collections.IList;
        var cloneTrees = treesField?.GetValue(cloned) as System.Collections.IList;
        Console.WriteLine($"Trees count   orig={origTrees?.Count ?? -1}  clone={cloneTrees?.Count ?? -1}");

        if (origTrees is not null && cloneTrees is not null && origTrees.Count == cloneTrees.Count && origTrees.Count > 0)
        {
            for (int t = 0; t < Math.Min(2, origTrees.Count); t++)
            {
                Console.WriteLine($"-- Tree #{t} --");
                Console.WriteLine($"  orig: {DescribeNode(origTrees[t])}");
                Console.WriteLine($"  clone:{DescribeNode(cloneTrees[t])}");
            }
        }

        // Show first 5 mismatched predictions
        int shown = 0;
        for (int i = 0; i < pred1.Length && shown < 5; i++)
        {
            if (pred1[i] != pred2[i])
            {
                Console.WriteLine($"  Sample {i}: orig pred={pred1[i]}, clone pred={pred2[i]}");
                shown++;
            }
        }
    }

    private static string Vec<T>(Vector<T>? v) =>
        v is null ? "<null>" : "[" + string.Join(",", System.Linq.Enumerable.Range(0, v.Length).Select(i => v[i]?.ToString())) + "]";

    private static string DescribeNode(object? node)
    {
        if (node is null) return "<null>";
        var type = node.GetType();
        var fi = type.GetProperty("FeatureIndex")?.GetValue(node);
        var th = type.GetProperty("Threshold")?.GetValue(node);
        var pc = type.GetProperty("PredictedClass")?.GetValue(node);
        var cp = type.GetProperty("ClassProbabilities")?.GetValue(node);
        var lc = type.GetProperty("LeftChild")?.GetValue(node);
        var rc = type.GetProperty("RightChild")?.GetValue(node);
        // Defend against ClassProbabilities exposing a scalar or a
        // dictionary via reflection — only treat it as a sequence when
        // it actually implements IEnumerable. Otherwise fall back to
        // the value's ToString() so the diagnostic still prints
        // something useful instead of throwing NRE on `.Cast<object>()`.
        string cpStr = cp switch
        {
            null => "<null>",
            System.Collections.IEnumerable enumerable =>
                "[" + string.Join(",", enumerable.Cast<object>().Select(o => o?.ToString())) + "]",
            _ => cp.ToString() ?? "<unknown>"
        };
        string leaf = (lc is null && rc is null) ? "LEAF" : "INNER";
        return $"{leaf} feat={fi} th={th} pred={pc} cp={cpStr}";
    }
}
