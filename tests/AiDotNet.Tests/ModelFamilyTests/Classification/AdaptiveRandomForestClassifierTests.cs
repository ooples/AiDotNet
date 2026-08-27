using System.Collections;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Classification.Online;
using AiDotNet.DriftDetection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Classification;

public class AdaptiveRandomForestClassifierTests : ClassificationModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new AdaptiveRandomForestClassifier<double>();

    [Fact]
    public void Clone_PreservesIndependentDriftDetectorStateAndContinuation()
    {
        var options = new AdaptiveRandomForestOptions<double>
        {
            NumTrees = 3,
            NumFeaturesPerTree = 2,
            LambdaPoisson = 1.0,
            GracePeriod = 4,
            WarningThreshold = 1.0,
            DriftThreshold = 1.5,
            RandomSeed = 173,
        };
        var source = new AdaptiveRandomForestClassifier<double>(options);

        for (int index = 0; index < 64; index++)
        {
            var sample = CreateStreamingSample(index);
            source.PartialFit(sample, index % 2);
        }

        var clone = Assert.IsType<AdaptiveRandomForestClassifier<double>>(source.Clone());

        AssertDetectorStateIsEqualAndIndependent(source, clone);
        Assert.Equal(source.SamplesSeen, clone.SamplesSeen);
        Assert.Equal(source.AverageTreeAccuracy, clone.AverageTreeAccuracy, precision: 12);

        // Continue both streams after cloning. The flipped concept exercises detector history and the
        // copied Random state used by Poisson resampling/tree replacement; either shared state or a
        // fresh detector/RNG makes the two models diverge during this loop.
        for (int index = 64; index < 96; index++)
        {
            var sample = CreateStreamingSample(index);
            double flippedLabel = (index + 1) % 2;
            source.PartialFit(sample, flippedLabel);
            clone.PartialFit(new Vector<double>(sample.ToArray()), flippedLabel);

            Assert.Equal(source.SamplesSeen, clone.SamplesSeen);
            Assert.Equal(source.TreesInWarning, clone.TreesInWarning);
            Assert.Equal(source.AverageTreeAccuracy, clone.AverageTreeAccuracy, precision: 12);

            var probe = new Matrix<double>(1, sample.Length);
            for (int feature = 0; feature < sample.Length; feature++)
                probe[0, feature] = sample[feature];
            Assert.Equal(source.Predict(probe).ToArray(), clone.Predict(probe).ToArray());
        }

        AssertDetectorStateIsEqualAndIndependent(source, clone);
    }

    private static Vector<double> CreateStreamingSample(int index)
        => new(new[]
        {
            (double)(index % 7),
            (double)((index * 3) % 11),
            (double)((index * index) % 13),
            (double)(index % 2),
        });

    private static void AssertDetectorStateIsEqualAndIndependent(
        AdaptiveRandomForestClassifier<double> source,
        AdaptiveRandomForestClassifier<double> clone)
    {
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.NonPublic;
        FieldInfo ensembleField = typeof(AdaptiveRandomForestClassifier<double>)
            .GetField("_ensemble", Flags)!;
        var sourceMembers = ((IEnumerable)ensembleField.GetValue(source)!).Cast<object>().ToArray();
        var cloneMembers = ((IEnumerable)ensembleField.GetValue(clone)!).Cast<object>().ToArray();
        Assert.Equal(sourceMembers.Length, cloneMembers.Length);

        for (int memberIndex = 0; memberIndex < sourceMembers.Length; memberIndex++)
        {
            Type memberType = sourceMembers[memberIndex].GetType();
            foreach (string propertyName in new[] { "DriftDetector", "WarningDetector" })
            {
                PropertyInfo detectorProperty = memberType.GetProperty(propertyName)!;
                var sourceDetector = Assert.IsType<DDMDriftDetector<double>>(
                    detectorProperty.GetValue(sourceMembers[memberIndex]));
                var cloneDetector = Assert.IsType<DDMDriftDetector<double>>(
                    detectorProperty.GetValue(cloneMembers[memberIndex]));
                Assert.NotSame(sourceDetector, cloneDetector);

                int comparedFields = 0;
                for (Type? detectorType = sourceDetector.GetType();
                     detectorType is not null && detectorType != typeof(object);
                     detectorType = detectorType.BaseType)
                {
                    foreach (FieldInfo field in detectorType.GetFields(
                                 Flags | BindingFlags.DeclaredOnly))
                    {
                        if (field.IsStatic || (!field.FieldType.IsPrimitive && !field.FieldType.IsEnum))
                            continue;
                        Assert.Equal(field.GetValue(sourceDetector), field.GetValue(cloneDetector));
                        comparedFields++;
                    }
                }

                Assert.True(comparedFields >= 10, "Expected to compare the complete numeric DDM state.");
            }
        }
    }
}
