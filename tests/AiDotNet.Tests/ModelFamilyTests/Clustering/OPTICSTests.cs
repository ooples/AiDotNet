using AiDotNet.Interfaces;
using AiDotNet.Clustering.Density;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Clustering;

public class OPTICSTests : ClusteringModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new OPTICS<double>(new AiDotNet.Clustering.Options.OPTICSOptions<double>
        {
            MinSamples = 3,
            ClusterEpsilon = 1.5, // After normalization, clusters ~3 sigma apart
            ExtractionMethod = AiDotNet.Clustering.Options.OPTICSExtractionMethod.DbscanStyle
        });

    // OPTICS is density-based — doesn't have centroid parameters
    protected override bool HasFlatParameters => false;
}
