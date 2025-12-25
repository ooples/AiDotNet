using Xunit;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.UnitTests.Engines;

/// <summary>
/// Unit tests for CpuHyperbolicManifoldEngine operations.
/// </summary>
public class CpuHyperbolicManifoldEngineTests
{
    private readonly CpuHyperbolicManifoldEngine _engine = CpuHyperbolicManifoldEngine.Instance;

    #region Poincare Ball Tests

    [Fact]
    public void MobiusAdd_OriginPlusPoint_ReturnsPoint()
    {
        // Arrange: x + 0 = x in Mobius addition
        var origin = new Vector<double>(new[] { 0.0, 0.0 });
        var point = new Vector<double>(new[] { 0.3, 0.4 });
        double curvature = -1.0;

        // Act
        var result = _engine.MobiusAdd(origin, point, curvature);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(0.3, result[0], precision: 10);
        Assert.Equal(0.4, result[1], precision: 10);
    }

    [Fact]
    public void MobiusAdd_PointPlusOrigin_ReturnsPoint()
    {
        // Arrange: 0 + x = x in Mobius addition
        var point = new Vector<double>(new[] { 0.3, 0.4 });
        var origin = new Vector<double>(new[] { 0.0, 0.0 });
        double curvature = -1.0;

        // Act
        var result = _engine.MobiusAdd(point, origin, curvature);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(0.3, result[0], precision: 10);
        Assert.Equal(0.4, result[1], precision: 10);
    }

    [Fact]
    public void PoincareDistance_SamePoint_ReturnsZero()
    {
        // Arrange
        var point = new Vector<double>(new[] { 0.3, 0.4 });
        double curvature = -1.0;

        // Act
        var distance = _engine.PoincareDistance(point, point, curvature);

        // Assert
        Assert.Equal(0.0, distance, precision: 10);
    }

    [Fact]
    public void PoincareDistance_SymmetricProperty()
    {
        // Arrange
        var x = new Vector<double>(new[] { 0.1, 0.2 });
        var y = new Vector<double>(new[] { 0.3, 0.4 });
        double curvature = -1.0;

        // Act
        var distXY = _engine.PoincareDistance(x, y, curvature);
        var distYX = _engine.PoincareDistance(y, x, curvature);

        // Assert: d(x,y) = d(y,x)
        Assert.Equal(distXY, distYX, precision: 10);
    }

    [Fact]
    public void PoincareExpLogMap_RoundTrip()
    {
        // Arrange
        var basePoint = new Vector<double>(new[] { 0.1, 0.1 });
        var tangent = new Vector<double>(new[] { 0.2, 0.3 });
        double curvature = -1.0;

        // Act: Apply exp then log
        var expResult = _engine.PoincareExpMap(basePoint, tangent, curvature);
        var logResult = _engine.PoincareLogMap(basePoint, expResult, curvature);

        // Assert: Should approximately recover tangent
        Assert.Equal(tangent[0], logResult[0], precision: 8);
        Assert.Equal(tangent[1], logResult[1], precision: 8);
    }

    [Fact]
    public void PoincareProject_PointInsideBall_ReturnsUnchanged()
    {
        // Arrange: Point inside ball
        var point = new Vector<double>(new[] { 0.3, 0.4 });  // norm = 0.5 < 1
        double curvature = -1.0;
        double epsilon = 1e-5;

        // Act
        var result = _engine.PoincareProject(point, curvature, epsilon);

        // Assert
        Assert.Equal(0.3, result[0], precision: 10);
        Assert.Equal(0.4, result[1], precision: 10);
    }

    [Fact]
    public void PoincareProject_PointOutsideBall_ProjectsBack()
    {
        // Arrange: Point outside ball (norm = sqrt(2) > 1)
        var point = new Vector<double>(new[] { 1.0, 1.0 });
        double curvature = -1.0;
        double epsilon = 1e-5;

        // Act
        var result = _engine.PoincareProject(point, curvature, epsilon);

        // Assert: Norm should be less than 1 - epsilon
        double normSq = result[0] * result[0] + result[1] * result[1];
        double maxNorm = 1.0 - epsilon;
        Assert.True(normSq <= maxNorm * maxNorm + 1e-10);
    }

    #endregion

    #region Hyperboloid Tests

    [Fact]
    public void HyperboloidProject_EnsuresConstraint()
    {
        // Arrange: Some point
        var point = new Vector<double>(new[] { 2.0, 0.5, 0.5 });
        double curvature = -1.0;

        // Act
        var result = _engine.HyperboloidProject(point, curvature);

        // Assert: -x0^2 + x1^2 + ... = -1/c should hold
        // For curvature -1: -x0^2 + x1^2 + x2^2 = -1 (on upper sheet where x0 > 0)
        double constraint = -result[0] * result[0] + result[1] * result[1] + result[2] * result[2];
        Assert.Equal(-1.0, constraint, precision: 8);
    }

    [Fact]
    public void HyperboloidDistance_SamePoint_ReturnsZero()
    {
        // Arrange: Create valid hyperboloid point
        var x = _engine.HyperboloidProject(new Vector<double>(new[] { 2.0, 0.5, 0.5 }), -1.0);
        double curvature = -1.0;

        // Act
        var distance = _engine.HyperboloidDistance(x, x, curvature);

        // Assert (use precision 6 to account for numerical precision in projection)
        Assert.Equal(0.0, distance, precision: 6);
    }

    [Fact]
    public void HyperboloidDistance_SymmetricProperty()
    {
        // Arrange
        var x = _engine.HyperboloidProject(new Vector<double>(new[] { 2.0, 0.3, 0.4 }), -1.0);
        var y = _engine.HyperboloidProject(new Vector<double>(new[] { 3.0, 0.5, 0.6 }), -1.0);
        double curvature = -1.0;

        // Act
        var distXY = _engine.HyperboloidDistance(x, y, curvature);
        var distYX = _engine.HyperboloidDistance(y, x, curvature);

        // Assert: d(x,y) = d(y,x)
        Assert.Equal(distXY, distYX, precision: 10);
    }

    #endregion

    #region Model Conversion Tests

    [Fact]
    public void PoincareToHyperboloid_Origin_ReturnsNorthPole()
    {
        // Arrange: Origin in Poincare ball
        var origin = new Vector<double>(new[] { 0.0, 0.0 });
        double curvature = -1.0;

        // Act
        var result = _engine.PoincareToHyperboloid(origin, curvature);

        // Assert: Should be (1, 0, 0) - the north pole
        Assert.Equal(3, result.Length);
        Assert.Equal(1.0, result[0], precision: 10);
        Assert.Equal(0.0, result[1], precision: 10);
        Assert.Equal(0.0, result[2], precision: 10);
    }

    [Fact]
    public void ConversionRoundTrip_PoincareToHyperboloidAndBack()
    {
        // Arrange
        var poincare = new Vector<double>(new[] { 0.3, 0.4 });
        double curvature = -1.0;

        // Act
        var hyperboloid = _engine.PoincareToHyperboloid(poincare, curvature);
        var backToPoincare = _engine.HyperboloidToPoincare(hyperboloid, curvature);

        // Assert
        Assert.Equal(poincare[0], backToPoincare[0], precision: 10);
        Assert.Equal(poincare[1], backToPoincare[1], precision: 10);
    }

    #endregion

    #region Batch Operations Tests

    [Fact]
    public void PoincareExpMapBatch_MultipleVectors_ComputesAll()
    {
        // Arrange
        var basePoints = new Matrix<double>(new double[,]
        {
            { 0.1, 0.1 },
            { 0.2, 0.2 }
        });
        var tangentVectors = new Matrix<double>(new double[,]
        {
            { 0.1, 0.2 },
            { 0.15, 0.25 }
        });
        double curvature = -1.0;

        // Act
        var result = _engine.PoincareExpMapBatch(basePoints, tangentVectors, curvature);

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(2, result.Columns);
    }

    [Fact]
    public void PoincareDistanceBatch_MultiplePoints_ComputesAll()
    {
        // Arrange
        var x = new Matrix<double>(new double[,]
        {
            { 0.1, 0.1 },
            { 0.2, 0.2 }
        });
        var y = new Matrix<double>(new double[,]
        {
            { 0.3, 0.3 },
            { 0.4, 0.4 }
        });
        double curvature = -1.0;

        // Act
        var result = _engine.PoincareDistanceBatch(x, y, curvature);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.True(result[0] > 0);
        Assert.True(result[1] > 0);
    }

    #endregion
}
