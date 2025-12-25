using Xunit;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Groups;

namespace AiDotNet.Tests.UnitTests.Engines;

/// <summary>
/// Unit tests for CpuAdvancedAlgebraEngine operations.
/// </summary>
public class CpuAdvancedAlgebraEngineTests
{
    private readonly CpuAdvancedAlgebraEngine _engine = CpuAdvancedAlgebraEngine.Instance;

    #region Octonion Tests

    [Fact]
    public void OctonionAddBatch_ComputesCorrectly()
    {
        // Arrange
        var left = new[]
        {
            new Octonion<double>(1, 0, 0, 0, 0, 0, 0, 0),
            new Octonion<double>(0, 1, 0, 0, 0, 0, 0, 0)
        };
        var right = new[]
        {
            new Octonion<double>(2, 0, 0, 0, 0, 0, 0, 0),
            new Octonion<double>(0, 2, 0, 0, 0, 0, 0, 0)
        };

        // Act
        var result = _engine.OctonionAddBatch(left, right);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(3.0, result[0].Scalar, precision: 10);  // 1 + 2
        Assert.Equal(3.0, result[1].E1, precision: 10);       // 1 + 2
    }

    [Fact]
    public void OctonionMultiplyBatch_RealNumbers_BehavesLikeScalarMultiply()
    {
        // Arrange: Real octonions (scalar only)
        var left = new[]
        {
            new Octonion<double>(2, 0, 0, 0, 0, 0, 0, 0),
            new Octonion<double>(3, 0, 0, 0, 0, 0, 0, 0)
        };
        var right = new[]
        {
            new Octonion<double>(3, 0, 0, 0, 0, 0, 0, 0),
            new Octonion<double>(4, 0, 0, 0, 0, 0, 0, 0)
        };

        // Act
        var result = _engine.OctonionMultiplyBatch(left, right);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(6.0, result[0].Scalar, precision: 10);   // 2 * 3
        Assert.Equal(12.0, result[1].Scalar, precision: 10);  // 3 * 4
        Assert.True(result[0].IsScalar);
        Assert.True(result[1].IsScalar);
    }

    [Fact]
    public void OctonionConjugateBatch_FlipsImaginaryParts()
    {
        // Arrange
        var octonions = new[]
        {
            new Octonion<double>(1, 2, 3, 4, 5, 6, 7, 8)
        };

        // Act
        var result = _engine.OctonionConjugateBatch(octonions);

        // Assert: Conjugate flips signs of e1-e7
        Assert.Equal(1, result.Length);
        Assert.Equal(1.0, result[0].Scalar, precision: 10);
        Assert.Equal(-2.0, result[0].E1, precision: 10);
        Assert.Equal(-3.0, result[0].E2, precision: 10);
        Assert.Equal(-4.0, result[0].E3, precision: 10);
        Assert.Equal(-5.0, result[0].E4, precision: 10);
        Assert.Equal(-6.0, result[0].E5, precision: 10);
        Assert.Equal(-7.0, result[0].E6, precision: 10);
        Assert.Equal(-8.0, result[0].E7, precision: 10);
    }

    [Fact]
    public void OctonionNormBatch_ComputesMagnitudes()
    {
        // Arrange: (3, 4, 0, 0, 0, 0, 0, 0) has magnitude 5
        var octonions = new[]
        {
            new Octonion<double>(3, 4, 0, 0, 0, 0, 0, 0),
            new Octonion<double>(1, 0, 0, 0, 0, 0, 0, 0)
        };

        // Act
        var result = _engine.OctonionNormBatch(octonions);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(5.0, result[0], precision: 10);
        Assert.Equal(1.0, result[1], precision: 10);
    }

    [Fact]
    public void OctonionMultiply_NonAssociativity()
    {
        // Arrange: Octonion multiplication is NOT associative: (a*b)*c != a*(b*c)
        // This is a defining property we should verify
        var e1 = new Octonion<double>(0, 1, 0, 0, 0, 0, 0, 0);
        var e2 = new Octonion<double>(0, 0, 1, 0, 0, 0, 0, 0);
        var e4 = new Octonion<double>(0, 0, 0, 0, 1, 0, 0, 0);

        // Act
        var leftAssoc = (e1 * e2) * e4;   // (e1*e2)*e4
        var rightAssoc = e1 * (e2 * e4);  // e1*(e2*e4)

        // Assert: They should NOT be equal (non-associativity)
        bool areEqual =
            Math.Abs(leftAssoc.Scalar - rightAssoc.Scalar) < 1e-10 &&
            Math.Abs(leftAssoc.E1 - rightAssoc.E1) < 1e-10 &&
            Math.Abs(leftAssoc.E2 - rightAssoc.E2) < 1e-10 &&
            Math.Abs(leftAssoc.E3 - rightAssoc.E3) < 1e-10 &&
            Math.Abs(leftAssoc.E4 - rightAssoc.E4) < 1e-10 &&
            Math.Abs(leftAssoc.E5 - rightAssoc.E5) < 1e-10 &&
            Math.Abs(leftAssoc.E6 - rightAssoc.E6) < 1e-10 &&
            Math.Abs(leftAssoc.E7 - rightAssoc.E7) < 1e-10;

        Assert.False(areEqual, "Octonion multiplication should be non-associative");
    }

    #endregion

    #region Multivector/Clifford Tests

    [Fact]
    public void MultivectorAddBatch_ComputesCorrectly()
    {
        // Arrange: G(2,0) - 2D Euclidean geometric algebra
        var algebra = new CliffordAlgebra(2, 0, 0);
        var left = new[]
        {
            Multivector<double>.CreateScalar(algebra, 1.0),
            Multivector<double>.CreateScalar(algebra, 2.0)
        };
        var right = new[]
        {
            Multivector<double>.CreateScalar(algebra, 3.0),
            Multivector<double>.CreateScalar(algebra, 4.0)
        };

        // Act
        var result = _engine.MultivectorAddBatch(left, right);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(4.0, result[0].Scalar, precision: 10);  // 1 + 3
        Assert.Equal(6.0, result[1].Scalar, precision: 10);  // 2 + 4
    }

    [Fact]
    public void GeometricProductBatch_Scalars_MultipliesScalars()
    {
        // Arrange
        var algebra = new CliffordAlgebra(2, 0, 0);
        var left = new[]
        {
            Multivector<double>.CreateScalar(algebra, 2.0),
            Multivector<double>.CreateScalar(algebra, 3.0)
        };
        var right = new[]
        {
            Multivector<double>.CreateScalar(algebra, 4.0),
            Multivector<double>.CreateScalar(algebra, 5.0)
        };

        // Act
        var result = _engine.GeometricProductBatch(left, right);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(8.0, result[0].Scalar, precision: 10);   // 2 * 4
        Assert.Equal(15.0, result[1].Scalar, precision: 10);  // 3 * 5
    }

    [Fact]
    public void MultivectorReverseBatch_AppliesReverseToAll()
    {
        // Arrange
        var algebra = new CliffordAlgebra(2, 0, 0);
        var mvs = new[]
        {
            Multivector<double>.CreateScalar(algebra, 5.0)
        };

        // Act
        var result = _engine.MultivectorReverseBatch(mvs);

        // Assert: Reverse of scalar is same scalar
        Assert.Equal(1, result.Length);
        Assert.Equal(5.0, result[0].Scalar, precision: 10);
    }

    [Fact]
    public void GradeProjectBatch_ExtractsGradeZero()
    {
        // Arrange: Create multivector with multiple grades
        var algebra = new CliffordAlgebra(2, 0, 0);  // 4 basis elements: 1, e1, e2, e12
        var coeffs = new Dictionary<int, double>
        {
            { 0, 5.0 },  // scalar
            { 1, 3.0 },  // e1
            { 2, 2.0 },  // e2
            { 3, 1.0 }   // e12
        };
        var mvs = new[] { new Multivector<double>(algebra, coeffs) };

        // Act
        var result = _engine.GradeProjectBatch(mvs, 0);

        // Assert: Should only have scalar part
        Assert.Equal(1, result.Length);
        Assert.Equal(5.0, result[0].Scalar, precision: 10);
        Assert.Equal(0.0, result[0][1], precision: 10);  // e1 should be 0
        Assert.Equal(0.0, result[0][2], precision: 10);  // e2 should be 0
        Assert.Equal(0.0, result[0][3], precision: 10);  // e12 should be 0
    }

    #endregion

    #region Lie Group Tests

    [Fact]
    public void So3ExpBatch_ZeroVectors_ReturnsIdentities()
    {
        // Arrange
        var group = new So3Group<double>();
        var tangents = new[]
        {
            new Vector<double>(new[] { 0.0, 0.0, 0.0 }),
            new Vector<double>(new[] { 0.0, 0.0, 0.0 })
        };

        // Act
        var result = _engine.So3ExpBatch(group, tangents);

        // Assert: exp(0) = Identity
        Assert.Equal(2, result.Length);
        foreach (var r in result)
        {
            Assert.Equal(1.0, r.Matrix[0, 0], precision: 10);
            Assert.Equal(1.0, r.Matrix[1, 1], precision: 10);
            Assert.Equal(1.0, r.Matrix[2, 2], precision: 10);
            Assert.Equal(0.0, r.Matrix[0, 1], precision: 10);
            Assert.Equal(0.0, r.Matrix[0, 2], precision: 10);
        }
    }

    [Fact]
    public void So3ExpLogBatch_RoundTrip()
    {
        // Arrange
        var group = new So3Group<double>();
        var tangents = new[]
        {
            new Vector<double>(new[] { 0.1, 0.2, 0.3 }),
            new Vector<double>(new[] { 0.4, 0.5, 0.1 })
        };

        // Act: exp then log
        var rotations = _engine.So3ExpBatch(group, tangents);
        var recoveredTangents = _engine.So3LogBatch(group, rotations);

        // Assert
        Assert.Equal(2, recoveredTangents.Length);
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(tangents[i][0], recoveredTangents[i][0], precision: 8);
            Assert.Equal(tangents[i][1], recoveredTangents[i][1], precision: 8);
            Assert.Equal(tangents[i][2], recoveredTangents[i][2], precision: 8);
        }
    }

    [Fact]
    public void So3ComposeBatch_IdentityComposition()
    {
        // Arrange
        var group = new So3Group<double>();
        var identity = So3<double>.Identity;
        var tangent = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
        var rotation = group.Exp(tangent);

        var lefts = new[] { rotation, identity };
        var rights = new[] { identity, rotation };

        // Act
        var results = _engine.So3ComposeBatch(group, lefts, rights);

        // Assert: R * I = R and I * R = R
        Assert.Equal(2, results.Length);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(rotation.Matrix[i, j], results[0].Matrix[i, j], precision: 10);
                Assert.Equal(rotation.Matrix[i, j], results[1].Matrix[i, j], precision: 10);
            }
        }
    }

    [Fact]
    public void Se3ExpBatch_ZeroVector_ReturnsIdentity()
    {
        // Arrange
        var group = new Se3Group<double>();
        var tangents = new[]
        {
            new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 })
        };

        // Act
        var result = _engine.Se3ExpBatch(group, tangents);

        // Assert: Identity rotation, zero translation
        Assert.Equal(1, result.Length);
        Assert.Equal(1.0, result[0].Rotation.Matrix[0, 0], precision: 10);
        Assert.Equal(0.0, result[0].Translation[0], precision: 10);
        Assert.Equal(0.0, result[0].Translation[1], precision: 10);
        Assert.Equal(0.0, result[0].Translation[2], precision: 10);
    }

    [Fact]
    public void Se3ExpLogBatch_RoundTrip()
    {
        // Arrange
        var group = new Se3Group<double>();
        var tangents = new[]
        {
            new Vector<double>(new[] { 0.1, 0.2, 0.1, 0.5, 0.3, 0.4 })
        };

        // Act
        var transforms = _engine.Se3ExpBatch(group, tangents);
        var recovered = _engine.Se3LogBatch(group, transforms);

        // Assert
        Assert.Equal(1, recovered.Length);
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(tangents[0][i], recovered[0][i], precision: 7);
        }
    }

    [Fact]
    public void So3AdjointBatch_ReturnsRotationMatrices()
    {
        // Arrange: For SO(3), the adjoint is just the rotation matrix itself
        var group = new So3Group<double>();
        var tangent = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
        var rotation = group.Exp(tangent);
        var rotations = new[] { rotation };

        // Act
        var adjoints = _engine.So3AdjointBatch(group, rotations);

        // Assert: Adjoint equals rotation matrix for SO(3)
        Assert.Equal(1, adjoints.Length);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(rotation.Matrix[i, j], adjoints[0][i, j], precision: 10);
            }
        }
    }

    #endregion
}
