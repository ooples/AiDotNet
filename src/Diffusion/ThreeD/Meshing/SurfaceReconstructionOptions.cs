namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Settings for reconstructing a surface mesh from a point cloud (<see cref="PointCloudSurfaceReconstruction"/>).
/// </summary>
/// <remarks>
/// <para>
/// The defaults follow the papers: screened Poisson uses Kazhdan &amp; Hoppe (2013)'s depth 8, point weight 4 and
/// bounding-box scale 1.1 (the defaults of their PoissonRecon code); ball pivoting, the signed-distance marching cubes
/// of Hoppe et al. (1992) and the alpha shape derive their radii from the measured sampling density when they are not
/// set, because their papers leave them to the data.
/// </para>
/// <para>
/// <b>For Beginners:</b> You rarely need to change these. Raise <see cref="PoissonDepth"/> for finer detail (each
/// step doubles the grid resolution and multiplies memory by eight), or set the radii by hand when the automatic
/// estimate from the point spacing does not suit your data.
/// </para>
/// </remarks>
public class SurfaceReconstructionOptions
{
    /// <summary>
    /// Neighbourhood size k for normal estimation and for the sampling-density estimate (Hoppe et al. 1992).
    /// </summary>
    public int NormalNeighborCount { get; set; } = 10;

    /// <summary>
    /// Octree depth of the screened Poisson solve: the indicator function is solved on a grid of 2^depth cells per
    /// axis. Kazhdan &amp; Hoppe (2013) default: 8. The grid is dense here, so memory grows as 8^depth (about 400 MB at
    /// depth 8).
    /// </summary>
    public int PoissonDepth { get; set; } = 8;

    /// <summary>
    /// The screening weight alpha of Kazhdan &amp; Hoppe (2013), eq. 4, which pulls the indicator function to the iso-value
    /// at the samples. Paper default: 4. Zero gives the unscreened Poisson reconstruction of Kazhdan et al. (2006).
    /// </summary>
    public double PoissonPointWeight { get; set; } = 4.0;

    /// <summary>
    /// The ratio between the side of the reconstruction cube and the largest side of the points' bounding box
    /// (PoissonRecon's --scale). Default: 1.1.
    /// </summary>
    public double PoissonScale { get; set; } = 1.1;

    /// <summary>
    /// The coarsest depth of the cascadic multigrid: the system is solved there first and each finer level starts
    /// from the prolonged coarser solution, as in Kazhdan &amp; Hoppe (2013), section 5. Default: 5.
    /// </summary>
    public int PoissonMinimumDepth { get; set; } = 5;

    /// <summary>
    /// The maximum conjugate-gradient iterations per multigrid level. Default: 100.
    /// </summary>
    public int PoissonSolverIterations { get; set; } = 100;

    /// <summary>
    /// The relative residual at which a level's conjugate-gradient solve stops. Default: 1e-6.
    /// </summary>
    public double PoissonSolverTolerance { get; set; } = 1e-6;

    /// <summary>
    /// Ball radii for ball pivoting (Bernardini et al. 1999), smallest first; each later radius re-pivots the
    /// boundary left by the previous one (the paper's section 5). Null derives two radii, 1.5 and 3 times the mean
    /// nearest-neighbour spacing.
    /// </summary>
    public double[]? BallPivotingRadii { get; set; }

    /// <summary>
    /// Grid cells along the longest side of the bounding box for the signed-distance marching cubes of Hoppe et al.
    /// (1992). Default: 64.
    /// </summary>
    public int SignedDistanceResolution { get; set; } = 64;

    /// <summary>
    /// Hoppe et al. (1992)'s rho + delta: a grid point whose projection onto its nearest tangent plane is farther than
    /// this from every sample is left undefined, so the surface is not extended across holes. Null uses twice the
    /// mean distance to the k-th nearest neighbour.
    /// </summary>
    public double? SignedDistanceSupportRadius { get; set; }

    /// <summary>
    /// The alpha radius of the alpha shape (Edelsbrunner &amp; Muecke 1994): Delaunay tetrahedra whose circumradius is
    /// below it are kept. Null uses twice the mean distance to the k-th nearest neighbour.
    /// </summary>
    public double? AlphaShapeRadius { get; set; }
}
