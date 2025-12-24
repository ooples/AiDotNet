using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace AiDotNet.Serving.Persistence;

/// <summary>
/// Design-time factory for EF Core tooling (migrations).
/// </summary>
public sealed class ServingDbContextFactory : IDesignTimeDbContextFactory<ServingDbContext>
{
    /// <summary>
    /// Creates a <see cref="ServingDbContext"/> instance for design-time tooling (migrations).
    /// </summary>
    /// <param name="args">Command-line arguments supplied by EF Core tooling.</param>
    /// <returns>A configured <see cref="ServingDbContext"/> instance.</returns>
    public ServingDbContext CreateDbContext(string[] args)
    {
        var optionsBuilder = new DbContextOptionsBuilder<ServingDbContext>();
        optionsBuilder.UseSqlite("Data Source=aidotnet.serving.design.db");
        return new ServingDbContext(optionsBuilder.Options);
    }
}

