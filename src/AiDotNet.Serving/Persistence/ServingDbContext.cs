using AiDotNet.Serving.Persistence.Entities;
using Microsoft.AspNetCore.DataProtection.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace AiDotNet.Serving.Persistence;

/// <summary>
/// EF Core database context for AiDotNet.Serving persistence.
/// </summary>
public sealed class ServingDbContext : DbContext, IDataProtectionKeyContext
{
    public ServingDbContext(DbContextOptions<ServingDbContext> options)
        : base(options)
    {
    }

    public DbSet<ApiKeyEntity> ApiKeys => Set<ApiKeyEntity>();

    public DbSet<ProtectedArtifactEntity> ProtectedArtifacts => Set<ProtectedArtifactEntity>();

    public DbSet<DataProtectionKey> DataProtectionKeys { get; set; } = null!;

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<ApiKeyEntity>(entity =>
        {
            entity.ToTable("ApiKeys");
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.KeyId).IsUnique();
            entity.Property(e => e.KeyId).HasMaxLength(64).IsRequired();
            entity.Property(e => e.Name).HasMaxLength(200).IsRequired();
            entity.Property(e => e.Salt).IsRequired();
            entity.Property(e => e.Hash).IsRequired();
        });

        modelBuilder.Entity<ProtectedArtifactEntity>(entity =>
        {
            entity.ToTable("ProtectedArtifacts");
            entity.HasKey(e => e.ArtifactName);
            entity.Property(e => e.ArtifactName).HasMaxLength(256).IsRequired();
            entity.Property(e => e.EncryptedPath).HasMaxLength(1024).IsRequired();
            entity.Property(e => e.KeyId).HasMaxLength(64).IsRequired();
            entity.Property(e => e.Algorithm).HasMaxLength(64).IsRequired();
            entity.Property(e => e.ProtectedKey).IsRequired();
            entity.Property(e => e.ProtectedNonce).IsRequired();
        });
    }
}

