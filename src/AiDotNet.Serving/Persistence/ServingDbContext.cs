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

    public DbSet<LicenseKeyEntity> LicenseKeys => Set<LicenseKeyEntity>();

    public DbSet<LicenseActivationEntity> LicenseActivations => Set<LicenseActivationEntity>();

    public DbSet<StripeCustomerEntity> StripeCustomers => Set<StripeCustomerEntity>();

    public DbSet<StripeSubscriptionEntity> StripeSubscriptions => Set<StripeSubscriptionEntity>();

    public DbSet<DataProtectionKey> DataProtectionKeys => Set<DataProtectionKey>();

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

        modelBuilder.Entity<LicenseKeyEntity>(entity =>
        {
            entity.ToTable("LicenseKeys");
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.KeyId).IsUnique();
            entity.Property(e => e.KeyId).HasMaxLength(64).IsRequired();
            entity.Property(e => e.CustomerName).HasMaxLength(200).IsRequired();
            entity.Property(e => e.CustomerEmail).HasMaxLength(320);
            entity.Property(e => e.Salt).IsRequired();
            entity.Property(e => e.Hash).IsRequired();
            entity.Property(e => e.Environment).HasMaxLength(64);
            entity.Property(e => e.Notes).HasMaxLength(2000);
        });

        modelBuilder.Entity<LicenseActivationEntity>(entity =>
        {
            entity.ToTable("LicenseActivations");
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => new { e.LicenseKeyId, e.MachineId }).IsUnique();
            entity.Property(e => e.MachineId).HasMaxLength(256).IsRequired();
            entity.Property(e => e.MachineName).HasMaxLength(256);
            entity.Property(e => e.Environment).HasMaxLength(64);
        });

        modelBuilder.Entity<StripeCustomerEntity>(entity =>
        {
            entity.ToTable("StripeCustomers");
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.StripeCustomerId).IsUnique();
            entity.Property(e => e.StripeCustomerId).HasMaxLength(128).IsRequired();
            entity.Property(e => e.Email).HasMaxLength(320).IsRequired();
            entity.Property(e => e.Name).HasMaxLength(200).IsRequired();
        });

        modelBuilder.Entity<StripeSubscriptionEntity>(entity =>
        {
            entity.ToTable("StripeSubscriptions");
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.StripeSubscriptionId).IsUnique();
            entity.Property(e => e.StripeSubscriptionId).HasMaxLength(128).IsRequired();
            entity.Property(e => e.StripeCustomerId).HasMaxLength(128).IsRequired();
            entity.Property(e => e.StripePriceId).HasMaxLength(128).IsRequired();
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

