using System;
using System.IO;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    public class BTreeIndexTests : IDisposable
    {
        private readonly string _testDirectory;

        public BTreeIndexTests()
        {
            _testDirectory = Path.Combine(Path.GetTempPath(), "btree_tests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_testDirectory);
        }

        public void Dispose()
        {
            if (Directory.Exists(_testDirectory))
                Directory.Delete(_testDirectory, true);
        }

        private string GetTestIndexPath()
        {
            return Path.Combine(_testDirectory, $"test_index_{Guid.NewGuid():N}.db");
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_WithValidPath_CreatesIndex()
        {
            // Arrange & Act
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Assert
            Assert.Equal(0, index.Count);
        }

        [Fact]
        public void Constructor_WithNullPath_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new BTreeIndex(null!));
        }

        [Fact]
        public void Constructor_WithNonexistentDirectory_CreatesDirectory()
        {
            // Arrange
            var nestedPath = Path.Combine(_testDirectory, "nested", "path", "index.db");

            // Act
            using var index = new BTreeIndex(nestedPath);
            index.Add("key1", 100);
            index.Flush();

            // Assert
            Assert.True(File.Exists(nestedPath));
        }

        #endregion

        #region Add Tests

        [Fact]
        public void Add_WithValidKeyAndOffset_IncreasesCount()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            index.Add("key1", 1024);

            // Assert
            Assert.Equal(1, index.Count);
        }

        [Fact]
        public void Add_WithNullKey_ThrowsArgumentException()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Add(null!, 100));
        }

        [Fact]
        public void Add_WithEmptyKey_ThrowsArgumentException()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Add("", 100));
        }

        [Fact]
        public void Add_WithWhitespaceKey_ThrowsArgumentException()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Add("   ", 100));
        }

        [Fact]
        public void Add_WithNegativeOffset_ThrowsArgumentException()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => index.Add("key1", -1));
        }

        [Fact]
        public void Add_WithDuplicateKey_UpdatesOffset()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);
            index.Add("key1", 1024);

            // Act
            index.Add("key1", 2048);

            // Assert
            Assert.Equal(1, index.Count);
            Assert.Equal(2048, index.Get("key1"));
        }

        #endregion

        #region Get Tests

        [Fact]
        public void Get_WithExistingKey_ReturnsCorrectOffset()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);
            index.Add("key1", 1024);
            index.Add("key2", 2048);

            // Act
            var offset1 = index.Get("key1");
            var offset2 = index.Get("key2");

            // Assert
            Assert.Equal(1024, offset1);
            Assert.Equal(2048, offset2);
        }

        [Fact]
        public void Get_WithNonexistentKey_ReturnsNegativeOne()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            var offset = index.Get("nonexistent");

            // Assert
            Assert.Equal(-1, offset);
        }

        [Fact]
        public void Get_WithNullKey_ReturnsNegativeOne()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            var offset = index.Get(null!);

            // Assert
            Assert.Equal(-1, offset);
        }

        #endregion

        #region Contains Tests

        [Fact]
        public void Contains_WithExistingKey_ReturnsTrue()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);
            index.Add("key1", 1024);

            // Act
            var contains = index.Contains("key1");

            // Assert
            Assert.True(contains);
        }

        [Fact]
        public void Contains_WithNonexistentKey_ReturnsFalse()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            var contains = index.Contains("nonexistent");

            // Assert
            Assert.False(contains);
        }

        [Fact]
        public void Contains_WithNullKey_ReturnsFalse()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            var contains = index.Contains(null!);

            // Assert
            Assert.False(contains);
        }

        #endregion

        #region Remove Tests

        [Fact]
        public void Remove_WithExistingKey_RemovesKeyAndReturnsTrue()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);
            index.Add("key1", 1024);

            // Act
            var result = index.Remove("key1");

            // Assert
            Assert.True(result);
            Assert.Equal(0, index.Count);
            Assert.False(index.Contains("key1"));
        }

        [Fact]
        public void Remove_WithNonexistentKey_ReturnsFalse()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            var result = index.Remove("nonexistent");

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Remove_WithNullKey_ReturnsFalse()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            var result = index.Remove(null!);

            // Assert
            Assert.False(result);
        }

        #endregion

        #region GetAllKeys Tests

        [Fact]
        public void GetAllKeys_WithMultipleKeys_ReturnsAllKeys()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);
            index.Add("key1", 1024);
            index.Add("key2", 2048);
            index.Add("key3", 3072);

            // Act
            var keys = index.GetAllKeys().ToList();

            // Assert
            Assert.Equal(3, keys.Count);
            Assert.Contains("key1", keys);
            Assert.Contains("key2", keys);
            Assert.Contains("key3", keys);
        }

        [Fact]
        public void GetAllKeys_WithEmptyIndex_ReturnsEmpty()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Act
            var keys = index.GetAllKeys();

            // Assert
            Assert.Empty(keys);
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllEntries()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);
            index.Add("key1", 1024);
            index.Add("key2", 2048);

            // Act
            index.Clear();

            // Assert
            Assert.Equal(0, index.Count);
            Assert.Empty(index.GetAllKeys());
        }

        #endregion

        #region Flush and Persistence Tests

        [Fact]
        public void Flush_SavesIndexToDisk()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using (var index = new BTreeIndex(indexPath))
            {
                index.Add("key1", 1024);
                index.Add("key2", 2048);

                // Act
                index.Flush();
            }

            // Assert
            Assert.True(File.Exists(indexPath));
        }

        [Fact]
        public void Constructor_WithExistingIndexFile_LoadsData()
        {
            // Arrange
            var indexPath = GetTestIndexPath();

            // Create and populate index
            using (var index = new BTreeIndex(indexPath))
            {
                index.Add("key1", 1024);
                index.Add("key2", 2048);
                index.Add("key3", 3072);
                index.Flush();
            }

            // Act - Create new index from same file
            using var loadedIndex = new BTreeIndex(indexPath);

            // Assert
            Assert.Equal(3, loadedIndex.Count);
            Assert.Equal(1024, loadedIndex.Get("key1"));
            Assert.Equal(2048, loadedIndex.Get("key2"));
            Assert.Equal(3072, loadedIndex.Get("key3"));
        }

        [Fact]
        public void Dispose_FlushesDataToDisk()
        {
            // Arrange
            var indexPath = GetTestIndexPath();

            // Create index and add data without explicit flush
            using (var index = new BTreeIndex(indexPath))
            {
                index.Add("key1", 1024);
                index.Add("key2", 2048);
                // Dispose will be called here
            }

            // Act - Load from disk
            using var loadedIndex = new BTreeIndex(indexPath);

            // Assert
            Assert.Equal(2, loadedIndex.Count);
            Assert.Equal(1024, loadedIndex.Get("key1"));
            Assert.Equal(2048, loadedIndex.Get("key2"));
        }

        [Fact]
        public void Flush_WithNoChanges_DoesNotWriteFile()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            using var index = new BTreeIndex(indexPath);

            // Track initial file timestamp
            DateTime? initialTimestamp = null;
            if (File.Exists(indexPath))
                initialTimestamp = File.GetLastWriteTimeUtc(indexPath);

            // Wait a bit to ensure timestamp would change
            System.Threading.Thread.Sleep(10);

            // Act
            index.Flush();

            // Assert
            if (File.Exists(indexPath))
            {
                var currentTimestamp = File.GetLastWriteTimeUtc(indexPath);
                if (initialTimestamp.HasValue)
                    Assert.Equal(initialTimestamp.Value, currentTimestamp);
            }
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void ComplexScenario_WithMultipleOperations_MaintainsConsistency()
        {
            // Arrange
            var indexPath = GetTestIndexPath();

            using (var index = new BTreeIndex(indexPath))
            {
                // Add entries
                index.Add("alice", 1000);
                index.Add("bob", 2000);
                index.Add("charlie", 3000);
                Assert.Equal(3, index.Count);

                // Update an entry
                index.Add("bob", 2500);
                Assert.Equal(3, index.Count);
                Assert.Equal(2500, index.Get("bob"));

                // Remove an entry
                index.Remove("charlie");
                Assert.Equal(2, index.Count);

                // Add new entry
                index.Add("diana", 4000);
                Assert.Equal(3, index.Count);

                index.Flush();
            }

            // Reload and verify
            using (var loadedIndex = new BTreeIndex(indexPath))
            {
                Assert.Equal(3, loadedIndex.Count);
                Assert.Equal(1000, loadedIndex.Get("alice"));
                Assert.Equal(2500, loadedIndex.Get("bob"));
                Assert.Equal(-1, loadedIndex.Get("charlie"));
                Assert.Equal(4000, loadedIndex.Get("diana"));
            }
        }

        [Fact]
        public void LargeIndex_WithThousandsOfEntries_PerformsCorrectly()
        {
            // Arrange
            var indexPath = GetTestIndexPath();
            const int entryCount = 10000;

            using (var index = new BTreeIndex(indexPath))
            {
                // Add many entries
                for (int i = 0; i < entryCount; i++)
                {
                    index.Add($"key_{i:D6}", i * 1024L);
                }

                // Act
                index.Flush();

                // Assert - Verify random samples
                Assert.Equal(entryCount, index.Count);
                Assert.Equal(0, index.Get("key_000000"));
                Assert.Equal(5000 * 1024L, index.Get("key_005000"));
                Assert.Equal(9999 * 1024L, index.Get("key_009999"));
            }

            // Reload and verify
            using (var loadedIndex = new BTreeIndex(indexPath))
            {
                Assert.Equal(entryCount, loadedIndex.Count);
                Assert.Equal(1234 * 1024L, loadedIndex.Get("key_001234"));
            }
        }

        #endregion
    }
}
