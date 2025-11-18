using AiDotNet.MetaLearning.Data;

namespace AiDotNet.Tests.UnitTests.MetaLearning.Data;

public class TaskTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesTask()
    {
        // Arrange
        var supportInput = new Matrix<double>(10, 5);
        var supportOutput = new Vector<double>(10);
        var queryInput = new Matrix<double>(15, 5);
        var queryOutput = new Vector<double>(15);
        int numWays = 5;
        int numShots = 2;
        int numQueryPerClass = 3;

        // Act
        var task = new Task<double, Matrix<double>, Vector<double>>(
            supportInput, supportOutput, queryInput, queryOutput,
            numWays, numShots, numQueryPerClass);

        // Assert
        Assert.NotNull(task);
        Assert.Equal(supportInput, task.SupportInput);
        Assert.Equal(supportOutput, task.SupportOutput);
        Assert.Equal(queryInput, task.QueryInput);
        Assert.Equal(queryOutput, task.QueryOutput);
        Assert.Equal(numWays, task.NumWays);
        Assert.Equal(numShots, task.NumShots);
        Assert.Equal(numQueryPerClass, task.NumQueryPerClass);
        Assert.NotNull(task.TaskId);
    }

    [Fact]
    public void Constructor_WithTaskId_UsesProvidedId()
    {
        // Arrange
        var supportInput = new Matrix<double>(10, 5);
        var supportOutput = new Vector<double>(10);
        var queryInput = new Matrix<double>(15, 5);
        var queryOutput = new Vector<double>(15);
        string expectedTaskId = "test-task-123";

        // Act
        var task = new Task<double, Matrix<double>, Vector<double>>(
            supportInput, supportOutput, queryInput, queryOutput,
            5, 2, 3, expectedTaskId);

        // Assert
        Assert.Equal(expectedTaskId, task.TaskId);
    }

    [Fact]
    public void Constructor_NullSupportInput_ThrowsArgumentNullException()
    {
        // Arrange
        Matrix<double>? supportInput = null;
        var supportOutput = new Vector<double>(10);
        var queryInput = new Matrix<double>(15, 5);
        var queryOutput = new Vector<double>(15);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new Task<double, Matrix<double>, Vector<double>>(
                supportInput!, supportOutput, queryInput, queryOutput,
                5, 2, 3));
    }

    [Fact]
    public void TotalSupportExamples_CalculatesCorrectly()
    {
        // Arrange
        var task = CreateTestTask(numWays: 5, numShots: 2, numQueryPerClass: 3);

        // Act
        int totalSupport = task.TotalSupportExamples;

        // Assert
        Assert.Equal(10, totalSupport); // 5 ways * 2 shots = 10
    }

    [Fact]
    public void TotalQueryExamples_CalculatesCorrectly()
    {
        // Arrange
        var task = CreateTestTask(numWays: 5, numShots: 2, numQueryPerClass: 3);

        // Act
        int totalQuery = task.TotalQueryExamples;

        // Assert
        Assert.Equal(15, totalQuery); // 5 ways * 3 query per class = 15
    }

    [Fact]
    public void Task_OneWayOneShot_ConfigurationWorks()
    {
        // Arrange & Act
        var task = CreateTestTask(numWays: 1, numShots: 1, numQueryPerClass: 1);

        // Assert
        Assert.Equal(1, task.NumWays);
        Assert.Equal(1, task.NumShots);
        Assert.Equal(1, task.TotalSupportExamples);
        Assert.Equal(1, task.TotalQueryExamples);
    }

    [Fact]
    public void Task_TenWayFiveShot_ConfigurationWorks()
    {
        // Arrange & Act
        var task = CreateTestTask(numWays: 10, numShots: 5, numQueryPerClass: 10);

        // Assert
        Assert.Equal(10, task.NumWays);
        Assert.Equal(5, task.NumShots);
        Assert.Equal(50, task.TotalSupportExamples);
        Assert.Equal(100, task.TotalQueryExamples);
    }

    private static Task<double, Matrix<double>, Vector<double>> CreateTestTask(
        int numWays, int numShots, int numQueryPerClass)
    {
        int supportSize = numWays * numShots;
        int querySize = numWays * numQueryPerClass;

        var supportInput = new Matrix<double>(supportSize, 5);
        var supportOutput = new Vector<double>(supportSize);
        var queryInput = new Matrix<double>(querySize, 5);
        var queryOutput = new Vector<double>(querySize);

        return new Task<double, Matrix<double>, Vector<double>>(
            supportInput, supportOutput, queryInput, queryOutput,
            numWays, numShots, numQueryPerClass);
    }
}
