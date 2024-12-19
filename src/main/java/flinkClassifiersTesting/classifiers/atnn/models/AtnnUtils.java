package flinkClassifiersTesting.classifiers.atnn.models;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;
import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;

public class AtnnUtils {

    public static RealMatrix createRandomMatrix(int m, int n) {
        double limit = Math.sqrt(6.0 / (m + n));
        if (Double.isNaN(limit)) {
            throw new RuntimeException("o kurde Nan, x: " + limit);
        }
        Random random = new Random();
        double[][] matrix = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = -limit + (2 * limit) * random.nextDouble();
            }
        }
        return createRealMatrix(matrix);
    }


    public static double calculateStandardDeviation(List<Double> numbers) {
        if (numbers == null || numbers.isEmpty()) {
            throw new IllegalArgumentException("List cannot be empty.");
        }


        double mean = numbers.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        double variance = numbers.stream()
                .mapToDouble(num -> Math.pow(num - mean, 2))
                .average()
                .orElse(0.0);

        if (variance < 0 || Double.isNaN(variance)) {
            throw new RuntimeException("o kurde Nan, x: " + variance);
        }
        return Math.sqrt(variance);
    }


    public static RealVector standardization(RealVector vector) {
        int vectorLen = vector.toArray().length;
        double mean = Arrays.stream(vector.toArray()).reduce(0, Double::sum) / vectorLen;
        double partialSum = Arrays.stream(vector.toArray())
                .map(x -> (x - mean) * (x - mean))
                .sum();
        double std = Math.sqrt(partialSum / vectorLen);
        double epsilon = 1e-7;
        return vector.mapAdd(-mean).mapMultiply(1.0 / (std + epsilon));
    }


    public static double matrixCosineSimilarity(RealMatrix arr1, RealMatrix arr2) {
        if (arr1.getRowDimension() != arr2.getRowDimension())
            throw new IllegalArgumentException("Matrix dimensions do not match");

        double[] farr1 = new double[arr1.getRowDimension() * arr1.getColumnDimension()];
        double[] farr2 = new double[arr2.getRowDimension() * arr2.getColumnDimension()];
        for (int i = 0; i < arr1.getRowDimension(); i++) {
            for (int j = 0; j < arr1.getColumnDimension(); j++) {
                farr1[i * arr1.getRowDimension() + j] = arr1.getEntry(i, j);
                farr2[i * arr2.getRowDimension() + j] = arr2.getEntry(i, j);
            }
        }

        double dotProduct = 0, norm1 = 0, norm2 = 0;

        for (int i = 0; i < farr1.length; i++) {
            dotProduct += farr1[i] * farr2[i];
            norm1 += farr1[i] * farr1[i];
            norm2 += farr2[i] * farr2[i];
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    public static RealVector softmax(RealVector x) {
        /*
            Funkcja ma kilka "haków" zabezpieczających przed przekroczeniem zakresu double, który występował...
            W teorii mogą one prowadzić do mniej efektywnego uczenia, w praktyce tego nie zaobserwowałem.
            W oryginalnej funkcji w Pythonie dołączonej do publikacji ATNN wydaje się, że błędy związane z
            przekroczeniem zakresu double również mogły występować.
         */
        // HAK 0: double epsilon = 1e-7;
        double epsilon = 1e-7;

        // HAK 1 - Double.isFinite:
        double[] expData = Arrays.stream(x.toArray()).map(Math::exp)
                .map(v -> Double.isFinite(v) ? v : Double.MAX_VALUE)
                .toArray();

        // HAK 2 - Double.isFinite:
        double sumExpTmp = Arrays.stream(expData).sum();
        double sumExp = Double.isFinite(sumExpTmp) ? sumExpTmp + epsilon : Double.MAX_VALUE;

        double[] result = Arrays.stream(expData).map(elem -> elem / sumExp + epsilon).toArray();

        return createRealVector(result);
    }

    public static double cross_entropy(RealVector yHat, RealVector y) {
        double[] yHatArr = yHat.toArray();
        double[] yArr = y.toArray();
        double sum = 0;
        for (int i = 0; i < yHatArr.length; i++) {
            sum += yArr[i] * Math.log(yHatArr[i]);
        }
        return -sum;
    }

    public static RealMatrix squareEachElement(RealMatrix matrix) {
        RealMatrix result = matrix.copy();

        result.walkInOptimizedOrder(new org.apache.commons.math3.linear.RealMatrixChangingVisitor() {
            @Override
            public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {}

            @Override
            public double visit(int row, int column, double value) {
                return value * value;
            }

            @Override
            public double end() {
                return 0;
            }
        });

        return result;
    }

    public static RealVector squareEachElement(RealVector vector) {
        RealVector result = vector.copy();
        result.mapToSelf(x -> x * x);
        return result;
    }

    public static RealMatrix getElementWiseMaximum(RealMatrix m1, RealMatrix m2) {
        int rows = m1.getRowDimension();
        int cols = m1.getColumnDimension();
        double[][] node_dev_hW_data1 = m1.getData();
        double[][] node_dev_hW_data2 = m2.getData();
        double[][] elementWiseMaximum = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                elementWiseMaximum[i][j] = Math.max(node_dev_hW_data1[i][j], node_dev_hW_data2[i][j]);
            }
        }
        return new Array2DRowRealMatrix(elementWiseMaximum);
    }

    public static RealMatrix ebeMultiply(RealMatrix m1, RealMatrix m2) {
        RealMatrix ebe = MatrixUtils.createRealMatrix(m1.getRowDimension(), m2.getColumnDimension());
        for (int i = 0; i < m1.getRowDimension(); i++) {
            for (int j = 0; j < m1.getColumnDimension(); j++) {
                ebe.setEntry(i, j, m1.getEntry(i, j) * m2.getEntry(i, j));
            }
        }
        return ebe;
    }

    public static void checkIfMatrixContainsNaN(RealMatrix matrix, String msg) {
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                if (Double.isNaN(matrix.getEntry(i, j))) {
                    throw new RuntimeException("matrix: " + matrix + "\n" + msg);
                }
            }
        }
    }

    private static void checkIfVectorContainsNaN(RealVector vector) {
        for (int i = 0; i < vector.getDimension(); i++) {
            if (Double.isNaN(vector.getEntry(i))) {
                throw new RuntimeException("vector: " + vector);
            }
        }
    }

    private static void checkIfVectorContainsNaN(RealVector vector, String msg) {
        for (int i = 0; i < vector.getDimension(); i++) {
            if (Double.isNaN(vector.getEntry(i))) {
                throw new RuntimeException("vector: " + vector + ",\n" + msg);
            }
        }
    }
}
