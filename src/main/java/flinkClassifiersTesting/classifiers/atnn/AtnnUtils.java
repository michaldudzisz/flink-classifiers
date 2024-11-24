package flinkClassifiersTesting.classifiers.atnn;
import java.util.*;
import java.nio.*;
import java.util.stream.Collectors;
//import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.math3.linear.*;

import static flinkClassifiersTesting.classifiers.atnn.AtnnUtils.matrixSim;
import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

public class AtnnUtils {
    // Creates a random matrix with given dimensions
    public static RealMatrix createRandomMatrix(int m, int n) {
        double limit = Math.sqrt(6.0 / (m + n));
        Random random = new Random();
        double[][] matrix = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = -limit + (2 * limit) * random.nextDouble();
            }
        }
        return createRealMatrix(matrix);
    }

    // Standardizes a matrix
    public static RealMatrix standardization(RealMatrix matrix) {
        double mean = Arrays.stream(matrix.getData()).flatMapToDouble(Arrays::stream).average().orElse(0);
        double std = Math.sqrt(Arrays.stream(matrix.getData())
                .flatMapToDouble(Arrays::stream)
                .map(x -> (x - mean) * (x - mean))
                .average().orElse(0));
        double epsilon = 1e-7;
        return matrix.scalarAdd(-mean).scalarMultiply(1.0 / (std + epsilon));
    }

    // Computes cosine similarity between two matrices
    public static double matrixSim(RealMatrix arr1, RealMatrix arr2) {
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

    // Softmax function
    public static RealMatrix softmax(RealMatrix x) {
        if (x.getRowDimension() != 1)
            throw new IllegalArgumentException("Input matrix rows number different than 1, it is: " + x.getRowDimension());

        double[] expData = Arrays.stream(x.getRow(0)).map(Math::exp).toArray();
        double sumExp = Arrays.stream(expData).sum();
        double[][] result = { Arrays.stream(expData).map(e -> e / sumExp).toArray() };
        return createRealMatrix(result);
    }

    // Cross entropy loss
    public static double crossEntropy(RealMatrix yHat, RealMatrix y) {
        if (y.getRowDimension() != 0 || yHat.getColumnDimension() != 0)
            throw new IllegalArgumentException("Input matrix rows number different than 1");

        double[] yHatArr = yHat.getRow(0);
        double[] yArr = y.getRow(0);
        double sum = 0;
        for (int i = 0; i < yHatArr.length; i++) {
            sum += yArr[i] * Math.log(yHatArr[i]);
        }
        return -sum;
    }
}



class Node {
    int branchType = 0;
    boolean isShare = false;
    double learnRate = 0.02;
    int hNeuronNum;
    int cNeuronNum;
    boolean isRootNode = false;
    List<Node> childList = new ArrayList<>();
    Node parent = null;
    RealMatrix hW = null;
    RealVector hb = null; // wektor, tyle co neuronow, może RealVector ?
    RealMatrix cW = null;
    RealVector cb = null;
    RealVector hideInput = null;
    RealVector hideOutput = null;
    RealVector classifierInput = null; // może inny typ
    RealVector classifierOutput = null; // może inny typ
    RealMatrix dev_hW = null;
    RealMatrix dev_cW = null;
    double weight = 0; // todo do sprawdzenia czy into czy double
    int depth = 0;
    int trainTimes = 0;
    double reduction = 0.999;
    double minLR = 0.001;
    double lastPredictLoss = 0;
    RealMatrix squareGrad_hW = null;
    RealVector squareGrad_hb = null;
    RealMatrix Fisher_hW = null;
    RealVector Fisher_hb = null;
    RealMatrix alertSquareGrad_hW = null;
    RealVector alertSquareGrad_hb = null;
    RealMatrix alertFisher_hW = null;
    RealVector alertFisher_hb = null;
    RealMatrix lastConcept_hW = null;
    RealVector lastConcept_hb = null;


    public Node(int hNeuronNum, int cNeuronNum, int branchType) {
        this.hNeuronNum = hNeuronNum;
        this.cNeuronNum = cNeuronNum;
        this.branchType = branchType;
        init_b();
    }

    private void init_b() {
        hb = new ArrayRealVector(hNeuronNum);
        cb = new ArrayRealVector(cNeuronNum);
        Fisher_hb = new ArrayRealVector(hNeuronNum);
        squareGrad_hb = new ArrayRealVector(hNeuronNum);
        alertSquareGrad_hb = new ArrayRealVector(hNeuronNum);
        alertFisher_hb = new ArrayRealVector(hNeuronNum);
        lastConcept_hb = new ArrayRealVector(hNeuronNum);
    }

    public void init_weight() {
        if (isRootNode) {
            hW = AtnnUtils.createRandomMatrix(hNeuronNum, hNeuronNum);
            cW = AtnnUtils.createRandomMatrix(cNeuronNum, cNeuronNum);
            squareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            Fisher_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            alertSquareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            alertFisher_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            lastConcept_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
        } else {
            hW = AtnnUtils.createRandomMatrix(hNeuronNum, parent.hNeuronNum);
            cW = AtnnUtils.createRandomMatrix(cNeuronNum, cNeuronNum);
            squareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            Fisher_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            alertSquareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            alertFisher_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            lastConcept_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
        }
    }

    // ReLU activation function
    public RealMatrix relu(RealMatrix x) {
        double[][] data = x.getData();
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = Math.max(0, data[i][j]);
            }
        }

        return AtnnUtils.standardization(createRealMatrix(data));
    }

    // Derivative of ReLU
    public RealMatrix devRelu(RealMatrix x) {
        double[][] data = x.getData();
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                data[i][j] = (data[i][j] > 0) ? 1 : 0;
            }
        }
        return createRealMatrix(data);
    }
}

class Model {
    int featureNum;
    int hNeuronNum;
    int cNeuronNum;
    double beta = 0.99;
    double smooth = 0.2;
    int trainTimes = 0;
    int activeBranch = 0;
    int branchNum = 0;
    int maxBranchNum = 20;
    Node model = null;
    Map<Integer, List<Node>> nodeList = new HashMap<>();
    List<Node> activeNodeList = new ArrayList<>();
    Map<Integer, List<Double>> lossList = new HashMap<>();
    int lossLen = 1000;
    int splitLen = 50;
    Map<Integer, Map<String, Double>> lossStatisticsList = new HashMap<>();
    List<Integer> branchList = new ArrayList<>();
    int driftAlert = 0;
    int alertNum = 0;
    int lastDriftTime = 0;
    int lamda = 5000;
    String dataSet = null;
    int confid = 3;

    public Model(int featureNum, int hNeuronNum, int cNeuronNum) {
        this.featureNum = featureNum;
        this.hNeuronNum = hNeuronNum;
        this.cNeuronNum = cNeuronNum;
        init_model();
    }

    public void init_model() {
        int branchType = 0;
        Node root = new Node(featureNum, cNeuronNum, branchType);
        root.isRootNode = true;
        root.init_weight();

        List<Node> valToAdd = new ArrayList<>();
        valToAdd.add(root);
        nodeList.put(0, valToAdd);

        model = root;
        Node parent = root;
        for (int i = 0; i < 2; i++) {
            add_child_node(parent, branchType, 0.0);
            parent = parent.childList.get(parent.childList.size() - 1); // przedostatni
        }

        activeBranch = branchType;
        init_node_weight();
    }

    private void add_child_node(Node parentNode, int branchType, double weight) { // todo tutaj 0 jesli niepodane
        Node child = new Node(hNeuronNum, cNeuronNum, branchType);
        child.parent = parentNode;
        child.depth = child.parent.depth + 1;
        child.init_weight();
        child.weight = weight;
        parentNode.childList.add(child);
        if (!nodeList.containsKey(branchType)) {
            nodeList.put(branchType, new ArrayList<>());
        }
        nodeList.get(branchType).add(child);
    }

    private void init_node_weight() {
        activeNodeList = get_active_node_list();
        double avgWeight = 1.0 / activeNodeList.size();
        for (Node node : activeNodeList) {
            node.weight = avgWeight;
        }
    }

    private List<Node> get_active_node_list() {
        return nodeList.get(activeBranch);
    }

    private Node weight_sim() {
        Node branchNode = model;
        double init_sim = 0.85;
        double min_sim = 0.7;
        List<Double> weightList = nodeList.get(0).stream().map(n -> n.weight).collect(Collectors.toList());
        int max_index = 0;

        for (int i = 0; i < weightList.size(); i++) {
            if (weightList.get(i) > weightList.stream().reduce(0.0, Double::sum) / weightList.size()) {
                max_index = i;
            }
        }

//        weightList = new ArrayList<>();
        for (Node node : nodeList.get(0).subList(1, nodeList.get(0).size())) {
            node.alertFisher_hW = node.alertSquareGrad_hW.scalarMultiply((double) 1 / alertNum);
            node.alertFisher_hb = node.alertSquareGrad_hb.mapMultiply((double) 1 / alertNum);
            double sim = matrixSim(node.alertFisher_hW, node.Fisher_hW);
//            weightList.add()
            if (node.depth <= max_index) {
                if (sim < init_sim) {
                    branchNode = node;
                    init_sim = sim;
                }
                if (sim < min_sim) {
                    branchNode = node;
                }
            }
        }

        Node parent = branchNode.parent;
        branchNode.isShare = true;

        while (parent != null) {
            parent.isShare = true;
            parent = parent.parent;
        }

        return branchNode;
    }


    private void add_branch() {
        Node parent = weight_sim();
        branchNum = branchNum + 1;
        branchList.add(branchNum);
        activeBranch = branchNum;
        add_child_node(parent, branchNum, 0.0);
//        parent = parent.childList.get(parent.childList.size() - 1);
        if (branchList.size() > maxBranchNum) {
            int delBranch = branchList.get(0);
            branchList = branchList.subList(1, branchList.size());
            del_branch(model, delBranch);
            lossStatisticsList.remove(delBranch);
            lossList.remove(delBranch);
            nodeList.remove(delBranch);
        }
    }


    private void del_branch(Node node, int branchType) {
        for (Node child : node.childList) {
            del_branch(child, branchType);
        }
        if (node.branchType == branchType) {
            node.parent.childList.remove(node);
        }
    }

}

public class Main {
    public static void main(String[] args) {
        int featureNum = 10, hNeuronNum = 256, cNeuronNum = 3;
        Model model = new Model(featureNum, hNeuronNum, cNeuronNum);

        // Mock training data
        List<RealMatrix> features = Arrays.asList(Utils.createRandomMatrix(1, featureNum));
        List<RealMatrix> labels = Arrays.asList(MatrixUtils.createColumnRealMatrix(new double[] { 0, 1, 0 }));

        model.trainModel(features, labels);

        System.out.println("Training complete.");
    }
}
