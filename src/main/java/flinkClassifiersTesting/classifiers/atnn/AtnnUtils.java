package flinkClassifiersTesting.classifiers.atnn;

import java.util.*;
import java.nio.*;
import java.util.stream.Collectors;
//import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.math3.linear.*;


import java.io.FileWriter;
import java.io.IOException;

import static flinkClassifiersTesting.classifiers.atnn.AtnnUtils.matrixSim;
import static flinkClassifiersTesting.classifiers.atnn.AtnnUtils.softmax;
import static org.apache.commons.math3.linear.MatrixUtils.*;

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
    public static RealMatrix standardization(RealMatrix matrix) { // todo czy to dobry kod w ogóle? to czata
        double mean = Arrays.stream(matrix.getData()).flatMapToDouble(Arrays::stream).average().orElse(0);
        double std = Math.sqrt(Arrays.stream(matrix.getData())
                .flatMapToDouble(Arrays::stream)
                .map(x -> (x - mean) * (x - mean))
                .average().orElse(0));
        double epsilon = 1e-7;
        return matrix.scalarAdd(-mean).scalarMultiply(1.0 / (std + epsilon));
    }



    public static double calculateStandardDeviation(List<Double> numbers) {
        // Sprawdzenie, czy lista nie jest pusta
        if (numbers == null || numbers.isEmpty()) {
            throw new IllegalArgumentException("Lista nie może być pusta.");
        }

        // Obliczanie średniej
        double mean = numbers.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        // Obliczanie sumy kwadratów różnic od średniej
        double variance = numbers.stream()
                .mapToDouble(num -> Math.pow(num - mean, 2))
                .average()
                .orElse(0.0);

        // Zwracanie pierwiastka kwadratowego z wariancji (odchylenie standardowe)
        return Math.sqrt(variance);
    }


    // Standardizes a matrix
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
    public static RealVector softmax(RealVector x) {
        double[] expData = Arrays.stream(x.toArray()).map(Math::exp).toArray();
        double sumExp = Arrays.stream(expData).sum();
        double[] result = Arrays.stream(expData).map(e -> e / sumExp).toArray();
        return createRealVector(result);
    }

    // Cross entropy loss todo czy to dobry wzor? czat zrobil
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
        // Tworzenie nowej macierzy z tą samą zawartością
        RealMatrix result = matrix.copy();

        // Podnoszenie do kwadratu każdego elementu
        result.walkInOptimizedOrder(new org.apache.commons.math3.linear.RealMatrixChangingVisitor() {
            @Override
            public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {
                // Nie wymaga inicjalizacji
            }

            @Override
            public double visit(int row, int column, double value) {
                return value * value; // Podnoszenie elementu do kwadratu
            }

            @Override
            public double end() {
                return 0; // Nie wymaga dodatkowych obliczeń
            }
        });

        return result;
    }

    public static RealVector squareEachElement(RealVector vector) {
        // Tworzenie kopii oryginalnego wektora
        RealVector result = vector.copy();

        // Podnoszenie każdego elementu do kwadratu
        result.mapToSelf(x -> x * x);

        return result;
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
        RealVector dev_cInput = null;
        RealVector dev_hInput = null;
        double weight = 0;
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
        public RealVector relu(RealVector x) {
            double[] data = x.toArray();
            for (int i = 0; i < data.length; i++) {
                data[i] = Math.max(0, data[i]);
            }

            return AtnnUtils.standardization(createRealVector(data));
            return AtnnUtils.standardization(createRealMatrix(data));
        }

        // Derivative of ReLU
        public RealVector dev_relu(RealVector x) {
            double[] data = x.toArray();
            for (int i = 0; i < data.length; i++) {
                data[i] = data[i] > 0 ? 1 : 0;
            }
            return createRealVector(data);
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
        boolean driftAlert = false;
        int alertNum = 0;
        int lastDriftTime = 0;
        int lamda = 5000;
        String dataSet = null;
        int confid = 3;
        int trainType = 0;

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

        private void forward_propagation(Node node, RealVector feature) {
            if (node.isRootNode) {
                node.hideInput = feature;
                node.hideOutput = feature;
                node.classifierInput = node.cW.operate(node.hideOutput).add(node.cb);
                node.classifierOutput = softmax(node.classifierInput);
            } else {
                node.hideInput = node.hW.operate(feature).add(node.hb);
                node.hideOutput = node.relu(node.hideInput);

                node.classifierInput = node.cW.operate(node.hideOutput).add(node.cb);
                node.classifierOutput = softmax(node.classifierInput);
            }

            for (Node child : node.childList) {
                if (!driftAlert && child.branchType != activeBranch && child.branchType != 0)
                    continue;
                forward_propagation(child, node.hideOutput);
            }
        }

        private void back_propagation(Node node, RealVector trueLabel) {
            for (Node child : node.childList) {
                if (child.branchType == 0 || child.branchType == activeBranch) {
                    back_propagation(child, trueLabel);
                }
            }

            node.lastPredictLoss = AtnnUtils.cross_entropy(node.classifierOutput, trueLabel);
            node.dev_cInput = node.classifierOutput.subtract(trueLabel).mapMultiply(node.weight);
            node.dev_cW = node.dev_cInput.outerProduct(node.hideOutput);

            if (node.isRootNode) {
                return;
            }

            node.dev_hInput = node.cW.transpose().operate(node.dev_cInput).ebeMultiply(node.dev_relu(node.hideInput)); // todo chyba zmyśliłem tę operację ebe, nie wiem, jaka powinna być
            node.dev_hW = node.dev_hInput.outerProduct(node.parent.hideOutput);

            if (driftAlert && node.branchType == 0) {
                // chodzi tylko o znalezienie element wise maximum
                int rows = node.dev_hW.getRowDimension();
                int cols = node.dev_hW.getColumnDimension();
                double[][] node_dev_hW_data = node.dev_hW.getData();
                double[][] elementWiseMaximum = new double[rows][cols];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        elementWiseMaximum[i][j] = Math.max(node_dev_hW_data[i][j], node_dev_hW_data[i][j]);
                    }
                }
                RealMatrix maxMatrix = new Array2DRowRealMatrix(elementWiseMaximum);

                node.alertSquareGrad_hW = node.alertSquareGrad_hW.add(maxMatrix);
                node.alertSquareGrad_hb = node.alertSquareGrad_hb.add(node.dev_hInput);
            }


            if (node.branchType != activeBranch) {
                node.dev_cW = node.dev_cW.scalarMultiply(0);
                node.dev_hInput = node.dev_hInput.mapMultiply(0);
                node.dev_hW = node.dev_hW.scalarMultiply(0);
            }

            for (Node child : node.childList) {
                if (child.branchType != 0 && child.branchType != activeBranch)
                    continue;
                RealVector child_dev_hInput = child.hW.transpose().operate(child.dev_hInput).ebeMultiply(node.dev_relu(node.hideInput)); // todo znowu nie jestem pewien operacji ebe
                node.dev_hInput = node.dev_hInput.add(child_dev_hInput);
            }

            node.dev_hW = node.dev_hInput.outerProduct(node.parent.hideOutput);

            if (node.branchType == 0 && node.trainTimes > 2000) {  // todo nie jestem pewien preMultiply na dole, to chyba to samo co ebe multiply w tutaj w dwoch nizej
                node.squareGrad_hW = node.squareGrad_hW.add(AtnnUtils.squareEachElement(node.dev_hW.add((node.Fisher_hW.preMultiply(node.hW.subtract(node.lastConcept_hW))).scalarMultiply(lamda))));
                node.squareGrad_hb = node.squareGrad_hb.add(AtnnUtils.squareEachElement(node.dev_hInput.add(node.Fisher_hb.ebeMultiply(node.hb.subtract(node.lastConcept_hb)))));
            }
        }

        private void update_model(Node node) {

            for (Node child : node.childList) {
                if (child.branchType == 0 || child.branchType == activeBranch) {
                    update_model(child);
                }
            }

            node.trainTimes = node.trainTimes + 1;
            double lr = node.learnRate;

            if (node.isRootNode) {
                node.cW = node.cW.subtract(node.dev_cW.scalarMultiply(lr));
                node.cb = node.cb.subtract(node.dev_cInput.mapMultiply(lr));
            } else {
                if (node.isShare) {
                    node.hW = node.hW.subtract(node.dev_hW.add((node.Fisher_hW.preMultiply(node.hW.subtract(node.lastConcept_hW))).scalarMultiply(lamda)).scalarMultiply(lr)); // todo tak samo nie jestem pewien preMultiply
                    node.cW = node.cW.subtract(node.dev_cW.scalarMultiply(lr));
                    node.hb = node.hb.subtract(node.dev_hInput.add(node.Fisher_hb.ebeMultiply(node.hb.subtract(node.lastConcept_hb))).mapMultiply(lr));
                    node.cb = node.cb.subtract(node.dev_cInput.mapMultiply(lr));
                } else {
                    node.hW = node.hW.subtract(node.dev_hW.scalarMultiply(lr));
                    node.cW = node.cW.subtract(node.dev_cW.scalarMultiply(lr));
                    node.hb = node.hb.subtract(node.dev_hInput.mapMultiply(lr));
                    node.cb = node.cb.subtract(node.dev_cInput.mapMultiply(lr));
                }
            }

            if (node.learnRate > node.minLR) {
                node.learnRate = node.learnRate * node.reduction;
            }

            node.dev_hW = null;
            node.dev_cW = null;
            node.dev_hInput = null;
            node.dev_cInput = null;
        }


        private String get_model_dict_structure(Node node) {
            String nodeName = "" + node.branchType + "-" + node.weight + "-" + node.isShare;
            if (node.childList.isEmpty()) {
                return nodeName;
            } else {
                return nodeName; // todo poprawić, w pythonie było inaczej
            }
        }

        private Map<Integer, RealVector> get_model_output() {
            Map<Integer, RealVector> modelOutput = new HashMap<>();
            for (Map.Entry<Integer, List<Node>> entry : nodeList.entrySet()) {
                int branch = entry.getKey();
                List<Node> nodeList = entry.getValue();

                RealVector branchOutput = new ArrayRealVector(nodeList.size());
                for (Node node : nodeList) {
                    RealVector output = node.classifierOutput;
                    branchOutput.add(output.mapMultiply(node.weight));
                }

                double sumOfBranchOutput = Arrays.stream(branchOutput.toArray()).sum();

                modelOutput.put(branch, branchOutput.mapMultiply(1 / sumOfBranchOutput));
            }
            return modelOutput;
        }

        private void update_branch_predict_loss(Map<Integer, RealVector> modelOutput, RealVector label) {
            for (Map.Entry<Integer, RealVector> entry : modelOutput.entrySet()) {
                int branch = entry.getKey();
                RealVector branchOutput = entry.getValue();
                if (!lossList.containsKey(branch)) {
                    lossList.put(branch, new ArrayList<>());
                }
                lossList.get(branch).add(cross_entropy(branchOutput, label));
                if (lossList.get(branch).size() > lossLen) {
                    lossList.put(branch, lossList.get(branch).subList(lossList.get(branch).size() - lossLen, lossList.get(branch).size()));
                }
            }
        }

        private void update_loss_statistics() {
            for (Map.Entry<Integer, List<Double>> entry : lossList.entrySet()) {
                int branch = entry.getKey();
                List<Double> losslist = entry.getValue();

                if (losslist.size() < lossLen) {
                    continue;
                }

                double mean = losslist.stream().reduce(0.0, Double::sum) / losslist.size();
                double variance = calculateStandardDeviation(losslist);
                List<Double> prev_loss = losslist.subList(splitLen, losslist.size()).stream().sorted().collect(Collectors.toList()).subList(0, losslist.size()-5); // todo czy to sie nie wywali przy -5?
                double prev_mean = prev_loss.stream().reduce(0.0, Double::sum) / prev_loss.size();
                double prev_var = calculateStandardDeviation(prev_loss);

                if (!lossStatisticsList.containsKey(branch)) {
                    Map<String, Double> dict = new HashMap<>();
                    dict.put("mean", mean);
                    dict.put("var", variance);
                    dict.put("prev_mean", prev_mean);
                    dict.put("prev_var", prev_var);
                    lossStatisticsList.put(branch, dict);
                } else {
                    if ((mean + variance) < lossStatisticsList.get(branch).get("mean") + lossStatisticsList.get(branch).get("var")) {
                        lossStatisticsList.get(branch).put("mean", mean);
                        lossStatisticsList.get(branch).put("var", variance);
                    }
                    lossStatisticsList.get(branch).put("prev_mean", prev_mean);
                    lossStatisticsList.get(branch).put("prev_var", prev_var);
                }
            }

            if (lossStatisticsList.containsKey(activeBranch)) {
                double d = lossStatisticsList.get(activeBranch).get("mean") + confid + lossStatisticsList.get(activeBranch).get("var");
                List<Double> lossWin = lossList.get(activeBranch).subList(lossList.get(activeBranch).size()-5, lossList.get(activeBranch).size()); // todo czy to sie nie wywali
                double min = Double.MAX_VALUE;
                for (Double aDouble : lossWin) {
                    if (aDouble < min) {
                        min = aDouble;
                    }
                }
                double driftFlag = min - d;
                if (!driftAlert) {
                    if (driftFlag > 0) {
                        driftAlert = true;
                    } else {
                        lossWin = lossList.get(activeBranch).subList(lossList.get(activeBranch).size() - splitLen, lossList.get(activeBranch).size());
                        List<Double> withoutLast5 = lossWin.subList(0, lossWin.size()-5);
                        double mean = withoutLast5.stream().reduce(0.0, Double::sum) / withoutLast5.size();
                        double variance = calculateStandardDeviation(withoutLast5);
                        if (mean + variance > d)
                            driftAlert = true;
                    }
                }

                if (driftAlert) {
                    alertNum = alertNum + 1;
                }
            }
        }

        private void reset_weight() {
            List<Node> nodes = get_active_node_list();
            for (Node node : nodes) {
                node.weight = 1.0 / nodes.size();
            }
        }


        private void conceptDetection() {
            if (!lossStatisticsList.containsKey(activeBranch)) {
                return;
            }

            int minLossBranch = lossStatisticsList.keySet().stream().min((k1, k2) -> lossStatisticsList.get(k1).get("prev_mean").compareTo(lossStatisticsList.get(k2).get("prev_mean"))).orElse(0); // todo nie wiem czy dobry default
            Map<String, Double> branchLoss = lossStatisticsList.get(minLossBranch);
            if (branchLoss.get("prev_var") + branchLoss.get("prev_mean") > branchLoss.get("mean") + confid * branchLoss.get("var")) {
                update_fisherMatrix();
                add_branch();
                trainType = 0;
                lastDriftTime = trainTimes;
                reset_weight();
            } else {
                if (minLossBranch != activeBranch) {
                    activeBranch = minLossBranch;
                }
            }

            driftAlert = false;
            alertNum = 0;
            for (Node node : nodeList.get(0)) {
                node.alertSquareGrad_hW = node.alertSquareGrad_hW.scalarMultiply(0);
                node.alertSquareGrad_hb = node.alertSquareGrad_hb.mapMultiply(0);
            }
        }


        private void update_fisherMatrix() {
            for (Node node : nodeList.get(0)) {
                node.lastConcept_hb = node.hb;
                node.lastConcept_hW = node.hW;
                node.Fisher_hW = node.squareGrad_hW.scalarMultiply((double) 1 /trainTimes);
                node.Fisher_hb = node.squareGrad_hb.mapMultiply((double) 1 /trainTimes);
            }
        }


        private RealVector train_model(RealVector feature, RealVector label) {
            Node model1 = model;
            forward_propagation(model1, feature);
            Map<Integer, RealVector> modelOutput = get_model_output();
            RealVector result = modelOutput.get(activeBranch);
            update_branch_predict_loss(modelOutput, label);
            update_loss_statistics();
            back_propagation(model1, label);
            update_model(model1);
            List<Node> activeNodeList = get_active_node_list();
            update_weight_by_loss(activeNodeList, label);
            trainTimes = trainTimes + 1;
            model_grow_and_prune();
            if (alertNum == splitLen)
                conceptDetection();
            return result;
        }


        private void model_grow_and_prune() {
            List<Node> activeNodeList = get_active_node_list();
            List<Double> wlist = activeNodeList.stream().map(node -> node.weight).collect(Collectors.toList());
            Node maxWeightNode = activeNodeList.stream().max(Comparator.comparingDouble(n -> n.weight)).orElse(null); // todo czy tu sie nie wywali
            assert maxWeightNode != null;
            if (maxWeightNode.childList.isEmpty()) {
                add_child_node(maxWeightNode, maxWeightNode.branchType, 1.0 / (activeNodeList.size() + 1));
            }
        }


        private void update_weight_by_loss(List<Node> nodeList, RealVector label) {
            double zt = 0;
            List<Double> losses = new ArrayList<>();
            for (Node node : nodeList) {
                losses.add(node.lastPredictLoss);
            }
            double M = losses.stream().reduce(0.0, Double::sum);
            losses = losses.stream().map(loss -> loss / M).collect(Collectors.toList());
            double min_loss = losses.stream().reduce(0.0, Double::min);
            double max_loss = losses.stream().reduce(0.0, Double::max);
            double range_of_loss = (max_loss - min_loss) + 1e-7;
            losses = losses.stream().map(loss -> (loss - min_loss) / range_of_loss).collect(Collectors.toList());

            for (int i = 0; i < nodeList.size(); i++) {
                double newWeight = nodeList.get(i).weight * (Math.pow(beta, losses.get(i)));
                if (newWeight < smooth / nodeList.size()) {
                    newWeight = smooth / nodeList.size();
                }
                nodeList.get(i).weight = newWeight;
                zt = zt + newWeight;
            }

            for (Node node : nodeList) {
                node.weight = node.weight / zt;
            }
        }

    }

}


public class OnlineLearning {

    public static void main(String[] args) {
        List<String> dataSetList = List.of("RBF2_0");
        for (String dataSet : dataSetList) {
            System.out.println("-----------------------------");
            System.out.println("start train : " + dataSet);
            onlineLearning(dataSet);
        }
    }

    public static void onlineLearning(String dataSet) {
        String dataSetName = dataSet;
        String fileName = "/datasets/" + dataSetName + ".csv";

        Data data = DataLoader.load(fileName);
        double[][] x_train = data.getXTrain();
        double[][] y_train = data.getYTrain();

        int statisticsLen = 100;
        int featureLen = x_train[0].length;
        int hNeuronNum = 256;
        int classNum = y_train[0].length;

        // Inicjalizacja modelu
        Model model = new Model(featureLen, hNeuronNum, classNum);
        model.initNodeWeight();

        int predictRightNumber = 0;
        int exampleNumber = 0;
        Map<Integer, Map<String, Double>> resultList = new HashMap<>();
        List<Boolean> blockResultList = new ArrayList<>();

        for (int i = 0; i < x_train.length; i++) {
            exampleNumber++;
            double[][] feature = transpose(new double[][]{x_train[i]});
            double[][] label = transpose(new double[][]{y_train[i]});
            double[][] predictResult = model.trainModel(feature, label);

            blockResultList.add(isMaxValueIndexEqual(predictResult, label));

            if (blockResultList.size() == statisticsLen || exampleNumber == x_train.length) {
                double predictRightRate = blockResultList.stream().filter(b -> b).count() / (double) blockResultList.size();
                predictRightNumber += (int) blockResultList.stream().filter(b -> b).count();

                System.out.println("*************************************************************************");
                System.out.println("size: " + exampleNumber);
                System.out.println("realtime: " + predictRightRate);
                System.out.println("cumulative: " + ((double) predictRightNumber / exampleNumber));

                Map<String, Double> stats = new HashMap<>();
                stats.put("realtime", predictRightRate);
                stats.put("cumulative", (double) predictRightNumber / exampleNumber);
                resultList.put(exampleNumber, stats);

                blockResultList.clear();
            }
        }

        saveData(resultList, dataSet + ".json");
    }

    // Transponowanie macierzy
    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    // Funkcja do porównania indeksów maksymalnych wartości
    public static boolean isMaxValueIndexEqual(double[][] predictResult, double[][] label) {
        int predictMaxIndex = getMaxIndex(predictResult);
        int labelMaxIndex = getMaxIndex(label);
        return predictMaxIndex == labelMaxIndex;
    }

    // Znajdowanie indeksu maksymalnej wartości w jednowymiarowej macierzy
    public static int getMaxIndex(double[][] vector) {
        double max = vector[0][0];
        int index = 0;
        for (int i = 1; i < vector.length; i++) {
            if (vector[i][0] > max) {
                max = vector[i][0];
                index = i;
            }
        }
        return index;
    }

    // Zapisywanie danych w formacie JSON
    public static void saveData(Map<Integer, Map<String, Double>> data, String fileName) {
        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write(data.toString());
        } catch (IOException e) {
            System.err.println("Error while saving data: " + e.getMessage());
        }
    }
}

// Przykladowa klasa Data - powinna być dostosowana do implementacji DataLoader
class Data {
    private final double[][] xTrain;
    private final double[][] yTrain;

    public Data(double[][] xTrain, double[][] yTrain) {
        this.xTrain = xTrain;
        this.yTrain = yTrain;
    }

    public double[][] getXTrain() {
        return xTrain;
    }

    public double[][] getYTrain() {
        return yTrain;
    }
}
