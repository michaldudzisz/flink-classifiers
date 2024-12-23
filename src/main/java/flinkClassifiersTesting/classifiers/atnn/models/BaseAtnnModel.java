package flinkClassifiersTesting.classifiers.atnn.models;

import java.util.*;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.*;

import static flinkClassifiersTesting.classifiers.atnn.models.AtnnUtils.*;

public class BaseAtnnModel {
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
    int lossLen = 500; // było 1000, w artykule napisali o 500
    int splitLen = 50;
    Map<Integer, Map<String, Double>> lossStatisticsList = new HashMap<>();
    List<Integer> branchList = new ArrayList<>();
    boolean driftAlert = false;
    int alertNum = 0;
    int lastDriftTime = 0;
    int lamda = 5000;
    String dataSet = null;
    int confid = 3;

    /*
        Drift reporting:
    */
    final String DRIFT_STATUS_NO_DRIFT = "0";
    final String DRIFT_STATUS_WARN = "warn";
    final String DRIFT_STATUS_ELIMINATED = "eliminated";
    final String DRIFT_STATUS_CURRENT_EVOLVING = "current_evolving";
    final String DRIFT_STATUS_NEW_DETECTED = "new_detected";
    final String DRIFT_STATUS_RECURRING = "recurring";
    String driftStatus = DRIFT_STATUS_NO_DRIFT;

    public BaseAtnnModel(int featureNum, int hNeuronNum, int cNeuronNum) {
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
            add_empty_child_node(parent, branchType, 0.0);
            parent = parent.childList.get(parent.childList.size() - 1);
        }

        activeBranch = branchType;
        init_node_weight();
    }

    protected void add_empty_child_node(Node parentNode, int branchType, double weight) {
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

    public void init_node_weight() {
        activeNodeList = get_active_node_list();
        double avgWeight = 1.0 / activeNodeList.size();
        for (Node node : activeNodeList) {
            node.weight = avgWeight;
            if (Double.isNaN(avgWeight)) {
                throw new RuntimeException("weight: " + avgWeight);
            }
        }
    }

    protected List<Node> get_active_node_list() {
        return nodeList.get(activeBranch);
    }

    protected Node weight_sim() {
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

        for (Node node : nodeList.get(0).subList(1, nodeList.get(0).size())) {
            node.alertFisher_hW = node.alertSquareGrad_hW.scalarMultiply((double) 1 / alertNum);
            node.alertFisher_hb = node.alertSquareGrad_hb.mapMultiply((double) 1 / alertNum);
//            if (node.alertFisher_hW.getRowDimension() != 256 || node.alertFisher_hW.getColumnDimension() != 256)
//                throw new IllegalArgumentException("rows: " + node.alertFisher_hW.getRowDimension() + ", cols: " + node.alertFisher_hW.getColumnDimension());
            double sim = matrixCosineSimilarity(node.alertFisher_hW, node.Fisher_hW);
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


    protected void add_empty_branch() {
        Node parent = weight_sim();
        branchNum = branchNum + 1;
        branchList.add(branchNum);
        activeBranch = branchNum;
        add_empty_child_node(parent, branchNum, 0.0);
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


    protected void del_branch(Node node, int branchType) {
        for (Node child : node.childList) {
            del_branch(child, branchType);
        }
        if (node.branchType == branchType) {
            node.parent.childList.remove(node);
        }
    }

    protected void forward_propagation(Node node, RealVector feature) {
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

    protected boolean should_back_propagate_node(Node node) {
        return node.branchType == 0 || node.branchType == activeBranch;
    }

    protected void back_propagation(Node node, RealVector trueLabel) {
        for (Node child : node.childList) {
            if (should_back_propagate_node(child)) {
                back_propagation(child, trueLabel);
            }
        }

        node.lastPredictLoss = AtnnUtils.cross_entropy(node.classifierOutput, trueLabel);
        node.dev_cInput = node.classifierOutput.subtract(trueLabel).mapMultiply(node.weight);
        node.dev_cW = node.dev_cInput.outerProduct(node.hideOutput);

        if (node.isRootNode)
            return;

        node.dev_hInput = node.cW.transpose().operate(node.dev_cInput).ebeMultiply(node.dev_relu(node.hideInput));

        node.dev_hW = node.dev_hInput.outerProduct(node.parent.hideOutput);

        if (driftAlert && node.branchType == 0) {
            RealMatrix maxMatrix = getElementWiseMaximum(node.dev_hW, node.dev_hW.scalarMultiply(-1)); // todo dec 18 dodałem minus
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
            RealVector child_dev_hInput = child.hW.transpose().operate(child.dev_hInput).ebeMultiply(node.dev_relu(node.hideInput));
            node.dev_hInput = node.dev_hInput.add(child_dev_hInput);
        }

        node.dev_hW = node.dev_hInput.outerProduct(node.parent.hideOutput);

        if (node.branchType == 0 && node.trainTimes > 2000) {
            node.squareGrad_hW = node.squareGrad_hW.add(squareEachElement(node.dev_hW.add(ebeMultiply(node.Fisher_hW, node.hW.subtract(node.lastConcept_hW)).scalarMultiply(lamda))));
            node.squareGrad_hb = node.squareGrad_hb.add(squareEachElement(node.dev_hInput.add(node.Fisher_hb.ebeMultiply(node.hb.subtract(node.lastConcept_hb)).mapMultiply(lamda))));
        }
    }

    protected void update_model(Node node) {

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
                node.hW = node.hW.subtract(node.dev_hW.add((ebeMultiply(node.Fisher_hW, node.hW.subtract(node.lastConcept_hW))).scalarMultiply(lamda)).scalarMultiply(lr));
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

        // todo dodane na czas sprawdzenia
//        if (trainTimes == 9_000 || trainTimes == 12_000 || trainTimes == 15_000 && (node.depth == 2 || node.depth == 3)) {
//            if (node.depth == 2) {
//                System.out.println("zwiekszam lr dla node.depth " + node.depth);
//            }
//            if (node.depth == 3) {
//                System.out.println("zwiekszam lr dla node.depth " + node.depth);
//            }
//
//            node.learnRate = 0.02;
//        }

        node.dev_hW = null;
        node.dev_cW = null;
        node.dev_hInput = null;
        node.dev_cInput = null;
    }

    protected String get_model_dict_structure(Node node) {
        String nodeName = "" + node.branchType + "-" + node.weight + "-" + node.isShare;
        if (node.childList.isEmpty()) {
            return nodeName;
        } else {
            return nodeName; // gdyby było potrzebne, to poprawić, w pythonie (kodzie dołączonym do artykułu) było inaczej
        }
    }

    protected Map<Integer, RealVector> get_model_output() {
        Map<Integer, RealVector> modelOutput = new HashMap<>();
        for (Map.Entry<Integer, List<Node>> entry : nodeList.entrySet()) {
            int branch = entry.getKey();
            List<Node> nodeList = entry.getValue();

            RealVector branchOutput = new ArrayRealVector(cNeuronNum);
            for (Node node : nodeList) {
                RealVector output = node.classifierOutput;
                branchOutput = branchOutput.add(output.mapMultiply(node.weight));
            }

            double sumOfBranchOutput = Arrays.stream(branchOutput.toArray()).sum();

            modelOutput.put(branch, branchOutput.mapMultiply(1 / sumOfBranchOutput));
        }
        return modelOutput;
    }

    protected void update_branch_predict_loss(Map<Integer, RealVector> modelOutput, RealVector label) {
        for (Map.Entry<Integer, RealVector> entry : modelOutput.entrySet()) {
            int branch = entry.getKey();
            RealVector branchOutput = entry.getValue();
            if (!lossList.containsKey(branch)) {
                lossList.put(branch, new ArrayList<>());
            }
            lossList.get(branch).add(AtnnUtils.cross_entropy(branchOutput, label));
            if (lossList.get(branch).size() > lossLen) {
                lossList.put(branch, lossList.get(branch).subList(lossList.get(branch).size() - lossLen, lossList.get(branch).size()));
            }
        }
    }

    protected void update_loss_statistics() {
        for (Map.Entry<Integer, List<Double>> entry : lossList.entrySet()) {
            int branch = entry.getKey();
            List<Double> losslist = entry.getValue();

            if (losslist.size() < lossLen) {
                continue;
            }

            double mean = losslist.stream().reduce(0.0, Double::sum) / losslist.size();
            double variance = AtnnUtils.calculateStandardDeviation(losslist);

            List<Double> prev_loss = losslist.subList(losslist.size() - splitLen, losslist.size());
            double prev_mean = prev_loss.stream().reduce(0.0, Double::sum) / prev_loss.size();
            double prev_var = AtnnUtils.calculateStandardDeviation(prev_loss);

            if (!lossStatisticsList.containsKey(branch)) {
                Map<String, Double> dict = new HashMap<>();
                dict.put("mean", mean);
                dict.put("var", variance);
                dict.put("prev_mean", prev_mean);
                dict.put("prev_var", prev_var);
                lossStatisticsList.put(branch, dict);
            } else {
                if ((mean + variance) < lossStatisticsList.get(branch).get("mean") + lossStatisticsList.get(branch).get("var")) { // todo to jest min
                    lossStatisticsList.get(branch).put("mean", mean);
                    lossStatisticsList.get(branch).put("var", variance);
                }
                lossStatisticsList.get(branch).put("prev_mean", prev_mean);
                lossStatisticsList.get(branch).put("prev_var", prev_var);
            }
        }

        if (lossStatisticsList.containsKey(activeBranch)) {
            driftAlertDetection();
        }

        if (!driftAlert) {
            driftStatus = DRIFT_STATUS_NO_DRIFT;
        }
    }

    protected void driftAlertDetection() {
        double driftWarnLevel = lossStatisticsList.get(activeBranch).get("mean") + 2 * lossStatisticsList.get(activeBranch).get("var"); // todo zmieniłem confid na 2, żeby było jak w wartykule
        List<Double> lossWin = lossList.get(activeBranch).subList(lossList.get(activeBranch).size() - splitLen, lossList.get(activeBranch).size());

        double activeBranchRecentMean = lossWin.stream().reduce(0.0, Double::sum) / lossWin.size();
        double activeBranchRecentVar = AtnnUtils.calculateStandardDeviation(lossWin);

        if (!driftAlert) {
            if (activeBranchRecentMean + activeBranchRecentVar > driftWarnLevel) { // todo zmyśliłem 10_000, żeby tego nigdy nie było
                driftAlert = true;
                driftStatus = DRIFT_STATUS_WARN;
            }
        } else {
            if (activeBranchRecentMean + activeBranchRecentVar < driftWarnLevel) {
                driftAlert = false;
                alertNum = 0;
                driftStatus = DRIFT_STATUS_ELIMINATED;
            }
        }

        if (driftAlert) {
            alertNum = alertNum + 1;
        }
    }

    protected void reset_weight() {
        List<Node> nodes = get_active_node_list();
        for (Node node : nodes) {
            node.weight = 1.0 / nodes.size();
        }
    }

    protected void conceptDetection() {
        if (!lossStatisticsList.containsKey(activeBranch)) {
            return;
        }

        int minLossBranch = lossStatisticsList.keySet().stream().min((k1, k2) -> lossStatisticsList.get(k1).get("prev_mean").compareTo(lossStatisticsList.get(k2).get("prev_mean"))).orElse(0); // todo nie wiem czy dobry default
        Map<String, Double> branchLoss = lossStatisticsList.get(minLossBranch);
        if (branchLoss.get("prev_var") + branchLoss.get("prev_mean") > branchLoss.get("mean") + confid * branchLoss.get("var")) {
            driftStatus = DRIFT_STATUS_NEW_DETECTED;
            update_fisherMatrix();
            add_empty_branch();
            lastDriftTime = trainTimes;
            reset_weight();
        } else {
            if (minLossBranch != activeBranch) {
                driftStatus = DRIFT_STATUS_RECURRING;
                activeBranch = minLossBranch;
            } else {
                driftStatus = DRIFT_STATUS_CURRENT_EVOLVING;
            }
        }

        driftAlert = false;
        alertNum = 0;
        for (Node node : nodeList.get(0)) {
            node.alertSquareGrad_hW = node.alertSquareGrad_hW.scalarMultiply(0);
            node.alertSquareGrad_hb = node.alertSquareGrad_hb.mapMultiply(0);
        }
    }

    protected void update_fisherMatrix() {
        for (Node node : nodeList.get(0)) {
            node.lastConcept_hb = node.hb;
            node.lastConcept_hW = node.hW;
            node.Fisher_hW = node.squareGrad_hW.scalarMultiply((double) 1 / trainTimes);
            node.Fisher_hb = node.squareGrad_hb.mapMultiply((double) 1 / trainTimes);
        }
    }

    public RealVector predict(RealVector feature) {
        forward_propagation(model, feature);
        Map<Integer, RealVector> modelOutput = get_model_output();
        return modelOutput.get(activeBranch);
    }

    public BranchesInfo getBranchesInfo() {
        return new BranchesInfo(
                branchList.size() + 1, // we add 1 because trunk is not considered as a separate branch in model code
                activeBranch,
                get_active_node_list().size(), // to zwróci tylko od miejsca złączenia z trunkiem
                driftStatus
        );
    }

    protected void printModelStructure() {
        System.out.println("\n*** Model structure ***\n");

        for (Node node : activeNodeList) {
            System.out.println("\nNode:\nnode.branchType: " + node.branchType + ", node.depth: " + node.depth);
            System.out.println("hW: ");
            System.out.println(node.hW);
            System.out.println("hb: ");
            System.out.println(node.hb);
            System.out.println("cW: ");
            System.out.println(node.cW);
            System.out.println("cb: ");
            System.out.println(node.cb);
            System.out.println("\n");
        }
        System.out.println("\n*** Model structure end ***\n");
    }

    public void train_model(RealVector feature, RealVector label) {
        forward_propagation(model, feature);
        Map<Integer, RealVector> modelOutput = get_model_output();
        RealVector result = modelOutput.get(activeBranch);
        update_branch_predict_loss(modelOutput, label);
        update_loss_statistics();
        back_propagation(model, label);
        update_model(model);
        List<Node> activeNodeList = get_active_node_list();
        update_weight_by_loss(activeNodeList, label);
        trainTimes = trainTimes + 1;
        model_grow_and_prune();
        if (alertNum == splitLen)
            conceptDetection();
        return;
    }


    protected void model_grow_and_prune() {
        List<Node> activeNodeList = get_active_node_list();
        List<Double> wlist = activeNodeList.stream().map(node -> node.weight).collect(Collectors.toList());
        Node maxWeightNode = activeNodeList.stream().max(Comparator.comparingDouble(n -> n.weight)).orElse(null); // todo czy tu sie nie wywali
        assert maxWeightNode != null;
        if (maxWeightNode.childList.isEmpty()) {
            add_empty_child_node(maxWeightNode, maxWeightNode.branchType, 1.0 / (activeNodeList.size() + 1));
        }
    }


    protected void update_weight_by_loss(List<Node> nodeList, RealVector label) {
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
            if (Double.isNaN(node.weight)) {
                throw new RuntimeException("node.weight: " + node.weight);
            }
        }
    }

}

