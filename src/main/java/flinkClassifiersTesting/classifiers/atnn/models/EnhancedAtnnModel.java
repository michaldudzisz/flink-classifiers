package flinkClassifiersTesting.classifiers.atnn.models;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EnhancedAtnnModel extends BaseAtnnModel {

    BranchesToTrainDuringDriftAlert branchesToTrainDuringDriftAlert = new BranchesToTrainDuringDriftAlert();

    public EnhancedAtnnModel(int featureNum, int hNeuronNum, int cNeuronNum) {
        super(featureNum, hNeuronNum, cNeuronNum);
    }

    @Override
    protected boolean should_back_propagate_node(Node node) {
        return (node.branchType == 0 || node.branchType == activeBranch) // normal path
                || (node.hasBeenForwarded && (node.branchType == branchesToTrainDuringDriftAlert.empty || node.branchType == branchesToTrainDuringDriftAlert.cloned));
    }

    @Override
    protected boolean should_forward_propagate_node(Node node) {
        return driftAlert
                || node.branchType == activeBranch
                || node.branchType == 0;
    }

    protected void choose_among_drift_branches() {
        int emptyBranch = branchesToTrainDuringDriftAlert.empty;
        double emptyBranchLoss = lossStatisticsList.get(emptyBranch).get("prev_mean");

        int clonedBranch = branchesToTrainDuringDriftAlert.cloned;
        double clonedBranchLoss = lossStatisticsList.get(clonedBranch).get("prev_mean");

        if (emptyBranchLoss < clonedBranchLoss) {
            activeBranch = emptyBranch;
        } else {
            activeBranch = clonedBranch;
        }
    }

    protected void removeBranch(int branch) {
        branchList.remove((Integer) branch);
        del_branch(model, branch);
        lossStatisticsList.remove(branch);
        lossList.remove(branch);
        nodeList.remove(branch);
        System.out.println("Removed branch: " + branch);
    }

    protected void clean_drift_alert_branches() {
        int emptyBranch = branchesToTrainDuringDriftAlert.empty;
        int clonedBranch = branchesToTrainDuringDriftAlert.cloned;

        if (activeBranch == emptyBranch) {
            removeBranch(clonedBranch);
        } else if (activeBranch == clonedBranch) {
            removeBranch(emptyBranch);
        } else {
            removeBranch(emptyBranch);
            removeBranch(clonedBranch);
        }

        branchesToTrainDuringDriftAlert = new BranchesToTrainDuringDriftAlert();

        if (branchList.size() > maxBranchNum) {
            int delBranch = branchWithMaxLoss();
            branchList = branchList.subList(1, branchList.size());
            del_branch(model, delBranch);
            lossStatisticsList.remove(delBranch);
            lossList.remove(delBranch);
            nodeList.remove(delBranch);
        }
    }

    protected int branchWithMaxLoss() {
        return lossStatisticsList.keySet().stream()
                .max((k1, k2) -> lossStatisticsList.get(k1).get("prev_mean").compareTo(lossStatisticsList.get(k2).get("prev_mean")))
                .orElseThrow();
    }

    @Override
    protected boolean shouldSkipUpdatingLossStatistics(List<Double> losslist) {
        return losslist.size() < splitLen;
    }

    @Override
    protected void conceptDetection() {
        if (!lossStatisticsList.containsKey(activeBranch)) {
            return;
        }

        // todo dec 29 - przepisywanie na zgodnie z artykułem
        Map<String, Double> activeBranchLoss = lossStatisticsList.get(activeBranch);
        double conceptDriftThreshold = activeBranchLoss.get("mean") + confid * activeBranchLoss.get("var");
        if (alertNum > splitLen && activeBranchLoss.get("prev_var") + activeBranchLoss.get("prev_mean") > conceptDriftThreshold) {
            int minLossBranch = lossStatisticsList.keySet().stream().min((k1, k2) -> lossStatisticsList.get(k1).get("prev_mean").compareTo(lossStatisticsList.get(k2).get("prev_mean"))).orElse(0); // todo nie wiem czy dobry default
            Map<String, Double> minLossMap = lossStatisticsList.get(minLossBranch);
            int oldActiveBranch = activeBranch;
            activeBranch = minLossBranch;
//            if (minLossMap.get("prev_var") + minLossMap.get("prev_mean") > conceptDriftThreshold) {
            if (activeBranch == oldActiveBranch) {
                driftStatus = DRIFT_STATUS_CURRENT_EVOLVING;
            } else if (activeBranch == branchesToTrainDuringDriftAlert.empty) {
                driftStatus = DRIFT_STATUS_NEW_DETECTED + "_empty";
            } else if (activeBranch == branchesToTrainDuringDriftAlert.cloned) {
                driftStatus = DRIFT_STATUS_NEW_DETECTED + "_cloned";
            } else {
                driftStatus = DRIFT_STATUS_RECURRING;
            }
            update_fisherMatrix();
            lastDriftTime = trainTimes;
//          reset_weight_in_active_branch(); // todo A CO Z TYM?



//            } else {
//                if (minLossBranch != activeBranch) {
//                    driftStatus = DRIFT_STATUS_RECURRING; // todo jakiś błąd tu jest, że to jest oznaczane a naprawdę jest nowy branczyk wybrany
//                    activeBranch = minLossBranch;
//                } else {
//                    driftStatus = DRIFT_STATUS_CURRENT_EVOLVING;
//                }
//            }

            clean_drift_alert_branches();
            driftAlert = false;
            alertNum = 0;
            for (Node node : nodeList.get(0)) {
                node.alertSquareGrad_hW = node.alertSquareGrad_hW.scalarMultiply(0);
                node.alertSquareGrad_hb = node.alertSquareGrad_hb.mapMultiply(0);
            }
        }
    }


    @Override
    protected void add_empty_branch() {
        Node parent = weight_sim(); // TODO TU JEST PROBLEM, ZAWSZE W ZŁYM MIEJSCU TO BĘDZIE BO POD SPODEM alertNum == 0
        // przykład jak to rozwiązać - dodać w momencie wykrycia dryfu a potem jakoś douczyć
        branchNum = branchNum + 1;
        branchList.add(branchNum);
//        activeBranch = branchNum;
        branchesToTrainDuringDriftAlert.empty = branchNum;
        add_empty_child_node(parent, branchNum, 1.0); // todo daje jeden jan 2
//        if (branchList.size() > maxBranchNum) { todo to zróbmy jak potwierdzimy dryf
//            int delBranch = branchList.get(0);
//            branchList = branchList.subList(1, branchList.size());
//            del_branch(model, delBranch);
//            lossStatisticsList.remove(delBranch);
//            lossList.remove(delBranch);
//            nodeList.remove(delBranch);
//        }
    }

    @Override
    protected void add_empty_child_node(Node parentNode, int branchType, double weight) { // todo tutaj zero ???????
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

    protected void add_cloned_branch() {
        branchNum = branchNum + 1;
        branchList.add(branchNum);
        branchesToTrainDuringDriftAlert.cloned = branchNum;

        List<Node> nodesInActiveBranch = get_active_node_list();
        Node firstNodeInActiveBranch = nodesInActiveBranch.get(0);

        Node upperNode;
        List<Node> nodesToCopy;
        if (activeBranch == 0) { // if trunk, first node is root, so we copy from first real node
            upperNode = nodesInActiveBranch.get(0);
            nodesToCopy = nodesInActiveBranch.subList(1, nodesInActiveBranch.size());
        } else { // if not trunk, get parent node in trunk
            upperNode = firstNodeInActiveBranch.parent;
            nodesToCopy = nodesInActiveBranch;
        }

        // wypisz informacje o nodzie
//        System.out.println("upper node: " + upperNode.branchType + ", depth: " + upperNode.depth + ", hideOutputDim: " + upperNode.hideOutput.getDimension());
        Node firstNodeToCopy = nodesToCopy.get(0);
//        System.out.println("first copied node: " + firstNodeToCopy.branchType + ", depth: " + firstNodeToCopy.depth + ", hideInputDim: " + firstNodeToCopy.hideInput.getDimension() + ", hw rows: " + firstNodeToCopy.hW.getRowDimension() + ", hw cols: " + firstNodeToCopy.hW.getColumnDimension());

        nodeList.put(branchNum, new ArrayList<>());
        for (Node node : nodesToCopy) {
//            System.out.println("Going to clone: branchType: " + node.branchType + ", depth: " + node.depth);
            Node newNode = node.copy(upperNode, branchNum);
            upperNode.childList.add(newNode);
            nodeList.get(branchNum).add(newNode);
            upperNode = newNode;
        }

//        if (branchList.size() > maxBranchNum) { todo to zróbmy jak potwierdzimy dryf
//            int delBranch = branchList.get(0);
//            branchList = branchList.subList(1, branchList.size());
//            del_branch(model, delBranch);
//            lossStatisticsList.remove(delBranch);
//            lossList.remove(delBranch);
//            nodeList.remove(delBranch);
//        }
    }

    @Override
    protected void driftAlertDetection() {
        double driftWarnLevel = lossStatisticsList.get(activeBranch).get("mean") + 2 * lossStatisticsList.get(activeBranch).get("var"); // todo zmieniłem confid na 2, żeby było jak w wartykule
        List<Double> lossWin = lossList.get(activeBranch).subList(lossList.get(activeBranch).size() - splitLen, lossList.get(activeBranch).size());

        double activeBranchRecentMean = lossWin.stream().reduce(0.0, Double::sum) / lossWin.size();
        double activeBranchRecentVar = AtnnUtils.calculateStandardDeviation(lossWin);

        if (!driftAlert) {
            if (activeBranchRecentMean + activeBranchRecentVar > driftWarnLevel) { // todo zmyśliłem 10_000, żeby tego nigdy nie było
                driftAlert = true;
                driftStatus = DRIFT_STATUS_WARN;
                add_empty_branch();
                add_cloned_branch();
            }
        } else {
            if (activeBranchRecentMean + activeBranchRecentVar < driftWarnLevel) {
                driftAlert = false;
                alertNum = 0;
                clean_drift_alert_branches();
                driftStatus = DRIFT_STATUS_ELIMINATED;
            }
        }

        if (driftAlert) {
            alertNum = alertNum + 1;
        }
    }

    @Override
    public BranchesInfo getBranchesInfo() {
        double activeLoss = 0;
        Map<String, Double> activeBranchStatistics = lossStatisticsList.get(activeBranch);
        if (activeBranchStatistics != null)
            activeLoss = activeBranchStatistics.get("prev_mean");
//        System.out.println("activeLoss: " + activeLoss);
        double emptyBranchLoss = 0;
        double clonedBranchLoss = 0;
        if (!branchesToTrainDuringDriftAlert.isEmpty() && nodeList.get(branchesToTrainDuringDriftAlert.empty).get(0).hasBeenForwarded) {
            int emptyBranch = branchesToTrainDuringDriftAlert.empty;
            int clonedBranch = branchesToTrainDuringDriftAlert.cloned;
//            System.out.println("emptyBranch: " + emptyBranch + ", clonedBranch: " + clonedBranch);
            if (lossStatisticsList.containsKey(emptyBranch) && lossStatisticsList.containsKey(clonedBranch)) {
                emptyBranchLoss = lossStatisticsList.get(emptyBranch).get("prev_mean");
                clonedBranchLoss = lossStatisticsList.get(clonedBranch).get("prev_mean");
            }
        }

        Map<String, String> eatnnLosses = new HashMap<>();
        eatnnLosses.put("active", String.valueOf(activeLoss));
        eatnnLosses.put("empty", String.valueOf(emptyBranchLoss));
        eatnnLosses.put("cloned", String.valueOf(clonedBranchLoss));

        return new BranchesInfo(
                branchList.size() + 1, // we add 1 because trunk is not considered as a separate branch in model code
                activeBranch,
                get_active_node_list().get(0).depth,
                get_active_node_list().size(), // to zwróci tylko od miejsca złączenia z trunkiem
                driftStatus,
                eatnnLosses
        );
    }

    static class BranchesToTrainDuringDriftAlert {
        int empty = -1;
        int cloned = -1;

        BranchesToTrainDuringDriftAlert() {
        }

        BranchesToTrainDuringDriftAlert(int empty, int cloned) {
            this.empty = empty;
            this.cloned = cloned;
        }

        public boolean isEmpty() {
            return empty == -1 && cloned == -1;
        }
    }

}
