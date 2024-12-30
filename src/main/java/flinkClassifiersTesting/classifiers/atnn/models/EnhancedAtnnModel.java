package flinkClassifiersTesting.classifiers.atnn.models;

import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static flinkClassifiersTesting.classifiers.atnn.models.AtnnUtils.softmax;

public class EnhancedAtnnModel extends BaseAtnnModel {

    BranchesToTrainDuringDriftAlert branchesToTrainDuringDriftAlert = new BranchesToTrainDuringDriftAlert();

    public EnhancedAtnnModel(int featureNum, int hNeuronNum, int cNeuronNum) {
        super(featureNum, hNeuronNum, cNeuronNum);
    }

    @Override
    protected boolean should_back_propagate_node(Node node) {
        return (node.branchType == 0 || node.branchType == activeBranch) // normal path
                || node.branchType == branchesToTrainDuringDriftAlert.empty
                || node.branchType == branchesToTrainDuringDriftAlert.cloned;
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

    protected void clean_drift_alert_branches() {
        int emptyBranch = branchesToTrainDuringDriftAlert.empty;
        int clonedBranch = branchesToTrainDuringDriftAlert.cloned;

        if (activeBranch == emptyBranch) {
            branchList.remove(clonedBranch);
            del_branch(model, clonedBranch);
        } else if (activeBranch == clonedBranch) {
            branchList.remove(emptyBranch);
            del_branch(model, emptyBranch);
        } else {
            branchList.remove(clonedBranch);
            del_branch(model, clonedBranch);
            branchList.remove(emptyBranch);
            del_branch(model, emptyBranch);
        }

        branchesToTrainDuringDriftAlert = new BranchesToTrainDuringDriftAlert();

        if (branchList.size() > maxBranchNum) {
            int delBranch = branchList.get(0);
            branchList = branchList.subList(1, branchList.size());
            del_branch(model, delBranch);
            lossStatisticsList.remove(delBranch);
            lossList.remove(delBranch);
            nodeList.remove(delBranch);
        }
    }

    @Override
    protected void conceptDetection() {
        if (!lossStatisticsList.containsKey(activeBranch)) {
            return;
        }

        // todo dec 29 - przepisywanie na zgodnie z artykułem
        Map<String, Double> activeBranchLoss = lossStatisticsList.get(activeBranch);
        double conceptDriftThreshold = activeBranchLoss.get("mean") + confid * activeBranchLoss.get("var");
        if (activeBranchLoss.get("prev_var") + activeBranchLoss.get("prev_mean") > conceptDriftThreshold) {
            int minLossBranch = lossStatisticsList.keySet().stream().min((k1, k2) -> lossStatisticsList.get(k1).get("prev_mean").compareTo(lossStatisticsList.get(k2).get("prev_mean"))).orElse(0); // todo nie wiem czy dobry default
            Map<String, Double> minLossMap = lossStatisticsList.get(minLossBranch);
            if (minLossMap.get("prev_var") + minLossMap.get("prev_mean") > conceptDriftThreshold) {
                driftStatus = DRIFT_STATUS_NEW_DETECTED;
                update_fisherMatrix();
                choose_among_drift_branches();
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
        Node parent = weight_sim();
        branchNum = branchNum + 1;
        branchList.add(branchNum);
        activeBranch = branchNum;
        branchesToTrainDuringDriftAlert.empty = branchNum;
        add_empty_child_node(parent, branchNum, 0.0);
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

    protected void add_cloned_branch() {
        branchNum = branchNum + 1;
        branchList.add(branchNum);
        branchesToTrainDuringDriftAlert.cloned = branchNum;

        List<Node> nodesInActiveBranch = get_active_node_list();
        Node upperNode = nodesInActiveBranch.get(0).parent;
        nodeList.put(branchNum, new ArrayList<>());
        for (Node node : nodesInActiveBranch) {
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
                driftStatus = DRIFT_STATUS_ELIMINATED;
            }
        }

        if (driftAlert) {
            alertNum = alertNum + 1;
        }
    }

    static class BranchesToTrainDuringDriftAlert {
        int empty;
        int cloned;
        BranchesToTrainDuringDriftAlert() {}
        BranchesToTrainDuringDriftAlert(int empty, int cloned) {
            this.empty = empty;
            this.cloned = cloned;
        }

        public boolean isEmpty() {
            return empty == 0 && cloned == 0;
        }
    }

}
