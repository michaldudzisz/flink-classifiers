package flinkClassifiersTesting.classifiers.atnn.models;

import java.util.Map;

public class BranchesInfo {
    final int allBranches;
    final int activeBranch;
    final int activeBranchGrowingPoint;
    final int activeBranchDepth;
    final Map<String, String> eatnnLosses;

    final String driftStatus;

    public BranchesInfo(
            int allBranches,
            int activeBranch,
            int activeBranchGrowingPoint,
            int activeBranchDepth,
            String driftStatus,
            Map<String, String> eatnnLosses
    ) {
        this.allBranches = allBranches;
        this.activeBranch = activeBranch;
        this.activeBranchGrowingPoint = activeBranchGrowingPoint;
        this.activeBranchDepth = activeBranchDepth;
        this.driftStatus = driftStatus;
        this.eatnnLosses = eatnnLosses;
    }

    public String getBranchesInfoString() {
        return "allBranches" + allBranches +
                "_activeBranch" + activeBranch +
                "_growingPoint" + activeBranchGrowingPoint +
                "_activeBranchDepth" + activeBranchDepth;
    }

    public String getBranchesInfoEAtnnString() {
        String losses = "active" + eatnnLosses.get("active") +
                "_empty" + eatnnLosses.get("empty") +
                "_cloned" + eatnnLosses.get("cloned");
        return "allBranches" + allBranches +
                "_activeBranch" + activeBranch +
                "_growingPoint" + activeBranchGrowingPoint +
                "_activeBranchDepth" + activeBranchDepth +
                "_" + losses;
    }

    public String getDriftStatusString() {
        return driftStatus;
    }
}
