package flinkClassifiersTesting.classifiers.atnn.models;

import java.util.Map;

public class BranchesInfo {
    final int allBranches;
    final int activeBranch;
    final int activeBranchGrowingPoint;
    final int activeBranchDepth;
    final Map<String, String> atnnLosses;
    final int clonedBranchNodesToLearn;
    final int emptyBranchNodesToLearn;
    final int normalNodesToLearn;

    final String driftStatus;

    public BranchesInfo(
            int allBranches,
            int activeBranch,
            int activeBranchGrowingPoint,
            int activeBranchDepth,
            String driftStatus,
            Map<String, String> eatnnLosses,
            int clonedBranchNodesToLearn,
            int emptyBranchNodesToLearn,
            int normalNodesToLearn
    ) {
        this.allBranches = allBranches;
        this.activeBranch = activeBranch;
        this.activeBranchGrowingPoint = activeBranchGrowingPoint;
        this.activeBranchDepth = activeBranchDepth;
        this.driftStatus = driftStatus;
        this.atnnLosses = eatnnLosses;
        this.clonedBranchNodesToLearn = clonedBranchNodesToLearn;
        this.emptyBranchNodesToLearn = emptyBranchNodesToLearn;
        this.normalNodesToLearn = normalNodesToLearn;
    }

    public String getBranchesInfoString() {
        return "allBranches" + allBranches +
                "_activeBranch" + activeBranch +
                "_growingPoint" + activeBranchGrowingPoint +
                "_activeBranchDepth" + activeBranchDepth +
                "_active" + atnnLosses.get("active");
    }

    public String getBranchesInfoEAtnnString() {
        String losses = "active" + atnnLosses.get("active") +
                "_empty" + atnnLosses.get("empty") +
                "_cloned" + atnnLosses.get("cloned");
        return "allBranches" + allBranches +
                "_activeBranch" + activeBranch +
                "_growingPoint" + activeBranchGrowingPoint +
                "_activeBranchDepth" + activeBranchDepth +
                "_" + losses;
    }

    public String getDriftStatusString() {
        return driftStatus;
    }

    public int getClonedBranchNodesToLearn() {
        return clonedBranchNodesToLearn;
    }

    public int getEmptyBranchNodesToLearn() {
        return emptyBranchNodesToLearn;
    }

    public int getNormalNodesToLearn() {
        return normalNodesToLearn;
    }
}
