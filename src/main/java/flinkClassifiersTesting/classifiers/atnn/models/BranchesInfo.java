package flinkClassifiersTesting.classifiers.atnn.models;

public class BranchesInfo {
    final int allBranches;
    final int activeBranch;
    final int activeBranchDepth;

    final String driftStatus;

    public BranchesInfo(int allBranches, int activeBranch, int activeBranchDepth, String driftStatus) {
        this.allBranches = allBranches;
        this.activeBranch = activeBranch;
        this.activeBranchDepth = activeBranchDepth;
        this.driftStatus = driftStatus;
    }

    public String getBranchesInfoString() {
        return "allBranches" + allBranches + "_activeBranch" + activeBranch + "_activeBranchDepth" + activeBranchDepth;
    }

    public String getDriftStatusString() {
        return driftStatus;
    }
}
