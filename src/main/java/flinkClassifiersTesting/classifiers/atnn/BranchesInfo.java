package flinkClassifiersTesting.classifiers.atnn;

public class BranchesInfo {
    final int allBranches;
    final int activeBranch;
    final int activeBranchDepth;

    public BranchesInfo(int allBranches, int activeBranch, int activeBranchDepth) {
        this.allBranches = allBranches;
        this.activeBranch = activeBranch;
        this.activeBranchDepth = activeBranchDepth;
    }

    public String toPerformanceString() {
        return "allBranches" + allBranches + "_activeBranch" + activeBranch + "_activeBranchDepth" + activeBranchDepth;
    }
}
