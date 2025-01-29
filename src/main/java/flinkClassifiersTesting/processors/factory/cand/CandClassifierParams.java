package flinkClassifiersTesting.processors.factory.cand;

import flinkClassifiersTesting.processors.factory.ClassifierParamsInterface;

public class CandClassifierParams implements ClassifierParamsInterface {

    public enum PoolSize { P10, P30 }

    public int mSize = 2;
    public PoolSize pSize = PoolSize.P30;
    public double backpropagationThreshold = 0;
    public String votingStatsFile = null;
    public boolean logAllMlpLosses = false;
    int iter = 0;

    public CandClassifierParams() {
    }

    public CandClassifierParams(
            PoolSize pSize,
            int mSize
    ) {
        this.mSize = mSize;
        this.pSize = pSize;
    }

    public CandClassifierParams(
            PoolSize pSize,
            int mSize,
            String outputVotingStatsFile
    ) {
        this.mSize = mSize;
        this.pSize = pSize;
        this.votingStatsFile = outputVotingStatsFile;
    }

    public CandClassifierParams(
            PoolSize pSize,
            int mSize,
            boolean logAllMlpLosses
    ) {
        this.mSize = mSize;
        this.pSize = pSize;
        this.logAllMlpLosses = logAllMlpLosses;
    }

    public CandClassifierParams(
            PoolSize pSize,
            int mSize,
            double backpropagationThreshold,
            int iter
    ) {
        this.mSize = mSize;
        this.pSize = pSize;
        this.backpropagationThreshold = backpropagationThreshold;
        this.iter = iter;
    }

    @Override
    public String directoryName() {
        return "Psize" + poolSizeToString(pSize) + "_Msize" + mSize + "_Bpth" + backpropagationThreshold + "_iter" + iter;
    }

    private static String poolSizeToString(PoolSize pSize) {
        String result = "";
        switch (pSize) {
            case P10:
                result = "10";
                break;
            case P30:
                result = "30";
                break;
        }
        return result;
    }

    @Override
    public String toString() {
        return "CandClassifierParams{" +
                "mSize=" + mSize +
                ", pSize=" + pSize +
                ", backpropagationThreshold=" + backpropagationThreshold +
                ", iter=" + iter +
                '}';
    }
}
