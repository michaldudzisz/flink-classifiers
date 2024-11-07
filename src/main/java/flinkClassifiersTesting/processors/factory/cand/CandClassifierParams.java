package flinkClassifiersTesting.processors.factory.cand;

import flinkClassifiersTesting.processors.factory.ClassifierParamsInterface;

public class CandClassifierParams implements ClassifierParamsInterface {

    public enum PoolSize { P10, P30 }

    public int mSize = 2;
    public PoolSize pSize = PoolSize.P30;

    public CandClassifierParams() {
    }

    public CandClassifierParams(PoolSize pSize, int mSize) {
        this.mSize = mSize;
        this.pSize = pSize;
    }

    @Override
    public String directoryName() {
        return "Psize" + poolSizeToString(pSize) + "_Msize" + mSize;
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
                '}';
    }
}
