package flinkClassifiersTesting.processors.factory.cand;

import flinkClassifiersTesting.processors.factory.ClassifierParamsInterface;

public class CandClassifierParams implements ClassifierParamsInterface {

    public CandClassifierParams() {
    }

    @Override
    public String directoryName() {
        return "dummyparam1";
    }
}
