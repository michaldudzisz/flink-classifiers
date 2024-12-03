package flinkClassifiersTesting.processors.factory.atnn;

import flinkClassifiersTesting.processors.factory.ClassifierParamsInterface;

public class AtnnClassifierParams implements ClassifierParamsInterface {

    public AtnnClassifierParams() {}


    @Override
    public String directoryName() {
        return "dummy1";
    }
}
