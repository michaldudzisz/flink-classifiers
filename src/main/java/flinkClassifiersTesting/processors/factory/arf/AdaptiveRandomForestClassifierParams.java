package flinkClassifiersTesting.processors.factory.arf;

import flinkClassifiersTesting.processors.factory.ClassifierParamsInterface;

public class AdaptiveRandomForestClassifierParams implements ClassifierParamsInterface {

    public int ensembleSize = 30; // lub 10 w artykule było
    public float lambda = 6.0f;
    public int featurePercentageToSplit = 60; // lub 10 w artykule było

    public AdaptiveRandomForestClassifierParams() {}

    public AdaptiveRandomForestClassifierParams(int ensembleSize, float lambda, int featurePercentageToSplit) {
        this.ensembleSize = ensembleSize;
        this.lambda = lambda;
        this.featurePercentageToSplit = featurePercentageToSplit;
    }

    @Override
    public String directoryName() {
        return "ensembleSize" + ensembleSize +
                "_lambda" + lambda +
                "_featurePercentageToSplit" + featurePercentageToSplit;
    }

    @Override
    public String toString() {
        return "AdaptiveRandomForestClassifierParams{" +
                "ensembleSize=" + ensembleSize +
                ", lambda=" + lambda +
                ", featurePercentageToSplit=" + featurePercentageToSplit +
                '}';
    }
}
