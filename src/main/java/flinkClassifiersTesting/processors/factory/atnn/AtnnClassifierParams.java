package flinkClassifiersTesting.processors.factory.atnn;

import flinkClassifiersTesting.processors.factory.ClassifierParamsInterface;

public class AtnnClassifierParams implements ClassifierParamsInterface {

    public double learningRate = 0.02;
    public int hiddenLayerSize = 256;
    public int lambda = 10;

    public AtnnClassifierParams() {}

    public AtnnClassifierParams(double learningRate, int hiddenLayerSize, int lambda) {
        this.learningRate = learningRate;
        this.hiddenLayerSize = hiddenLayerSize;
        this.lambda = lambda;
    }

    @Override
    public String directoryName() {
        return "lr" + learningRate + "_hls" + hiddenLayerSize + "_lambda" + lambda;
    }
}
