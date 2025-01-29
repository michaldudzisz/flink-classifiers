package flinkClassifiersTesting.processors.factory.atnn;

import flinkClassifiersTesting.processors.factory.ClassifierParamsInterface;

public class AtnnClassifierParams implements ClassifierParamsInterface {

    public double learningRate = 0.02;
    public int hiddenLayerSize = 256;
    public int lambda = 10;
    public double gamma = 1;
    public int iter = 0;

    public AtnnClassifierParams() {}

    public AtnnClassifierParams(double learningRate, int hiddenLayerSize, int lambda, double gamma, int iter) {
        this.learningRate = learningRate;
        this.hiddenLayerSize = hiddenLayerSize;
        this.lambda = lambda;
        this.gamma = gamma;
        this.iter = iter;
    }

    @Override
    public String directoryName() {
        return "lr" + learningRate + "_hls" + hiddenLayerSize + "_lambda" + lambda + "_gamma" + gamma + "_iter" + iter;
    }

    @Override
    public String toString() {
        return "AtnnClassifierParams{" +
                "learningRate=" + learningRate +
                ", hiddenLayerSize=" + hiddenLayerSize +
                ", lambda=" + lambda +
                ", gamma=" + gamma +
                ", iter=" + iter +
                '}';
    }
}
