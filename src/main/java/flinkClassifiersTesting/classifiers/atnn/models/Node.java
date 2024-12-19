package flinkClassifiersTesting.classifiers.atnn.models;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;

import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;

public class Node {
    final double initialLearnRate = 0.02; // 0.0006 0.02

    int branchType = 0;
    boolean isShare = false;
    double learnRate = initialLearnRate;
    double minLR = 0.001; // 0.0005 <- wartość z canda 0.001
    int hNeuronNum;
    int cNeuronNum;
    boolean isRootNode = false;
    List<Node> childList = new ArrayList<>(); // to są tylko bezpośrednie dzieci
    Node parent = null;
    RealMatrix hW = null;
    RealVector hb = null;
    RealMatrix cW = null;
    RealVector cb = null;
    RealVector hideInput = null;
    RealVector hideOutput = null;
    RealVector classifierInput = null;
    RealVector classifierOutput = null;
    RealMatrix dev_hW = null;
    RealMatrix dev_cW = null;
    RealVector dev_cInput = null;
    RealVector dev_hInput = null;
    double weight = 0;
    int depth = 0;
    int trainTimes = 0;
    double reduction = 0.999;
    double lastPredictLoss = 0;
    RealMatrix squareGrad_hW = null;
    RealVector squareGrad_hb = null;
    RealMatrix Fisher_hW = null;
    RealVector Fisher_hb = null;
    RealMatrix alertSquareGrad_hW = null;
    RealVector alertSquareGrad_hb = null;
    RealMatrix alertFisher_hW = null;
    RealVector alertFisher_hb = null;
    RealMatrix lastConcept_hW = null;
    RealVector lastConcept_hb = null;


    public Node(int hNeuronNum, int cNeuronNum, int branchType) {
        this.hNeuronNum = hNeuronNum;
        this.cNeuronNum = cNeuronNum;
        this.branchType = branchType;
        init_b();
    }

    private Node() {}

    private void init_b() {
        hb = new ArrayRealVector(hNeuronNum);
        for (double d : hb.toArray()) {
            if (Double.isNaN(d)) {
                throw new RuntimeException("hb: " + hb);
            }
        }
        cb = new ArrayRealVector(cNeuronNum);
        Fisher_hb = new ArrayRealVector(hNeuronNum);
        squareGrad_hb = new ArrayRealVector(hNeuronNum);
        alertSquareGrad_hb = new ArrayRealVector(hNeuronNum);
        alertFisher_hb = new ArrayRealVector(hNeuronNum);
        lastConcept_hb = new ArrayRealVector(hNeuronNum);
    }

    public void init_weight() {
        if (isRootNode) {
            hW = AtnnUtils.createRandomMatrix(hNeuronNum, hNeuronNum);
            cW = AtnnUtils.createRandomMatrix(cNeuronNum, hNeuronNum);
            squareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            Fisher_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            alertSquareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            alertFisher_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
            lastConcept_hW = new Array2DRowRealMatrix(hNeuronNum, hNeuronNum);
        } else {
            hW = AtnnUtils.createRandomMatrix(hNeuronNum, parent.hNeuronNum);
            cW = AtnnUtils.createRandomMatrix(cNeuronNum, hNeuronNum);
            squareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            Fisher_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            alertSquareGrad_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            alertFisher_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
            lastConcept_hW = new Array2DRowRealMatrix(hNeuronNum, parent.hNeuronNum);
        }
    }

    public RealVector relu(RealVector x) {
        double[] data = x.toArray();

        for (double d : data) {
            if (Double.isNaN(d)) {
                throw new RuntimeException("x: " + x);
            }
        }

        for (int i = 0; i < data.length; i++) {
            data[i] = Math.max(0, data[i]);
        }

        return AtnnUtils.standardization(createRealVector(data));
    }

    // Derivative of ReLU
    public RealVector dev_relu(RealVector x) {
        double[] data = x.toArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] > 0 ? 1 : 0;
        }
        return createRealVector(data);
    }

    public Node copy(Node parent, int branchType) {
        Node copied = new Node();
        copied.branchType = branchType;
        copied.isShare = false;
        copied.learnRate = this.initialLearnRate; // powrót do dużego lr
        copied.minLR = this.minLR;
        copied.hNeuronNum = this.hNeuronNum;
        copied.cNeuronNum = this.cNeuronNum;
        copied.isRootNode = this.isRootNode;
        copied.childList = new ArrayList<>();
        copied.parent = parent;
        copied.hW = this.hW;
        copied.hb = this.hb;
        copied.cW = this.cW;
        copied.cb = this.cb;
        copied.hideInput = this.hideInput;
        copied.hideOutput = this.hideOutput;
        copied.classifierInput = this.classifierInput;
        copied.classifierOutput = this.classifierOutput;
        copied.dev_hW = this.dev_hW;
        copied.dev_cW = this.dev_cW;
        copied.dev_cInput = this.dev_cInput;
        copied.dev_hInput = this.dev_hInput;
        copied.weight = this.weight; // todo jak są zmieniane w czasie te wagi?
        copied.depth = this.depth;
        copied.trainTimes = 0; // todo czy na pewno 0? Gdzie to jest używane?
        copied.reduction = this.reduction;
        copied.lastPredictLoss = this.lastPredictLoss;
        copied.squareGrad_hW = this.squareGrad_hW;
        copied.squareGrad_hb = this.squareGrad_hb;
        copied.Fisher_hW = this.Fisher_hW;
        copied.Fisher_hb = this.Fisher_hb;
        copied.alertSquareGrad_hW = this.alertSquareGrad_hW;
        copied.alertSquareGrad_hb = this.alertSquareGrad_hb;
        copied.alertFisher_hW = this.alertFisher_hW;
        copied.alertFisher_hb = this.alertFisher_hb;
        copied.lastConcept_hW = this.lastConcept_hW;
        copied.lastConcept_hb = this.lastConcept_hb;
        return copied;
    }
}
