package flinkClassifiersTesting.classifiers.atnn;

import flinkClassifiersTesting.classifiers.atnn.models.BranchesInfo;
import flinkClassifiersTesting.classifiers.atnn.models.BaseAtnnModel;
import flinkClassifiersTesting.classifiers.atnn.models.EnhancedAtnnModel;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.atnn.AtnnClassifierParams;
import flinkClassifiersTesting.processors.factory.cand.CandClassifierParams;
import org.apache.commons.math3.linear.RealVector;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.*;
import java.util.stream.IntStream;

import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.BRANCH_STRUCTURE;
import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.DRIFT_STATUS;
import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;


public class Atnn extends BaseClassifierClassifyAndTrain {

    int featureLen;
    int classNum;

    int hNeuronNum;
    double initialLearningRate;
    int lambda;

    int exampleNumber = 0;

    BaseAtnnModel model;

    public Atnn(
            Map<String, Integer> classEncoder,
            AtnnClassifierParams params
    ) {
        classNum = classEncoder.keySet().size(); // classes may be defined not as in class encoder, can fix it later
        hNeuronNum = params.hiddenLayerSize;
        initialLearningRate = params.learningRate;
        lambda = params.lambda;
    }


    @Override
    protected ArrayList<Tuple2<String, Object>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Object>> performances) {
        // przyjmuje i zwraca performance
        if (model == null) {
            featureLen = example.getAttributes().length;
            model = new BaseAtnnModel(featureLen, hNeuronNum, classNum, initialLearningRate, lambda);
            model.init_node_weight();
        }
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        model.train_model(feature, label);
        return performances;
    }

    private RealVector oneHotEncodeExample(Example example) {
        int classNum = example.getMappedClass();
        double[] encoded = new double[this.classNum];
        for (int i = 0; i < this.classNum; i++) {
            if (classNum == i)
                encoded[i] = 1.0;
            else
                encoded[i] = 0.0;
        }
        return createRealVector(encoded);
    }

    @Override
    protected Tuple2<Integer, ArrayList<Tuple2<String, Object>>> classifyImplementation(Example example) {
        if (model == null) {
            featureLen = example.getAttributes().length;
            model = new BaseAtnnModel(featureLen, hNeuronNum, classNum, initialLearningRate, lambda);
            model.init_node_weight();
        }
        exampleNumber += 1;
        System.out.println("i: " + exampleNumber);
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        double[] result = model.predict(feature).toArray();
        int predictedClass = IntStream.range(0, result.length).reduce((i, j) -> result[i] > result[j] ? i : j).getAsInt();
        ArrayList<Tuple2<String, Object>> performances = new ArrayList<>();
        // w performances chciałbym zwrócić tak: allBranches10_chosenBranch2_chosenBranchDepth4
        BranchesInfo branchesInfo = model.getBranchesInfo();
        performances.add(new Tuple2<>(BRANCH_STRUCTURE, branchesInfo.getBranchesInfoString()));
        performances.add(new Tuple2<>(DRIFT_STATUS, branchesInfo.getDriftStatusString()));
        return new Tuple2<>(predictedClass, performances);
    }

    @Override
    public String generateClassifierParams() {
        return "parametry";
    }

    @Override
    public void bootstrapTrainImplementation(Example example) {
        if (model == null) {
            featureLen = example.getAttributes().length;
            model = new BaseAtnnModel(featureLen, hNeuronNum, classNum, initialLearningRate, lambda);
            model.init_node_weight();
        }
        exampleNumber += 1;
        System.out.println("bi: " + exampleNumber);
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        model.train_model(feature, label);
    }
}
