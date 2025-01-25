package flinkClassifiersTesting.classifiers.atnn;

import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.BRANCH_STRUCTURE;
import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.CLONED_NODES_TO_TRAIN;
import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.DRIFT_STATUS;
import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.EMPTY_NODES_TO_TRAIN;
import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.NORMAL_NODES_TO_TRAIN;
import flinkClassifiersTesting.classifiers.atnn.models.BranchesInfo;
import flinkClassifiersTesting.classifiers.atnn.models.EnhancedAtnnModel2;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.atnn.AtnnClassifierParams;
import java.util.ArrayList;
import java.util.Map;
import java.util.stream.IntStream;
import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.flink.api.java.tuple.Tuple2;

/*
This class is a flinkClassifiersTesting wrapper for moa.classifiers.deeplearning.CAND
*/
public class EAtnn2 extends BaseClassifierClassifyAndTrain {

    int featureLen;
    int classNum;

    int hNeuronNum;
    double initialLearningRate;
    int lambda;

    int exampleNumber = 0;
    int bootstrapExampleNumber = 0;

    double gamma = 1.0;

    EnhancedAtnnModel2 model;
    EnhancedAtnnModel2 bootstrapModel;

    public EAtnn2(
            Map<String, Integer> classEncoder,
            AtnnClassifierParams params
    ) {
        classNum = classEncoder.keySet().size(); // classes may be defined not as in class encoder, can fix it later
        hNeuronNum = params.hiddenLayerSize;
        initialLearningRate = params.learningRate;
        lambda = params.lambda;
        gamma = params.gamma;
    }


    @Override
    protected ArrayList<Tuple2<String, Object>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Object>> performances) {
        // przyjmuje i zwraca performance
        if (model == null) {
            featureLen = example.getAttributes().length;
            model = new EnhancedAtnnModel2(featureLen, hNeuronNum, classNum, initialLearningRate, lambda, gamma);
            model.init_node_weight();
        }
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        model.train_model(feature, label);
        // todo jan 2
        // w performances chciałbym zwrócić tak: allBranches10_chosenBranch2_chosenBranchDepth4
        BranchesInfo branchesInfo = model.getBranchesInfo();
        performances.add(new Tuple2<>(BRANCH_STRUCTURE, branchesInfo.getBranchesInfoEAtnnString()));
        performances.add(new Tuple2<>(DRIFT_STATUS, branchesInfo.getDriftStatusString()));
        performances.add(new Tuple2<>(CLONED_NODES_TO_TRAIN, branchesInfo.getClonedBranchNodesToLearn()));
        performances.add(new Tuple2<>(EMPTY_NODES_TO_TRAIN, branchesInfo.getEmptyBranchNodesToLearn()));
        performances.add(new Tuple2<>(NORMAL_NODES_TO_TRAIN, branchesInfo.getNormalNodesToLearn()));
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
            model = new EnhancedAtnnModel2(featureLen, hNeuronNum, classNum, initialLearningRate, lambda, gamma);
            model.init_node_weight();
        }
        exampleNumber += 1;
        if (exampleNumber % 1000 == 0) {
            System.out.println("i: " + exampleNumber);
        }
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        double[] result = model.predict(feature).toArray();
        int predictedClass = IntStream.range(0, result.length).reduce((i, j) -> result[i] > result[j] ? i : j).getAsInt();
        ArrayList<Tuple2<String, Object>> performances = new ArrayList<>();
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
            model = new EnhancedAtnnModel2(featureLen, hNeuronNum, classNum, initialLearningRate, lambda, gamma);
            model.init_node_weight();
        }
        if (bootstrapModel == null) {
            featureLen = example.getAttributes().length;
            bootstrapModel = new EnhancedAtnnModel2(featureLen, hNeuronNum, classNum, initialLearningRate, lambda, gamma);
            bootstrapModel.init_node_weight();
        }
        bootstrapExampleNumber += 1;
        System.out.println("bi: " + bootstrapExampleNumber);
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        bootstrapModel.train_model(feature, label); // todo zmienić, bo dwa razy sie bedzie wykonywalo
    }
}