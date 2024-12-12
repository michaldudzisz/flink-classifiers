package flinkClassifiersTesting.classifiers.atnn;

import com.yahoo.labs.samoa.instances.*;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.cand.CandClassifierParams;
import moa.classifiers.deeplearning.CAND;
import moa.classifiers.deeplearning.MLP;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields.BRANCH_STRUCTURE;
import static flinkClassifiersTesting.classifiers.cand.CandClassifierFields.*;
import static flinkClassifiersTesting.classifiers.helpers.MOAAdapter.mapExampleToInstance;
import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;

/*
This class is a flinkClassifiersTesting wrapper for moa.classifiers.deeplearning.CAND
*/
public class Atnn extends BaseClassifierClassifyAndTrain {

    int statisticsLen = 100;

    // mnist
//    int featureLen = 784;
//    int classNum = 10;

    // elec
//    int featureLen = 6;
//    int classNum = 2;

    // sea abr
//    int featureLen = 3;
//    int classNum = 2;

    int featureLen;
    int classNum;

    int hNeuronNum = 256;

    int exampleNumber = 0;

    Model model;

    public Atnn(Map<String, Integer> classEncoder) {
        classNum = classEncoder.keySet().size(); // classes may be defined not as in class encoder, can fix it later
//        lastResults[0] = 0.001;
    }


    @Override
    protected ArrayList<Tuple2<String, Object>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Object>> performances) {
        // przyjmuje i zwraca performance
        if (model == null) {
            featureLen = example.getAttributes().length;
            model = new Model(featureLen, hNeuronNum, classNum);
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
            model = new Model(featureLen, hNeuronNum, classNum);
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
        performances.add(new Tuple2<>(BRANCH_STRUCTURE, branchesInfo.toPerformanceString()));
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
            model = new Model(featureLen, hNeuronNum, classNum);
            model.init_node_weight();
        }
        exampleNumber += 1;
        System.out.println("bi: " + exampleNumber);
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        model.train_model(feature, label); // todo zmienić, bo dwa razy sie bedzie wykonywalo
    }
}
