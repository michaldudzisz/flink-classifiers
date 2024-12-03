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

import static flinkClassifiersTesting.classifiers.cand.CandClassifierFields.*;
import static flinkClassifiersTesting.classifiers.helpers.MOAAdapter.mapExampleToInstance;
import static org.apache.commons.math3.linear.MatrixUtils.createRealVector;

/*
This class is a flinkClassifiersTesting wrapper for moa.classifiers.deeplearning.CAND
*/
public class Atnn extends BaseClassifierClassifyAndTrain {

    int statisticsLen = 100;

    int featureLen = 784;
    int hNeuronNum = 256;
    int classNum = 10;

    int predictRightNumber = 0;
    int exampleNumber = 0;

    Model model;

    double[] lastResults = new double[classNum];

    public Atnn() {
        model = new Model(featureLen, hNeuronNum, classNum);
        model.init_node_weight();
//        lastResults[0] = 0.001;
    }


    @Override
    protected ArrayList<Tuple2<String, Object>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Object>> performances) {
        // przyjmuje i zwraca performance
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        lastResults = model.train_model(feature, label).toArray(); // todo zmienić, bo dwa razy sie bedzie wykonywalo
        return performances;
    }

    private RealVector oneHotEncodeExample(Example example) {
        int classNum = example.getMappedClass();
        double[] encoded = new double[this.classNum]; // todo dla eleca dwie, potem rozszerzymy
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
        exampleNumber += 1;
        System.out.println("i: " + exampleNumber);
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        double[] result = lastResults; // model.train_model(feature, label).toArray(); // todo zmienić, bo dwa razy sie bedzie wykonywalo
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
        exampleNumber += 1;
        System.out.println("bi: " + exampleNumber);
        RealVector feature = createRealVector(example.getAttributes());
        RealVector label = oneHotEncodeExample(example);
        lastResults = model.train_model(feature, label).toArray(); // todo zmienić, bo dwa razy sie bedzie wykonywalo
    }
}
