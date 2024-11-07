package flinkClassifiersTesting.classifiers.cand;

import com.yahoo.labs.samoa.instances.*;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.cand.CandClassifierParams;
import moa.classifiers.deeplearning.CAND;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.flink.api.java.tuple.Tuple2;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/*
This class is a flinkClassifiersTesting wrapper for moa.classifiers.deeplearning.CAND
*/
public class Cand extends BaseClassifierClassifyAndTrain {

    private CAND cand = null;
    private Instances datasetProperties = null;

    int i = 0;

    public Cand(CandClassifierParams params) {
        cand = new CAND();

        if (params.pSize == CandClassifierParams.PoolSize.P10)
            cand.largerPool.setChosenIndex(0); // P = 10
        else
            cand.largerPool.setChosenIndex(1); // P = 30

        cand.numberOfMLPsToTrainOption.setValue(params.mSize); // M

        cand.numberOfLayersInEachMLP.setValue(1); // 1
        cand.numberOfInstancesToTrainAllMLPsAtStartOption.setValue(100); // 100 - MAX
        cand.miniBatchSize.setValue(1); // incrementally
        cand.useOneHotEncode.setValue(true); // t
        cand.useNormalization.setValue(true); // t
//        cand.backPropLossThreshold.setValue(0.3); // 0.3
        cand.deviceTypeOption.setChosenIndex(1); // CPU
        cand.statsDumpFileName.setValue("cand_stats.txt");
    }

    @Override
    protected ArrayList<Tuple2<String, Long>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Long>> performances) {
        // przyjmuje i zwraca performance
        if (i == 0) {
            Instance moaInstance = mapExampleToInstance(example);
            cand.trainOnInstanceImpl(moaInstance);
        }
        return performances;
    }

    @Override
    protected Tuple2<Integer, ArrayList<Tuple2<String, Long>>> classifyImplementation(Example example) {
        Instance moaInstance = mapExampleToInstance(example);
        double[] votesForEachClass = cand.getVotesForInstance(moaInstance);
        int predictedClass = IntStream.range(0, votesForEachClass.length).reduce((i, j) -> votesForEachClass[i] > votesForEachClass[j] ? i : j).getAsInt();
        return new Tuple2<>(
                predictedClass,
                new ArrayList<>() // To są parametry tego jak mu poszło, można zobaczyć klasę BaseDynamicWeightedMajority
        );
    }

    @Override
    public String generateClassifierParams() {
        return "parametry";
    }

    @Override
    public void bootstrapTrainImplementation(Example example) {
        trainImplementation(example, example.getMappedClass(), null);
    }

    private Instance mapExampleToInstance(Example example) {
        if (datasetProperties == null) { // ukradzione z csv
            ArrayList<Attribute> attributes = new ArrayList<>(example.getAttributes().length);
            for (int i = 0; i < example.getAttributes().length; i++) {
                attributes.add(new Attribute("Dim " + i));
            }
            ArrayList<String> classLabels = new ArrayList<>(List.of("0", "1"));
            attributes.add(new Attribute("class", classLabels));
            datasetProperties = new Instances("elec", attributes, 0);
            datasetProperties.setClassIndex(example.getAttributes().length);
        }

        double[] attributes = example.getAttributes();
        double[] classValue = new double[] {example.getMappedClass()};
        double[] attributesWithClass = ArrayUtils.addAll(attributes, classValue);
        double weight = 1;
        Instance instance = new DenseInstance(weight, attributesWithClass);
        instance.setDataset(datasetProperties);

        return instance;
    }
}
