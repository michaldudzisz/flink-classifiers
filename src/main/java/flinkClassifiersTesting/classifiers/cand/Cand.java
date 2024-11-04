package flinkClassifiersTesting.classifiers.cand;

import com.yahoo.labs.samoa.instances.*;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import moa.classifiers.deeplearning.CAND;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/*
This class is a flinkClassifiersTesting wrapper for moa.classifiers.deeplearning.CAND
 */
public class Cand extends BaseClassifierClassifyAndTrain {

    private final CAND cand;
    private Instances datasetProperties = null;

    public Cand() {
        cand = new CAND();
        cand.largerPool.setChosenIndex(1);
        cand.numberOfMLPsToTrainOption.setValue(2);
        cand.numberOfLayersInEachMLP.setValue(1);
        cand.numberOfInstancesToTrainAllMLPsAtStartOption.setValue(100);
        cand.miniBatchSize.setValue(1);
        cand.useOneHotEncode.setValue(true);
        cand.useNormalization.setValue(true);
        cand.backPropLossThreshold.setValue(0.3);
        cand.deviceTypeOption.setChosenIndex(1);
    }

    @Override
    protected ArrayList<Tuple2<String, Long>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Long>> performances) {
        // przyjmuje i zwraca performance
        // może się tu odbywać normalizacja wejścia
        Instance moaInstance = mapExampleToInstance(example);
        cand.trainOnInstanceImpl(moaInstance); // todo czy normalizacja jest zrobiona już w środku
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
            ArrayList<String> classLabels = new ArrayList<>(List.of("0", "1")); // todo może tak?
            attributes.add(new Attribute("class", classLabels));
            datasetProperties = new Instances("elec", attributes, 0);
            datasetProperties.setClassIndex(example.getAttributes().length);
//            cand.setModelContext(new InstancesHeader(this.datasetProperties)); // todo nie działa
        }

        double[] attributes = example.getAttributes();
        double[] classValue = new double[] {example.getMappedClass()};
        double[] attributesWithClass = ArrayUtils.addAll(attributes, classValue);
        double weight = 1;
        Instance instance = new DenseInstance(weight, attributesWithClass);
        instance.setDataset(datasetProperties);

//        InstanceImpl xd = new InstanceImpl(weight, example.getAttributes());
//
//        System.out.println("xd.dataset()" + xd.dataset()); // todo TO JEST NULL
//        System.out.println("xd.numClasses()" + xd.numClasses());

        return instance;
    }
}
