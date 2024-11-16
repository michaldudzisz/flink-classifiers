package flinkClassifiersTesting.classifiers.cand;

import com.yahoo.labs.samoa.instances.*;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.cand.CandClassifierParams;
import moa.classifiers.deeplearning.CAND;
import moa.classifiers.deeplearning.MLP;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.apache.flink.api.java.tuple.Tuple2;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import static flinkClassifiersTesting.classifiers.cand.CandClassifierFields.*;

/*
This class is a flinkClassifiersTesting wrapper for moa.classifiers.deeplearning.CAND
*/
public class Cand extends BaseClassifierClassifyAndTrain {

    private CAND cand = null;
    private Instances datasetProperties = null;

    public Cand(CandClassifierParams params) {
        System.setProperty("ai.djl.pytorch.native_helper", "ai.djl.pytorch.jni.NativeHelper");

        cand = new CAND();

        // configured parameters
        cand.largerPool.setChosenIndex(params.pSize == CandClassifierParams.PoolSize.P10 ? 0 : 1); // |P|
        cand.numberOfMLPsToTrainOption.setValue(params.mSize); // |M|
        cand.backPropLossThreshold.setValue(params.backpropagationThreshold); // Math.pow(10, 10);
        cand.numberOfInstancesToTrainAllMLPsAtStartOption.setValue(500); // 10% of elec

        // output files
        if (params.votingStatsFile != null)
            cand.votesDumpFileName.setValue(params.votingStatsFile);
//        cand.statsDumpFileName.setValue("cand_stats.txt");


        cand.miniBatchSize.setValue(1); // incrementally
        cand.numberOfLayersInEachMLP.setValue(1); // 1
        cand.numberOfInstancesToTrainAllMLPsAtStartOption.setValue(1_000); // 1000
        cand.useOneHotEncode.setValue(true); // t
        cand.useNormalization.setValue(true); // t
        cand.deviceTypeOption.setChosenIndex(1); // CPU
    }

    @Override
    protected ArrayList<Tuple2<String, Long>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Long>> performances) {
        // przyjmuje i zwraca performance
        Instance moaInstance = mapExampleToInstance(example);
        cand.trainOnInstanceImpl(moaInstance);
        return performances;
    }

    @Override
    protected Tuple2<Integer, ArrayList<Tuple2<String, Long>>> classifyImplementation(Example example) {
        Instance moaInstance = mapExampleToInstance(example);
        double[] votesForEachClass = cand.getVotesForInstance(moaInstance);
        int predictedClass = IntStream.range(0, votesForEachClass.length).reduce((i, j) -> votesForEachClass[i] > votesForEachClass[j] ? i : j).getAsInt();
        ArrayList<Tuple2<String, Long>> performances = getBestMLPParams();

        return new Tuple2<>(predictedClass, performances);
    }

    @Override
    public String generateClassifierParams() {
        return "parametry";
    }

    @Override
    public void bootstrapTrainImplementation(Example example) {
        cand.getVotesForInstance(mapExampleToInstance(example)); // todo we do this so that MOA's Cand can increment samplesSeen counter
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

    private MLP findCurrentlyBestMLP() {
        MLP[] nns;
        try {
            nns = (MLP[]) FieldUtils.readField(cand, "nn", true);
        } catch (IllegalAccessException e) {
            throw new RuntimeException("not allowed to read Cand's private field nn.", e);
        }

        return Arrays.stream(nns).min(Comparator.comparingDouble(MLP::getLossEstimation)).get();
    }

    private ArrayList<Tuple2<String, Long>> getBestMLPParams() {
        MLP bestMLP = findCurrentlyBestMLP();
        ArrayList<Tuple2<String, Long>> params = new ArrayList<>();
        params.add(new Tuple2<>(USED_OPTIMIZER, getOptimizerFromNNName(bestMLP.modelName)));
        params.add(new Tuple2<>(LAYER_SIZE, getHiddenLayerSizeFromNNName(bestMLP.modelName)));
        params.add(new Tuple2<>(LEARNING_RATE, getLearningRateFromNNName(bestMLP.modelName)));
        return params;
    }
}
