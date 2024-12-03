package flinkClassifiersTesting.classifiers.cand;

import com.yahoo.labs.samoa.instances.*;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.cand.CandClassifierParams;
import moa.classifiers.deeplearning.CAND;
import moa.classifiers.deeplearning.MLP;
import org.apache.commons.lang3.reflect.FieldUtils;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static flinkClassifiersTesting.classifiers.cand.CandClassifierFields.*;
import static flinkClassifiersTesting.classifiers.helpers.MOAAdapter.mapExampleToInstance;

/*
This class is a flinkClassifiersTesting wrapper for moa.classifiers.deeplearning.CAND
*/
public class Cand extends BaseClassifierClassifyAndTrain {

    private final CAND cand;
    private Instances datasetProperties = null;
    private boolean logAllMlpLosses = false;

    int samplesProcessed = 0;

    public Cand(
            int classNumber,
            int attributesNumber,
            String datasetName,
            Map<String, Integer> classEncoder,
            CandClassifierParams params
    ) {
        System.setProperty("ai.djl.pytorch.native_helper", "ai.djl.pytorch.jni.NativeHelper");

        cand = new CAND();
        setDatasetProperties(classNumber, attributesNumber, datasetName, classEncoder);

        // configured parameters
        cand.largerPool.setChosenIndex(params.pSize == CandClassifierParams.PoolSize.P10 ? 0 : 1); // |P|
        cand.numberOfMLPsToTrainOption.setValue(params.mSize); // |M|
        cand.backPropLossThreshold.setValue(params.backpropagationThreshold); // Math.pow(10, 10);
        cand.numberOfInstancesToTrainAllMLPsAtStartOption.setValue(500); // 10% of elec

        // output files
        if (params.votingStatsFile != null)
            cand.votesDumpFileName.setValue(params.votingStatsFile);
//        cand.statsDumpFileName.setValue("cand_stats.txt");

        if (params.logAllMlpLosses)
            logAllMlpLosses = true;

        cand.miniBatchSize.setValue(1); // 1 - incrementally
        cand.numberOfLayersInEachMLP.setValue(1); // 1
        cand.numberOfInstancesToTrainAllMLPsAtStartOption.setValue(1_000); // 1000
        cand.useOneHotEncode.setValue(true); // t
        cand.useNormalization.setValue(true); // t
        cand.deviceTypeOption.setChosenIndex(1); // CPU
    }

    @Override
    protected ArrayList<Tuple2<String, Object>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Object>> performances) {
        // przyjmuje i zwraca performance
        Instance moaInstance = mapExampleToInstance(example, datasetProperties);
        cand.trainOnInstanceImpl(moaInstance);
        return performances;
    }

    @Override
    protected Tuple2<Integer, ArrayList<Tuple2<String, Object>>> classifyImplementation(Example example) {
        Instance moaInstance = mapExampleToInstance(example, datasetProperties);
        double[] votesForEachClass = cand.getVotesForInstance(moaInstance);
        int predictedClass = IntStream.range(0, votesForEachClass.length).reduce((i, j) -> votesForEachClass[i] > votesForEachClass[j] ? i : j).getAsInt();

        samplesProcessed++;
        System.out.println("i: " + samplesProcessed);

        ArrayList<Tuple2<String, Object>> performances = new ArrayList<>();
        performances.add(getBestMLPParams());

        if (logAllMlpLosses)
            performances.add(getAllMLPLosses());
        else
            performances.add(emptyMLPLosses());

        return new Tuple2<>(predictedClass, performances);
    }

    @Override
    public String generateClassifierParams() {
        return "parametry";
    }

    @Override
    public void bootstrapTrainImplementation(Example example) {
        // we do the line below so that MOA's Cand can increment samplesSeen counter
        cand.getVotesForInstance(mapExampleToInstance(example, datasetProperties));
        trainImplementation(example, example.getMappedClass(), null);
    }

    private void setDatasetProperties(
            int classNumber,
            int attributesNumber,
            String datasetName,
            Map<String, Integer> classEncoder
    ) {
        ArrayList<Attribute> attributes = new ArrayList<>(attributesNumber);
        for (int i = 0; i < attributesNumber; i++) {
            attributes.add(new Attribute("Dim " + i));
        }
        // let's use this for now this way. The drawback is that mapping may not be performed as in classEncoder.
        List<String> classLabels = new ArrayList<>(classEncoder.keySet());
        attributes.add(new Attribute("class", classLabels));
        datasetProperties = new Instances(datasetName, attributes, 0);
        datasetProperties.setClassIndex(attributesNumber);
    }

    private MLP[] getAllMLPs() {
        MLP[] nns;
        try {
            nns = (MLP[]) FieldUtils.readField(cand, "nn", true);
        } catch (IllegalAccessException e) {
            throw new RuntimeException("not allowed to read Cand's private field nn.", e);
        }
        return nns;
    }

    private Tuple2<String, Object> getBestMLPParams() {
        MLP[] nns = getAllMLPs();
        MLP bestMLP = findCurrentlyBestMLP(nns);
        return new Tuple2<>(BEST_MLP_NAME, bestMLP.modelName);
    }

    private MLP findCurrentlyBestMLP(MLP[] nns) {
        return Arrays.stream(nns).min(Comparator.comparingDouble(MLP::getLossEstimation)).get();
    }

    private Tuple2<String, Object> getAllMLPLosses() {
        MLP[] nns = getAllMLPs();
        List<Tuple2<String, Double>> losses = Arrays.stream(nns)
                .map(nn -> new Tuple2<>(nn.modelName, nn.getLossEstimation()))
                .collect(Collectors.toList());
        return new Tuple2<>(CandClassifierFields.MLP_LOSSES, makeMlpLossesString(losses));
    }

    private Tuple2<String, Object> emptyMLPLosses() {
        return new Tuple2<>(CandClassifierFields.MLP_LOSSES, "nd");
    }
}
