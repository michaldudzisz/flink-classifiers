package flinkClassifiersTesting.classifiers.arf;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import flinkClassifiersTesting.classifiers.base.BaseClassifierClassifyAndTrain;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.arf.AdaptiveRandomForestClassifierParams;
import moa.classifiers.meta.AdaptiveRandomForest;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static flinkClassifiersTesting.classifiers.helpers.MOAAdapter.mapExampleToInstance;


public class AdaptiveRandomForestAdapter extends BaseClassifierClassifyAndTrain {
    private final AdaptiveRandomForest arf;
    private Instances datasetProperties = null;

    public AdaptiveRandomForestAdapter(
            int classNumber,
            int attributesNumber,
            String datasetName,
            Map<String, Integer> classEncoder,
            AdaptiveRandomForestClassifierParams params
    ) {
//        System.setProperty("ai.djl.pytorch.native_helper", "ai.djl.pytorch.jni.NativeHelper");

        arf = new AdaptiveRandomForest();
        arf.ensembleSizeOption.setValue(params.ensembleSize);
        arf.lambdaOption.setValue(params.lambda);
        arf.mFeaturesModeOption.setChosenIndex(3); // jak w artykule - 3 - percentage
        arf.mFeaturesPerTreeSizeOption.setValue(params.featurePercentageToSplit);

        // todo zobaczyc czy sa opcje odpalania w parallel jak w artykule, bo pisza, ze mozna
        // todo defaulty adwina sa inne niz w artykule

        setDatasetProperties(classNumber, attributesNumber, datasetName, classEncoder);
        arf.prepareForUse();
    }

    @Override
    protected ArrayList<Tuple2<String, Object>> trainImplementation(Example example, int predictedClass, ArrayList<Tuple2<String, Object>> performances) {
        Instance moaInstance = mapExampleToInstance(example, datasetProperties);
        arf.trainOnInstanceImpl(moaInstance);
        return performances;
    }

    @Override
    protected Tuple2<Integer, ArrayList<Tuple2<String, Object>>> classifyImplementation(Example example) {
        Instance moaInstance = mapExampleToInstance(example, datasetProperties);
        double[] votesForEachClass = arf.getVotesForInstance(moaInstance);

        int predictedClass = 0;
        try {
            predictedClass = IntStream.range(0, votesForEachClass.length).reduce((i, j) -> votesForEachClass[i] > votesForEachClass[j] ? i : j).getAsInt();
        } catch (Exception e) { // todo z jakiegoś powodu czasem arf.getVotesForInstance(moaInstance); zwraca 0, obejście tego to wybór np losowej klasy
            System.out.println("\n****************");
            System.out.println("AdaptiveRandomForestAdapter::classifyImplementation: " + e.getMessage());
            System.out.println("votesForEachClass: " + Arrays.toString(votesForEachClass));
            System.out.println("\n****************");
        }

        ArrayList<Tuple2<String, Object>> performances = new ArrayList<>();
        return new Tuple2<>(predictedClass, performances);
    }

    @Override
    public String generateClassifierParams() {
        return "lol po co to";
    }

    @Override
    public void bootstrapTrainImplementation(Example example) {
        // done this for Cand, do it here
        arf.getVotesForInstance(mapExampleToInstance(example, datasetProperties));
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
}
