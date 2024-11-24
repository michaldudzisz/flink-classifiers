package flinkClassifiersTesting.classifiers.driftDetection;

import org.apache.flink.api.java.tuple.Tuple2;
import flinkClassifiersTesting.classifiers.bstHoeffding.standard.BstHoeffdingTree;
import flinkClassifiersTesting.classifiers.hoeffding.Node;
import flinkClassifiersTesting.classifiers.hoeffding.NodeStatistics;
import flinkClassifiersTesting.classifiers.hoeffding.StatisticsBuilderInterface;
import flinkClassifiersTesting.inputs.Example;

import java.util.ArrayList;

public abstract class WindowedDetectorBstHoeffdingTree<N_S extends NodeStatistics, B extends StatisticsBuilderInterface<N_S>> extends BstHoeffdingTree<N_S, B> {
    private final WindowedDetector driftDetector;
    private Node<N_S, B> rootSubstitute;
    private int updateMaxAccuracyEachSamples;
    private long nSamplesSinceSubstituteTrainingStart;
    private int bootstrapSubstituteTraining;

    public WindowedDetectorBstHoeffdingTree(int classesNumber, double delta, int attributesNumber, double tau, long nMin, B statisticsBuilder, int windowSize, double warningFrac, double driftFrac, int updateMaxAccuracyEachSamples, int bootstrapSubstituteTraining) {
        super(classesNumber, delta, attributesNumber, tau, nMin, statisticsBuilder);
        this.updateMaxAccuracyEachSamples = updateMaxAccuracyEachSamples;
        this.bootstrapSubstituteTraining = bootstrapSubstituteTraining;
        this.driftDetector = new WindowedDetector(windowSize, warningFrac, driftFrac);
        this.rootSubstitute = null;
        this.nSamplesSinceSubstituteTrainingStart = 0;
    }

    @Override
    protected Tuple2<Integer, ArrayList<Tuple2<String, Object>>> classifyImplementation(Example example, ArrayList<Tuple2<String, Object>> performances) throws RuntimeException {
        Tuple2<Integer, ArrayList<Tuple2<String, Object>>> classifyResults = super.classifyImplementation(example, performances);

        driftDetector.updateWindow(example.getMappedClass() == classifyResults.f0);

        if (rootSubstitute == null && driftDetector.warningDetected() && nSamplesSinceSubstituteTrainingStart == 0) { //kinda naive approach - better to test in window of k samples between warning and current sample - even to make sure sub had enough bootstrap time
            rootSubstitute = new Node<>(statisticsBuilder, null);
            classifyResults.f1.add(Tuple2.of(WindowedDetectorHoeffdingTreeFields.SUBSTITUTE_TRAINING_BEGAN, 1L));
            classifyResults.f1.add(Tuple2.of(WindowedDetectorHoeffdingTreeFields.REPLACED_CLASSIFIER, 0L));
        } else {
            classifyResults.f1.add(Tuple2.of(WindowedDetectorHoeffdingTreeFields.SUBSTITUTE_TRAINING_BEGAN, 0L));

            if (rootSubstitute != null && driftDetector.driftDetected() && nSamplesSinceSubstituteTrainingStart >= bootstrapSubstituteTraining) {
                root = rootSubstitute;
                rootSubstitute = null;
                n = 0L;
                nSamplesSinceSubstituteTrainingStart = 0L;
                driftDetector.clearWindow();
                classifyResults.f1.add(Tuple2.of(WindowedDetectorHoeffdingTreeFields.REPLACED_CLASSIFIER, 1L));
            } else classifyResults.f1.add(Tuple2.of(WindowedDetectorHoeffdingTreeFields.REPLACED_CLASSIFIER, 0L));
        }

        return classifyResults;
    }

    @Override
    protected ArrayList<Tuple2<String, Object>> trainImplementation(Example example) throws RuntimeException {
        ArrayList<Tuple2<String, Object>> trainResults = super.trainImplementation(example);

        if (rootSubstitute != null) {
            nSamplesSinceSubstituteTrainingStart++;
            Node<N_S, B> substituteLeaf = getLeaf(example, rootSubstitute);
            updateLeaf(example, substituteLeaf, new ArrayList<>());
        }

        if (n % updateMaxAccuracyEachSamples == 0)
            driftDetector.updateMaxAccuracy();

        return trainResults;
    }
}
