package flinkClassifiersTesting.processors.atnn;

import flinkClassifiersTesting.classifiers.arf.AdaptiveRandomForestAdapter;
import flinkClassifiersTesting.classifiers.atnn.Atnn;
import flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields;
import flinkClassifiersTesting.classifiers.base.BaseClassifierFields;
import flinkClassifiersTesting.processors.base.BaseProcessFunctionClassifyAndTrain;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;

import java.util.List;
import java.util.Map;

public abstract class AtnnProcessFunction extends BaseProcessFunctionClassifyAndTrain<Atnn> {

    public AtnnProcessFunction(String name, String dataset, long bootstrapSamplesLimit, Map<String, Integer> encoder) {
        super(name, dataset, bootstrapSamplesLimit, encoder);
    }

    @Override
    protected void registerClassifier() {
        TypeInformation<Atnn> classifierInfo = TypeInformation.of(new TypeHint<>() {
        });
        classifierState = getRuntimeContext().getState(new ValueStateDescriptor<>("Atnn", classifierInfo));
    }

    @Override
    public List<String> csvColumnsHeader() {
        return List.of(
                AtnnClassifierFields.BRANCH_STRUCTURE,
                AtnnClassifierFields.DRIFT_STATUS,
                BaseClassifierFields.CLASSIFICATION_DURATION,
                BaseClassifierFields.TRAINING_DURATION
        );
    }

}
