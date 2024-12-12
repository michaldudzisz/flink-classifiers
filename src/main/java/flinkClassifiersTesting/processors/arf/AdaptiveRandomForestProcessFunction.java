package flinkClassifiersTesting.processors.arf;

import flinkClassifiersTesting.classifiers.arf.AdaptiveRandomForestAdapter;
import flinkClassifiersTesting.classifiers.base.BaseClassifierFields;
import flinkClassifiersTesting.processors.base.BaseProcessFunctionClassifyAndTrain;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;

import java.util.List;
import java.util.Map;


public abstract class AdaptiveRandomForestProcessFunction extends BaseProcessFunctionClassifyAndTrain<AdaptiveRandomForestAdapter> {

    public AdaptiveRandomForestProcessFunction(String name, String dataset, long bootstrapSamplesLimit, Map<String, Integer> encoder) {
        super(name, dataset, bootstrapSamplesLimit, encoder);
    }

    @Override
    protected void registerClassifier() {
        TypeInformation<AdaptiveRandomForestAdapter> classifierInfo = TypeInformation.of(new TypeHint<>() {
        });
        classifierState = getRuntimeContext().getState(new ValueStateDescriptor<>("AdaptiveRandomForest", classifierInfo));
    }

    @Override
    public List<String> csvColumnsHeader() {
        return List.of(
                BaseClassifierFields.CLASSIFICATION_DURATION,
                BaseClassifierFields.TRAINING_DURATION
        );
    }

}
