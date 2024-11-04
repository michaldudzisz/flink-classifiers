package flinkClassifiersTesting.processors.cand;

import flinkClassifiersTesting.classifiers.base.BaseClassifierFields;
import flinkClassifiersTesting.classifiers.cand.Cand;
import flinkClassifiersTesting.processors.base.BaseProcessFunctionClassifyAndTrain;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;

import java.util.List;
import java.util.Map;

public abstract class CandProcessFunction extends BaseProcessFunctionClassifyAndTrain<Cand> {

    public CandProcessFunction(String name, String dataset, long bootstrapSamplesLimit, Map<String, Integer> encoder) {
        super(name, dataset, bootstrapSamplesLimit, encoder);
    }

    @Override
    protected void registerClassifier() {
        TypeInformation<Cand> classifierInfo = TypeInformation.of(new TypeHint<>() {
        });
        classifierState = getRuntimeContext().getState(new ValueStateDescriptor<>("cand", classifierInfo));
    }

    @Override
    public List<String> csvColumnsHeader() {
        return List.of(BaseClassifierFields.CLASSIFICATION_DURATION, BaseClassifierFields.TRAINING_DURATION);
    }

}
