package flinkClassifiersTesting.processors.atnn;

import flinkClassifiersTesting.classifiers.atnn.AtnnClassifierFields;
import flinkClassifiersTesting.classifiers.atnn.EAtnn2;
import flinkClassifiersTesting.classifiers.base.BaseClassifierFields;
import flinkClassifiersTesting.processors.base.BaseProcessFunctionClassifyAndTrain;
import java.util.List;
import java.util.Map;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.TypeHint;
import org.apache.flink.api.common.typeinfo.TypeInformation;

public abstract class EAtnn2ProcessFunction extends BaseProcessFunctionClassifyAndTrain<EAtnn2> {

    public EAtnn2ProcessFunction(String name, String dataset, long bootstrapSamplesLimit, Map<String, Integer> encoder) {
        super(name, dataset, bootstrapSamplesLimit, encoder);
    }

    @Override
    protected void registerClassifier() {
        TypeInformation<EAtnn2> classifierInfo = TypeInformation.of(new TypeHint<>() {
        });
        classifierState = getRuntimeContext().getState(new ValueStateDescriptor<>("EAtnn2", classifierInfo));
    }

    @Override
    public List<String> csvColumnsHeader() {
        return List.of(
                BaseClassifierFields.CLASSIFICATION_DURATION,
                AtnnClassifierFields.BRANCH_STRUCTURE,
                AtnnClassifierFields.DRIFT_STATUS,
                AtnnClassifierFields.CLONED_NODES_TO_TRAIN,
                AtnnClassifierFields.EMPTY_NODES_TO_TRAIN,
                AtnnClassifierFields.NORMAL_NODES_TO_TRAIN,
                BaseClassifierFields.TRAINING_DURATION
        );
    }

}
