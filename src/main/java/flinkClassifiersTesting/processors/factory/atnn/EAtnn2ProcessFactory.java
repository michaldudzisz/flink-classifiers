package flinkClassifiersTesting.processors.factory.atnn;

import flinkClassifiersTesting.classifiers.atnn.EAtnn2;
import flinkClassifiersTesting.processors.atnn.EAtnn2ProcessFunction;
import flinkClassifiersTesting.processors.factory.ProcessFunctionsFromParametersFactory;
import java.util.List;
import java.util.Map;

public class EAtnn2ProcessFactory {
    public static ProcessFunctionsFromParametersFactory<AtnnClassifierParams, EAtnn2, EAtnn2ProcessFunction> eatnn2(List<AtnnClassifierParams> parameters) {
        String name = "eatnn2";

        return new ProcessFunctionsFromParametersFactory<>(name, parameters) {
            @Override
            public EAtnn2ProcessFunction createProcessFunction(AtnnClassifierParams params, int classNumber, int attributesNumber, String dataset, long samplesLimit, Map<String, Integer> classEncoder) {
                return new EAtnn2ProcessFunction(name, dataset, samplesLimit, classEncoder) {
                    @Override
                    protected EAtnn2 createClassifier() {
                        return new EAtnn2(classEncoder, params);
                    }
                };
            }
        };
    }
}
