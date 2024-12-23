package flinkClassifiersTesting.processors.factory.atnn;

import flinkClassifiersTesting.classifiers.atnn.Atnn;
import flinkClassifiersTesting.classifiers.atnn.EAtnn;
import flinkClassifiersTesting.processors.atnn.AtnnProcessFunction;
import flinkClassifiersTesting.processors.atnn.EAtnnProcessFunction;
import flinkClassifiersTesting.processors.factory.ProcessFunctionsFromParametersFactory;

import java.util.List;
import java.util.Map;

public class EAtnnProcessFactory {
    public static ProcessFunctionsFromParametersFactory<AtnnClassifierParams, EAtnn, EAtnnProcessFunction> atnn(List<AtnnClassifierParams> parameters) {
        String name = "eatnn";

        return new ProcessFunctionsFromParametersFactory<>(name, parameters) {
            @Override
            public EAtnnProcessFunction createProcessFunction(AtnnClassifierParams params, int classNumber, int attributesNumber, String dataset, long samplesLimit, Map<String, Integer> classEncoder) {
                return new EAtnnProcessFunction(name, dataset, samplesLimit, classEncoder) {
                    @Override
                    protected EAtnn createClassifier() {
                        return new EAtnn(classEncoder);
                    }
                };
            }
        };
    }
}
