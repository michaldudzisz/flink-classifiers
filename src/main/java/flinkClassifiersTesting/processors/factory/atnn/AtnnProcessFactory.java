package flinkClassifiersTesting.processors.factory.atnn;

import flinkClassifiersTesting.classifiers.arf.AdaptiveRandomForestAdapter;
import flinkClassifiersTesting.classifiers.atnn.Atnn;
import flinkClassifiersTesting.processors.arf.AdaptiveRandomForestProcessFunction;
import flinkClassifiersTesting.processors.atnn.AtnnProcessFunction;
import flinkClassifiersTesting.processors.factory.ProcessFunctionsFromParametersFactory;

import java.util.List;
import java.util.Map;

public class AtnnProcessFactory {
    public static ProcessFunctionsFromParametersFactory<AtnnClassifierParams, Atnn, AtnnProcessFunction> atnn(List<AtnnClassifierParams> parameters) {
        String name = "atnn";

        return new ProcessFunctionsFromParametersFactory<>(name, parameters) {
            @Override
            public AtnnProcessFunction createProcessFunction(AtnnClassifierParams params, int classNumber, int attributesNumber, String dataset, long samplesLimit, Map<String, Integer> classEncoder) {
                return new AtnnProcessFunction(name, dataset, samplesLimit, classEncoder) {
                    @Override
                    protected Atnn createClassifier() {
                        return new Atnn(classEncoder);
                    }
                };
            }
        };
    }
}
