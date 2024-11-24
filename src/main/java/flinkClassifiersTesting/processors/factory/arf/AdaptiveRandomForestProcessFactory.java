package flinkClassifiersTesting.processors.factory.arf;

import flinkClassifiersTesting.classifiers.arf.AdaptiveRandomForestAdapter;
import flinkClassifiersTesting.processors.arf.AdaptiveRandomForestProcessFunction;
import flinkClassifiersTesting.processors.factory.ProcessFunctionsFromParametersFactory;

import java.util.List;
import java.util.Map;

public class AdaptiveRandomForestProcessFactory {
    public static ProcessFunctionsFromParametersFactory<AdaptiveRandomForestClassifierParams, AdaptiveRandomForestAdapter, AdaptiveRandomForestProcessFunction> arf(List<AdaptiveRandomForestClassifierParams> parameters) {
        String name = "AdaptiveRandomForest";

        return new ProcessFunctionsFromParametersFactory<>(name, parameters) {
            @Override
            public AdaptiveRandomForestProcessFunction createProcessFunction(AdaptiveRandomForestClassifierParams params, int classNumber, int attributesNumber, String dataset, long samplesLimit, Map<String, Integer> classEncoder) {
                return new AdaptiveRandomForestProcessFunction(name, dataset, samplesLimit, classEncoder) {
                    @Override
                    protected AdaptiveRandomForestAdapter createClassifier() {
                        return new AdaptiveRandomForestAdapter(classNumber, attributesNumber, dataset, classEncoder, params);
                    }
                };
            }
        };
    }
}
