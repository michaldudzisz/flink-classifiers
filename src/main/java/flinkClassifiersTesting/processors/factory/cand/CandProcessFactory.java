package flinkClassifiersTesting.processors.factory.cand;

import flinkClassifiersTesting.classifiers.cand.Cand;
import flinkClassifiersTesting.processors.cand.CandProcessFunction;
import flinkClassifiersTesting.processors.factory.ProcessFunctionsFromParametersFactory;

import java.util.List;
import java.util.Map;

public class CandProcessFactory {
    public static ProcessFunctionsFromParametersFactory<CandClassifierParams, Cand, CandProcessFunction> cand(List<CandClassifierParams> parameters) {
        String name = "cand";

        return new ProcessFunctionsFromParametersFactory<>(name, parameters) {
            @Override
            public CandProcessFunction createProcessFunction(CandClassifierParams params, int classNumber, int attributesNumber, String dataset, long samplesLimit, Map<String, Integer> classEncoder) {
                return new CandProcessFunction(name, dataset, samplesLimit, classEncoder) {
                    @Override
                    protected Cand createClassifier() {
                        return new Cand(classNumber, attributesNumber, dataset, classEncoder, params);
                    }
                };
            }
        };
    }
}
