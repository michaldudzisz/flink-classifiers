package flinkClassifiersTesting.classifiers.cand;

import org.apache.flink.api.java.tuple.Tuple2;

import java.util.List;

public class CandClassifierFields {

    protected CandClassifierFields() {}

    public final static String BEST_MLP_NAME = "bestMLPName"; // name: for example L1_N8_SGD_0.50000
    public final static String MLP_LOSSES = "MLPLosses";

    public static String makeMlpLossesString(List<Tuple2<String, Double>> losses) {
        StringBuilder s = new StringBuilder();
        if (!losses.isEmpty()) {
            s = new StringBuilder(losses.get(0).f0 + ":" + losses.get(0).f1); // "nn_name:loss_value"
        }

        for (int i = 1; i < losses.size(); i++) {
            s.append(";").append(losses.get(i).f0).append(":").append(losses.get(i).f1); // "nn_name:loss_value"
        }

        return s.toString();
    }
}
