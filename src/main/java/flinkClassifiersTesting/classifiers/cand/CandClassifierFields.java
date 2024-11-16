package flinkClassifiersTesting.classifiers.cand;

public class CandClassifierFields {

    protected CandClassifierFields() {}

    public final static String USED_OPTIMIZER = "usedOptimizer";
    public final static String LAYER_SIZE = "hiddenLayerSize";
    public final static String LEARNING_RATE = "lr";

    public final static Long SGD = 0L;
    public final static Long ADAM = 1L;

//    public final static Long N2E8 = 0L;
//    public final static Long N2E9 = 1L;
//    public final static Long N2E10 = 2L;
//
    public final static Long LR5Em1 = 0L;
    public final static Long LR5Em2 = 1L;
    public final static Long LR5Em3 = 2L;
    public final static Long LR5Em4 = 3L;
    public final static Long LR5Em5 = 4L;

    public static Long getHiddenLayerSizeFromNNName(String name) {
        // name: for example L1_N8_SGD_0.50000
        return Long.parseLong(name.split("_")[1].substring(1));
    }

    public static Long getOptimizerFromNNName(String name) {
        // name: for example L1_N8_SGD_0.50000
        String optimizer = name.split("_")[2];
        Long chosenConstant;
        if (optimizer.equals("SGD"))
            chosenConstant = SGD;
        else if (optimizer.equals("ADAM"))
            chosenConstant = ADAM;
        else throw new RuntimeException("Invalid optimizer: " + optimizer);
        System.out.println("getOptimizerFromNNName. Got name: " + name + ". Parsing as optimizer: " + optimizer + ". Chosen constant: " + chosenConstant);
        return chosenConstant;
    }

    public static Long getLearningRateFromNNName(String name) {
        // name: for example L1_N8_SGD_0.50000
        double lr = Double.parseDouble(name.split("_")[3]);
        Long chosenConstant;
        if (lr == 0.5) chosenConstant = LR5Em1;
        else if (lr == 0.05) chosenConstant = LR5Em2;
        else if (lr == 0.005) chosenConstant = LR5Em3;
        else if (lr == 0.0005) chosenConstant = LR5Em4;
        else if (lr == 0.00005) chosenConstant = LR5Em5;
        else throw new RuntimeException("Invalid learning rate: " + lr);
        System.out.println("getLearningRateFromNNName. Got name: " + name + ". Parsing as lr: " + lr + ". Chosen constant: " + chosenConstant);
        return chosenConstant;
    }
}
