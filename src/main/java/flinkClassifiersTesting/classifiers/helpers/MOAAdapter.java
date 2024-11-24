package flinkClassifiersTesting.classifiers.helpers;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import flinkClassifiersTesting.inputs.Example;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MOAAdapter {
    public static Instance mapExampleToInstance(Example example, Instances datasetProperties) {
        if (datasetProperties == null) {
            throw new RuntimeException("Dataset properties not set, cannot properly create MOA instance");
        }

        double[] attributes = example.getAttributes();
        double[] classValue = new double[] {example.getMappedClass()};
        double[] attributesWithClass = ArrayUtils.addAll(attributes, classValue);
        double weight = 1;
        Instance instance = new DenseInstance(weight, attributesWithClass);
        instance.setDataset(datasetProperties);

        return instance;
    }
}
