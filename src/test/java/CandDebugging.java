import flinkClassifiersTesting.classifiers.cand.Cand;
import flinkClassifiersTesting.inputs.Example;
import flinkClassifiersTesting.processors.factory.vfdt.VfdtClassifierParams;
import org.junit.Test;

import static org.apache.flink.types.PojoTestUtils.assertSerializedAsPojo;

public class CandDebugging {
    @Test
    public void candRuns() {
        // given
        Cand cand = new Cand();

        // and
        int mappedClass = 0; // todo 5 nie działa, tylko 0 działa
        double[] attributes = {2., 1.};
        Example example = new Example(mappedClass, attributes);

        // then
        cand.bootstrapTrainImplementation(example);


        assertSerializedAsPojo(VfdtClassifierParams.class);
    }
}

