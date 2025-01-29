/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package flinkClassifiersTesting;

import flinkClassifiersTesting.processors.factory.FlinkProcessFactory;
import flinkClassifiersTesting.processors.factory.arf.AdaptiveRandomForestClassifierParams;
import flinkClassifiersTesting.processors.factory.arf.AdaptiveRandomForestProcessFactory;
import flinkClassifiersTesting.processors.factory.atnn.AtnnClassifierParams;
import flinkClassifiersTesting.processors.factory.atnn.AtnnProcessFactory;
import flinkClassifiersTesting.processors.factory.atnn.EAtnn2ProcessFactory;
import flinkClassifiersTesting.processors.factory.atnn.EAtnnProcessFactory;
import flinkClassifiersTesting.processors.factory.cand.CandClassifierParams;
import flinkClassifiersTesting.processors.factory.cand.CandProcessFactory;
import flinkClassifiersTesting.processors.factory.vfdt.VfdtClassifierParams;
import flinkClassifiersTesting.processors.factory.vfdt.VfdtProcessFactory;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;

public class DataStreamJob {
    private static String getBaseDirectory() throws URISyntaxException {
        URI jarUri = DataStreamJob.class
                .getProtectionDomain()
                .getCodeSource()
                .getLocation()
                .toURI();
        File file = new File(jarUri);

        File parentDirectory = file.getParentFile();

        File parentOfParentDirectory = parentDirectory.getParentFile();

        return parentOfParentDirectory.getAbsolutePath();
    }

    public static void main(String[] args) throws Exception {
        String basePath = getBaseDirectory();
//        String dataset = "elec-maly";
//        String dataset = "elec";
        String dataset = "mnist_abrupt_atnn_like";
//        String dataset = "sea_abr";
//        String dataset = "sea_grad";
//        String dataset = "mnist_grad";
//        String dataset = "elec-malutki";
//        String dataset = "mnist_grad_mniejszy";
//        String dataset = "mnist_grad_malutki";
//        String dataset = "mnist_grad_powolny_4x_szybszy";
//        String dataset = "mnist_grad_atnnowy_prosty";
//        String dataset = "sea_grad";
//        String dataset = "incremental_drift_synth_attr2_speed5.0";
        String datasetPath = basePath + "/datasets/" + dataset + ".csv";
//        long bootstrapSamplesLimit = 500L;
        long bootstrapSamplesLimit = 0L; // 2000L;// 5_000L;

//        String[] datasets = {
//                "elec",
//                "weather_norm",
//                "covtype_norm",
//                "sea_abr",
//                "sea_inc",
//                "mnist_abrupt_atnn_like",
//                "mnist_inc_20k_0.1x",
//                "mnist_inc_20k_0.5x",
//                "mnist_inc_20k_1x",
//                "mnist_inc_20k_2x",
//        };
//
//        List<CandClassifierParams> candParams = List.of(
//                new CandClassifierParams(CandClassifierParams.PoolSize.P30, 30, false),
//                new CandClassifierParams(CandClassifierParams.PoolSize.P10, 10, false)
//        );
//        for (String d : datasets) {
//            datasetPath = basePath + "/datasets/" + d + ".csv";
//            FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, CandProcessFactory.cand(candParams));
//        }



        String[] datasets2 = {
                "elec", // 1 min
                "weather_norm", // 1 min
                "covtype_norm", // 7 min
                "sea_abr", // 2 min
                "sea_inc", // 3 min
                "mnist_abrupt_atnn_like", // 13 min
                "mnist_inc_20k_0.1x", // 9 min
                "mnist_inc_20k_0.5x", // 9 min
                "mnist_inc_20k_1x", // 9 min
                "mnist_inc_20k_2x", // 9 min
        };


        for (int i = 0; i < 3; i++) {
            List<CandClassifierParams> candParams2 = List.of(
                    new CandClassifierParams(CandClassifierParams.PoolSize.P30, 10, 0.2, i),
                    new CandClassifierParams(CandClassifierParams.PoolSize.P30, 10, 0.6, i),
                    new CandClassifierParams(CandClassifierParams.PoolSize.P30, 10, 1.0, i)
            ); // 3h na wszystkie 3
            for (String d : datasets2) {
                datasetPath = basePath + "/datasets/" + d + ".csv";
                FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, CandProcessFactory.cand(candParams2));
            }
        }





        // ATNN artykułowo: 0.02, 256, 5000
//        List<AtnnClassifierParams> atnnParams = List.of(
//                new AtnnClassifierParams(2E-2, 256, 0),
//                new AtnnClassifierParams(2E-2, 256, 50),
//                new AtnnClassifierParams(2E-2, 256, 500),
//                new AtnnClassifierParams(2E-2, 256, 5000),
//                new AtnnClassifierParams(2E-2, 256, 50000)
//        );
//        FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, AtnnProcessFactory.atnn(atnnParams));






//        List<VfdtClassifierParams> vfdtParams = List.of(new VfdtClassifierParams(0.2, 0.1, 10));
//        FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, VfdtProcessFactory.vfdt(vfdtParams));

//        List<AdaptiveRandomForestClassifierParams> arfParams = List.of(
//                new AdaptiveRandomForestClassifierParams(30, 6.0f, 60)
//        );
//        FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, AdaptiveRandomForestProcessFactory.arf(arfParams));
//

//        String[] datasets = {
//                "incremental_drift_synth_attr2_speed0.05_len20000",
//                "incremental_drift_synth_attr2_speed0.2_len20000",
//                "incremental_drift_synth_attr2_speed0.5_len20000",
//                "incremental_drift_synth_attr2_speed1.0_len20000",
//                "incremental_drift_synth_attr2_speed2.0_len20000",
//                "incremental_drift_synth_attr2_speed5.0_len20000",
//                "incremental_drift_synth_attr2_speed10.0_len20000",
//        };


        String[] artificialDatasets = {
                "incremental_drift_synth_attr2_speed0.05_len20000",
                "incremental_drift_synth_attr2_speed0.2_len20000",
                "incremental_drift_synth_attr2_speed0.5_len20000",
                "incremental_drift_synth_attr2_speed1.0_len20000",
                "incremental_drift_synth_attr2_speed2.0_len20000",
                "incremental_drift_synth_attr2_speed5.0_len20000",
                "incremental_drift_synth_attr2_speed10.0_len20000",
                "gradual_drift_synth_attr2_a50_len20000",
                "gradual_drift_synth_attr2_a200_len20000",
                "gradual_drift_synth_attr2_a500_len20000",
                "gradual_drift_synth_attr2_a1000_len20000",
                "gradual_drift_synth_attr2_a1500_len20000",
                "gradual_drift_synth_attr2_a2000_len20000",
        };
//
//        List<AtnnClassifierParams> atnnParams = List.of(
//                new AtnnClassifierParams(2E-2, 256, 5000, 0)
//        );

//        for (int i = 0; i < 3; i++) {
//            for (String d : artificialDatasets) {
//                datasetPath = basePath + "/datasets/" + d + ".csv";
//                FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, EAtnn2ProcessFactory.eatnn2(atnnParams));
//                FlinkProcessFactory.runJobs(datasetPath, bootstrapSamplesLimit, AtnnProcessFactory.atnn(atnnParams));
//            }
//        }





    }
}
